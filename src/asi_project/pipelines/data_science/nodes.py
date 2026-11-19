import logging
import shutil
import time
from pathlib import Path

import pandas as pd

from asi_project.config import DatasetContract, ProjectConfig, get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline
from asi_project.tracking import (
    dir_size_mb,
    flatten_metrics,
    get_data_fingerprint,
    get_git_info,
    init_wandb_run,
    measure_prediction_time,
    pickled_size_mb,
)

logger = logging.getLogger(__name__)


def _project() -> tuple[ProjectConfig, DatasetContract]:
    config = load_project_config()
    return config, get_contract(config)


def split_data(df: pd.DataFrame, split_params: dict):
    config, contract = _project()
    X, y = make_xy(df, contract, config.model_features)
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        contract,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, numeric_features: list, categorical_features: list
):
    pipeline = build_baseline_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return metrics


def compute_data_fingerprint(raw_data_path: str) -> dict:
    config, contract = _project()
    return {
        "dataset": contract.dataset_id,
        "file": raw_data_path,
        "raw_sha256": get_data_fingerprint(config.root / raw_data_path),
    }


def _base_run_config(data_fingerprint: dict, split_params: dict) -> dict:
    config, contract = _project()
    return {
        **get_git_info(config.root),
        "data_fingerprint": data_fingerprint,
        "dataset": contract.dataset_id,
        "target": contract.target,
        "positive_class": contract.positive_class,
        "features": list(config.model_features),
        "split": split_params,
    }


def log_baseline_run(
    trained_model,
    metrics: dict,
    X_test: pd.DataFrame,
    split_params: dict,
    data_fingerprint: dict,
    wandb_params: dict,
) -> dict:
    run_config = {
        **_base_run_config(data_fingerprint, split_params),
        "model": "logistic_regression",
        "model_params": trained_model.named_steps["classifier"].get_params(),
    }
    with init_wandb_run(
        wandb_params,
        name="baseline-logreg",
        job_type="baseline",
        config=run_config,
        tags=["baseline", "sprint3"],
    ) as run:
        summary = {
            **flatten_metrics(metrics),
            **measure_prediction_time(trained_model, X_test),
            "artifact_size_mb": pickled_size_mb(trained_model),
        }
        run.log(summary)
        return {"run_id": str(run.id), "summary": summary}


def train_autogluon_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    autogluon_params: dict,
    split_params: dict,
    data_fingerprint: dict,
    wandb_params: dict,
    baseline_run: dict,
) -> dict:
    if not autogluon_params.get("enabled", False):
        logger.info("AutoGluon wyłączony (autogluon.enabled=false) — pomijam eksperymenty.")
        return {}

    import wandb
    from autogluon.tabular import TabularPredictor

    label = y_train.name or "target"
    train_df = X_train.assign(**{label: y_train.to_numpy()})
    output_root = Path(autogluon_params["output_dir"])
    results: dict = {}

    for exp in autogluon_params["experiments"]:
        name = exp["name"]
        fit_kwargs = {"presets": exp["presets"], "time_limit": int(exp["time_limit"])}
        if exp.get("hyperparameters"):
            fit_kwargs["hyperparameters"] = exp["hyperparameters"]

        model_dir = output_root / name
        deploy_dir = output_root / f"{name}_deploy"
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(deploy_dir, ignore_errors=True)

        run_config = {
            **_base_run_config(data_fingerprint, split_params),
            "model": "autogluon",
            "experiment": name,
            "eval_metric": autogluon_params["eval_metric"],
            **fit_kwargs,
        }
        with init_wandb_run(
            wandb_params,
            name=name,
            job_type="autogluon",
            config=run_config,
            tags=["autogluon", "sprint3"],
        ) as run:
            predictor = TabularPredictor(
                label=label,
                problem_type="binary",
                eval_metric=autogluon_params["eval_metric"],
                positive_class=1,
                path=str(model_dir),
                verbosity=autogluon_params.get("verbosity", 1),
            )
            start = time.perf_counter()
            predictor.fit(train_data=train_df, **fit_kwargs)
            fit_seconds = time.perf_counter() - start

            predictor.clone_for_deployment(path=str(deploy_dir))

            y_pred = predictor.predict(X_test).astype(int)
            metrics = compute_metrics(y_test, y_pred)
            summary = {
                **flatten_metrics(metrics),
                **measure_prediction_time(predictor, X_test),
                "artifact_size_mb": dir_size_mb(deploy_dir),
                "artifact_size_full_mb": dir_size_mb(model_dir),
                "fit_seconds": fit_seconds,
            }
            run.log(summary)
            run.log({"leaderboard": wandb.Table(dataframe=predictor.leaderboard(silent=True))})
            run.summary["best_model"] = predictor.model_best
            run_id = str(run.id)

        results[name] = {
            **metrics,
            "best_model": predictor.model_best,
            "model_dir": str(model_dir),
            "deploy_dir": str(deploy_dir),
            "run_id": run_id,
            **{k: v for k, v in summary.items() if not k.startswith("test/")},
        }
        logger.info(
            "AutoGluon %s: f1=%.4f, best=%s, deploy=%.1f MB (full=%.1f MB)",
            name,
            metrics["f1"],
            predictor.model_best,
            summary["artifact_size_mb"],
            summary["artifact_size_full_mb"],
        )
    return results
