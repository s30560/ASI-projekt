import logging
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from asi_project.config import DatasetContract, ProjectConfig, get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline
from asi_project.model_package import save_model_package
from asi_project.tracking import (
    dir_size_mb,
    flatten_metrics,
    get_data_fingerprint,
    get_git_info,
    init_wandb_run,
    measure_load_memory,
    measure_prediction_time,
)

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_PARAMS = {
    "baseline_package_dir": "data/06_models/baseline_package",
    "memory_sample_rows": 5,
}


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


def _comparison_params(comparison_params: dict | None) -> dict:
    return {**DEFAULT_COMPARISON_PARAMS, **(comparison_params or {})}


def log_baseline_run(
    trained_model,
    metrics: dict,
    X_test: pd.DataFrame,
    split_params: dict,
    data_fingerprint: dict,
    wandb_params: dict,
    comparison_params: dict | None = None,
) -> dict:
    config, contract = _project()
    comparison = _comparison_params(comparison_params)
    package_dir = save_model_package(
        trained_model,
        Path(config.root) / comparison["baseline_package_dir"],
        feature_names=list(config.model_features),
        target_name=contract.target,
        metrics={k: v for k, v in metrics.items() if not isinstance(v, dict)},
        extra_metadata={"positive_class": contract.positive_class},
    )
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
            "artifact_size_mb": dir_size_mb(package_dir),
            **measure_load_memory(
                "sklearn_package",
                package_dir,
                X_test.head(int(comparison["memory_sample_rows"])),
            ),
        }
        run.log(summary)
        return {
            "run_id": str(run.id),
            "model": "logistic_regression",
            "package_dir": str(package_dir),
            "summary": summary,
        }


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
    comparison_params: dict | None = None,
) -> dict:
    if not autogluon_params.get("enabled", False):
        logger.info("AutoGluon wyłączony (autogluon.enabled=false): pomijam eksperymenty.")
        return {}

    import wandb
    from autogluon.tabular import TabularPredictor

    label = y_train.name or "target"
    train_df = X_train.assign(**{label: y_train.to_numpy()})
    output_root = Path(autogluon_params["output_dir"])
    memory_sample = X_test.head(int(_comparison_params(comparison_params)["memory_sample_rows"]))
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
                **measure_load_memory("autogluon", deploy_dir, memory_sample),
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


COMPARISON_COLUMNS = {
    "model": "Model",
    "best_model": "Najlepszy model wewnętrzny",
    "f1": "F1 (główna)",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "predict_time_ms_batch": "Predykcja zbioru test [ms]",
    "predict_time_ms_single_row": "Predykcja 1 wiersza [ms]",
    "artifact_size_mb": "Rozmiar artefaktu [MB]",
    "load_memory_mb": "Pamięć po wczytaniu [MB]",
    "cold_start_s": "Wczytanie + 1. predykcja [s]",
    "fit_seconds": "Czas treningu [s]",
}
_METRIC_COLUMNS = {"f1", "precision", "recall", "accuracy"}


def _comparison_row(name: str, values: dict) -> dict:
    row = {"model": name}
    for column in list(COMPARISON_COLUMNS)[1:]:
        row[column] = values.get(column, values.get(f"test/{column}"))
    return row


def _format_cell(column: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if column in _METRIC_COLUMNS else f"{value:.2f}"
    return str(value)


def comparison_to_markdown(table: pd.DataFrame) -> str:
    headers = [COMPARISON_COLUMNS.get(column, column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("-" for _ in headers) + "|",
    ]
    for record in table.to_dict(orient="records"):
        cells = [_format_cell(column, record[column]) for column in table.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_model_comparison(
    baseline_run: dict,
    autogluon_metrics: dict,
    split_params: dict,
    data_fingerprint: dict,
    wandb_params: dict,
) -> tuple[pd.DataFrame, str]:
    baseline_name = f"baseline ({baseline_run.get('model', 'logistic_regression')})"
    rows = [_comparison_row(baseline_name, baseline_run["summary"])]
    rows += [_comparison_row(name, values) for name, values in (autogluon_metrics or {}).items()]
    table = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    markdown = comparison_to_markdown(table)
    logger.info("Porównanie modeli:\n%s", markdown)

    mode = os.environ.get("WANDB_MODE") or wandb_params.get("mode", "online")
    if mode != "disabled":
        import wandb

        with init_wandb_run(
            wandb_params,
            name="model-comparison",
            job_type="comparison",
            config=_base_run_config(data_fingerprint, split_params),
            tags=["comparison", "sprint3"],
        ) as run:
            run.log({"model_comparison": wandb.Table(dataframe=table)})
    return table, markdown
