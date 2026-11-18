import pandas as pd

from asi_project.config import DatasetContract, ProjectConfig, get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.metrics import compute_metrics
from asi_project.model import build_baseline_pipeline


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
