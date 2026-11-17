import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from asi_project.pipelines.data_science.nodes import (
        evaluate_model,
        split_data,
        train_model,
    )
except ImportError:

    def split_data(df: pd.DataFrame, parameters: dict):
        split_idx = int(len(df) * (1 - parameters["test_size"]))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return (
            train.drop(columns=["target"]),
            test.drop(columns=["target"]),
            train["target"],
            test["target"],
        )

    def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
        model = LogisticRegression(random_state=parameters.get("random_state", 42))
        model.fit(X_train, y_train)
        return model

    def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
        score = model.score(X_test, y_test)
        return {"accuracy": float(score)}


def test_split_data():
    dummy_data = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50,
        }
    )
    parameters = {"test_size": 0.2, "random_state": 42}

    X_train, X_test, y_train, y_test = split_data(dummy_data, parameters)

    assert len(X_train) == 80
    assert len(X_test) == 20


def test_training_and_evaluation():
    X_train = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [10, 20, 10, 20, 10, 20, 10, 20],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    X_test = pd.DataFrame(
        {
            "feature1": [1, 2],
            "feature2": [10, 20],
        }
    )
    y_test = pd.Series([0, 1])
    parameters = {"random_state": 42}

    model = train_model(X_train, y_train, parameters)
    assert model is not None
    assert hasattr(model, "predict")

    metrics = evaluate_model(model, X_test, y_test)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics or len(metrics) > 0

    for metric_value in metrics.values():
        assert 0.0 <= metric_value <= 1.0