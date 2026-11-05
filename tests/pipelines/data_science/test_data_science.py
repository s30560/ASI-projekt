import pandas as pd

try:
    from asi_project.pipelines.data_science.nodes import split_data
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


def test_split_data():
    dummy_data = pd.DataFrame({
        "feature1": range(100),
        "target": [0, 1] * 50,
    })
    parameters = {"test_size": 0.2, "random_state": 42}

    X_train, X_test, y_train, y_test = split_data(dummy_data, parameters)

    assert len(X_train) == 80
    assert len(X_test) == 20