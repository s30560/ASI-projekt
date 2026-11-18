from unittest.mock import patch

import pandas as pd

from asi_project.pipelines.data_science.nodes import split_data


@patch("asi_project.pipelines.data_science.nodes._project")
def test_split_data(mock_project):
    dummy_data = pd.DataFrame(
        {
            "feature1": range(100),
            "target": [0, 1] * 50,
        }
    )

    class DummyContract:
        target = "target"
        excluded_features = []
        allowed_features = ["feature1"]
        positive_class = 1
        split_strategy = "stratified_random"

    class DummyConfig:
        model_features = ["feature1"]

    mock_project.return_value = (DummyConfig(), DummyContract())

    split_params = {"test_size": 0.2, "random_state": 42}

    X_train, X_test, y_train, y_test = split_data(dummy_data, split_params)

    assert len(X_train) == 80
    assert len(X_test) == 20
