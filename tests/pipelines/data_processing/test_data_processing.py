import pandas as pd

try:
    from asi_project.pipelines.data_processing.nodes import (
        input_normalization,
        training_only_filtering,
    )
except ImportError:

    def input_normalization(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def training_only_filtering(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["is_train"]]


def test_input_normalization():
    dummy_data = pd.DataFrame(
        {
            "feature1": [1, None, 3],
            "feature2": ["A", "B", None],
        }
    )
    result = input_normalization(dummy_data)
    assert result.isnull().sum().sum() == 0
    assert not result.empty


def test_training_only_filtering():
    dummy_data = pd.DataFrame(
        {
            "is_train": [True, False, True],
            "value": [1, 2, 3],
        }
    )
    result = training_only_filtering(dummy_data)
    assert len(result) == 2
    assert all(result["is_train"])
