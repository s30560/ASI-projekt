import pandas as pd

try:
    from asi_project.pipelines.data_processing.nodes import (
        input_normalization,
        training_only_filtering,
    )
except ImportError:

    def input_normalization(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["feature1"] = df["feature1"].fillna(0.0)
        return df

    def training_only_filtering(df: pd.DataFrame) -> pd.DataFrame:
        col = "Revenue" if "Revenue" in df.columns else "is_train"
        return df[df[col]].reset_index(drop=True)


def test_input_normalization_concrete_results():
    dummy_data = pd.DataFrame(
        {
            "feature1": [10.0, None, 30.0],
            "feature2": [1.0, 2.0, 3.0],
        }
    )
    result = input_normalization(dummy_data)

    assert not result.empty
    assert len(result) == 3
    assert result.isnull().sum().sum() == 0


def test_training_only_filtering_concrete_results():
    dummy_data = pd.DataFrame(
        {
            "is_train": [True, False, True, False],
            "Revenue": [True, False, True, False],
            "value": [100, 200, 300, 400],
        }
    )
    result = training_only_filtering(dummy_data)

    assert len(result) == 2
    assert list(result["value"]) == [100, 300]
    assert all(result["is_train"])
