import pandas as pd

from asi_project.config import DatasetContract, ProjectConfig, get_contract, load_project_config
from asi_project.data import (
    assert_valid,
    filter_training_rows,
    normalize_column_names,
    normalize_input,
)


def _project() -> tuple[ProjectConfig, DatasetContract]:
    """Konfiguracja zespołu"""
    config = load_project_config()
    return config, get_contract(config)


def validate_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    config, contract = _project()
    df = normalize_column_names(raw_data)
    assert_valid(df, contract, config)
    return df


def input_normalization(validated_data: pd.DataFrame) -> pd.DataFrame:
    return normalize_input(validated_data)


def training_only_filtering(normalized_data: pd.DataFrame) -> pd.DataFrame:
    _, contract = _project()
    return filter_training_rows(normalized_data, contract)
