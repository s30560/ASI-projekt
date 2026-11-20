import math

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from asi_project.model_package import save_model_package
from asi_project.pipelines.data_science.nodes import COMPARISON_COLUMNS, build_model_comparison
from asi_project.tracking import measure_load_memory, measure_prediction_time


@pytest.fixture
def toy_data():
    X = pd.DataFrame({"a": [float(i) for i in range(40)], "b": [float(i % 3) for i in range(40)]})
    y = pd.Series([0, 1] * 20, name="target")
    return X, y


def test_measure_prediction_time_reports_batch_and_single_row(toy_data):
    X, y = toy_data
    model = LogisticRegression().fit(X, y)
    timings = measure_prediction_time(model, X, repeats=3, single_row_samples=5)
    assert set(timings) == {
        "predict_time_ms_batch",
        "predict_time_ms_per_row",
        "predict_time_ms_single_row",
    }
    assert all(value >= 0 for value in timings.values())


def test_measure_load_memory_in_clean_process(toy_data, tmp_path):
    X, y = toy_data
    package = save_model_package(LogisticRegression().fit(X, y), tmp_path / "pkg", list(X.columns))
    result = measure_load_memory("sklearn_package", package, X.head(5))
    assert set(result) == {"load_memory_mb", "process_rss_after_load_mb", "cold_start_s"}
    assert result["load_memory_mb"] > 0
    assert result["process_rss_after_load_mb"] > result["load_memory_mb"]


def test_measure_load_memory_missing_artifact_returns_empty(tmp_path):
    assert measure_load_memory("sklearn_package", tmp_path / "missing", pd.DataFrame()) == {}


def test_measure_load_memory_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        measure_load_memory("xgboost", tmp_path, pd.DataFrame())


BASELINE_RUN = {
    "run_id": "r0",
    "model": "logistic_regression",
    "summary": {
        "test/f1": 0.6,
        "test/precision": 0.5,
        "test/recall": 0.75,
        "test/accuracy": 0.85,
        "test/cm_tp": 3,
        "predict_time_ms_batch": 4.0,
        "predict_time_ms_single_row": 1.5,
        "artifact_size_mb": 0.01,
        "load_memory_mb": 40.0,
        "cold_start_s": 0.8,
    },
}


def test_comparison_with_baseline_only(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    table, markdown = build_model_comparison(BASELINE_RUN, {}, {}, {}, {"mode": "disabled"})
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert len(table) == 1
    assert table.loc[0, "f1"] == 0.6
    assert table.loc[0, "load_memory_mb"] == 40.0
    lines = markdown.splitlines()
    assert lines[0].startswith("| Model |")
    assert lines[2].startswith("| baseline (logistic_regression) | - |")
    assert lines[2].endswith("| - |")


def test_comparison_includes_autogluon_configs(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    autogluon_metrics = {
        name: {
            "f1": f1,
            "precision": 0.6,
            "recall": 0.7,
            "accuracy": 0.9,
            "confusion_matrix": {"tp": 1},
            "best_model": "WeightedEnsemble_L2",
            "predict_time_ms_batch": 50.0,
            "predict_time_ms_single_row": 20.0,
            "artifact_size_mb": 12.0,
            "load_memory_mb": 300.0,
            "cold_start_s": 4.0,
            "fit_seconds": 100.0,
        }
        for name, f1 in [("ag_a", 0.65), ("ag_b", 0.66)]
    }
    table, markdown = build_model_comparison(
        BASELINE_RUN, autogluon_metrics, {}, {}, {"mode": "disabled"}
    )
    assert table["model"].tolist() == ["baseline (logistic_regression)", "ag_a", "ag_b"]
    required = ["f1", "predict_time_ms_single_row", "artifact_size_mb", "load_memory_mb"]
    assert not table[required].isna().any().any()
    assert math.isclose(table.loc[2, "f1"], 0.66)
    assert len(markdown.strip().splitlines()) == 2 + 3
