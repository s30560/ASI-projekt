import hashlib
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from asi_project.pipelines.data_science.nodes import train_autogluon_experiments
from asi_project.tracking import flatten_metrics, get_data_fingerprint, get_git_info


def test_data_fingerprint_matches_sha256(tmp_path):
    data = tmp_path / "data.csv"
    data.write_bytes(b"a,b\n1,2\n")
    assert get_data_fingerprint(data) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_data_fingerprint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_data_fingerprint(tmp_path / "missing.csv")


def test_git_info_returns_sha(repo_root):
    info = get_git_info(repo_root)
    assert info["git_sha"] and info["git_sha"] != "unknown"


def test_flatten_metrics():
    flat = flatten_metrics({"f1": 0.5, "confusion_matrix": {"tp": 3, "fn": 1}})
    assert flat == {"test/f1": 0.5, "test/cm_tp": 3, "test/cm_fn": 1}


def test_autogluon_disabled_returns_empty():
    X = pd.DataFrame({"f": [1.0, 2.0]})
    y = pd.Series([0, 1], name="Revenue")
    result = train_autogluon_experiments(
        X, y, X, y, {"enabled": False}, {}, {}, {"mode": "disabled"}, {}
    )
    assert result == {}


@patch("asi_project.pipelines.data_science.nodes._project")
def test_autogluon_node_with_mocked_predictor(mock_project, tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    config = MagicMock(root=tmp_path, model_features=("f",))
    contract = MagicMock(dataset_id="toy", target="Revenue", positive_class="1")
    mock_project.return_value = (config, contract)

    X = pd.DataFrame({"f": [float(i) for i in range(10)]})
    y = pd.Series([0, 1] * 5, name="Revenue")

    predictor = MagicMock()
    predictor.predict.side_effect = lambda df: pd.Series([0, 1] * (len(df) // 2))
    predictor.model_best = "LightGBM"
    predictor.leaderboard.return_value = pd.DataFrame({"model": ["LightGBM"], "score_val": [0.5]})
    fake_module = types.ModuleType("autogluon.tabular")
    fake_module.TabularPredictor = MagicMock(return_value=predictor)

    params = {
        "enabled": True,
        "eval_metric": "f1",
        "output_dir": str(tmp_path / "ag"),
        "experiments": [
            {"name": "exp_a", "presets": "medium_quality", "time_limit": 5},
            {"name": "exp_b", "presets": "good_quality", "time_limit": 5},
        ],
    }
    with patch.dict(sys.modules, {"autogluon.tabular": fake_module}):
        result = train_autogluon_experiments(
            X,
            y,
            X,
            y,
            params,
            {"test_size": 0.2},
            {"raw_sha256": "abc"},
            {"project": "test"},
            {"run_id": "x"},
        )

    assert set(result) == {"exp_a", "exp_b"}
    assert result["exp_a"]["f1"] == 1.0
    assert predictor.fit.call_args.kwargs["time_limit"] == 5
