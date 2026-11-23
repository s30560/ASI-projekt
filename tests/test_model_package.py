from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.linear_model import LogisticRegression

from asi_project.model_package import (
    download_production_package,
    load_model_package,
    log_model_package_to_wandb,
    predict_with_package,
    save_model_package,
)


def test_save_and_reload_model_package(tmp_path: Path):
    """Verify saving, reloading, and required metadata fields."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 10.0, 20.0]})
    y_train = pd.Series([0, 1, 0, 1])
    model = LogisticRegression(random_state=42).fit(X_train, y_train)

    pkg_dir = tmp_path / "model_pkg"
    save_model_package(
        model=model,
        package_dir=pkg_dir,
        feature_names=["f1", "f2"],
        target_name="Revenue",
        positive_class=True,
        metrics={"f1": 0.85},
        git_sha="abc1234",
        data_fingerprint="sha256-sample",
    )

    assert (pkg_dir / "model.pkl").exists()
    assert (pkg_dir / "metadata.json").exists()

    loaded_model, metadata = load_model_package(pkg_dir)
    assert hasattr(loaded_model, "predict")
    assert metadata["model_class"] == "LogisticRegression"
    assert metadata["features"] == ["f1", "f2"]
    assert metadata["target"] == "Revenue"
    assert metadata["positive_class"] is True
    assert metadata["metrics"]["f1"] == 0.85
    assert metadata["git_sha"] == "abc1234"
    assert metadata["data_fingerprint"] == "sha256-sample"
    assert "python" in metadata["library_versions"]
    assert "scikit-learn" in metadata["library_versions"]


def test_wandb_artifact_logging_with_production_alias(tmp_path: Path, monkeypatch):
    """Verify logging model package to W&B with production alias in offline mode."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    pkg_dir = tmp_path / "model_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "metadata.json").write_text("{}", encoding="utf-8")

    mock_wandb = MagicMock()
    mock_artifact = MagicMock()
    mock_wandb.Artifact.return_value = mock_artifact

    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        log_model_package_to_wandb(
            pkg_dir,
            artifact_name="model-package",
            aliases=["production"],
        )

        mock_wandb.Artifact.assert_called_once_with(name="model-package", type="model")
        mock_artifact.add_dir.assert_called_once_with(str(pkg_dir))
        mock_wandb.run.log_artifact.assert_called_once_with(
            mock_artifact,
            aliases=["production"],
        )


def test_smoke_test_production_package_ci(tmp_path: Path, monkeypatch):
    """Verify local save, offline load, 5-row prediction, and output schema."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 10.0, 20.0]})
    y_train = pd.Series([0, 1, 0, 1])
    model = LogisticRegression(random_state=42).fit(X_train, y_train)

    prod_pkg_dir = tmp_path / "production_pkg"
    save_model_package(model, prod_pkg_dir, feature_names=["f1", "f2"])

    monkeypatch.setenv("WANDB_MODE", "offline")
    downloaded_dir = download_production_package(
        artifact_path="model-package:production",
        fallback_dir=prod_pkg_dir,
    )

    sample_5_rows = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 1.5, 3.5, 4.0],
            "f2": [10.0, 20.0, 15.0, 25.0, 10.0],
        }
    )
    predictions_df = predict_with_package(downloaded_dir, sample_5_rows)

    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) == 5
    assert list(predictions_df.columns) == ["prediction"]
    assert predictions_df["prediction"].isnull().sum() == 0
    assert set(predictions_df["prediction"]).issubset({0, 1})
