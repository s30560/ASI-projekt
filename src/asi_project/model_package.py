import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def _get_git_sha() -> str:
    """Return current Git commit SHA or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _get_library_versions() -> dict[str, str]:
    """Collect versions of Python and core ML libraries."""
    versions = {"python": platform.python_version()}
    packages = (
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "wandb",
        "autogluon.tabular",
    )
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


SKLEARN_MODEL_FILE = "model.pkl"
AUTOGLUON_MODEL_DIR = "model"


def _is_autogluon(model: Any) -> bool:
    """AutoGluon TabularPredictor is a directory on disk, not a picklable estimator."""
    return type(model).__name__ == "TabularPredictor" and hasattr(model, "path")


def save_model_package(
    model: Any,
    package_dir: str | Path,
    feature_names: list[str],
    target_name: str = "target",
    positive_class: Any = 1,
    metrics: dict[str, float] | None = None,
    git_sha: str | None = None,
    data_fingerprint: str = "unknown",
    library_versions: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save trained model and metadata.json into a package directory.

    sklearn-like models are stored as ``model.pkl`` (joblib); an AutoGluon
    ``TabularPredictor`` (ideally after ``clone_for_deployment``) is copied
    as the ``model/`` directory. ``model_format`` in metadata tells the loader which.
    """
    package_path = Path(package_dir)
    package_path.mkdir(parents=True, exist_ok=True)
    (package_path / SKLEARN_MODEL_FILE).unlink(missing_ok=True)
    shutil.rmtree(package_path / AUTOGLUON_MODEL_DIR, ignore_errors=True)

    if _is_autogluon(model):
        model_format = "autogluon"
        shutil.copytree(Path(model.path), package_path / AUTOGLUON_MODEL_DIR)
    else:
        model_format = "joblib"
        joblib.dump(model, package_path / SKLEARN_MODEL_FILE)

    metadata = {
        "model_class": type(model).__name__,
        "model_format": model_format,
        "created_at": datetime.now(UTC).isoformat(),
        "features": list(feature_names),
        "target": target_name,
        "positive_class": positive_class,
        "metrics": metrics or {},
        "git_sha": git_sha or _get_git_sha(),
        "data_fingerprint": data_fingerprint,
        "library_versions": library_versions or _get_library_versions(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(package_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return package_path


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load model object and metadata dictionary from a package directory."""
    package_path = Path(package_dir)
    metadata_file = package_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Incomplete model package in {package_path} (metadata.json is required)."
        )
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    model_format = metadata.get("model_format", "joblib")
    if model_format == "autogluon":
        model_dir = package_path / AUTOGLUON_MODEL_DIR
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Incomplete model package in {package_path} "
                f"({AUTOGLUON_MODEL_DIR}/ directory is required for AutoGluon)."
            )
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(model_dir), verbosity=0)
    elif model_format == "joblib":
        model_file = package_path / SKLEARN_MODEL_FILE
        if not model_file.exists():
            raise FileNotFoundError(
                f"Incomplete model package in {package_path} ({SKLEARN_MODEL_FILE} is required)."
            )
        model = joblib.load(model_file)
    else:
        raise ValueError(f"Unknown model_format in metadata.json: {model_format!r}")

    return model, metadata


def predict_with_package(package_dir: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load package, validate input features, and return predictions DataFrame."""
    model, metadata = load_model_package(package_dir)
    expected_features = metadata["features"]

    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in input data: {missing}")

    preds = model.predict(df[expected_features])
    preds = pd.Series(preds).to_numpy()
    return pd.DataFrame({"prediction": preds}, index=df.index)


def log_model_package_to_wandb(
    package_dir: str | Path,
    artifact_name: str = "model-package",
    aliases: list[str] | None = None,
) -> None:
    """Log model package directory as a W&B Artifact (supports online and offline modes)."""
    if os.environ.get("WANDB_MODE") == "disabled":
        return

    import wandb

    if aliases is None:
        aliases = ["production"]

    if wandb.run is None:
        return

    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_dir(str(package_dir))
    wandb.run.log_artifact(artifact, aliases=aliases)


def download_production_package(
    artifact_path: str = "model-package:production",
    fallback_dir: str | Path = "data/06_models/package",
) -> Path:
    """Download production package from W&B or return local fallback in offline/CI mode."""
    if os.environ.get("WANDB_MODE") in {"disabled", "offline"}:
        return Path(fallback_dir)

    import wandb

    if wandb.run is not None:
        artifact = wandb.run.use_artifact(artifact_path, type="model")
    else:
        # Append entity/project from env when short artifact name is provided.
        full_path = artifact_path
        if "/" not in artifact_path:
            project = os.environ.get("WANDB_PROJECT")
            entity = os.environ.get("WANDB_ENTITY")
            if entity and project:
                full_path = f"{entity}/{project}/{artifact_path}"
            elif project:
                full_path = f"{project}/{artifact_path}"
        api = wandb.Api()
        artifact = api.artifact(full_path, type="model")

    return Path(artifact.download())
