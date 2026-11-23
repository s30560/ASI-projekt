import json
import os
import platform
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
    """Save trained model (model.pkl) and metadata (metadata.json) into a directory."""
    package_path = Path(package_dir)
    package_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, package_path / "model.pkl")

    metadata = {
        "model_class": type(model).__name__,
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
    model_file = package_path / "model.pkl"
    metadata_file = package_path / "metadata.json"

    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"Incomplete model package in {package_path} "
            "(model.pkl and metadata.json are required)."
        )

    model = joblib.load(model_file)
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def predict_with_package(package_dir: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load package, validate input features, and return predictions DataFrame."""
    model, metadata = load_model_package(package_dir)
    expected_features = metadata["features"]

    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in input data: {missing}")

    preds = model.predict(df[expected_features])
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
