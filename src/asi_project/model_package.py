import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model_package(
    model: Any,
    package_dir: str | Path,
    feature_names: list[str],
    target_name: str = "target",
    metrics: dict[str, float] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Zapisuje pakiet modelu (model.pkl + metadata.json) we wskazanym katalogu."""
    package_path = Path(package_dir)
    package_path.mkdir(parents=True, exist_ok=True)

    model_file = package_path / "model.pkl"
    metadata_file = package_path / "metadata.json"

    joblib.dump(model, model_file)

    metadata = {
        "model_class": type(model).__name__,
        "created_at": datetime.now(UTC).isoformat(),
        "schema": {
            "features": list(feature_names),
            "target": target_name,
        },
        "metrics": metrics or {},
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return package_path


def load_model_package(package_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Wczytuje pakiet modelu (zwraca krotkę: obiekt modelu oraz słownik metadata)."""
    package_path = Path(package_dir)
    model_file = package_path / "model.pkl"
    metadata_file = package_path / "metadata.json"

    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"Brak pełnego pakietu modelu w {package_path} (wymagane model.pkl i metadata.json)."
        )

    model = joblib.load(model_file)
    with open(metadata_file, encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def predict_with_package(package_dir: str | Path, df: pd.DataFrame) -> pd.DataFrame:
    """Wczytuje pakiet, weryfikuje kolumny wejściowe i zwraca DataFrame z predykcją."""
    model, metadata = load_model_package(package_dir)
    expected_features = metadata["schema"]["features"]

    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"Brakujące kolumny w danych wejściowych: {missing}")

    X = df[expected_features]
    preds = model.predict(X)

    return pd.DataFrame({"prediction": preds}, index=df.index)


def log_model_package_to_wandb(
    package_dir: str | Path,
    artifact_name: str = "model-package",
    aliases: list[str] | None = None,
) -> None:
    """Loguje katalog pakietu modelu jako W&B Artifact (domyślnie z aliasem 'production')."""
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
    """Pobiera pakiet modelu z W&B po aliasie production (lub używa lokalnego fallbacku w CI)."""
    if os.environ.get("WANDB_MODE") == "disabled":
        return Path(fallback_dir)

    import wandb

    if wandb.run is not None:
        artifact = wandb.run.use_artifact(artifact_path, type="model")
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")

    return Path(artifact.download())
