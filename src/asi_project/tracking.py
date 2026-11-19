import hashlib
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


def get_git_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": os.environ.get("GITHUB_SHA", "unknown"), "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def get_data_fingerprint(data_path: str | Path) -> str:
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Brak pliku danych do fingerprintu: {path}")
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def flatten_metrics(metrics: dict[str, Any], prefix: str = "test") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{prefix}/cm_{sub_key}"] = sub_value
        else:
            flat[f"{prefix}/{key}"] = value
    return flat


def measure_prediction_time(model: Any, X: pd.DataFrame, repeats: int = 5) -> dict[str, float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        timings.append(time.perf_counter() - start)
    median = sorted(timings)[len(timings) // 2]
    return {
        "predict_time_ms_batch": median * 1000,
        "predict_time_ms_per_row": median * 1000 / max(len(X), 1),
    }


def pickled_size_mb(obj: Any) -> float:
    return len(pickle.dumps(obj)) / 1024**2


def dir_size_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2


def init_wandb_run(
    wandb_params: dict[str, Any],
    *,
    name: str,
    job_type: str,
    config: dict[str, Any],
    tags: list[str] | None = None,
):
    import wandb

    mode = os.environ.get("WANDB_MODE") or wandb_params.get("mode", "online")
    return wandb.init(
        project=wandb_params.get("project"),
        entity=wandb_params.get("entity") or None,
        group=wandb_params.get("group"),
        name=name,
        job_type=job_type,
        config=config,
        tags=tags or [],
        mode=mode,
        reinit="finish_previous",
    )
