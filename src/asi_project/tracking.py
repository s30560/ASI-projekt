import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MB = 1024**2
MEMORY_PROBE_KINDS = ("sklearn_package", "autogluon")
_PROBE_PREFIX = "MEMORY_PROBE_RESULT:"


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


def measure_prediction_time(
    model: Any, X: pd.DataFrame, repeats: int = 5, single_row_samples: int = 20
) -> dict[str, float]:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        timings.append(time.perf_counter() - start)
    median = sorted(timings)[len(timings) // 2]

    single = []
    for i in range(min(len(X), single_row_samples)):
        row = X.iloc[[i]]
        start = time.perf_counter()
        model.predict(row)
        single.append(time.perf_counter() - start)
    single_median = sorted(single)[len(single) // 2] if single else float("nan")

    return {
        "predict_time_ms_batch": median * 1000,
        "predict_time_ms_per_row": median * 1000 / max(len(X), 1),
        "predict_time_ms_single_row": single_median * 1000,
    }


def pickled_size_mb(obj: Any) -> float:
    return len(pickle.dumps(obj)) / MB


def dir_size_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / MB


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


def _load_for_probe(kind: str, model_path: str | Path) -> Any:
    if kind == "sklearn_package":
        from asi_project.model_package import load_model_package

        model, _ = load_model_package(model_path)
        return model
    if kind == "autogluon":
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(str(model_path), verbosity=0)
    raise ValueError(f"Nieznany rodzaj modelu: {kind!r}. Dozwolone: {MEMORY_PROBE_KINDS}.")


def _memory_probe(kind: str, model_path: str | Path, sample_path: str | Path) -> dict[str, float]:
    import psutil

    process = psutil.Process()
    sample = pd.read_pickle(sample_path)
    rss_start = process.memory_info().rss
    start = time.perf_counter()
    model = _load_for_probe(kind, model_path)
    model.predict(sample)
    cold_start = time.perf_counter() - start
    rss_end = process.memory_info().rss
    return {
        "load_memory_mb": (rss_end - rss_start) / MB,
        "process_rss_after_load_mb": rss_end / MB,
        "cold_start_s": cold_start,
    }


def measure_load_memory(
    kind: str, model_path: str | Path, sample: pd.DataFrame, timeout: int = 600
) -> dict[str, float]:
    if kind not in MEMORY_PROBE_KINDS:
        raise ValueError(f"Nieznany rodzaj modelu: {kind!r}. Dozwolone: {MEMORY_PROBE_KINDS}.")
    path = Path(model_path)
    if not path.exists():
        logger.warning("Pomiar pamięci pominięty: brak artefaktu %s", path)
        return {}

    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.pkl"
        sample.to_pickle(sample_path)
        command = [sys.executable, "-m", "asi_project.tracking", kind, str(path), str(sample_path)]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=timeout
            )
        except (OSError, subprocess.SubprocessError) as exc:
            stderr = getattr(exc, "stderr", "") or ""
            logger.warning("Pomiar pamięci dla %s nie powiódł się: %s\n%s", path, exc, stderr)
            return {}

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_PROBE_PREFIX):
            return json.loads(line.removeprefix(_PROBE_PREFIX))
    logger.warning("Pomiar pamięci dla %s nie zwrócił wyniku.", path)
    return {}


if __name__ == "__main__":
    probe_kind, probe_model_path, probe_sample_path = sys.argv[1:4]
    result = _memory_probe(probe_kind, probe_model_path, probe_sample_path)
    print(_PROBE_PREFIX + json.dumps(result), flush=True)
