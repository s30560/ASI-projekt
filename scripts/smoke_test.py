import argparse
from pathlib import Path

import pandas as pd

from asi_project.model_package import (
    download_production_package,
    load_model_package,
    predict_with_package,
)


def run_smoke_test(
    artifact_path: str,
    data_path: Path,
    fallback_dir: Path,
    n_rows: int = 5,
) -> None:
    """Load production model package, run inference on sample rows, and validate output."""
    package_dir = download_production_package(
        artifact_path=artifact_path,
        fallback_dir=fallback_dir,
    )
    _, metadata = load_model_package(package_dir)

    df = pd.read_csv(data_path).head(n_rows)
    predictions = predict_with_package(package_dir, df)

    assert isinstance(predictions, pd.DataFrame), "Output must be a pandas DataFrame."
    assert len(predictions) == n_rows, f"Expected {n_rows} rows, got {len(predictions)}."
    assert list(predictions.columns) == ["prediction"], "Unexpected output columns."
    assert predictions["prediction"].isnull().sum() == 0, "Predictions contain NaN values."

    print(f"Smoke test passed | model: {metadata['model_class']} | rows: {len(predictions)}")
    print(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for production model package.")
    parser.add_argument("--artifact", default="model-package:production")
    parser.add_argument("--data", type=Path, default=Path("data/01_raw/sample.csv"))
    parser.add_argument("--fallback-dir", type=Path, default=Path("data/06_models/package"))
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    run_smoke_test(args.artifact, args.data, args.fallback_dir, args.rows)


if __name__ == "__main__":
    main()