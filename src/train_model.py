from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .churn_model import (
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        calculate_metrics,
        train_model,
    )
else:
    from churn_model import (
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
        NUMERIC_FEATURES,
        calculate_metrics,
        train_model,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "raw.csv"
MODEL_PATH = ROOT / "models" / "Model.pkl"
THRESHOLD = 0.40


def build_artifact(frame: pd.DataFrame) -> dict:
    bundle = train_model(frame)
    features = bundle["training_features"]
    threshold_metrics = {}
    for threshold in [round(value, 2) for value in np.arange(0.10, 0.91, 0.05)]:
        metrics = calculate_metrics(
            bundle["y_test"], bundle["probabilities"], threshold
        )
        metrics["test_rows"] = len(bundle["y_test"])
        threshold_metrics[f"{threshold:.2f}"] = metrics

    return {
        "model": bundle["model"],
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "numeric_defaults": {
            column: float(features[column].median()) for column in NUMERIC_FEATURES
        },
        "categorical_options": {
            column: sorted(features[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "threshold_metrics": threshold_metrics,
        "threshold": THRESHOLD,
        "training_rows": bundle["row_count"],
        "model_type": "LogisticRegression",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save the logistic-regression churn model."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV with Telco churn data (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {args.data_path}")

    artifact = build_artifact(pd.read_csv(args.data_path))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    for metric, value in artifact["threshold_metrics"][f"{THRESHOLD:.2f}"].items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
