from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required customer fields: {', '.join(missing)}")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    for column in CATEGORICAL_FEATURES:
        values = features[column].astype("string").str.strip()
        values = values.replace(
            {"No internet service": "No", "No phone service": "No"}
        )
        normalized = values.str.lower()
        values = values.mask(normalized.eq("yes"), "Yes")
        values = values.mask(normalized.eq("no"), "No")
        values = values.mask(normalized.eq("male"), "Male")
        values = values.mask(normalized.eq("female"), "Female")
        features[column] = values

    return features


def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]:
    if "Churn" not in frame.columns:
        raise ValueError("Training CSV must include a 'Churn' column with Yes/No labels.")

    labels = (
        frame["Churn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )
    valid_rows = labels.notna()
    dropped_rows = int((~valid_rows).sum())
    if not valid_rows.any():
        raise ValueError("No usable Churn labels found. Labels must be Yes or No.")

    features = clean_feature_frame(frame.loc[valid_rows]).reset_index(drop=True)
    target = labels.loc[valid_rows].astype(int).reset_index(drop=True)
    labeled_data = features.copy()
    labeled_data["Churn"] = target
    return features, target, labeled_data, dropped_rows


def train_model(frame: pd.DataFrame) -> dict[str, Any]:
    features, target, labeled_data, dropped_rows = prepare_training_data(frame)
    class_counts = target.value_counts()
    if len(class_counts) != 2:
        raise ValueError("Training data must include both churn and non-churn customers.")
    if int(class_counts.min()) < 5:
        raise ValueError("Please provide at least 5 labeled examples in each Churn class.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "model": model,
        "x_test": x_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "probabilities": probabilities,
        "training_features": features,
        "labeled_data": labeled_data,
        "dropped_rows": dropped_rows,
        "row_count": len(features),
        "churn_rate": float(target.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def calculate_metrics(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "flagged": int(predictions.sum()),
    }
