"""Data drift, performance, and group error monitoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


NUMERIC_MONITOR_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_MONITOR_COLUMNS = [
    "gender", "SeniorCitizen", "Contract", "InternetService", "PaymentMethod"
]


def build_reference_profile(df: pd.DataFrame, bins: int = 10) -> dict:
    profile = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "numeric": {},
        "categorical": {},
    }
    for column in NUMERIC_MONITOR_COLUMNS:
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([values.min() - 1, values.max() + 1])
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=edges)
        profile["numeric"][column] = {
            "edges": [None if np.isinf(x) else float(x) for x in edges],
            "proportions": (counts / max(counts.sum(), 1)).tolist(),
        }
    for column in CATEGORICAL_MONITOR_COLUMNS:
        if column in df:
            profile["categorical"][column] = (
                df[column].fillna("__MISSING__").astype(str)
                .value_counts(normalize=True).to_dict()
            )
    return profile


def _psi(expected, actual, epsilon: float = 1e-6) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    actual = np.clip(np.asarray(actual, dtype=float), epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def calculate_drift(profile: dict, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, reference in profile.get("numeric", {}).items():
        if column not in current:
            continue
        edges = np.array([
            -np.inf if value is None and index == 0 else
            np.inf if value is None else value
            for index, value in enumerate(reference["edges"])
        ])
        counts, _ = np.histogram(
            pd.to_numeric(current[column], errors="coerce").dropna(), bins=edges
        )
        actual = counts / max(counts.sum(), 1)
        score = _psi(reference["proportions"], actual)
        rows.append({"feature": column, "type": "numeric", "drift_score": score})
    for column, reference in profile.get("categorical", {}).items():
        if column not in current:
            continue
        actual = (
            current[column].fillna("__MISSING__").astype(str)
            .value_counts(normalize=True).to_dict()
        )
        categories = sorted(set(reference) | set(actual))
        score = 0.5 * sum(
            abs(reference.get(category, 0) - actual.get(category, 0))
            for category in categories
        )
        rows.append({"feature": column, "type": "categorical", "drift_score": score})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["status"] = np.select(
        [result["drift_score"] >= 0.25, result["drift_score"] >= 0.10],
        ["critical", "warning"],
        default="stable",
    )
    return result.sort_values("drift_score", ascending=False).reset_index(drop=True)


def group_error_analysis(
    y_true,
    probabilities,
    raw_data: pd.DataFrame,
    group_columns: list[str] | None = None,
    threshold: float = 0.5,
    min_group_size: int = 20,
) -> pd.DataFrame:
    """Report observed group metrics; this is not a legal fairness certification."""
    group_columns = group_columns or ["gender", "SeniorCitizen", "Contract"]
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    rows = []
    for column in group_columns:
        if column not in raw_data:
            continue
        for group, indexes in raw_data.groupby(column, dropna=False).groups.items():
            indexes = np.asarray(list(indexes), dtype=int)
            if len(indexes) < min_group_size:
                continue
            observed = y_true[indexes]
            scores = probabilities[indexes]
            predicted = predictions[indexes]
            rows.append({
                "attribute": column,
                "group": str(group),
                "n": int(len(indexes)),
                "churn_rate": float(observed.mean()),
                "positive_prediction_rate": float(predicted.mean()),
                "precision": float(precision_score(observed, predicted, zero_division=0)),
                "recall": float(recall_score(observed, predicted, zero_division=0)),
                "f1": float(f1_score(observed, predicted, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(observed, scores))
                    if len(np.unique(observed)) == 2 else None
                ),
            })
    return pd.DataFrame(rows)


def save_reference_profile(profile: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)
    return path
