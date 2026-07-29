"""Leakage-safe baseline model comparison utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """Compare baseline models with SMOTE applied inside every CV fold."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows = []

    for name, estimator in _candidate_models().items():
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", estimator),
        ])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        row["fit_time_seconds"] = float(np.mean(scores["fit_time"]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["f1", "roc_auc"], ascending=False
    ).reset_index(drop=True)


def save_model_comparison(
    comparison: pd.DataFrame,
    filename: str = "model_comparison.json",
) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "f1",
        "best_model": comparison.iloc[0]["model"],
        "results": comparison.to_dict("records"),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"✅ Model karşılaştırması kaydedildi: {path}")
    return path


def load_model_comparison(
    filename: str = "model_comparison.json",
) -> pd.DataFrame | None:
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return pd.DataFrame(payload["results"])
