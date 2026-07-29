"""Probability calibration utilities for churn scores."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class ProbabilityCalibrator:
    """Platt-style calibrator fitted on out-of-fold model probabilities."""

    def __init__(self):
        self.model = LogisticRegression(random_state=42)

    @staticmethod
    def _logit(probabilities) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(probabilities / (1 - probabilities)).reshape(-1, 1)

    def fit(self, probabilities, y_true):
        self.model.fit(self._logit(probabilities), np.asarray(y_true))
        return self

    def predict(self, probabilities) -> np.ndarray:
        return self.model.predict_proba(self._logit(probabilities))[:, 1]


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Return weighted absolute confidence/observed-rate error."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = max(len(y_true), 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        include_upper = upper == 1
        mask = (probabilities >= lower) & (
            probabilities <= upper if include_upper else probabilities < upper
        )
        if mask.any():
            error += mask.sum() / total * abs(
                probabilities[mask].mean() - y_true[mask].mean()
            )
    return float(error)


def fit_oof_calibrator(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 3,
) -> tuple[ProbabilityCalibrator, np.ndarray]:
    """Fit calibration on leakage-safe out-of-fold predictions."""
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", clone(estimator)),
    ])
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    oof_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    calibrator = ProbabilityCalibrator().fit(oof_probabilities, y_train)
    return calibrator, oof_probabilities


def calibration_report(y_true, raw_probabilities, calibrated_probabilities) -> dict:
    """Build a JSON-safe before/after calibration report."""
    y_true = np.asarray(y_true)

    def metrics(values):
        values = np.asarray(values, dtype=float)
        return {
            "brier_score": float(brier_score_loss(y_true, values)),
            "log_loss": float(log_loss(y_true, values)),
            "expected_calibration_error": expected_calibration_error(y_true, values),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": int(len(y_true)),
        "raw": metrics(raw_probabilities),
        "calibrated": metrics(calibrated_probabilities),
    }


def save_calibration(
    calibrator: ProbabilityCalibrator,
    report: dict,
    directory: Path = MODELS_DIR,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "probability_calibrator.pkl"
    report_path = directory / "calibration_report.json"
    joblib.dump(calibrator, model_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return model_path, report_path


def load_calibrator(path: Path | None = None) -> ProbabilityCalibrator | None:
    path = path or MODELS_DIR / "probability_calibrator.pkl"
    return joblib.load(path) if path.exists() else None
