"""
XGBoost Model Modülü
Telco müşteri churn tahmini için XGBoost modeli eğitir ve değerlendirir.
"""

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_recall_curve, roc_curve,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  params: dict = None) -> xgb.XGBClassifier:
    """
    XGBoost modelini eğitir.

    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        params: XGBoost hiperparametreleri (None ise varsayılan)

    Returns:
        Eğitilmiş XGBClassifier
    """
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        verbose=False,
    )
    print("✅ XGBoost modeli eğitildi.")
    return model


def evaluate_model(model: xgb.XGBClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    Modeli test seti üzerinde değerlendirir.

    Returns:
        dict: F1, ROC-AUC, classification report, confusion matrix
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # ROC curve verileri
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "y_proba": y_proba,
    }

    print("\n" + "="*50)
    print("\nMODEL DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred))

    return metrics


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_splits: int = 5,
) -> dict:
    """
    Optuna ile XGBoost hiperparametrelerini optimize eder.

    Returns:
        dict: En iyi hiperparametreler
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", xgb.XGBClassifier(**params)),
        ])
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        return float(
            cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1
            ).mean()
        )

    print(f"\nOptuna ile {n_trials} deneme yapılıyor...")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    })
    print(f"✅ En iyi F1: {study.best_value:.4f}")
    print(f"   Parametreler: {best_params}")
    return best_params


def find_cost_optimal_threshold(
    y_true,
    y_proba,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> dict:
    """İş maliyetini en aza indiren churn karar eşiğini hesaplar."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 91)

    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(threshold),
            "cost": float(fn * false_negative_cost + fp * false_positive_cost),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        })

    best = min(rows, key=lambda row: (row["cost"], -row["recall"]))
    return {**best, "curve": rows}


def save_evaluation_report(
    metrics: dict,
    model_params: dict | None = None,
    filename: str = "evaluation_report.json",
) -> Path:
    """JSON uyumlu model değerlendirme ve deney raporu kaydeder."""
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "f1_score": float(metrics["f1_score"]),
            "roc_auc": float(metrics["roc_auc"]),
            "confusion_matrix": np.asarray(metrics["confusion_matrix"]).tolist(),
            "classification_report": metrics["classification_report"],
        },
        "model_params": model_params or {},
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"✅ Değerlendirme raporu kaydedildi: {path}")
    return path


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def save_model(
    model: xgb.XGBClassifier,
    filename: str = "xgboost_model.pkl",
    metrics: dict | None = None,
):
    """Modeli ve yeniden üretilebilirlik metadatasını diske kaydeder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "feature_names": list(getattr(model, "feature_names_in_", [])),
        "n_features": int(getattr(model, "n_features_in_", 0)),
        "versions": {
            "xgboost": _package_version("xgboost"),
            "scikit-learn": _package_version("scikit-learn"),
            "pandas": _package_version("pandas"),
        },
    }
    if metrics:
        metadata["metrics"] = {
            key: float(metrics[key])
            for key in ("f1_score", "roc_auc")
            if key in metrics
        }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"✅ Model kaydedildi: {path}")
    return path


def validate_model_features(model: xgb.XGBClassifier, X: pd.DataFrame) -> list[str]:
    """Model ile işlenmiş veri arasındaki sözleşme hatalarını döndürür."""
    issues = []
    expected_count = int(getattr(model, "n_features_in_", X.shape[1]))
    if expected_count != X.shape[1]:
        issues.append(
            f"Özellik sayısı uyuşmuyor: model={expected_count}, veri={X.shape[1]}"
        )

    expected_names = list(getattr(model, "feature_names_in_", []))
    if expected_names and expected_names != list(X.columns):
        issues.append("Özellik adları veya sırası model artefaktıyla uyuşmuyor.")
    return issues


def load_model(filename: str = "xgboost_model.pkl") -> xgb.XGBClassifier:
    """Kaydedilmiş modeli yükler."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {path}")
    model = joblib.load(path)
    print(f"✅ Model yüklendi: {path}")
    return model


def predict_proba_single(model: xgb.XGBClassifier,
                          X: pd.DataFrame) -> tuple[int, float]:
    """
    Tek bir müşteri için churn tahmini yapar.

    Returns:
        (tahmin: 0|1, olasılık: float)
    """
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    return pred, float(proba)
