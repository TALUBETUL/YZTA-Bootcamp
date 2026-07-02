"""
XGBoost Model Modülü
Telco müşteri churn tahmini için XGBoost modeli eğitir ve değerlendirir.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_recall_curve, roc_curve
)
import joblib
import optuna
from pathlib import Path


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
            "use_label_encoder": False,
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
    print("📊 MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred))

    return metrics


def optimize_hyperparams(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          n_trials: int = 50) -> dict:
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
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)

    print(f"🔍 Optuna ile {n_trials} deneme yapılıyor...")
    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    })
    print(f"✅ En iyi F1: {study.best_value:.4f}")
    print(f"   Parametreler: {best_params}")
    return best_params


def save_model(model: xgb.XGBClassifier, filename: str = "xgboost_model.pkl"):
    """Modeli diske kaydeder."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"✅ Model kaydedildi: {path}")
    return path


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
