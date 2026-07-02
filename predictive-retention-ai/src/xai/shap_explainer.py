"""
SHAP Açıklanabilirlik Modülü
XGBoost modeli için SHAP değerlerini hesaplar ve yorumlar.
"""

import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parents[2] / "data" / "figures"


def get_shap_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """
    XGBoost modeli için SHAP TreeExplainer oluşturur.

    Returns:
        shap.TreeExplainer
    """
    explainer = shap.TreeExplainer(model)
    print("✅ SHAP TreeExplainer oluşturuldu.")
    return explainer


def compute_shap_values(explainer: shap.TreeExplainer,
                         X: pd.DataFrame) -> np.ndarray:
    """
    Verilen özellikler için SHAP değerlerini hesaplar.

    Returns:
        np.ndarray: SHAP değerleri
    """
    shap_values = explainer.shap_values(X)
    return shap_values


def get_top_factors(shap_values: np.ndarray,
                     feature_names: list[str],
                     top_n: int = 5) -> list[dict]:
    """
    Tek bir müşteri için en etkili N faktörü döndürür.

    Args:
        shap_values: Tek satırlık SHAP değerleri (1D array)
        feature_names: Özellik adları listesi
        top_n: Döndürülecek faktör sayısı

    Returns:
        list[dict]: [{"feature": str, "shap_value": float, "direction": str}]
    """
    factor_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
    })
    factor_df["abs_shap"] = factor_df["shap_value"].abs()
    factor_df = factor_df.sort_values("abs_shap", ascending=False).head(top_n)
    factor_df["direction"] = factor_df["shap_value"].apply(
        lambda x: "risk artırıcı ↑" if x > 0 else "risk azaltıcı ↓"
    )
    return factor_df[["feature", "shap_value", "direction"]].to_dict("records")


def format_shap_factors_for_llm(factors: list[dict],
                                  churn_probability: float) -> str:
    """
    SHAP faktörlerini LLM prompt'u için metin formatına çevirir.

    Returns:
        str: LLM'e gönderilecek bağlam metni
    """
    lines = [f"Churn Riski: %{churn_probability*100:.1f}"]
    lines.append("Ana Risk Faktörleri (SHAP Analizi):")
    for i, f in enumerate(factors, 1):
        lines.append(f"  {i}. {f['feature']}: SHAP={f['shap_value']:.3f} ({f['direction']})")
    return "\n".join(lines)


def plot_summary(shap_values: np.ndarray,
                  X: pd.DataFrame,
                  save: bool = True) -> plt.Figure:
    """
    Tüm veri seti için SHAP summary (beeswarm) grafiği çizer.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        print("✅ SHAP summary grafiği kaydedildi.")

    return fig


def plot_waterfall_single(explainer: shap.TreeExplainer,
                           X_single: pd.DataFrame,
                           save: bool = False) -> plt.Figure:
    """
    Tek bir müşteri için SHAP waterfall grafiği çizer.
    """
    shap_values = explainer(X_single)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight")

    return fig


def get_global_feature_importance(shap_values: np.ndarray,
                                   feature_names: list[str],
                                   top_n: int = 15) -> pd.DataFrame:
    """
    Global feature importance (ortalama |SHAP|) döndürür.

    Returns:
        pd.DataFrame: Özellik adı ve ortalama |SHAP| değeri
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)
    return importance_df
