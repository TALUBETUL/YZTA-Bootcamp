"""
SHAP Açıklanabilirlik Modülü
XGBoost modeli için SHAP değerlerini hesaplar ve yorumlar.
Sprint 3: Plotly tabanlı waterfall ve beeswarm grafikleri eklendi.
"""

import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
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


# ─── Plotly Görselleştirme Fonksiyonları (Sprint 3) ──────────────────────────

def plot_waterfall_plotly(
    shap_values_single: np.ndarray,
    feature_names: list,
    base_value: float,
    top_n: int = 10,
) -> go.Figure:
    """
    Tek bir müşteri için Plotly tabanlı SHAP Waterfall grafiği.
    Pozitif SHAP → kırmızı (risk artırıcı), negatif → yeşil (risk azaltıcı).
    """
    # Top N faktörleri |SHAP|'a göre seç
    abs_shap = np.abs(shap_values_single)
    top_idx = np.argsort(abs_shap)[-top_n:]
    sv_top = shap_values_single[top_idx]
    fn_top = [feature_names[i] for i in top_idx]

    # Değere göre sırala (negatiften pozitife)
    order = np.argsort(sv_top)
    sv_sorted = sv_top[order]
    fn_sorted = [fn_top[i] for i in order]

    # Gösterilmeyen özelliklerin toplam etkisi
    others_val = float(shap_values_single.sum()) - float(sv_top.sum())
    final_pred = base_value + float(shap_values_single.sum())

    all_names = fn_sorted + ["Diğer Özellikler", f"Tahmin: %{final_pred*100:.1f}"]
    all_vals = list(sv_sorted) + [others_val, None]
    all_measures = ["relative"] * len(sv_sorted) + ["relative", "total"]
    all_text = [f"{v:+.3f}" for v in sv_sorted] + [f"{others_val:+.3f}", f"%{final_pred*100:.1f}"]

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=all_measures,
        x=all_vals,
        y=all_names,
        base=base_value,
        increasing={"marker": {"color": "#ff6b6b", "line": {"color": "#ff4b4b", "width": 1}}},
        decreasing={"marker": {"color": "#6bffb8", "line": {"color": "#00c864", "width": 1}}},
        totals={"marker": {"color": "#4A3728", "line": {"color": "#a090ee", "width": 1}}},
        connector={"visible": False},
        text=all_text,
        textposition="outside",
    ))

    fig.add_vline(
        x=base_value, line_dash="dot",
        line_color="rgba(255,255,255,0.35)",
        annotation_text=f"Base: {base_value:.3f}",
        annotation_font_color="#8C7560",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=f"SHAP Waterfall — Risk Bileşenleri (Top {top_n})",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,248,0.5)",
        font_color="#2C2416",
        xaxis=dict(title="SHAP Değeri", gridcolor="rgba(44,36,22,0.1)"),
        yaxis=dict(gridcolor="rgba(44,36,22,0.05)"),
        height=460,
        margin=dict(l=20, r=60, t=60, b=40),
    )
    return fig


def plot_beeswarm_plotly(
    shap_values_all: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 12,
    seed: int = 42,
) -> go.Figure:
    """
    Tüm test seti için Plotly tabanlı SHAP Beeswarm (strip plot).
    Her nokta bir müşteri. Renk = normalize özellik değeri (mavi=düşük, kırmızı=yüksek).
    """
    np.random.seed(seed)
    feature_names = list(X.columns)

    # Top N feature seç (mean |SHAP|)
    mean_abs = np.abs(shap_values_all).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-top_n:][::-1]  # büyükten küçüğe

    fig = go.Figure()

    for rank, idx in enumerate(top_indices):
        feat_name = feature_names[idx]
        sv = shap_values_all[:, idx]
        feat_vals = X.iloc[:, idx].values.astype(float)

        # Özellik değerlerini normalize et (0→1)
        v_min, v_max = feat_vals.min(), feat_vals.max()
        feat_norm = (feat_vals - v_min) / (v_max - v_min + 1e-9)

        # Y ekseni jitter (beeswarm etkisi)
        y_center = top_n - 1 - rank
        jitter = np.random.uniform(-0.3, 0.3, size=len(sv))
        y_pos = y_center + jitter

        # Renk: mavi (düşük) → kırmızı (yüksek)
        colors = [
            f"rgb({int(50 + 200*v)}, {int(100*(1-v)+50)}, {int(220*(1-v)+35)})"
            for v in feat_norm
        ]

        fig.add_trace(go.Scatter(
            x=sv,
            y=y_pos,
            mode="markers",
            name=feat_name,
            marker=dict(color=colors, size=4, opacity=0.55),
            showlegend=False,
            hovertemplate=f"<b>{feat_name}</b><br>SHAP: %{{x:.3f}}<extra></extra>",
        ))

    feature_labels = [feature_names[i] for i in top_indices]

    fig.update_layout(
        title=f"SHAP Beeswarm — Top {top_n} Özellik (Tüm Test Seti)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,248,0.5)",
        font_color="#2C2416",
        xaxis=dict(
            title="SHAP Değeri",
            gridcolor="rgba(44,36,22,0.1)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            zerolinewidth=1,
        ),
        yaxis=dict(
            tickvals=list(range(top_n)),
            ticktext=feature_labels[::-1],
            gridcolor="rgba(44,36,22,0.05)",
        ),
        height=520,
        margin=dict(l=20, r=20, t=60, b=40),
    )

    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
    return fig
