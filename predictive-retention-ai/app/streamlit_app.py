"""
Predictive Retention AI — Ana Streamlit Uygulaması
3 sayfa: Dashboard | Müşteri Analizi | Mesaj Üretici
Sprint 3: Segmentasyon, gelişmiş filtreler, Plotly SHAP, benzer müşteri
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_pipeline, preprocess_single_customer
from src.models.xgboost_model import (
    train_xgboost, evaluate_model, save_model, load_model, predict_proba_single
)
from src.xai.shap_explainer import (
    get_shap_explainer, compute_shap_values, get_top_factors,
    format_shap_factors_for_llm, get_global_feature_importance,
    plot_waterfall_plotly, plot_beeswarm_plotly,
)
from src.llm.prompt_builder import build_retention_prompt
from src.llm.retention_writer import generate_retention_message
from src.features.segmentation import (
    run_kmeans_segmentation, get_segment_profiles, plot_segments_plotly,
)

# ─── Sayfa Konfigürasyonu ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Retention AI",
    page_icon="chart-line",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Stilleri ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    /* — GLOBAL — */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Ana arka plan — sıcak bej */
    .stApp {
        background-color: #F5F0E8 !important;
        color: #2C2416;
    }
    .main {
        background-color: #F5F0E8 !important;
    }

    /* — SIDEBAR — */
    [data-testid="stSidebar"] {
        background-color: #FFFDF8;
        border-right: 1px solid #E8DDD0;
    }
    [data-testid="stSidebar"] * {
        color: #4A3728 !important;
    }

    /* Sidebar logo alanı */
    [data-testid="stSidebar"] h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.4em !important;
        font-weight: 400 !important;
        color: #2C2416 !important;
        letter-spacing: -0.02em;
        line-height: 1.3;
    }

    /* Başlıklar */
    h1 { font-family: 'DM Serif Display', serif !important;
         color: #2C2416 !important; font-weight: 400 !important; }
    h2, h3, h4 { color: #3D2E1E !important; font-weight: 600 !important; }

    /* Metrik kartları */
    [data-testid="metric-container"] {
        background: #FFFDF8;
        border: 1px solid #E8DDD0;
        border-radius: 14px;
        padding: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 1px 4px rgba(44,36,22,0.06);
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(44,36,22,0.1);
    }
    [data-testid="metric-container"] label {
        color: #8C7560 !important;
        font-size: 0.78em !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #2C2416 !important;
        font-weight: 700 !important;
    }

    /* Butonlar */
    .stButton > button * {
        color: #F5F0E8 !important;
    }
    .stButton > button {
        background: #2C2416;
        color: #F5F0E8 !important;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 500;
        font-size: 0.88em;
        letter-spacing: 0.02em;
        transition: all 0.25s ease;
        box-shadow: none;
    }
    .stButton > button:hover {
        background: #4A3728;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(44,36,22,0.18);
    }

    /* Radio (navigasyon) — pill style */
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        gap: 4px;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background: transparent;
        border-radius: 8px;
        padding: 9px 14px;
        cursor: pointer;
        transition: background 0.18s ease;
        font-size: 0.9em !important;
        font-weight: 500 !important;
        color: #6B5744 !important;
        border: 1px solid transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: #F0E8DC;
        border-color: #DDD0BF;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] + label,
    [data-testid="stSidebar"] [data-baseweb="radio"] label[data-checked="true"] {
        background: #2C2416 !important;
        color: #F5F0E8 !important;
        border-color: transparent;
    }
    /* Aktif radio için görsel vurgu */
    [data-testid="stSidebar"] .st-emotion-cache-j7qwjs {
        display: none;
    }

    /* Sidebar slider & multiselect */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #C9B99A !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background: #2C2416 !important;
    }

    /* Multiselect tags */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background: #E8DDD0 !important;
        color: #2C2416 !important;
        border-radius: 6px !important;
    }

    /* Tab stili */
    .stTabs [data-baseweb="tab-list"] {
        background: #EDE5D8;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #6B5744;
        font-weight: 500;
        font-size: 0.88em;
        padding: 7px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #2C2416 !important;
        color: #F5F0E8 !important;
    }

    /* Info kutusu */
    .info-box {
        background: #FFFDF8;
        border: 1px solid #DDD0BF;
        border-left: 3px solid #C9B99A;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 10px 0;
        color: #4A3728;
        font-size: 0.9em;
    }

    /* DataFrame */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E8DDD0;
    }

    /* Divider */
    hr { border-color: #E8DDD0; margin: 18px 0; }

    /* st.info / st.success / st.warning — warm overrides */
    [data-testid="stNotification"] {
        background: #FFFDF8;
        border-color: #DDD0BF;
        color: #4A3728;
        border-radius: 10px;
    }

    /* Text area */
    .stTextArea textarea {
        background: #FFFDF8;
        border-color: #DDD0BF;
        border-radius: 10px;
        color: #2C2416;
        font-family: 'DM Sans', sans-serif;
    }
    .stTextArea textarea:focus {
        border-color: #C9B99A;
        box-shadow: 0 0 0 2px rgba(201,185,154,0.3);
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input {
        background: #FFFDF8;
        border-color: #DDD0BF;
        border-radius: 8px;
        color: #2C2416;
    }
    .stSelectbox select {
        background: #FFFDF8;
        border-color: #DDD0BF;
    }

    /* Sidebar section labels */
    .sidebar-section {
        font-size: 0.72em;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #A08060;
        font-weight: 600;
        margin: 16px 0 6px 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


@st.cache_data(show_spinner=False)
def load_data():
    """Veri setini yükler (cache'li)."""
    return load_raw_data()


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Eğitilmiş modeli yükler."""
    model_path = MODELS_DIR / "xgboost_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


@st.cache_data(show_spinner=False)
def load_processed_data():
    """İşlenmiş verileri yükler."""
    X_test_path = DATA_DIR / "processed" / "X_test.csv"
    y_test_path = DATA_DIR / "processed" / "y_test.csv"
    if X_test_path.exists() and y_test_path.exists():
        return pd.read_csv(X_test_path), pd.read_csv(y_test_path).squeeze()
    return None, None


def risk_badge(prob: float) -> str:
    if prob >= 0.7:
        return "Yüksek Risk"
    elif prob >= 0.4:
        return "Orta Risk"
    else:
        return "Düşük Risk"


def risk_color(prob: float) -> str:
    if prob >= 0.7:
        return "#ff4b4b"
    elif prob >= 0.4:
        return "#ffa500"
    else:
        return "#00c864"


@st.cache_data(show_spinner=False)
def load_test_ids():
    """CustomerID ↔ test_row eşleştirmesini yükler."""
    p = DATA_DIR / "processed" / "test_ids.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def get_raw_test_subset():
    """Ham veriden test satırlarını döndürür (filtreler için)."""
    X_test, _ = load_processed_data()
    df_raw = load_data()
    if X_test is None or df_raw is None:
        return None
    return df_raw.iloc[X_test.index].reset_index(drop=True)


def find_similar_customers(X_test: pd.DataFrame, customer_idx: int, n: int = 5) -> list:
    """Euclidean mesafeye göre en benzer N müşteriyi bulur."""
    target = X_test.iloc[customer_idx].values
    diffs = X_test.values - target
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    dists[customer_idx] = np.inf  # kendişini dışla
    return list(np.argsort(dists)[:n])


def apply_filters(
    X_test: pd.DataFrame,
    y_proba: np.ndarray,
    y_test,
    raw_subset,
    contract_sel: list,
    tenure_range: tuple,
    charges_range: tuple,
) -> tuple:
    """Sidebar filtrelerini X_test ve y_proba'ya uygular."""
    mask = np.ones(len(X_test), dtype=bool)

    if raw_subset is not None:
        # Contract filtresi
        if contract_sel:
            contract_col = "Contract" if "Contract" in raw_subset.columns else None
            if contract_col:
                mask &= raw_subset[contract_col].isin(contract_sel).values

        # Tenure filtresi
        if "tenure" in raw_subset.columns:
            mask &= (raw_subset["tenure"] >= tenure_range[0]).values
            mask &= (raw_subset["tenure"] <= tenure_range[1]).values

        # MonthlyCharges filtresi
        if "MonthlyCharges" in raw_subset.columns:
            mask &= (raw_subset["MonthlyCharges"] >= charges_range[0]).values
            mask &= (raw_subset["MonthlyCharges"] <= charges_range[1]).values

    X_f = X_test[mask].reset_index(drop=True)
    y_f = y_proba[mask]
    y_t = y_test.values[mask] if y_test is not None else None
    return X_f, y_f, y_t, mask


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo & başlık
    st.markdown(
        '<div style="padding: 8px 0 20px 0;">&nbsp;&nbsp;'
        '<span style="font-family:\'DM Serif Display\',serif;font-size:1.3em;'
        'font-weight:400;color:#2C2416;letter-spacing:-0.02em;">Churnoloji</span>'
        '<br>&nbsp;&nbsp;<span style="font-size:0.72em;color:#A08060;'
        'letter-spacing:0.08em;text-transform:uppercase;font-weight:500;">'
        'Retention AI</span></div>',
        unsafe_allow_html=True,
    )

    # — Navigasyon —
    st.markdown('<div class="sidebar-section">Sayfalar</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Ana Dashboard", "Müşteri Analizi", "Mesaj Üretici"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:#E8DDD0;margin:18px 0;">', unsafe_allow_html=True)

    # — Model durumu —
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    model_exists = (MODELS_DIR / "xgboost_model.pkl").exists()
    if model_exists:
        st.markdown(
            '<div style="background:#F0EAE0;border:1px solid #DDD0BF;border-radius:8px;'
            'padding:9px 14px;font-size:0.85em;color:#4A7C59;font-weight:500;">'
            'Model hazır</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Model eğitilmedi")
        if st.button("Modeli Eğit", key="train_btn"):
            st.session_state["train_model"] = True

    st.markdown('<hr style="border-color:#E8DDD0;margin:18px 0;">', unsafe_allow_html=True)

    # — Risk eşiği —
    st.markdown('<div class="sidebar-section">Risk Eşiği</div>', unsafe_allow_html=True)
    risk_threshold = st.slider(
        "Yüksek risk sınırı",
        min_value=0.3, max_value=0.9, value=0.7, step=0.05,
        format="%.2f",
    )
    st.session_state["risk_threshold"] = risk_threshold

    st.markdown('<hr style="border-color:#E8DDD0;margin:18px 0;">', unsafe_allow_html=True)

    # — Filtreler —
    st.markdown('<div class="sidebar-section">Filtreler</div>', unsafe_allow_html=True)

    contract_options = ["Month-to-month", "One year", "Two year"]
    contract_sel = st.multiselect(
        "Sözleşme Tipi",
        options=contract_options,
        default=contract_options,
        key="filter_contract",
    )

    tenure_range = st.slider(
        "Abonelik Süresi (ay)",
        min_value=0, max_value=72,
        value=(0, 72), step=1,
        key="filter_tenure",
    )

    charges_range = st.slider(
        "Aylık Ücret ($)",
        min_value=0, max_value=120,
        value=(0, 120), step=5,
        key="filter_charges",
    )

    if st.button("Filtreleri Sıfırla", key="reset_filters"):
        st.session_state["filter_contract"] = contract_options
        st.session_state["filter_tenure"] = (0, 72)
        st.session_state["filter_charges"] = (0, 120)
        st.rerun()

    st.markdown('<hr style="border-color:#E8DDD0;margin:18px 0;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.73em;color:#B0977D;line-height:1.6;">'
        'IBM Telco Churn Dataset<br>7.043 müşteri &middot; 21 özellik</div>',
        unsafe_allow_html=True,
    )

# ─── Model Eğitimi ───────────────────────────────────────────────────────────

if st.session_state.get("train_model") and not (MODELS_DIR / "xgboost_model.pkl").exists():
    with st.spinner("Model eğitiliyor... (bu işlem 1-2 dakika sürebilir)"):
        try:
            df = load_data()
            results = preprocess_pipeline(df)
            model = train_xgboost(results["X_train"], results["y_train"])
            metrics = evaluate_model(model, results["X_test"], results["y_test"])
            save_model(model)
            st.session_state["last_metrics"] = metrics
            st.session_state["feature_names"] = results["feature_names"]
            load_trained_model.clear()
            load_processed_data.clear()
            st.success(f"Model başarıyla eğitildi. F1 Skoru: {metrics['f1_score']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
            st.session_state["train_model"] = False
            st.rerun()
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# ─── SAYFA 1: ANA DASHBOARD ──────────────────────────────────────────────────

if page == "Ana Dashboard":
    st.markdown("# Müşteri Churn Risk Dashboard")
    st.markdown("*Tüm müşteriler için risk skorları, filtreler ve segmentasyon*")
    st.markdown("---")

    model = load_trained_model()

    if model is None:
        st.info("Henüz bir model eğitilmedi. Analize başlamak için sol menüden Modeli Eğit butonuna tıklayın.")
    else:
        X_test_full, y_test_full = load_processed_data()

        if X_test_full is None:
            st.warning("İşlenmiş test verisi bulunamadı. Lütfen modeli yeniden eğitin.")
        else:
            y_proba_full = model.predict_proba(X_test_full)[:, 1]
            raw_subset = get_raw_test_subset()

            contract_sel = st.session_state.get("filter_contract", ["Month-to-month", "One year", "Two year"])
            tenure_range = st.session_state.get("filter_tenure", (0, 72))
            charges_range = st.session_state.get("filter_charges", (0, 120))

            X_test, y_proba, y_test_arr, mask = apply_filters(
                X_test_full, y_proba_full, y_test_full, raw_subset,
                contract_sel, tenure_range, charges_range,
            )

            if len(X_test) == 0:
                st.warning("Seçilen filtre kriterleriyle eşleşen müşteri bulunamadı.")
            else:
                if mask.sum() < len(mask):
                    st.info(f"Filtre aktif: **{len(X_test):,}** / {len(X_test_full):,} müşteri gösteriliyor")

                threshold = st.session_state.get("risk_threshold", 0.7)
                n_high = (y_proba >= 0.7).sum()
                n_medium = ((y_proba >= 0.4) & (y_proba < 0.7)).sum()
                n_low = (y_proba < 0.4).sum()
                avg_risk = y_proba.mean()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Yüksek Risk", f"{n_high}", f"%{n_high/len(y_proba)*100:.1f}")
                with col2:
                    st.metric("Orta Risk", f"{n_medium}", f"%{n_medium/len(y_proba)*100:.1f}")
                with col3:
                    st.metric("Düşük Risk", f"{n_low}", f"%{n_low/len(y_proba)*100:.1f}")
                with col4:
                    st.metric("Ortalama Risk", f"%{avg_risk*100:.1f}")

                st.markdown("---")
                tab_risk, tab_seg = st.tabs(["Risk Tablosu", "Müşteri Segmentleri"])

                with tab_risk:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### Risk Dağılımı")
                        risk_labels = pd.cut(y_proba, bins=[0, 0.4, 0.7, 1.0],
                                             labels=["Düşük", "Orta", "Yüksek"])
                        fig_pie = px.pie(
                            values=risk_labels.value_counts().values,
                            names=risk_labels.value_counts().index,
                            color=risk_labels.value_counts().index,
                            color_discrete_map={"Yüksek": "#ff4b4b", "Orta": "#ffa500", "Düşük": "#00c864"},
                            hole=0.4,
                        )
                        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                              plot_bgcolor="rgba(0,0,0,0)",
                                              font_color="#2C2416",
                                              legend=dict(bgcolor="rgba(0,0,0,0)"))
                        st.plotly_chart(fig_pie, use_container_width=True, theme=None)

                    with col_b:
                        st.markdown("#### Churn Olasılığı Dağılımı")
                        fig_hist = px.histogram(x=y_proba, nbins=40,
                                                labels={"x": "Churn Olasılığı", "y": "Müşteri Sayısı"},
                                                color_discrete_sequence=["#667eea"])
                        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ff4b4b",
                                           annotation_text="Risk Eşiği", annotation_font_color="#ff4b4b")
                        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                               plot_bgcolor="rgba(255,253,248,0.5)", font_color="#2C2416")
                        st.plotly_chart(fig_hist, use_container_width=True, theme=None)

                    st.markdown("---")
                    st.markdown(f"#### Yüksek Riskli Müşteriler (Eşik: %{threshold*100:.0f})")
                    high_risk_idx = np.where(y_proba >= threshold)[0]

                    if len(high_risk_idx) == 0:
                        st.info("Seçilen eşikte yüksek riskli müşteri bulunamadı.")
                    else:
                        risk_df = X_test.iloc[high_risk_idx].copy()
                        risk_df["Churn Olasılığı"] = y_proba[high_risk_idx]
                        risk_df["Risk Seviyesi"] = risk_df["Churn Olasılığı"].apply(risk_badge)
                        if y_test_arr is not None:
                            risk_df["Gerçek Değer"] = y_test_arr[high_risk_idx]
                        display_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges",
                                        "Churn Olasılığı", "Risk Seviyesi", "Gerçek Değer"]
                                        if c in risk_df.columns]
                        display_df = risk_df[display_cols].copy()
                        display_df["Churn Olasılığı"] = display_df["Churn Olasılığı"].apply(
                            lambda x: f"%{x*100:.1f}")
                        st.dataframe(display_df.head(50), use_container_width=True, height=300)
                        csv = risk_df.to_csv(index=False).encode("utf-8")
                        st.download_button("CSV İndir", csv,
                                           "yuksek_riskli_musteriler.csv", "text/csv", key="download_risk")

                with tab_seg:
                    st.markdown("#### K-Means Müşteri Segmentasyonu")
                    st.markdown("*tenure, MonthlyCharges, TotalCharges ve Churn Riski baz alınarak 4 segment.*")
                    with st.spinner("Segmentasyon hesaplanıyor..."):
                        seg_df = run_kmeans_segmentation(X_test, y_proba, n_clusters=4)
                        profiles = get_segment_profiles(seg_df)

                    seg_colors = ["#ff4b4b", "#ffa500", "#4361ee", "#00c864"]
                    cols_seg = st.columns(len(profiles))
                    for i, (_, row) in enumerate(profiles.iterrows()):
                        with cols_seg[i]:
                            color = seg_colors[i % len(seg_colors)]
                            st.markdown(
                                f'<div style="background:rgba(255,255,255,0.06);border-left:4px solid {color};'
                                f'border-radius:10px;padding:14px 16px;">'
                                f'<div style="font-size:1em;font-weight:600;color:{color};">{row["segment_name"]}</div>'
                                f'<div style="font-size:1.5em;font-weight:700;color:#2C2416;margin:4px 0;">%{row["Ort. Risk"]*100:.1f}</div>'
                                f'<div style="color:#8C7560;font-size:0.82em;">{int(row["Müşteri Sayısı"])} müşteri</div>'
                                f'</div>', unsafe_allow_html=True)

                    st.markdown("")
                    st.plotly_chart(plot_segments_plotly(seg_df), use_container_width=True, theme=None)
                    st.markdown("#### Segment Profilleri")
                    dp = profiles.drop(columns=["segment"]).copy()
                    dp["Ort. Risk"] = dp["Ort. Risk"].apply(lambda x: f"%{x*100:.1f}")
                    for col in ["Ort. Tenure (ay)", "Ort. Aylık Ücret ($)", "Ort. Toplam Ücret ($)"]:
                        if col in dp.columns:
                            dp[col] = dp[col].apply(lambda x: f"{x:.1f}")
                    st.dataframe(dp, use_container_width=True, hide_index=True)




# ─── SAYFA 2: MÜŞTERİ ANALİZİ ───────────────────────────────────────────────

elif page == "Müşteri Analizi":
    st.markdown("# Müşteri Churn Analizi")
    st.markdown("*Tek müşteri için detaylı risk analizi — SHAP Waterfall, Beeswarm ve Benzer Müşteriler*")
    st.markdown("---")

    model = load_trained_model()

    if model is None:
        st.info("Analiz için öncelikle sol panelden modeli eğitmeniz gerekmektedir.")
    else:
        X_test, y_test = load_processed_data()

        if X_test is None:
            st.warning("İşlenmiş müşteri verisi bulunamadı.")
        else:
            y_proba = model.predict_proba(X_test)[:, 1]

            # ── Sekmeli Müşteri Seçimi
            tab_idx, tab_rand = st.tabs(["Index ile Seç", "Rastgele Seç"])
            with tab_idx:
                c1, _ = st.columns([1, 2])
                with c1:
                    customer_idx = st.number_input(
                        f"Müşteri Index'i (0–{len(X_test)-1})",
                        min_value=0, max_value=len(X_test)-1,
                        value=int(st.session_state.get("customer_idx", 0)),
                        step=1, key="customer_idx",
                    )
            with tab_rand:
                st.selectbox(
                    "Risk Grubundan Seç",
                    ["Yüksek Risk (>%70)", "Orta Risk (%40–70)", "Düşük Risk (<%40)", "Tümü"],
                    key="rand_risk_filter",
                )
                
                def pick_random():
                    rf = st.session_state.get("rand_risk_filter", "Tümü")
                    if "Yüksek" in rf:
                        pool = np.where(y_proba >= 0.7)[0]
                    elif "Orta" in rf:
                        pool = np.where((y_proba >= 0.4) & (y_proba < 0.7))[0]
                    elif "Düşük" in rf:
                        pool = np.where(y_proba < 0.4)[0]
                    else:
                        pool = np.arange(len(y_proba))
                    if len(pool) > 0:
                        st.session_state["customer_idx"] = int(np.random.choice(pool))
                        
                st.button("Rastgele Seç", key="random_pick", on_click=pick_random)

            customer_idx = int(st.session_state.get("customer_idx", 0))
            customer_data = X_test.iloc[[customer_idx]]
            churn_prob = float(y_proba[customer_idx])
            true_label = int(y_test.values[customer_idx]) if y_test is not None else None

            # ── Gauge + Info
            st.markdown("---")
            col_gauge, col_info = st.columns([1, 2])
            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    title={"text": "Churn Risk Skoru", "font": {"color": "#4A3728", "size": 18}},
                    number={"suffix": "%", "font": {"color": "#2C2416", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8C7560"},
                        "bar": {"color": risk_color(churn_prob)},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "rgba(44,36,22,0.1)",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(0,200,100,0.15)"},
                            {"range": [40, 70], "color": "rgba(255,165,0,0.15)"},
                            {"range": [70, 100], "color": "rgba(255,75,75,0.15)"},
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#2C2416",
                    height=280, margin=dict(t=60, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True, theme=None)

            with col_info:
                st.markdown(f"### {risk_badge(churn_prob)}")
                st.markdown(f"**Churn Olasılığı:** `%{churn_prob*100:.1f}`")
                if true_label is not None:
                    st.markdown(f"**Gerçek Değer:** {'Aktif Müşteri' if true_label == 0 else 'Ayrıldı (Churn)'}")

                st.markdown("#### Müşteri Özellikleri")
                for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                    if col in customer_data.columns:
                        st.markdown(f"- **{col}:** `{customer_data[col].values[0]:.2f}`")

                st.markdown("#### Top Risk Faktörleri (SHAP)")
                with st.spinner("SHAP değerleri hesaplanıyor..."):
                    explainer = get_shap_explainer(model)
                    shap_vals = explainer.shap_values(customer_data)
                    shap_1d = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
                    factors = get_top_factors(shap_1d, list(customer_data.columns), top_n=5)

                max_shap = max(abs(f["shap_value"]) for f in factors) + 1e-9
                for f in factors:
                    color = "#D34E4E" if f["shap_value"] > 0 else "#408060"
                    bar_pct = int(abs(f["shap_value"]) / max_shap * 100)
                    icon = "↑" if f["shap_value"] > 0 else "↓"
                    st.markdown(
                        f'<div style="background:rgba(44,36,22,0.05);border-radius:8px;padding:8px 12px;'
                        f'margin:4px 0;border-left:3px solid {color};">'
                        f'<div style="display:flex;justify-content:space-between;">'
                        f'<b style="color:{color};">{icon} {f["feature"]}</b>'
                        f'<span style="color:#8C7560;font-size:0.82em;">{f["shap_value"]:+.3f}</span></div>'
                        f'<div style="background:rgba(255,255,255,0.08);border-radius:4px;height:5px;margin-top:5px;">'
                        f'<div style="background:{color};width:{bar_pct}%;height:5px;border-radius:4px;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                st.session_state["current_customer_data"] = customer_data.iloc[0].to_dict()
                st.session_state["current_shap_factors"] = factors
                st.session_state["current_churn_prob"] = churn_prob

            # ── SHAP Waterfall
            st.markdown("---")
            st.markdown("#### SHAP Waterfall Grafiği")
            ev = explainer.expected_value
            if isinstance(ev, np.ndarray):
                base_val = float(ev[0])   # XGBoost binary: [0.] tek elemanlı array
            else:
                base_val = float(ev)
            st.plotly_chart(
                plot_waterfall_plotly(shap_1d, list(customer_data.columns), base_val, top_n=10),
                use_container_width=True,
            )

            # ── Global Feature Importance + Beeswarm
            st.markdown("---")
            st.markdown("#### Global Feature Importance & Beeswarm (Tüm Test Seti)")
            with st.spinner("Tüm test seti üzerinde SHAP hesaplanıyor..."):
                shap_all = explainer.shap_values(X_test)

            col_fi, col_bee = st.columns(2)
            with col_fi:
                importance_df = get_global_feature_importance(shap_all, list(X_test.columns), top_n=12)
                fig_fi = px.bar(
                    importance_df.sort_values("mean_abs_shap"),
                    x="mean_abs_shap", y="feature", orientation="h",
                    title="Global Feature Importance (mean |SHAP|)",
                    color="mean_abs_shap",
                    color_continuous_scale=[[0, "#4361ee"], [0.5, "#4A3728"], [1, "#ff4b4b"]],
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,253,248,0.5)",
                    font_color="#2C2416", coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(44,36,22,0.1)", title="mean |SHAP|"),
                    yaxis=dict(gridcolor="rgba(44,36,22,0.05)"),
                    height=420, margin=dict(l=10, r=10, t=50, b=30),
                )
                st.plotly_chart(fig_fi, use_container_width=True, theme=None)

            with col_bee:
                st.plotly_chart(
                    plot_beeswarm_plotly(shap_all, X_test, top_n=12),
                    use_container_width=True,
                )

            # ── Benzer Müşteriler
            st.markdown("---")
            st.markdown("#### Benzer Müşteriler")
            similar_idxs = find_similar_customers(X_test, customer_idx, n=5)
            sim_rows = []
            for si in similar_idxs:
                row = {"Index": si, "Churn Olasılığı": f"%{y_proba[si]*100:.1f}", "Risk": risk_badge(y_proba[si])}
                if "tenure" in X_test.columns:
                    row["Tenure"] = f"{X_test.iloc[si]['tenure']:.2f}"
                if "MonthlyCharges" in X_test.columns:
                    row["Aylık Ücret"] = f"{X_test.iloc[si]['MonthlyCharges']:.2f}"
                sim_rows.append(row)
            st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

            sel_similar = st.selectbox(
                "Bu müşteriyi analiz et",
                options=[f"Index {si}" for si in similar_idxs],
                key="pick_similar",
            )
            if st.button("Seçili Müşteriyi Yükle", key="load_similar"):
                st.session_state["customer_idx"] = int(sel_similar.split()[-1])
                st.rerun()

            st.markdown("---")
            if st.button("Bu Müşteri İçin Geri Kazanım Mesajı Oluştur", key="goto_msg"):
                st.info("Sol menüden Mesaj Üretici sayfasına geçin.")

# ─── SAYFA 3: MESAJ ÜRETİCİ ──────────────────────────────────────────────────

elif page == "Mesaj Üretici":
    st.markdown("# Kişiselleştirilmiş Geri Kazanım Mesajı")
    st.markdown("*SHAP analiziyle tespit edilen risk faktörlerine göre Groq LLM ile mesaj üret*")
    st.markdown("---")

    if "current_churn_prob" not in st.session_state:
        st.info("Önce **Müşteri Analizi** sayfasından bir müşteri seçin.")
    else:
        churn_prob = st.session_state["current_churn_prob"]
        factors = st.session_state["current_shap_factors"]
        customer_data = st.session_state["current_customer_data"]
        rc = risk_color(churn_prob)

        # ── Özet Kartları
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.06);border-radius:14px;padding:20px;">'
                f'<div style="color:#8C7560;font-size:0.82em;text-transform:uppercase;letter-spacing:1px;">Risk Skoru</div>'
                f'<div style="font-size:2.8em;font-weight:800;color:{rc};">%{churn_prob*100:.1f}</div>'
                f'<div style="font-size:1.1em;margin:4px 0;">{risk_badge(churn_prob)}</div>'
                f'<hr style="border-color:rgba(44,36,22,0.1);margin:12px 0;">'
                f'<div style="color:#4A3728;">Tenure: <b>{customer_data.get("tenure", 0):.0f} ay</b></div>'
                f'<div style="color:#4A3728;">Aylık: <b>${customer_data.get("MonthlyCharges", 0):.2f}</b></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("#### SHAP Risk Faktörleri")
            max_sv = max(abs(f["shap_value"]) for f in factors) + 1e-9
            for f in factors:
                color = "#D34E4E" if f["shap_value"] > 0 else "#408060"
                bar_w = int(abs(f["shap_value"]) / max_sv * 100)
                icon = "↑" if f["shap_value"] > 0 else "↓"
                st.markdown(
                    f'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                    f'<span style="color:{color};font-weight:600;">{icon} {f["feature"]}</span>'
                    f'<span style="color:#8C7560;font-size:0.82em;">{f["direction"]}</span></div>'
                    f'<div style="background:rgba(255,255,255,0.08);border-radius:4px;height:6px;">'
                    f'<div style="background:{color};width:{bar_w}%;height:6px;border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Mesaj Ayarları
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            customer_name = st.text_input("Müşteri İsmi", value="Sayın Müşterimiz", key="msg_name")
        with col_b:
            tone = st.selectbox("Mesaj Tonu", ["samimi", "resmi", "kısa"], key="msg_tone")
        with col_c:
            model_name = st.selectbox(
                "Groq Modeli",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                key="msg_model",
            )

        if st.button("Mesaj Oluştur", key="generate_msg"):
            prompt = build_retention_prompt(
                customer_info={**customer_data, "name": customer_name},
                shap_factors=factors, churn_probability=churn_prob, tone=tone,
            )
            with st.spinner("Mesaj yazılıyor..."):
                message = generate_retention_message(user_prompt=prompt, model=model_name)
            st.session_state["generated_message"] = message

        # ── Üretilen Mesaj
        if "generated_message" in st.session_state:
            st.markdown("---")
            msg = st.session_state["generated_message"]
            words = len(msg.split())
            read_t = max(1, round(words / 200))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Kelime", words)
            mc2.metric("Karakter", len(msg))
            mc3.metric("Okuma", f"~{read_t} dk")

            st.markdown("#### E-posta Önizleme")
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.97);color:#1a1a2e;border-radius:12px;'
                f'padding:28px 32px;font-family:Georgia,serif;line-height:1.75;font-size:0.96em;'
                f'box-shadow:0 8px 32px rgba(255,253,248,0.5);">'
                f'<div style="border-bottom:2px solid #2C2416;padding-bottom:12px;margin-bottom:16px;">'
                f'<span style="font-size:0.82em;color:#666;">Kimden: Müşteri Deneyimi Ekibi</span><br>'
                f'<span style="font-size:0.82em;color:#666;">Konu: Size Özel Teklifimiz</span></div>'
                f'{msg.replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Düzenle:**")
            edited_message = st.text_area(
                "", value=msg, height=220, key="message_editor", label_visibility="collapsed"
            )

            col_dl, col_clr, _ = st.columns([1, 1, 3])
            with col_dl:
                st.download_button(
                    "İndir (.txt)", edited_message,
                    "geri_kazanim_mesaji.txt", "text/plain", key="dl_msg",
                )
            with col_clr:
                if st.button("Temizle", key="clear_msg"):
                    del st.session_state["generated_message"]
                    st.rerun()

            st.markdown(
                '<div class="info-box"><b>İpucu:</b> E-posta önizlemesini tarayıcıdan kopyalayıp '
                'CRM sisteminize yapıştırabilirsiniz.</div>',
                unsafe_allow_html=True,
            )
