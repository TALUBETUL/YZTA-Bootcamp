"""
Predictive Retention AI — Ana Streamlit Uygulaması
3 sayfa: Dashboard | Müşteri Analizi | Mesaj Üretici
Sprint 3: Segmentasyon, gelişmiş filtreler, Plotly SHAP, benzer müşteri
"""

import base64
import html
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_curve,
)
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_pipeline, preprocess_single_customer
from src.models.xgboost_model import (
    train_xgboost, evaluate_model, save_model, load_model, predict_proba_single,
    validate_model_features, find_cost_optimal_threshold, save_evaluation_report,
)
from src.models.model_comparison import (
    compare_models, load_model_comparison, save_model_comparison,
)
from src.models.calibration import (
    calibration_report, fit_oof_calibrator, load_calibrator, save_calibration,
)
from src.xai.shap_explainer import (
    get_shap_explainer, compute_shap_values, get_top_factors,
    format_shap_factors_for_llm, get_global_feature_importance,
    plot_waterfall_plotly, plot_beeswarm_plotly,
)
from src.llm.prompt_builder import build_retention_prompt, build_batch_summary_prompt
from src.llm.retention_writer import (
    generate_retention_message, generate_batch_summary,
)
from src.features.segmentation import (
    run_kmeans_segmentation, get_segment_profiles, plot_segments_plotly,
)
from src.features.retention_decisioning import recommend_next_best_action
from src.monitoring.model_monitor import (
    build_reference_profile, calculate_drift, group_error_analysis,
    save_reference_profile,
)
from src.operations.governance import OperationsStore, validate_retention_message
from src.operations.crm import approved_records_dataframe, send_to_crm_webhook

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

    /* Sidebar slider labels
       Keep these selectors semantic: broad nested-div selectors also paint
       Streamlit's value and min/max labels, making their text unreadable. */
    [data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] {
        background: transparent !important;
        color: #6B5744 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSliderThumbValue"] *,
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"] * {
        color: #6B5744 !important;
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


@st.cache_resource(show_spinner=False)
def load_probability_calibrator():
    """Load the sidecar calibrator while keeping XGBoost explainable by SHAP."""
    return load_calibrator()


@st.cache_data(show_spinner=False)
def load_drift_reference():
    path = MODELS_DIR / "drift_reference.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return None


@st.cache_data(show_spinner=False)
def load_processed_data():
    """İşlenmiş model verisini ve aynı sıradaki ham test satırlarını yükler."""
    X_test_path = DATA_DIR / "processed" / "X_test.csv"
    y_test_path = DATA_DIR / "processed" / "y_test.csv"
    raw_test_path = DATA_DIR / "processed" / "raw_test.csv"
    if X_test_path.exists() and y_test_path.exists() and raw_test_path.exists():
        return (
            pd.read_csv(X_test_path),
            pd.read_csv(y_test_path).squeeze(),
            pd.read_csv(raw_test_path),
        )
    return None, None, None


@st.cache_data(show_spinner=False)
def load_base_training_data():
    """SMOTE öncesi eğitim verisini model karşılaştırması için yükler."""
    X_path = DATA_DIR / "processed" / "X_train_base.csv"
    y_path = DATA_DIR / "processed" / "y_train_base.csv"
    if X_path.exists() and y_path.exists():
        return pd.read_csv(X_path), pd.read_csv(y_path).squeeze()
    return None, None


def copy_to_clipboard_button(text: str, key: str = "copy_message"):
    """Render a browser-side clipboard button without sending text elsewhere."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <button id="{key}" style="
            background:#2C2416;color:#F5F0E8;border:0;border-radius:9px;
            padding:10px 18px;font:500 14px sans-serif;cursor:pointer;">
            Panoya Kopyala
        </button>
        <span id="{key}-status" style="margin-left:10px;color:#6B5744;font:13px sans-serif;"></span>
        <script>
        const button = document.getElementById("{key}");
        const status = document.getElementById("{key}-status");
        button.addEventListener("click", async () => {{
            const bytes = Uint8Array.from(atob("{encoded}"), c => c.charCodeAt(0));
            const value = new TextDecoder().decode(bytes);
            try {{
                await navigator.clipboard.writeText(value);
                status.textContent = "Kopyalandı";
            }} catch (error) {{
                status.textContent = "Tarayıcı izni gerekli";
            }}
        }});
        </script>
        """,
        height=48,
    )


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
    """Model test satırlarıyla birebir hizalı ham müşteri verisini döndürür."""
    _, _, raw_test = load_processed_data()
    return raw_test


def get_runtime_artifact_issues() -> list[str]:
    """Eksik, eski veya birbiriyle uyumsuz çalışma artefaktlarını tespit eder."""
    required_paths = {
        "model": MODELS_DIR / "xgboost_model.pkl",
        "model metadata": MODELS_DIR / "model_metadata.json",
        "scaler": MODELS_DIR / "scaler.pkl",
        "işlenmiş test verisi": DATA_DIR / "processed" / "X_test.csv",
        "test hedefleri": DATA_DIR / "processed" / "y_test.csv",
        "SMOTE öncesi eğitim verisi": DATA_DIR / "processed" / "X_train_base.csv",
        "SMOTE öncesi eğitim hedefleri": DATA_DIR / "processed" / "y_train_base.csv",
        "ham eğitim referansı": DATA_DIR / "processed" / "raw_train.csv",
        "ham test eşleşmesi": DATA_DIR / "processed" / "raw_test.csv",
        "preprocessing metadata": DATA_DIR / "processed" / "preprocessing_metadata.json",
        "olasılık kalibratörü": MODELS_DIR / "probability_calibrator.pkl",
        "kalibrasyon raporu": MODELS_DIR / "calibration_report.json",
        "drift referansı": MODELS_DIR / "drift_reference.json",
    }
    issues = [
        f"Eksik: {label}"
        for label, path in required_paths.items()
        if not path.exists()
    ]
    if issues:
        return issues

    try:
        model = load_trained_model()
        X_test, y_test, raw_test = load_processed_data()
        issues.extend(validate_model_features(model, X_test))
        if not (len(X_test) == len(y_test) == len(raw_test)):
            issues.append("İşlenmiş veri, hedef ve ham müşteri satır sayıları uyuşmuyor.")
    except Exception as exc:
        issues.append(f"Artefaktlar yüklenemedi: {exc}")
    return issues


def build_single_prediction(
    model,
    customer: dict,
    feature_names: list[str],
) -> dict:
    """Transform and explain one raw customer using the training contract."""
    processed = preprocess_single_customer(
        customer,
        feature_names=feature_names,
    )
    feature_issues = validate_model_features(model, processed)
    if feature_issues:
        raise ValueError(" ".join(feature_issues))
    _, raw_probability = predict_proba_single(model, processed)
    calibrator = load_probability_calibrator()
    probability = (
        float(calibrator.predict([raw_probability])[0])
        if calibrator is not None else raw_probability
    )
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(processed)[0]
    factors = get_top_factors(
        shap_values, list(processed.columns), top_n=5
    )
    return {
        "probability": probability,
        "raw_probability": raw_probability,
        "factors": factors,
        "customer": customer,
    }


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
    raw_f = raw_subset[mask].reset_index(drop=True) if raw_subset is not None else None
    return X_f, y_f, y_t, raw_f, mask


# ─── Sidebar ─────────────────────────────────────────────────────────────────

artifact_issues = get_runtime_artifact_issues()
artifacts_ready = not artifact_issues

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
        "Sayfalar",
        ["Ana Dashboard", "Müşteri Analizi", "Mesaj Üretici", "Retention Operasyonları"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:#E8DDD0;margin:18px 0;">', unsafe_allow_html=True)

    # — Model durumu —
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    if artifacts_ready:
        st.markdown(
            '<div style="background:#F0EAE0;border:1px solid #DDD0BF;border-radius:8px;'
            'padding:9px 14px;font-size:0.85em;color:#4A7C59;font-weight:500;">'
            'Model hazır</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Model/veri artefaktları hazır değil")
        with st.expander("Detay"):
            for issue in artifact_issues:
                st.caption(f"• {issue}")
        if st.button("Veriyi ve Modeli Hazırla", key="train_btn"):
            st.session_state["train_model"] = True

    st.markdown('<hr style="border-color:#E8DDD0;margin:18px 0;">', unsafe_allow_html=True)

    # — Risk eşiği —
    st.markdown('<div class="sidebar-section">Risk Eşiği</div>', unsafe_allow_html=True)
    risk_threshold = st.slider(
        "Yüksek risk sınırı",
        min_value=0.05, max_value=0.95,
        value=float(st.session_state.get("risk_threshold", 0.7)), step=0.01,
        format="%.2f",
        key="risk_threshold",
    )

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

if st.session_state.get("train_model"):
    with st.spinner("Model eğitiliyor... (bu işlem 1-2 dakika sürebilir)"):
        try:
            df = load_data()
            results = preprocess_pipeline(df)
            model = train_xgboost(results["X_train"], results["y_train"])
            metrics = evaluate_model(model, results["X_test"], results["y_test"])
            save_model(model, metrics=metrics)
            save_evaluation_report(metrics, model_params=model.get_params())
            calibrator, _ = fit_oof_calibrator(
                model,
                results["X_train_unbalanced"],
                results["y_train_unbalanced"],
            )
            calibrated_test = calibrator.predict(metrics["y_proba"])
            save_calibration(
                calibrator,
                calibration_report(results["y_test"], metrics["y_proba"], calibrated_test),
            )
            save_reference_profile(
                build_reference_profile(results["raw_train"]),
                MODELS_DIR / "drift_reference.json",
            )
            st.session_state["last_metrics"] = metrics
            st.session_state["feature_names"] = results["feature_names"]
            load_trained_model.clear()
            load_processed_data.clear()
            load_base_training_data.clear()
            load_probability_calibrator.clear()
            load_drift_reference.clear()
            load_test_ids.clear()
            get_raw_test_subset.clear()
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

    model = load_trained_model() if artifacts_ready else None

    if model is None:
        st.info("Analize başlamak için sol menüden Veriyi ve Modeli Hazırla butonuna tıklayın.")
    else:
        X_test_full, y_test_full, raw_subset = load_processed_data()

        if X_test_full is None:
            st.warning("Uyumlu test verisi bulunamadı. Lütfen modeli yeniden hazırlayın.")
        else:
            raw_proba_full = model.predict_proba(X_test_full)[:, 1]
            calibrator = load_probability_calibrator()
            y_proba_full = (
                calibrator.predict(raw_proba_full)
                if calibrator is not None else raw_proba_full
            )

            contract_sel = st.session_state.get("filter_contract", ["Month-to-month", "One year", "Two year"])
            tenure_range = st.session_state.get("filter_tenure", (0, 72))
            charges_range = st.session_state.get("filter_charges", (0, 120))

            X_test, y_proba, y_test_arr, raw_filtered, mask = apply_filters(
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
                tab_risk, tab_seg, tab_perf = st.tabs([
                    "Risk Tablosu", "Müşteri Segmentleri", "Model Performansı",
                ])

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
                        st.plotly_chart(fig_pie, width="stretch", theme=None)

                    with col_b:
                        st.markdown("#### Churn Olasılığı Dağılımı")
                        fig_hist = px.histogram(x=y_proba, nbins=40,
                                                labels={"x": "Churn Olasılığı", "y": "Müşteri Sayısı"},
                                                color_discrete_sequence=["#667eea"])
                        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#ff4b4b",
                                           annotation_text="Risk Eşiği", annotation_font_color="#ff4b4b")
                        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                               plot_bgcolor="rgba(255,253,248,0.5)", font_color="#2C2416")
                        st.plotly_chart(fig_hist, width="stretch", theme=None)

                    st.markdown("---")
                    st.markdown(f"#### Yüksek Riskli Müşteriler (Eşik: %{threshold*100:.0f})")
                    high_risk_idx = np.where(y_proba >= threshold)[0]

                    if len(high_risk_idx) == 0:
                        st.info("Seçilen eşikte yüksek riskli müşteri bulunamadı.")
                    else:
                        risk_df = raw_filtered.iloc[high_risk_idx].copy()
                        risk_df["Churn Olasılığı"] = y_proba[high_risk_idx]
                        risk_df["Risk Seviyesi"] = risk_df["Churn Olasılığı"].apply(risk_badge)
                        if y_test_arr is not None:
                            risk_df["Gerçek Değer"] = y_test_arr[high_risk_idx]
                        display_cols = [c for c in ["customerID", "Contract", "tenure",
                                        "MonthlyCharges", "TotalCharges",
                                        "Churn Olasılığı", "Risk Seviyesi", "Gerçek Değer"]
                                        if c in risk_df.columns]
                        display_df = risk_df[display_cols].copy()
                        display_df["Churn Olasılığı"] = display_df["Churn Olasılığı"].apply(
                            lambda x: f"%{x*100:.1f}")
                        st.dataframe(display_df.head(50), width="stretch", height=300)
                        csv = risk_df.to_csv(index=False).encode("utf-8")
                        st.download_button("CSV İndir", csv,
                                           "yuksek_riskli_musteriler.csv", "text/csv", key="download_risk")

                        st.markdown("#### Yönetici Özet Raporu")
                        if st.button("Yönetici Özeti Oluştur", key="generate_manager_summary"):
                            summary_customers = [
                                {
                                    "id": row.get("customerID", "?"),
                                    "churn_prob": row["Churn Olasılığı"],
                                    "tenure": row.get("tenure", "?"),
                                    "MonthlyCharges": row.get("MonthlyCharges", "?"),
                                }
                                for _, row in risk_df.head(10).iterrows()
                            ]
                            summary_prompt = build_batch_summary_prompt(summary_customers)
                            with st.spinner("Yönetici özeti hazırlanıyor..."):
                                st.session_state["manager_summary"] = generate_batch_summary(
                                    summary_prompt
                                )
                        if "manager_summary" in st.session_state:
                            summary = st.text_area(
                                "Rapor",
                                value=st.session_state["manager_summary"],
                                height=220,
                                key="manager_summary_editor",
                            )
                            st.download_button(
                                "Raporu İndir",
                                summary,
                                "yonetici_ozet_raporu.txt",
                                "text/plain",
                                key="download_manager_summary",
                            )

                with tab_seg:
                    st.markdown("#### K-Means Müşteri Segmentasyonu")
                    st.markdown("*tenure, MonthlyCharges, TotalCharges ve Churn Riski baz alınarak 4 segment.*")
                    with st.spinner("Segmentasyon hesaplanıyor..."):
                        seg_df = run_kmeans_segmentation(raw_filtered, y_proba, n_clusters=4)
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
                    st.plotly_chart(plot_segments_plotly(seg_df), width="stretch", theme=None)
                    st.markdown("#### Segment Profilleri")
                    dp = profiles.drop(columns=["segment"]).copy()
                    dp["Ort. Risk"] = dp["Ort. Risk"].apply(lambda x: f"%{x*100:.1f}")
                    for col in ["Ort. Tenure (ay)", "Ort. Aylık Ücret ($)", "Ort. Toplam Ücret ($)"]:
                        if col in dp.columns:
                            dp[col] = dp[col].apply(lambda x: f"{x:.1f}")
                    st.dataframe(dp, width="stretch", hide_index=True)

                with tab_perf:
                    st.markdown("#### Sınıflandırma Performansı")
                    eval_threshold = st.session_state.get("risk_threshold", 0.7)
                    y_eval = np.asarray(y_test_arr)
                    y_pred_eval = (y_proba >= eval_threshold).astype(int)
                    cm = confusion_matrix(y_eval, y_pred_eval, labels=[0, 1])

                    metric_cols = st.columns(4)
                    metric_cols[0].metric(
                        "Precision", f"{precision_score(y_eval, y_pred_eval, zero_division=0):.3f}"
                    )
                    metric_cols[1].metric(
                        "Recall", f"{recall_score(y_eval, y_pred_eval, zero_division=0):.3f}"
                    )
                    metric_cols[2].metric(
                        "F1", f"{f1_score(y_eval, y_pred_eval, zero_division=0):.3f}"
                    )

                    fpr, tpr, _ = roc_curve(y_eval, y_proba)
                    precision_curve, recall_curve, _ = precision_recall_curve(y_eval, y_proba)
                    roc_auc_value = auc(fpr, tpr)
                    pr_auc_value = auc(recall_curve[::-1], precision_curve[::-1])
                    metric_cols[3].metric("PR-AUC", f"{pr_auc_value:.3f}")

                    chart_left, chart_right = st.columns(2)
                    with chart_left:
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            x=["Aktif Tahmin", "Churn Tahmin"],
                            y=["Aktif Gerçek", "Churn Gerçek"],
                            color_continuous_scale=["#FFFDF8", "#D34E4E"],
                            title=f"Confusion Matrix — Eşik {eval_threshold:.2f}",
                        )
                        fig_cm.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_cm, width="stretch")

                    with chart_right:
                        fig_curves = go.Figure()
                        fig_curves.add_trace(go.Scatter(
                            x=fpr, y=tpr, name=f"ROC (AUC={roc_auc_value:.3f})"
                        ))
                        fig_curves.add_trace(go.Scatter(
                            x=recall_curve,
                            y=precision_curve,
                            name=f"Precision-Recall (AUC={pr_auc_value:.3f})",
                        ))
                        fig_curves.update_layout(
                            title="ROC ve Precision-Recall Eğrileri",
                            xaxis_title="FPR / Recall",
                            yaxis_title="TPR / Precision",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,253,248,0.5)",
                        )
                        st.plotly_chart(fig_curves, width="stretch")

                    st.markdown("---")
                    st.markdown("#### İş Maliyetine Göre Önerilen Eşik")
                    cost_a, cost_b = st.columns(2)
                    fn_cost = cost_a.number_input(
                        "Kaçırılan churn maliyeti", min_value=1.0,
                        value=5.0, step=1.0, key="fn_cost",
                    )
                    fp_cost = cost_b.number_input(
                        "Gereksiz aksiyon maliyeti", min_value=0.1,
                        value=1.0, step=0.5, key="fp_cost",
                    )
                    optimal = find_cost_optimal_threshold(
                        y_eval, y_proba, fn_cost, fp_cost
                    )
                    st.info(
                        f"Önerilen eşik: **{optimal['threshold']:.2f}** · "
                        f"Toplam maliyet: **{optimal['cost']:.1f}** · "
                        f"Recall: **{optimal['recall']:.3f}** · "
                        f"False negative: **{optimal['false_negatives']}**"
                    )

                    def set_recommended_threshold(value):
                        st.session_state["risk_threshold"] = float(value)

                    st.button(
                        "Önerilen Eşiği Uygula",
                        key="apply_cost_threshold",
                        on_click=set_recommended_threshold,
                        args=(optimal["threshold"],),
                    )

                    st.markdown("---")
                    st.markdown("#### Baseline Model Karşılaştırması")
                    comparison = load_model_comparison()
                    if comparison is None:
                        st.caption(
                            "Logistic Regression, Random Forest ve XGBoost leakage-safe "
                            "cross-validation ile henüz karşılaştırılmadı."
                        )
                        if st.button("Model Karşılaştırmasını Çalıştır", key="run_comparison"):
                            X_base, y_base = load_base_training_data()
                            if X_base is None:
                                st.error("SMOTE öncesi eğitim verisi bulunamadı. Modeli yeniden hazırlayın.")
                            else:
                                with st.spinner("Modeller karşılaştırılıyor..."):
                                    comparison = compare_models(X_base, y_base)
                                    save_model_comparison(comparison)
                    if comparison is not None:
                        comparison_display = comparison[
                            ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
                        ].copy()
                        st.dataframe(
                            comparison_display.style.format({
                                col: "{:.3f}" for col in comparison_display.columns if col != "model"
                            }),
                            width="stretch",
                            hide_index=True,
                        )




# ─── SAYFA 2: MÜŞTERİ ANALİZİ ───────────────────────────────────────────────

elif page == "Müşteri Analizi":
    st.markdown("# Müşteri Churn Analizi")
    st.markdown("*Tek müşteri için detaylı risk analizi — SHAP Waterfall, Beeswarm ve Benzer Müşteriler*")
    st.markdown("---")

    model = load_trained_model() if artifacts_ready else None

    if model is None:
        st.info("Analiz için öncelikle sol panelden modeli eğitmeniz gerekmektedir.")
    else:
        X_test, y_test, raw_test = load_processed_data()

        if X_test is None:
            st.warning("İşlenmiş müşteri verisi bulunamadı.")
        else:
            raw_proba = model.predict_proba(X_test)[:, 1]
            calibrator = load_probability_calibrator()
            y_proba = calibrator.predict(raw_proba) if calibrator is not None else raw_proba

            # ── Sekmeli Müşteri Seçimi
            def clear_custom_customer():
                st.session_state["use_custom_customer"] = False

            tab_idx, tab_id, tab_rand, tab_live = st.tabs([
                "Index ile Seç", "CustomerID ile Ara", "Rastgele Seç", "Yeni Müşteri",
            ])
            with tab_idx:
                c1, _ = st.columns([1, 2])
                with c1:
                    customer_idx = st.number_input(
                        f"Müşteri Index'i (0–{len(X_test)-1})",
                        min_value=0, max_value=len(X_test)-1,
                        value=int(st.session_state.get("customer_idx", 0)),
                        step=1, key="customer_idx",
                        on_change=clear_custom_customer,
                    )
            with tab_id:
                search_customer_id = st.text_input(
                    "CustomerID",
                    placeholder="Örn. 7590-VHVEG",
                    key="customer_id_search",
                )
                if st.button("Müşteriyi Bul", key="find_customer_id"):
                    matches = raw_test.index[
                        raw_test["customerID"].astype(str).str.upper()
                        == search_customer_id.strip().upper()
                    ].tolist()
                    if matches:
                        st.session_state["customer_idx"] = int(matches[0])
                        st.session_state["use_custom_customer"] = False
                        st.session_state.pop("searched_customer_result", None)
                        st.success(f"Müşteri bulundu: test index {matches[0]}")
                    else:
                        all_customers = load_data()
                        all_matches = all_customers[
                            all_customers["customerID"].astype(str).str.upper()
                            == search_customer_id.strip().upper()
                        ]
                        if all_matches.empty:
                            st.error("Bu CustomerID veri setinde bulunamadı.")
                        else:
                            try:
                                searched_raw = all_matches.iloc[0].drop(
                                    labels=["Churn"]
                                ).to_dict()
                                searched_result = build_single_prediction(
                                    model, searched_raw, list(X_test.columns)
                                )
                                st.session_state["searched_customer_result"] = searched_result
                                st.session_state["current_customer_data"] = searched_raw
                                st.session_state["current_shap_factors"] = searched_result["factors"]
                                st.session_state["current_churn_prob"] = searched_result["probability"]
                                st.session_state["use_custom_customer"] = True
                                st.success("Müşteri bulundu ve canlı tahmin oluşturuldu.")
                            except Exception as exc:
                                st.error(f"Müşteri tahmin edilemedi: {exc}")
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
                        st.session_state["use_custom_customer"] = False
                        
                st.button("Rastgele Seç", key="random_pick", on_click=pick_random)

            with tab_live:
                st.markdown("#### Yeni Müşteri İçin Canlı Tahmin")
                with st.form("live_customer_form"):
                    lc1, lc2, lc3 = st.columns(3)
                    gender = lc1.selectbox("Gender", ["Female", "Male"])
                    senior = lc2.selectbox("Senior Citizen", ["No", "Yes"])
                    partner = lc3.selectbox("Partner", ["No", "Yes"])
                    dependents = lc1.selectbox("Dependents", ["No", "Yes"])
                    tenure = lc2.number_input("Tenure (ay)", 0, 72, 12)
                    phone_service = lc3.selectbox("Phone Service", ["Yes", "No"])
                    multiple_lines = lc1.selectbox(
                        "Multiple Lines", ["No", "Yes", "No phone service"]
                    )
                    internet_service = lc2.selectbox(
                        "Internet Service", ["DSL", "Fiber optic", "No"]
                    )
                    service_options = ["No", "Yes", "No internet service"]
                    online_security = lc3.selectbox("Online Security", service_options)
                    online_backup = lc1.selectbox("Online Backup", service_options)
                    device_protection = lc2.selectbox("Device Protection", service_options)
                    tech_support = lc3.selectbox("Tech Support", service_options)
                    streaming_tv = lc1.selectbox("Streaming TV", service_options)
                    streaming_movies = lc2.selectbox("Streaming Movies", service_options)
                    contract = lc3.selectbox(
                        "Contract", ["Month-to-month", "One year", "Two year"]
                    )
                    paperless = lc1.selectbox("Paperless Billing", ["Yes", "No"])
                    payment_method = lc2.selectbox(
                        "Payment Method",
                        [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)",
                        ],
                    )
                    monthly_charges = lc3.number_input(
                        "Monthly Charges ($)", 0.0, 150.0, 65.0, 1.0
                    )
                    total_charges = lc1.number_input(
                        "Total Charges ($)", 0.0, 10000.0,
                        float(65 * 12), 10.0,
                    )
                    live_submit = st.form_submit_button("Canlı Tahmin Yap")

                if live_submit:
                    live_customer = {
                        "customerID": "LIVE-CUSTOMER",
                        "gender": gender,
                        "SeniorCitizen": int(senior == "Yes"),
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                    }
                    try:
                        live_result = build_single_prediction(
                            model, live_customer, list(X_test.columns)
                        )
                        st.session_state["live_prediction_result"] = live_result
                        st.session_state["current_customer_data"] = live_customer
                        st.session_state["current_shap_factors"] = live_result["factors"]
                        st.session_state["current_churn_prob"] = live_result["probability"]
                        st.session_state["use_custom_customer"] = True
                    except Exception as exc:
                        st.error(f"Canlı tahmin oluşturulamadı: {exc}")

                if "live_prediction_result" in st.session_state:
                    live_result = st.session_state["live_prediction_result"]
                    st.success(
                        f"Churn olasılığı: %{live_result['probability']*100:.1f} · "
                        f"{risk_badge(live_result['probability'])}"
                    )
                    live_factor_df = pd.DataFrame(live_result["factors"])
                    st.dataframe(live_factor_df, width="stretch", hide_index=True)
                    st.caption("Bu müşteri Mesaj Üretici sayfasında kullanıma hazır.")

            if "searched_customer_result" in st.session_state:
                searched_result = st.session_state["searched_customer_result"]
                searched_id = searched_result["customer"].get("customerID", "?")
                st.success(
                    f"CustomerID {searched_id} · Canlı churn olasılığı: "
                    f"%{searched_result['probability']*100:.1f} · "
                    f"{risk_badge(searched_result['probability'])}"
                )
                st.dataframe(
                    pd.DataFrame(searched_result["factors"]),
                    width="stretch",
                    hide_index=True,
                )

            customer_idx = int(st.session_state.get("customer_idx", 0))
            customer_data = X_test.iloc[[customer_idx]]
            customer_raw = raw_test.iloc[[customer_idx]]
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
                st.plotly_chart(fig_gauge, width="stretch", theme=None)

            with col_info:
                st.markdown(f"### {risk_badge(churn_prob)}")
                st.markdown(f"**Churn Olasılığı:** `%{churn_prob*100:.1f}`")
                if true_label is not None:
                    st.markdown(f"**Gerçek Değer:** {'Aktif Müşteri' if true_label == 0 else 'Ayrıldı (Churn)'}")

                st.markdown("#### Müşteri Özellikleri")
                if "customerID" in customer_raw.columns:
                    st.markdown(f"- **CustomerID:** `{customer_raw['customerID'].iloc[0]}`")
                for col in ["Contract", "tenure", "MonthlyCharges", "TotalCharges"]:
                    if col in customer_raw.columns:
                        value = customer_raw[col].iloc[0]
                        rendered = f"{value:.2f}" if isinstance(value, (int, float, np.number)) else str(value)
                        st.markdown(f"- **{col}:** `{rendered}`")

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

                if not st.session_state.get("use_custom_customer", False):
                    st.session_state["current_customer_data"] = customer_raw.iloc[0].to_dict()
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
                width="stretch",
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
                st.plotly_chart(fig_fi, width="stretch", theme=None)

            with col_bee:
                st.plotly_chart(
                    plot_beeswarm_plotly(shap_all, X_test, top_n=12),
                    width="stretch",
                )

            # ── Benzer Müşteriler
            st.markdown("---")
            st.markdown("#### Benzer Müşteriler")
            similar_idxs = find_similar_customers(X_test, customer_idx, n=5)
            sim_rows = []
            for si in similar_idxs:
                row = {"Index": si, "Churn Olasılığı": f"%{y_proba[si]*100:.1f}", "Risk": risk_badge(y_proba[si])}
                if "customerID" in raw_test.columns:
                    row["CustomerID"] = raw_test.iloc[si]["customerID"]
                if "tenure" in raw_test.columns:
                    row["Tenure"] = f"{raw_test.iloc[si]['tenure']:.0f}"
                if "MonthlyCharges" in raw_test.columns:
                    row["Aylık Ücret"] = f"${raw_test.iloc[si]['MonthlyCharges']:.2f}"
                sim_rows.append(row)
            st.dataframe(pd.DataFrame(sim_rows), width="stretch", hide_index=True)

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
        recommended_offer = recommend_next_best_action(customer_data, churn_prob)

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
        st.markdown("#### Next Best Action ve Teklif Ekonomisi")
        offer_cols = st.columns(4)
        offer_cols[0].metric("Önerilen Aksiyon", recommended_offer["name"])
        offer_cols[1].metric("Tahmini Müşteri Değeri", f"${recommended_offer['customer_value']:.2f}")
        offer_cols[2].metric("Teklif Maliyeti", f"${recommended_offer['offer_cost']:.2f}")
        offer_cols[3].metric("Beklenen Net Değer", f"${recommended_offer['expected_net_value']:.2f}")
        st.caption(
            "Uplift ve net değer şu anda şeffaf senaryo varsayımlarına dayanır; "
            "nedensel sonuç olarak yorumlanmamalıdır. Gerçek uplift A/B sonuçlarıyla ölçülür."
        )

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
                shap_factors=factors,
                churn_probability=churn_prob,
                tone=tone,
                recommended_offer=recommended_offer,
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
                f'{html.escape(msg).replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Düzenle:**")
            edited_message = st.text_area(
                "", value=msg, height=220, key="message_editor", label_visibility="collapsed"
            )

            safety = validate_retention_message(
                edited_message, approved_offer=recommended_offer
            )
            if safety["safe_to_approve"]:
                st.success("Mesaj otomatik doğruluk ve güvenlik kontrollerinden geçti.")
            else:
                st.error("Mesajda onayı engelleyen güvenlik sorunları var.")
            for issue in safety["issues"]:
                renderer = st.error if issue["severity"] == "error" else st.warning
                renderer(issue["message"])

            col_copy, col_dl, col_clr, _ = st.columns([1.2, 1, 1, 2])
            with col_copy:
                copy_to_clipboard_button(edited_message, key="copy_retention_message")
            with col_dl:
                st.download_button(
                    "İndir (.txt)", edited_message,
                    "geri_kazanim_mesaji.txt", "text/plain", key="dl_msg",
                )
            with col_clr:
                if st.button("Temizle", key="clear_msg"):
                    del st.session_state["generated_message"]
                    st.rerun()

            customer_id = str(customer_data.get("customerID", "LIVE-CUSTOMER"))
            if st.button(
                "Mesaj ve Teklifi Onaya Gönder",
                key="submit_for_approval",
                disabled=not safety["safe_to_approve"],
            ):
                record_id = OperationsStore().create_message(
                    customer_id,
                    edited_message,
                    recommended_offer,
                    safety,
                )
                st.success(f"Onay kaydı oluşturuldu: #{record_id}")

            st.markdown(
                '<div class="info-box"><b>İpucu:</b> E-posta önizlemesini tarayıcıdan kopyalayıp '
                'CRM sisteminize yapıştırabilirsiniz.</div>',
                unsafe_allow_html=True,
            )

# ─── SAYFA 4: RETENTION OPERASYONLARI ───────────────────────────────────────

elif page == "Retention Operasyonları":
    st.markdown("# Retention Operasyonları")
    st.markdown(
        "*Model güveni, deney ölçümü, insan onayı, denetim geçmişi ve CRM aktarımı*"
    )
    st.markdown("---")

    tab_trust, tab_experiment, tab_approval, tab_crm = st.tabs([
        "Model Güveni", "A/B Test ve Uplift", "Onay ve Denetim", "CRM Aktarımı",
    ])
    store = OperationsStore()

    with tab_trust:
        if not artifacts_ready:
            st.warning("P2 model artefaktları için modeli sol menüden yeniden hazırlayın.")
        else:
            model = load_trained_model()
            X_test, y_test, raw_test = load_processed_data()
            raw_probabilities = model.predict_proba(X_test)[:, 1]
            calibrator = load_probability_calibrator()
            calibrated_probabilities = calibrator.predict(raw_probabilities)

            st.markdown("#### Olasılık Kalibrasyonu")
            report_path = MODELS_DIR / "calibration_report.json"
            with report_path.open(encoding="utf-8") as handle:
                cal_report = json.load(handle)
            cal_cols = st.columns(3)
            cal_cols[0].metric(
                "Brier Score",
                f"{cal_report['calibrated']['brier_score']:.4f}",
                delta=(
                    f"{cal_report['calibrated']['brier_score'] - cal_report['raw']['brier_score']:+.4f}"
                ),
                delta_color="inverse",
            )
            cal_cols[1].metric(
                "Log Loss", f"{cal_report['calibrated']['log_loss']:.4f}"
            )
            cal_cols[2].metric(
                "Calibration Error",
                f"{cal_report['calibrated']['expected_calibration_error']:.4f}",
            )

            calibration_df = pd.DataFrame({
                "Tahmin": calibrated_probabilities,
                "Gerçek": np.asarray(y_test),
            })
            calibration_df["Aralık"] = pd.cut(
                calibration_df["Tahmin"], bins=np.linspace(0, 1, 11), include_lowest=True
            )
            reliability = (
                calibration_df.groupby("Aralık", observed=True)
                .agg(Tahmin=("Tahmin", "mean"), Gerçekleşen=("Gerçek", "mean"), N=("Gerçek", "size"))
                .reset_index()
            )
            reliability["Aralık"] = reliability["Aralık"].astype(str)
            fig_cal = px.line(
                reliability, x="Tahmin", y="Gerçekleşen", markers=True,
                hover_data=["Aralık", "N"], title="Güvenilirlik Eğrisi",
            )
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="İdeal",
                line={"dash": "dash", "color": "#8C7560"},
            ))
            st.plotly_chart(fig_cal, width="stretch")

            st.markdown("---")
            left_monitor, right_monitor = st.columns(2)
            with left_monitor:
                st.markdown("#### Veri Drift")
                drift = calculate_drift(load_drift_reference(), raw_test)
                st.dataframe(
                    drift.style.format({"drift_score": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )
                if not drift.empty and (drift["status"] == "critical").any():
                    st.error("Kritik drift görüldü; yeniden eğitim değerlendirilmelidir.")
                else:
                    st.success("Kritik veri drift sinyali yok.")

            with right_monitor:
                st.markdown("#### Fairness / Segment Hata Analizi")
                fairness = group_error_analysis(
                    y_test,
                    calibrated_probabilities,
                    raw_test,
                    threshold=st.session_state.get("risk_threshold", 0.7),
                )
                st.dataframe(
                    fairness.style.format({
                        "churn_rate": "{:.3f}",
                        "positive_prediction_rate": "{:.3f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1": "{:.3f}",
                        "roc_auc": "{:.3f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Bu tablo gözlenen grup hatalarını gösterir; hukuki bir fairness sertifikası değildir."
                )

    with tab_experiment:
        st.markdown("#### Retention Kampanyası A/B Ataması")
        campaign_id = st.text_input(
            "Kampanya ID", value="retention-pilot-001", key="campaign_id"
        )
        experiment_customer_id = st.text_input(
            "CustomerID", value="", key="experiment_customer_id"
        )
        if st.button("Varyant Ata", key="assign_variant"):
            if campaign_id.strip() and experiment_customer_id.strip():
                variant = store.assign_variant(
                    campaign_id.strip(), experiment_customer_id.strip()
                )
                st.session_state["assigned_variant"] = variant
                st.success(f"Deterministik atama: {variant}")
            else:
                st.error("Kampanya ID ve CustomerID gereklidir.")

        if "assigned_variant" in st.session_state:
            retained = st.radio(
                "Gözlenen sonuç",
                ["Müşteri kaldı", "Müşteri ayrıldı"],
                horizontal=True,
                key="experiment_outcome",
            )
            if st.button("Sonucu Kaydet", key="save_experiment_outcome"):
                try:
                    store.record_outcome(
                        campaign_id.strip(),
                        experiment_customer_id.strip(),
                        retained == "Müşteri kaldı",
                    )
                    st.success("Kampanya sonucu kaydedildi.")
                except ValueError as exc:
                    st.error(str(exc))

        if campaign_id.strip():
            experiment = store.experiment_summary(campaign_id.strip())
            exp_rows = pd.DataFrame([
                {"Varyant": name, **values}
                for name, values in (
                    ("control", experiment["control"]),
                    ("treatment", experiment["treatment"]),
                )
            ])
            st.dataframe(exp_rows, width="stretch", hide_index=True)
            measured_uplift = experiment["measured_uplift"]
            if measured_uplift is None:
                st.info("Ölçülen uplift için iki varyantta da sonuç verisi gerekir.")
            else:
                st.metric("Ölçülen Retention Uplift", f"{measured_uplift:+.1%}")
                if min(
                    experiment["control"]["measured"],
                    experiment["treatment"]["measured"],
                ) < 30:
                    st.warning("Örneklem küçük; bu sonucu karar vermek için kullanmayın.")

        st.markdown("---")
        st.markdown("#### Tek Müşteri Next Best Action Senaryosu")
        if artifacts_ready:
            model = load_trained_model()
            X_test, _, raw_test = load_processed_data()
            options = raw_test["customerID"].astype(str).tolist()
            selected_customer = st.selectbox(
                "Müşteri", options, key="nba_customer"
            )
            selected_index = options.index(selected_customer)
            raw_probability = model.predict_proba(X_test.iloc[[selected_index]])[:, 1]
            probability = float(load_probability_calibrator().predict(raw_probability)[0])
            action = recommend_next_best_action(
                raw_test.iloc[selected_index].to_dict(), probability
            )
            action_table = pd.DataFrame(action["alternatives"])
            st.success(
                f"Öneri: {action['name']} · Beklenen net değer: "
                f"${action['expected_net_value']:.2f}"
            )
            st.dataframe(
                action_table[[
                    "name", "offer_cost", "scenario_uplift",
                    "expected_benefit", "expected_net_value",
                ]].style.format({
                    "offer_cost": "${:.2f}",
                    "scenario_uplift": "{:.1%}",
                    "expected_benefit": "${:.2f}",
                    "expected_net_value": "${:.2f}",
                }),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Senaryo uplift değerleri varsayımdır; A/B test sonuçları geldikçe "
                "ölçülen değerlerle değiştirilmelidir."
            )

    with tab_approval:
        st.markdown("#### Mesaj ve Teklif Onay Kuyruğu")
        records = store.list_messages()
        pending = [record for record in records if record["status"] == "draft"]
        if not pending:
            st.info("Onay bekleyen kayıt yok.")
        else:
            pending_map = {
                f"#{record['id']} · {record['customer_id']}": record
                for record in pending
            }
            selected_label = st.selectbox(
                "Onay kaydı", list(pending_map), key="approval_record"
            )
            selected_record = pending_map[selected_label]
            st.text_area(
                "Mesaj", selected_record["message"], height=200,
                disabled=True, key="approval_message",
            )
            st.json(selected_record["offer"], expanded=False)
            reviewer = st.text_input("İnceleyen", key="reviewer_name")
            approve_col, reject_col = st.columns(2)
            if approve_col.button("Onayla", key="approve_message"):
                try:
                    store.review_message(
                        selected_record["id"], "approved", reviewer
                    )
                    st.success("Mesaj ve teklif onaylandı.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if reject_col.button("Reddet", key="reject_message"):
                try:
                    store.review_message(
                        selected_record["id"], "rejected", reviewer
                    )
                    st.success("Mesaj ve teklif reddedildi.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

        st.markdown("#### Denetim Geçmişi")
        events = store.list_audit_events()
        if events:
            audit_df = pd.DataFrame(events).drop(columns=["details_json"], errors="ignore")
            st.dataframe(audit_df, width="stretch", hide_index=True)
        else:
            st.caption("Henüz denetim olayı yok.")

    with tab_crm:
        st.markdown("#### Onaylı CRM Aktarımı")
        approved_records = store.list_messages(status="approved")
        export_df = approved_records_dataframe(approved_records)
        if export_df.empty:
            st.info("CRM'e aktarılabilecek onaylı kayıt yok.")
        else:
            st.dataframe(export_df, width="stretch", hide_index=True)
            st.download_button(
                "Onaylı Kayıtları CSV İndir",
                export_df.to_csv(index=False).encode("utf-8"),
                "crm_retention_export.csv",
                "text/csv",
                key="crm_csv",
            )
            webhook_url = st.text_input(
                "CRM HTTPS Webhook", type="password", key="crm_webhook"
            )
            record_options = {
                f"#{record['id']} · {record['customer_id']}": record
                for record in approved_records
            }
            crm_label = st.selectbox(
                "Gönderilecek kayıt", list(record_options), key="crm_record"
            )
            if st.button("CRM'e Gönder", key="send_crm"):
                try:
                    status = send_to_crm_webhook(
                        record_options[crm_label], webhook_url
                    )
                    st.success(f"CRM webhook yanıtı: HTTP {status}")
                except Exception as exc:
                    st.error(f"CRM aktarımı başarısız: {exc}")
