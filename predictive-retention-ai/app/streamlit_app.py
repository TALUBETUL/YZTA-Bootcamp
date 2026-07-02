"""
Predictive Retention AI — Ana Streamlit Uygulaması
3 sayfa: Dashboard | Müşteri Analizi | Mesaj Üretici
"""

import sys
import os
from pathlib import Path

# Proje kökünü Python path'e ekle
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
    format_shap_factors_for_llm, get_global_feature_importance
)
from src.llm.prompt_builder import build_retention_prompt
from src.llm.retention_writer import generate_retention_message

# ─── Sayfa Konfigürasyonu ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Retention AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Stilleri ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Ana arka plan */
    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e8e8f0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }

    /* Başlık */
    h1, h2, h3 { color: #c8b8ff !important; }

    /* Metrik kartları */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(200,184,255,0.2);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(200,184,255,0.5);
    }

    /* Butonlar */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.6);
    }

    /* Risk kartları */
    .risk-high {
        background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(255,75,75,0.05));
        border-left: 4px solid #ff4b4b;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .risk-medium {
        background: linear-gradient(135deg, rgba(255,165,0,0.15), rgba(255,165,0,0.05));
        border-left: 4px solid #ffa500;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(0,200,100,0.15), rgba(0,200,100,0.05));
        border-left: 4px solid #00c864;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Info kutuları */
    .info-box {
        background: rgba(102,126,234,0.1);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }

    /* Tab stili */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a0a0c0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* DataFrame stili */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1); }
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
        return "🔴 Yüksek Risk"
    elif prob >= 0.4:
        return "🟡 Orta Risk"
    else:
        return "🟢 Düşük Risk"


def risk_color(prob: float) -> str:
    if prob >= 0.7:
        return "#ff4b4b"
    elif prob >= 0.4:
        return "#ffa500"
    else:
        return "#00c864"


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🧠 Predictive\nRetention AI")
    st.markdown("---")

    st.markdown("### 📌 Navigasyon")
    page = st.radio(
        "",
        ["📊 Ana Dashboard", "👤 Müşteri Analizi", "✉️ Mesaj Üretici"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Ayarları")

    model_exists = (MODELS_DIR / "xgboost_model.pkl").exists()
    if model_exists:
        st.success("✅ Model hazır")
    else:
        st.warning("⚠️ Model eğitilmedi")
        if st.button("🚀 Modeli Eğit", key="train_btn"):
            st.session_state["train_model"] = True

    st.markdown("---")
    st.markdown("### 🎛️ Risk Eşiği")
    risk_threshold = st.slider(
        "Yüksek Risk Eşiği",
        min_value=0.3, max_value=0.9, value=0.7, step=0.05,
        format="%.2f",
    )
    st.session_state["risk_threshold"] = risk_threshold

    st.markdown("---")
    st.caption("IBM Telco Churn Dataset\n7.043 müşteri | 21 özellik")

# ─── Model Eğitimi ───────────────────────────────────────────────────────────

if st.session_state.get("train_model") and not (MODELS_DIR / "xgboost_model.pkl").exists():
    with st.spinner("🔄 Model eğitiliyor... (bu işlem 1-2 dakika sürebilir)"):
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
            st.success(f"✅ Model eğitildi! F1: {metrics['f1_score']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
            st.session_state["train_model"] = False
            st.rerun()
        except Exception as e:
            st.error(f"❌ Hata: {e}")

# ─── SAYFA 1: ANA DASHBOARD ──────────────────────────────────────────────────

if page == "📊 Ana Dashboard":
    st.markdown("# 📊 Müşteri Churn Risk Dashboard")
    st.markdown("*Tüm müşteriler için risk skorları ve genel istatistikler*")
    st.markdown("---")

    model = load_trained_model()
    df_raw = load_data()

    if model is None:
        st.info("ℹ️ Henüz bir model eğitilmedi. Sol panelden **Modeli Eğit** butonuna tıklayın.")
    else:
        # Test verisi üzerinde tahminler
        X_test, y_test = load_processed_data()

        if X_test is None:
            st.warning("⚠️ İşlenmiş test verisi bulunamadı. Modeli yeniden eğitin.")
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= st.session_state.get("risk_threshold", 0.7)).astype(int)

            # ── Özet Metrikler
            col1, col2, col3, col4 = st.columns(4)
            n_high = (y_proba >= 0.7).sum()
            n_medium = ((y_proba >= 0.4) & (y_proba < 0.7)).sum()
            n_low = (y_proba < 0.4).sum()
            avg_risk = y_proba.mean()

            with col1:
                st.metric("🔴 Yüksek Risk", f"{n_high}", f"%{n_high/len(y_proba)*100:.1f}")
            with col2:
                st.metric("🟡 Orta Risk", f"{n_medium}", f"%{n_medium/len(y_proba)*100:.1f}")
            with col3:
                st.metric("🟢 Düşük Risk", f"{n_low}", f"%{n_low/len(y_proba)*100:.1f}")
            with col4:
                st.metric("📈 Ortalama Risk", f"%{avg_risk*100:.1f}")

            st.markdown("---")

            # ── Grafikler
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### Risk Dağılımı")
                risk_labels = pd.cut(
                    y_proba,
                    bins=[0, 0.4, 0.7, 1.0],
                    labels=["Düşük", "Orta", "Yüksek"],
                )
                fig_pie = px.pie(
                    values=risk_labels.value_counts().values,
                    names=risk_labels.value_counts().index,
                    color=risk_labels.value_counts().index,
                    color_discrete_map={"Yüksek": "#ff4b4b", "Orta": "#ffa500", "Düşük": "#00c864"},
                    hole=0.4,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8e8f0",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                st.markdown("#### Churn Olasılığı Histogramı")
                fig_hist = px.histogram(
                    x=y_proba, nbins=40,
                    labels={"x": "Churn Olasılığı", "y": "Müşteri Sayısı"},
                    color_discrete_sequence=["#667eea"],
                )
                fig_hist.add_vline(
                    x=st.session_state.get("risk_threshold", 0.7),
                    line_dash="dash", line_color="#ff4b4b",
                    annotation_text="Risk Eşiği",
                    annotation_font_color="#ff4b4b",
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.3)",
                    font_color="#e8e8f0",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── Yüksek Riskli Müşteri Tablosu
            st.markdown("---")
            st.markdown("#### 🔴 Yüksek Riskli Müşteriler")

            threshold = st.session_state.get("risk_threshold", 0.7)
            high_risk_idx = np.where(y_proba >= threshold)[0]

            if len(high_risk_idx) == 0:
                st.info("Seçilen eşikte yüksek riskli müşteri bulunamadı.")
            else:
                risk_df = X_test.iloc[high_risk_idx].copy()
                risk_df["Churn Olasılığı"] = y_proba[high_risk_idx]
                risk_df["Risk Seviyesi"] = risk_df["Churn Olasılığı"].apply(risk_badge)
                risk_df["Gerçek Değer"] = y_test.values[high_risk_idx] if y_test is not None else "?"

                display_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                                "Churn Olasılığı", "Risk Seviyesi", "Gerçek Değer"]
                display_cols = [c for c in display_cols if c in risk_df.columns]

                display_df = risk_df[display_cols].copy()
                display_df["Churn Olasılığı"] = display_df["Churn Olasılığı"].apply(lambda x: f"%{x*100:.1f}")

                st.dataframe(
                    display_df.head(50),
                    use_container_width=True,
                    height=300,
                )

                # CSV export
                csv = risk_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 CSV İndir",
                    csv,
                    "yuksek_riskli_musteriler.csv",
                    "text/csv",
                    key="download_risk",
                )

# ─── SAYFA 2: MÜŞTERİ ANALİZİ ───────────────────────────────────────────────

elif page == "👤 Müşteri Analizi":
    st.markdown("# 👤 Müşteri Churn Analizi")
    st.markdown("*Tek bir müşteri için detaylı risk analizi ve SHAP açıklamaları*")
    st.markdown("---")

    model = load_trained_model()

    if model is None:
        st.info("ℹ️ Sol panelden modeli eğitin.")
    else:
        X_test, y_test = load_processed_data()

        if X_test is None:
            st.warning("⚠️ İşlenmiş veri bulunamadı.")
        else:
            y_proba = model.predict_proba(X_test)[:, 1]

            # Müşteri seçimi
            col_sel, _ = st.columns([1, 2])
            with col_sel:
                customer_idx = st.number_input(
                    "Müşteri Index'i (0 - {})".format(len(X_test)-1),
                    min_value=0, max_value=len(X_test)-1,
                    value=0, step=1,
                    key="customer_idx"
                )

            customer_data = X_test.iloc[[customer_idx]]
            churn_prob = float(y_proba[customer_idx])
            true_label = int(y_test.values[customer_idx]) if y_test is not None else None

            # ── Risk Skoru Göstergesi
            st.markdown("---")
            col_gauge, col_info = st.columns([1, 2])

            with col_gauge:
                # Gauge grafiği
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=churn_prob * 100,
                    title={"text": "Churn Risk Skoru", "font": {"color": "#c8b8ff", "size": 18}},
                    number={"suffix": "%", "font": {"color": "#e8e8f0", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#a0a0c0"},
                        "bar": {"color": risk_color(churn_prob)},
                        "bgcolor": "rgba(0,0,0,0)",
                        "bordercolor": "rgba(255,255,255,0.1)",
                        "steps": [
                            {"range": [0, 40], "color": "rgba(0,200,100,0.15)"},
                            {"range": [40, 70], "color": "rgba(255,165,0,0.15)"},
                            {"range": [70, 100], "color": "rgba(255,75,75,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": 0.75,
                            "value": churn_prob * 100,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e8e8f0",
                    height=280,
                    margin=dict(t=60, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_info:
                st.markdown(f"### {risk_badge(churn_prob)}")
                st.markdown(f"**Churn Olasılığı:** `%{churn_prob*100:.1f}`")
                if true_label is not None:
                    actual = "✅ Churn YOK" if true_label == 0 else "❌ Churn VAR"
                    st.markdown(f"**Gerçek Değer:** {actual}")

                # Müşteri özellikleri
                st.markdown("#### 📋 Müşteri Özellikleri")
                show_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
                for col in show_cols:
                    if col in customer_data.columns:
                        val = customer_data[col].values[0]
                        st.markdown(f"- **{col}:** `{val:.2f}`")

                # SHAP ile risk faktörleri
                st.markdown("#### 🔍 Risk Faktörleri (SHAP)")
                with st.spinner("SHAP değerleri hesaplanıyor..."):
                    explainer = get_shap_explainer(model)
                    shap_vals = explainer.shap_values(customer_data)
                    factors = get_top_factors(
                        shap_vals[0] if shap_vals.ndim > 1 else shap_vals,
                        list(customer_data.columns),
                        top_n=5,
                    )

                for f in factors:
                    color = "#ff6b6b" if f["shap_value"] > 0 else "#6bffb8"
                    direction_icon = "↑" if f["shap_value"] > 0 else "↓"
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.05); border-radius:8px; '
                        f'padding:8px 12px; margin:4px 0; border-left:3px solid {color};">'
                        f'<b style="color:{color};">{direction_icon} {f["feature"]}</b> '
                        f'<span style="color:#a0a0c0; font-size:0.85em;">'
                        f'SHAP: {f["shap_value"]:.3f} — {f["direction"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Session state'e kaydet (mesaj üretici için)
                st.session_state["current_customer_data"] = customer_data.iloc[0].to_dict()
                st.session_state["current_shap_factors"] = factors
                st.session_state["current_churn_prob"] = churn_prob

            # ── SHAP Feature Importance Bar Chart
            st.markdown("---")
            st.markdown("#### 📊 SHAP Etki Grafiği")

            feature_importance = pd.DataFrame({
                "Özellik": [f["feature"] for f in factors],
                "SHAP Değeri": [f["shap_value"] for f in factors],
                "Renk": ["Artırıcı" if f["shap_value"] > 0 else "Azaltıcı" for f in factors],
            }).sort_values("SHAP Değeri")

            fig_bar = px.bar(
                feature_importance,
                x="SHAP Değeri",
                y="Özellik",
                color="Renk",
                color_discrete_map={"Artırıcı": "#ff6b6b", "Azaltıcı": "#6bffb8"},
                orientation="h",
                title=f"Müşteri #{customer_idx} — Churn Risk Faktörleri",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0.3)",
                font_color="#e8e8f0",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                height=350,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Mesaj üretici'ye geç butonu
            st.markdown("---")
            if st.button("✉️ Bu Müşteri İçin Geri Kazanım Mesajı Oluştur", key="goto_msg"):
                st.session_state["selected_page"] = "✉️ Mesaj Üretici"
                st.info("ℹ️ Sol menüden **✉️ Mesaj Üretici** sayfasına geçin.")

# ─── SAYFA 3: MESAJ ÜRETİCİ ──────────────────────────────────────────────────

elif page == "✉️ Mesaj Üretici":
    st.markdown("# ✉️ Kişiselleştirilmiş Geri Kazanım Mesajı")
    st.markdown("*SHAP analiziyle tespit edilen risk faktörlerine göre LLM ile mesaj üret*")
    st.markdown("---")

    # Müşteri verisi kontrolü
    if "current_churn_prob" not in st.session_state:
        st.info("ℹ️ Önce **👤 Müşteri Analizi** sayfasından bir müşteri seçin.")
    else:
        churn_prob = st.session_state["current_churn_prob"]
        factors = st.session_state["current_shap_factors"]
        customer_data = st.session_state["current_customer_data"]

        # Mevcut analiz özeti
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 📊 Mevcut Müşteri Özeti")
            st.markdown(f"- **Risk Skoru:** `%{churn_prob*100:.1f}`")
            st.markdown(f"- **Risk Seviyesi:** {risk_badge(churn_prob)}")
            st.markdown(f"- **Tenure:** `{customer_data.get('tenure', '?'):.0f}` ay")
            st.markdown(f"- **Aylık Ücret:** `{customer_data.get('MonthlyCharges', '?'):.2f}` TL")

        with col2:
            st.markdown("#### 🔍 SHAP Risk Faktörleri")
            for f in factors[:3]:
                color = "#ff6b6b" if f["shap_value"] > 0 else "#6bffb8"
                st.markdown(
                    f'<span style="color:{color};">■</span> **{f["feature"]}** — {f["direction"]}',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Mesaj ayarları
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            customer_name = st.text_input("Müşteri İsmi (isteğe bağlı)", value="Sayın Müşterimiz")
        with col_b:
            tone = st.selectbox("Mesaj Tonu", ["samimi", "resmi", "kısa"])
        with col_c:
            model_name = st.selectbox(
                "LLM Modeli",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            )

        # Mesaj oluştur
        if st.button("🚀 Mesaj Oluştur (Groq AI)", key="generate_msg"):
            customer_info_with_name = {**customer_data, "name": customer_name}

            prompt = build_retention_prompt(
                customer_info=customer_info_with_name,
                shap_factors=factors,
                churn_probability=churn_prob,
                tone=tone,
            )

            with st.spinner("✍️ AI mesaj yazıyor..."):
                message = generate_retention_message(user_prompt=prompt, model=model_name)

            st.session_state["generated_message"] = message

        # Üretilen mesaj
        if "generated_message" in st.session_state:
            st.markdown("---")
            st.markdown("#### ✉️ Üretilen Geri Kazanım E-postası")

            edited_message = st.text_area(
                "Mesajı düzenleyebilirsiniz:",
                value=st.session_state["generated_message"],
                height=350,
                key="message_editor",
            )

            col_copy, col_clear = st.columns([1, 4])
            with col_copy:
                st.download_button(
                    "📥 Mesajı İndir",
                    edited_message,
                    "geri_kazanim_mesaji.txt",
                    "text/plain",
                )
            with col_clear:
                if st.button("🗑️ Temizle", key="clear_msg"):
                    del st.session_state["generated_message"]
                    st.rerun()

            st.markdown(
                '<div class="info-box">💡 <b>İpucu:</b> Mesajı düzenledikten sonra '
                'kopyalayıp CRM sisteminize yapıştırabilirsiniz.</div>',
                unsafe_allow_html=True,
            )
