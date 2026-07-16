"""
Müşteri Segmentasyon Modülü (Sprint 3)
K-Means ile churn riski + müşteri özellikleri bazında segmentasyon.
PCA ile 2D görselleştirme.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px


# Segment renkleri ve sembolleri
SEGMENT_COLORS = ["#ff4b4b", "#ffa500", "#4361ee", "#00c864"]
SEGMENT_NAMES_ORDERED = [
    "Yüksek Riskli",
    "Orta-Yüksek Riskli",
    "Orta-Düşük Riskli",
    "Düşük Riskli",
]


def run_kmeans_segmentation(
    X_test: pd.DataFrame,
    y_proba: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    K-Means ile müşteri segmentasyonu yapar.

    Segmentasyon için: tenure + MonthlyCharges + TotalCharges + churn_proba
    PCA ile 2D görselleştirme koordinatları da hesaplanır.

    Returns:
        pd.DataFrame: pca_x, pca_y, segment, segment_name, churn_proba + orijinal özellikler
    """
    # Segmentasyon için kullanılacak sütunlar
    seg_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in X_test.columns]

    seg_data = X_test[seg_cols].copy()
    seg_data["churn_proba"] = y_proba

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(seg_data)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    labels = kmeans.fit_predict(X_scaled)

    # PCA (2D görselleştirme)
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)

    result_df = pd.DataFrame({
        "pca_x": X_2d[:, 0],
        "pca_y": X_2d[:, 1],
        "segment": labels,
        "churn_proba": y_proba,
    })

    for col in seg_cols:
        result_df[col] = X_test[col].values

    # Segment isimlerini ata (risk skoruna göre sırala)
    seg_risk = result_df.groupby("segment")["churn_proba"].mean().sort_values(ascending=False)
    name_map = {seg_id: SEGMENT_NAMES_ORDERED[rank] for rank, (seg_id, _) in enumerate(seg_risk.items())}
    result_df["segment_name"] = result_df["segment"].map(name_map)

    return result_df


def get_segment_profiles(segment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her segment için özet istatistikler döndürür.
    """
    agg = {
        "churn_proba": ["mean", "count"],
    }
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in segment_df.columns:
            agg[col] = "mean"

    profiles = segment_df.groupby(["segment", "segment_name"]).agg(agg)
    profiles.columns = ["_".join(c).strip("_") for c in profiles.columns]
    profiles = profiles.reset_index()

    rename_map = {
        "churn_proba_mean": "Ort. Risk",
        "churn_proba_count": "Müşteri Sayısı",
        "tenure_mean": "Ort. Tenure (ay)",
        "MonthlyCharges_mean": "Ort. Aylık Ücret ($)",
        "TotalCharges_mean": "Ort. Toplam Ücret ($)",
    }
    profiles = profiles.rename(columns=rename_map)
    return profiles.sort_values("Ort. Risk", ascending=False).reset_index(drop=True)


def plot_segments_plotly(segment_df: pd.DataFrame) -> go.Figure:
    """
    PCA 2D scatter plot ile müşteri segmentlerini görselleştirir.
    """
    unique_names = (
        segment_df.groupby("segment_name")["churn_proba"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    color_map = {name: SEGMENT_COLORS[i % len(SEGMENT_COLORS)] for i, name in enumerate(unique_names)}

    hover_data = {"churn_proba": ":.1%", "pca_x": False, "pca_y": False}
    for col in ["tenure", "MonthlyCharges"]:
        if col in segment_df.columns:
            hover_data[col] = True

    fig = px.scatter(
        segment_df,
        x="pca_x",
        y="pca_y",
        color="segment_name",
        color_discrete_map=color_map,
        hover_data=hover_data,
        labels={
            "pca_x": "PCA Bileşen 1",
            "pca_y": "PCA Bileşen 2",
            "segment_name": "Segment",
            "churn_proba": "Churn Riski",
        },
        title="Müşteri Segmentleri — PCA Görünümü",
        opacity=0.65,
        category_orders={"segment_name": unique_names},
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,248,0.5)",
        font_color="#2C2416",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showticklabels=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showticklabels=False),
        legend=dict(bgcolor="rgba(44,36,22,0.05)", bordercolor="rgba(44,36,22,0.1)"),
        height=480,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig
