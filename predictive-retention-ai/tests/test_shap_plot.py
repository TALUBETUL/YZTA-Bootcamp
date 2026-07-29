import numpy as np

from src.xai.shap_explainer import plot_waterfall_plotly


def test_waterfall_converts_zero_log_odds_to_fifty_percent():
    figure = plot_waterfall_plotly(
        shap_values_single=np.array([0.0, 0.0]),
        feature_names=["a", "b"],
        base_value=0.0,
        top_n=2,
    )

    assert figure.data[0].y[-1] == "Churn olasılığı: %50.0"
