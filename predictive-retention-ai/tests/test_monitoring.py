import numpy as np
import pandas as pd

from src.monitoring.model_monitor import (
    build_reference_profile, calculate_drift, group_error_analysis,
)


def _raw_frame():
    return pd.DataFrame({
        "tenure": np.arange(40),
        "MonthlyCharges": np.linspace(20, 100, 40),
        "TotalCharges": np.linspace(50, 4000, 40),
        "gender": ["Male", "Female"] * 20,
        "SeniorCitizen": [0, 1] * 20,
        "Contract": ["Month-to-month", "One year"] * 20,
    })


def test_identical_data_has_no_critical_drift():
    frame = _raw_frame()
    drift = calculate_drift(build_reference_profile(frame), frame)
    assert not drift.empty
    assert (drift["status"] == "stable").all()


def test_group_error_analysis_reports_protected_segments():
    frame = _raw_frame()
    y_true = np.array([0, 1] * 20)
    probabilities = np.linspace(0.05, 0.95, 40)
    report = group_error_analysis(
        y_true, probabilities, frame, min_group_size=5
    )
    assert {"gender", "SeniorCitizen", "Contract"} <= set(report["attribute"])
    assert {"precision", "recall", "f1", "roc_auc"} <= set(report.columns)
