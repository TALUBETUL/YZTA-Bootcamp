import numpy as np

from src.models.calibration import (
    ProbabilityCalibrator, calibration_report, expected_calibration_error,
)


def test_calibrator_returns_valid_probabilities_and_report():
    probabilities = np.array([0.05, 0.20, 0.45, 0.65, 0.80, 0.95])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    calibrator = ProbabilityCalibrator().fit(probabilities, y_true)
    calibrated = calibrator.predict(probabilities)

    assert calibrated.shape == probabilities.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    report = calibration_report(y_true, probabilities, calibrated)
    assert report["sample_size"] == 6
    assert set(report) >= {"raw", "calibrated"}


def test_expected_calibration_error_is_zero_for_matching_bins():
    assert expected_calibration_error([0, 1], [0, 1], n_bins=2) < 1e-5
