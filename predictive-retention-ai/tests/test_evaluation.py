import json

import numpy as np

import src.models.xgboost_model as model_module
from src.models.xgboost_model import (
    find_cost_optimal_threshold,
    save_evaluation_report,
)


def test_cost_threshold_prioritizes_expensive_false_negatives():
    result = find_cost_optimal_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.1, 0.6, 0.4, 0.9]),
        false_negative_cost=10,
        false_positive_cost=1,
    )

    assert result["false_negatives"] == 0
    assert result["cost"] == 1
    assert result["recall"] == 1


def test_evaluation_report_is_json_serializable(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODELS_DIR", tmp_path)
    metrics = {
        "f1_score": 0.6,
        "roc_auc": 0.8,
        "confusion_matrix": np.array([[8, 2], [3, 7]]),
        "classification_report": {"accuracy": 0.75},
    }

    path = save_evaluation_report(metrics, model_params={"max_depth": 4})
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["metrics"]["confusion_matrix"] == [[8, 2], [3, 7]]
    assert payload["model_params"]["max_depth"] == 4
