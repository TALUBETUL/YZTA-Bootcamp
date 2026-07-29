import pandas as pd

from src.models.xgboost_model import validate_model_features


class FakeModel:
    n_features_in_ = 2
    feature_names_in_ = ["a", "b"]


def test_model_feature_contract_accepts_matching_frame():
    frame = pd.DataFrame([[1, 2]], columns=["a", "b"])

    assert validate_model_features(FakeModel(), frame) == []


def test_model_feature_contract_rejects_wrong_order():
    frame = pd.DataFrame([[1, 2]], columns=["b", "a"])

    issues = validate_model_features(FakeModel(), frame)

    assert any("sırası" in issue for issue in issues)
