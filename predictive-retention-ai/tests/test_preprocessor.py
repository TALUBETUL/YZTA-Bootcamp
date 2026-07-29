import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_raw_data
import src.data.preprocessor as preprocessor
from src.data.preprocessor import encode_features, preprocess_single_customer


def test_gender_encoding_preserves_male_and_female_values():
    frame = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "Churn": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "Contract": ["Month-to-month", "One year"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
        }
    )

    encoded = encode_features(frame)

    assert encoded["gender"].tolist() == [1, 0]
    assert encoded["Churn"].tolist() == [0, 1]


def test_raw_test_rows_and_customer_ids_stay_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocessor, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(preprocessor, "MODELS_DIR", tmp_path / "models")

    result = preprocessor.preprocess_pipeline(load_raw_data(), use_smote=False)
    raw_test = pd.read_csv(tmp_path / "processed" / "raw_test.csv")
    test_ids = pd.read_csv(tmp_path / "processed" / "test_ids.csv")

    assert len(result["X_test"]) == len(result["y_test"]) == len(raw_test)
    assert raw_test["customerID"].tolist() == test_ids["customerID"].tolist()
    assert raw_test["tenure"].between(0, 72).all()


def test_live_customer_is_reindexed_to_training_feature_contract(tmp_path):
    scaler = StandardScaler().fit(
        pd.DataFrame(
            [[1, 50.0, 50.0], [12, 80.0, 960.0]],
            columns=["tenure", "MonthlyCharges", "TotalCharges"],
        )
    )
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    feature_names = [
        "gender", "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "InternetService_DSL", "Contract_Month-to-month",
        "PaymentMethod_Electronic check",
    ]
    customer = {
        "customerID": "LIVE",
        "gender": "Male",
        "SeniorCitizen": 0,
        "tenure": 6,
        "MonthlyCharges": 65.0,
        "TotalCharges": 390.0,
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
    }

    processed = preprocess_single_customer(
        customer,
        scaler_path=scaler_path,
        feature_names=feature_names,
    )

    assert processed.columns.tolist() == feature_names
    assert processed["gender"].iloc[0] == 1
    assert processed["InternetService_DSL"].iloc[0] == 1
