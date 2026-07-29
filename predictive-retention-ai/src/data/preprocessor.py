"""
Veri Ön İşleme Modülü
Telco Churn veri setini modele hazır hale getirir.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


# Sütun tanımları
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService",
               "PaperlessBilling", "MultipleLines"]

ORDINAL_BINARY_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                       "TechSupport", "StreamingTV", "StreamingMovies"]

CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"
ID_COL = "customerID"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temel veri temizliği yapar.
    - TotalCharges sütunundaki boş string'leri NaN'e çevirip medyan ile doldurur
    - customerID sütununu kaldırır
    """
    df = df.copy()

    # TotalCharges: string → numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    print(f"✅ Veri temizlendi. TotalCharges NaN sayısı: {df['TotalCharges'].isnull().sum()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorik değişkenleri sayısala çevirir.
    - Binary sütunlar: Yes/No → 1/0
    - Çok kategorili sütunlar: One-Hot Encoding
    """
    df = df.copy()

    # Churn → 0/1
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # Binary sütunlar (gender ayrı ele alınır)
    for col in BINARY_COLS:
        if col in df.columns and col != "gender":
            df[col] = (df[col] == "Yes").astype(int)

    # "No internet service" / "No phone service" → 0, "Yes" → 1, "No" → 0
    for col in ORDINAL_BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0})

    # Gender ayrıca ele alınır
    if "gender" in df.columns:
        df["gender"] = (df["gender"] == "Male").astype(int)

    # One-hot encoding
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Boolean sütunları int'e çevir
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Kalan tüm NaN değerlerini 0 ile doldur (mapping'den kaçanlar)
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️ Encoding sonrası {nan_count} NaN bulundu, 0 ile dolduruluyor.")
        df = df.fillna(0)

    print(f"✅ Encoding tamamlandı. Toplam özellik sayısı: {df.shape[1]}")
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   save_scaler: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sayısal sütunları StandardScaler ile normalleştirir.
    """
    scaler = StandardScaler()
    numeric_cols = [c for c in NUMERIC_COLS if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
        print("✅ Scaler kaydedildi: models/scaler.pkl")

    return X_train, X_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE ile azınlık sınıfını (churn=1) oversample eder.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"✅ SMOTE uygulandı.")
    print(f"   Önceki dağılım → Churn=0: {(y_train==0).sum()}, Churn=1: {(y_train==1).sum()}")
    print(f"   Sonraki dağılım → Churn=0: {(y_resampled==0).sum()}, Churn=1: {(y_resampled==1).sum()}")

    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=TARGET_COL)


def preprocess_pipeline(df: pd.DataFrame,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         use_smote: bool = True) -> dict:
    """
    Tüm ön işleme adımlarını tek seferde çalıştırır.

    Returns:
        dict: X_train, X_test, y_train, y_test, feature_names
    """
    # 1. Temizlik
    df = clean_data(df)
    raw_df = df.copy()

    # 2. customerID sütununu kaydet ve kaldır
    customer_ids = df[ID_COL].values if ID_COL in df.columns else None
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # 3. Encoding
    df = encode_features(df)

    # 4. Hedef ve özellik ayrımı
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5b. Ham test satırlarını aynı sırada sakla. UI ve filtreler model için
    # ölçeklenmiş değerleri değil, bu gerçek müşteri değerlerini kullanır.
    raw_train = raw_df.loc[X_train.index].reset_index(drop=True)
    raw_test = raw_df.loc[X_test.index].reset_index(drop=True)
    test_ids_df = None
    if ID_COL in raw_test.columns:
        test_ids_df = pd.DataFrame({
            ID_COL: raw_test[ID_COL],
            "test_row": range(len(raw_test)),
        })

    # 6. Ölçeklendirme
    X_train, X_test = scale_features(X_train, X_test)

    # Model karşılaştırması ve leakage-safe CV için dengelenmemiş kopyayı koru.
    X_train_unbalanced = X_train.copy()
    y_train_unbalanced = y_train.copy()

    # 7. SMOTE
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)

    # 8. İşlenmiş verileri kaydet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_train_unbalanced.to_csv(PROCESSED_DIR / "X_train_base.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_train_unbalanced.to_csv(PROCESSED_DIR / "y_train_base.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    raw_train.to_csv(PROCESSED_DIR / "raw_train.csv", index=False)
    raw_test.to_csv(PROCESSED_DIR / "raw_test.csv", index=False)
    if test_ids_df is not None:
        test_ids_df.to_csv(PROCESSED_DIR / "test_ids.csv", index=False)
    metadata = {
        "random_state": random_state,
        "test_size": test_size,
        "use_smote": use_smote,
        "feature_names": feature_cols,
        "test_rows": len(X_test),
    }
    with (PROCESSED_DIR / "preprocessing_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print("✅ İşlenmiş veriler kaydedildi: data/processed/")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_cols,
        "customer_ids": customer_ids,
        "raw_test": raw_test,
        "raw_train": raw_train,
        "X_train_unbalanced": X_train_unbalanced,
        "y_train_unbalanced": y_train_unbalanced,
    }


def _load_training_feature_names() -> list[str]:
    metadata_path = PROCESSED_DIR / "preprocessing_metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            return list(json.load(handle)["feature_names"])

    train_path = PROCESSED_DIR / "X_train.csv"
    if train_path.exists():
        return pd.read_csv(train_path, nrows=0).columns.tolist()

    raise FileNotFoundError(
        "Eğitim özellikleri bulunamadı. Önce modeli ve veriyi hazırlayın."
    )


def preprocess_single_customer(
    customer_dict: dict,
    scaler_path: str | Path | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Tek bir müşteri kaydını tahmin için hazırlar.
    Streamlit arayüzünde canlı tahmin için kullanılır.

    Args:
        customer_dict: Müşteri özelliklerini içeren sözlük
        scaler_path: Eğitilmiş scaler dosyasının yolu

    Returns:
        pd.DataFrame: Modele hazır tek satırlık DataFrame
    """
    if scaler_path is None:
        scaler_path = MODELS_DIR / "scaler.pkl"

    df = pd.DataFrame([customer_dict])

    # Temizlik ve encoding
    df = clean_data(df)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    df = encode_features_single(df)
    feature_names = feature_names or _load_training_feature_names()
    df = df.reindex(columns=feature_names, fill_value=0)

    # Ölçeklendirme
    scaler = joblib.load(scaler_path)
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def encode_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tek müşteri için encoding (get_dummies yerine manuel).
    Eğitim setindeki sütun düzeniyle uyumlu olması için.
    """
    df = df.copy()

    for col in BINARY_COLS:
        if col in df.columns and col != "gender":
            df[col] = (df[col] == "Yes").astype(int)

    if "gender" in df.columns:
        df["gender"] = (df["gender"] == "Male").astype(int)

    for col in ORDINAL_BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0,
                                    "No internet service": 0, "No phone service": 0})

    # One-hot encoding
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df
