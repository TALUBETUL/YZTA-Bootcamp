"""
Veri Yükleyici Modülü
IBM Telco Customer Churn veri setini yükler.
"""

import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.xls"


def load_raw_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Ham veri setini yükler.

    Args:
        filepath: Veri dosyasının yolu (.xls veya .xlsx)

    Returns:
        pd.DataFrame: Ham veri çerçevesi
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {filepath}")

    # Gerçek format tespiti (uzantı yanıltıcı olabilir)
    try:
        # Önce CSV olarak dene (dosya CSV ama .xls uzantılı olabilir)
        df = pd.read_csv(filepath)
        # CSV başarılıysa içeriği doğrula
        if df.shape[1] < 5:
            raise ValueError("Çok az sütun — muhtemelen CSV değil")
    except Exception:
        # CSV başarısız → Excel engine'leri dene
        try:
            df = pd.read_excel(filepath, engine="openpyxl")
        except Exception:
            df = pd.read_excel(filepath, engine="xlrd")

    print(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Veri seti hakkında temel bilgileri döndürür.

    Args:
        df: Veri çerçevesi

    Returns:
        dict: Temel istatistikler
    """
    churn_col = "Churn"
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    if churn_col in df.columns:
        churn_dist = df[churn_col].value_counts(normalize=True).to_dict()
        info["churn_rate"] = churn_dist.get("Yes", churn_dist.get(1, 0))
    return info
