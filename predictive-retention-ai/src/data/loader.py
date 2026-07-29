"""
Veri Yükleyici Modülü
IBM Telco Customer Churn veri setini yükler.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.xls"
RAW_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / DATASET_FILENAME,
    REPOSITORY_ROOT / DATASET_FILENAME,
)
RAW_DATA_PATH = RAW_DATA_CANDIDATES[0]


def resolve_raw_data_path(filepath: str | Path | None = None) -> Path:
    """Return an existing dataset path, supporting both documented locations."""
    if filepath is not None:
        path = Path(filepath)
        if path.exists():
            return path
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")

    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = "\n- ".join(str(path) for path in RAW_DATA_CANDIDATES)
    raise FileNotFoundError(f"Veri dosyası bulunamadı. Kontrol edilen yollar:\n- {checked}")


def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Ham veri setini yükler.

    Args:
        filepath: Veri dosyasının yolu (.xls veya .xlsx)

    Returns:
        pd.DataFrame: Ham veri çerçevesi
    """
    filepath = resolve_raw_data_path(filepath)

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
