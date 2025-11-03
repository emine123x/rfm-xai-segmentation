from __future__ import annotations
from pathlib import Path
import pandas as pd

# Proje kökünü otomatik bul (src'nin 1 üstü)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def load_raw_excel(path: Path | str | None = None, sheet: str = "Year 2010-2011") -> pd.DataFrame:
    """Ham Online Retail II verisini oku (.xlsx)."""
    if path is None:
        path = RAW_DIR / "online_retail_II.xlsx"
    df = pd.read_excel(path, sheet_name=sheet)
    return df

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Ham veriyi temizle: eksik müşteri, iade faturalar, negatif miktar/fiyat."""
    # Sütun adları farklı isimle gelebilir (Invoice vs InvoiceNo, Price vs UnitPrice)
    df = df.rename(columns={
        "Customer ID": "CustomerID",
        "InvoiceNo": "Invoice",
        "UnitPrice": "Price"
    })
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    return df

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Temizlenmiş df'den RFM tablosu döndür."""
    import datetime as dt
    snapshot = df["InvoiceDate"].max() + dt.timedelta(days=1)
    rfm = (df.groupby("CustomerID")
             .agg(Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
                  Frequency=("Invoice", "nunique"),
                  Monetary=("TotalPrice", "sum"))
             .reset_index())
    return rfm

def save_csv(df: pd.DataFrame, rel_path: str) -> Path:
    """Data klasörü altında güvenli CSV kaydı."""
    out_path = PROJECT_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

def load_processed_rfm(name: str = "rfm_clustered.csv") -> pd.DataFrame:
    """İşlenmiş RFM dosyasını oku."""
    path = PROCESSED_DIR / name
    return pd.read_csv(path)
