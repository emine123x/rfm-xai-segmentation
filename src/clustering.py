from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def scale_rfm(rfm: pd.DataFrame):
    """RFM -> log1p + StandardScaler. X_scaled, scaler, X_log döndürür."""
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X_log = X.apply(np.log1p)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    return X_scaled, scaler, X_log

def elbow_inertia(X_scaled, k_max: int = 10, random_state: int = 42):
    """1..k_max için inertia listesi döndürür."""
    inertia = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return list(range(1, k_max + 1)), inertia

def kmeans_cluster(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """KMeans uygula, rfm'e Cluster sütunu ekle ve modeli/scaler'ı döndür."""
    X_scaled, scaler, _ = scale_rfm(rfm)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    rfm = rfm.copy()
    rfm["Cluster"] = labels
    return rfm, km, scaler

def profile_clusters(rfm_with_labels: pd.DataFrame) -> pd.DataFrame:
    """Kümeleri özetle (mean/median/count)."""
    prof = (rfm_with_labels.groupby("Cluster")[["Recency","Frequency","Monetary"]]
            .agg(["mean","median","count"]).round(1))
    return prof
