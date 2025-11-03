"""
RFM-XAI-Segmentation package
Kullanışlı üst seviye import'lar.
"""
from .features import (
    PROJECT_ROOT, RAW_DIR, PROCESSED_DIR,
    load_raw_excel, clean_raw, compute_rfm,
    save_csv, load_processed_rfm
)
from .clustering import (
    scale_rfm, elbow_inertia, kmeans_cluster, profile_clusters
)
from .viz import (
    ensure_figdir, plot_elbow, plot_boxplots, plot_pairplot
)
from .xai import (
    train_rf_classifier, shap_summary_bar, shap_summary_dot, shap_explain_row
)
