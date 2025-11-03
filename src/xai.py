from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

def train_rf_classifier(rfm_with_labels: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """RFM (Cluster etiketli) ile RF modeli eğit. Model, X_test, y_test ve raporu döndürür."""
    X = rfm_with_labels[["Recency","Frequency","Monetary"]]
    y = rfm_with_labels["Cluster"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=False)
    cm = confusion_matrix(y_te, y_pred)
    return rf, X_te, y_te, report, cm

def shap_summary_bar(model, X_test, save_to: str | None = None):
    """Sınıf-bazlı SHAP bar özeti."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    if save_to:
        plt.gcf().tight_layout()
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()

def shap_summary_dot(model, X_test, save_to: str | None = None):
    """SHAP dot summary (dağılım)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    if save_to:
        plt.gcf().tight_layout()
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()

def shap_explain_row(model, X_test: pd.DataFrame, row_idx: int = 0):
    """Tek bir müşterinin açıklaması (force plot). Jupyter'de interaktif gösterir."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Çok sınıflı modellerde sınıf seçimi gerekebilir; burada modelin tahmin ettiği sınıfı alıyoruz.
    pred_class = int(model.predict(X_test.iloc[[row_idx]])[0])
    shap.initjs()
    return shap.force_plot(explainer.expected_value[pred_class],
                           shap_values[pred_class][row_idx, :],
                           X_test.iloc[row_idx, :],
                           matplotlib=True)

