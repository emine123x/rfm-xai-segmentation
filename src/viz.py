from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_figdir(figdir: Path | str = "reports/figures") -> Path:
    p = Path(figdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_elbow(K, inertia, save_to: str | None = None):
    plt.figure()
    plt.plot(K, inertia, marker="o")
    plt.xlabel("Küme Sayısı (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Yöntemi ile Küme Sayısı")
    if save_to:
        ensure_figdir(Path(save_to).parent)
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()

def plot_boxplots(rfm, save_prefix: str | None = None):
    for col in ["Recency", "Frequency", "Monetary"]:
        plt.figure(figsize=(10,6))
        sns.boxplot(x="Cluster", y=col, data=rfm)
        plt.title(f"Clusterlara Göre {col} Dağılımı")
        if save_prefix:
            ensure_figdir(Path(save_prefix).parent)
            plt.savefig(f"{save_prefix}_{col.lower()}.png", dpi=150, bbox_inches="tight")
        plt.show()

def plot_pairplot(rfm, save_to: str | None = None):
    g = sns.pairplot(rfm, vars=["Recency","Frequency","Monetary"], hue="Cluster", palette="tab10")
    if save_to:
        ensure_figdir(Path(save_to).parent)
        g.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.show()
