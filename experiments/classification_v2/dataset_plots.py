"""
Dataset Visualization — one figure per dataset + a combined overview.

For each UCI dataset:
  - Class balance bar
  - Feature value distributions (violin or boxplot)
  - 2D PCA scatter

For synthetic datasets: scatter plots of the raw 2D data.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from datasets import PAPER_DATASETS, load_uci_split

FIGURES_DIR = Path(__file__).parent / "figures" / "datasets"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

C0, C1 = "#4e79a7", "#f28e2b"

DS_INFO = {
    "iris":         "Iris Flowers — 3 species binarised (Setosa vs rest)\nFeatures: sepal length/width, petal length/width",
    "wine":         "Wine Quality — 3 cultivars binarised (Class 1 vs rest)\nFeatures: 13 chemical measurements",
    "breastcancer": "Breast Cancer Wisconsin — Malignant vs Benign\nFeatures: 30 tumour morphology measurements",
    "ionosphere":   "Ionosphere Radar — Good vs Bad signal\nFeatures: 34 radar return measurements (feature 2 = constant 0)",
    "spambase":     "Spam Email — Spam vs Not Spam\nFeatures: 57 word/character frequency + run-length statistics",
    "banknote":     "Banknote Authentication — Real vs Fake\nFeatures: 4 wavelet transform statistics",
    "heart":        "Heart Disease (Cleveland) — Disease vs No Disease\nFeatures: 13 clinical measurements",
    "haberman":     "Haberman Cancer Survival — Survived ≥5yr vs Died\nFeatures: 3 (age, operation year, positive axillary nodes)",
    "mammographic": "Mammographic Mass — Malignant vs Benign\nFeatures: 5 (BI-RADS, age, shape, margin, density)",
    "parkinsons":   "Parkinson's Disease — Has PD vs Healthy\nFeatures: 22 voice measurements",
}


def plot_one_dataset(dataset_name, ax_balance, ax_pca, ax_violin):
    X_tr, y_tr, X_te, y_te, info = load_uci_split(dataset_name)
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([
        y_tr * 0 + 1 if y_tr.dtype == np.int64 else y_tr,
        y_te * 0 + 1 if y_te.dtype == np.int64 else y_te,
    ])
    # Reload to get {-1,+1} labels for class balance
    from datasets import load_uci_split as lus
    
    X_tr_r, y_tr_r, X_te_r, y_te_r, _ = lus(dataset_name)
    y_raw = np.concatenate([y_tr_r, y_te_r])
    X_raw = np.vstack([X_tr_r, X_te_r])

    pos = (y_raw == 1).sum()
    neg = (y_raw == -1).sum()

    # Class balance
    ax_balance.bar(["Positive\n(class +1)", "Negative\n(class -1)"],
                   [pos, neg], color=[C0, C1], edgecolor="white", lw=0.5, width=0.5)
    ax_balance.set_title("Class balance", fontsize=8)
    ax_balance.set_ylabel("Count", fontsize=7)
    for bar, val in zip(ax_balance.patches, [pos, neg]):
        ax_balance.text(bar.get_x()+bar.get_width()/2, val+1, str(val),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_balance.spines[["top","right"]].set_visible(False)
    ax_balance.tick_params(labelsize=7)

    # 2D PCA scatter
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_raw)
    var = pca.explained_variance_ratio_ * 100
    mask_pos = y_raw == 1
    ax_pca.scatter(Xp[mask_pos, 0], Xp[mask_pos, 1], c=C0, s=8, alpha=0.5,
                   edgecolors="none", label=f"Positive (n={pos})")
    ax_pca.scatter(Xp[~mask_pos, 0], Xp[~mask_pos, 1], c=C1, s=8, alpha=0.5,
                   edgecolors="none", label=f"Negative (n={neg})")
    ax_pca.set_xlabel(f"PC1 ({var[0]:.1f}% var)", fontsize=7)
    ax_pca.set_ylabel(f"PC2 ({var[1]:.1f}% var)", fontsize=7)
    ax_pca.set_title("2D PCA projection", fontsize=8)
    ax_pca.legend(fontsize=6, markerscale=1.5)
    ax_pca.spines[["top","right"]].set_visible(False)
    ax_pca.tick_params(labelsize=6)

    # Feature violin (top 6 by variance, or all if p<=6)
    p = X_raw.shape[1]
    var_feat = X_raw.var(axis=0)
    top_idx = np.argsort(-var_feat)[:min(6, p)]
    X_top = X_raw[:, top_idx]
    parts = ax_violin.violinplot([X_top[y_raw==1, k] for k in range(len(top_idx))],
                                  positions=range(len(top_idx)),
                                  showmedians=True, showextrema=False)
    for pc_v in parts["bodies"]:
        pc_v.set_facecolor(C0); pc_v.set_alpha(0.6)
    parts2 = ax_violin.violinplot([X_top[y_raw==-1, k] for k in range(len(top_idx))],
                                   positions=range(len(top_idx)),
                                   showmedians=True, showextrema=False)
    for pc_v in parts2["bodies"]:
        pc_v.set_facecolor(C1); pc_v.set_alpha(0.6)
    ax_violin.set_xticks(range(len(top_idx)))
    ax_violin.set_xticklabels([f"f{i}" for i in top_idx], fontsize=7)
    ax_violin.set_title(f"Top {len(top_idx)} features by variance\n(blue=pos, orange=neg)", fontsize=8)
    ax_violin.set_ylabel("Standardised value", fontsize=7)
    ax_violin.spines[["top","right"]].set_visible(False)
    ax_violin.tick_params(labelsize=6)


def plot_all_datasets():
    """One figure per dataset."""
    for ds in PAPER_DATASETS:
        print(f"  Plotting {ds} ...")
        fig = plt.figure(figsize=(13, 4), facecolor="white")
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
        ax_b = fig.add_subplot(gs[0, 0])
        ax_p = fig.add_subplot(gs[0, 1])
        ax_v = fig.add_subplot(gs[0, 2])
        try:
            plot_one_dataset(ds, ax_b, ax_p, ax_v)
        except Exception as e:
            print(f"    Warning: {e}")
        fig.suptitle(DS_INFO[ds], fontsize=10, fontweight="bold", x=0.5, y=1.02)
        plt.tight_layout()
        out = str(FIGURES_DIR / f"dataset_{ds}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {out}")


def plot_overview():
    """5×2 grid — all datasets in one figure for everything.md."""
    fig, axes = plt.subplots(5, 4, figsize=(20, 22), facecolor="white")

    for row, ds in enumerate(PAPER_DATASETS[:5]):
        X_tr, y_tr, X_te, y_te, info = load_uci_split(ds)
        y_raw = np.concatenate([y_tr, y_te])
        X_raw = np.vstack([X_tr, X_te])

        pos = (y_raw == 1).sum(); neg = (y_raw == -1).sum()

        # Balance
        ax = axes[row, 0]
        ax.bar(["Pos", "Neg"], [pos, neg], color=[C0, C1], edgecolor="w", width=0.5)
        ax.set_title(f"{ds.capitalize()}\nn={info['n_samples']}, p={info['n_features']}",
                     fontsize=8, fontweight="bold")
        ax.set_ylabel("Count", fontsize=7); ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)

        # PCA
        ax = axes[row, 1]
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X_raw)
        var = pca.explained_variance_ratio_ * 100
        mask = y_raw == 1
        ax.scatter(Xp[mask,0], Xp[mask,1], c=C0, s=5, alpha=0.45, edgecolors="none")
        ax.scatter(Xp[~mask,0], Xp[~mask,1], c=C1, s=5, alpha=0.45, edgecolors="none")
        ax.set_xlabel(f"PC1 {var[0]:.0f}%", fontsize=6)
        ax.set_ylabel(f"PC2 {var[1]:.0f}%", fontsize=6)
        ax.set_title("PCA", fontsize=8); ax.tick_params(labelsize=5)
        ax.spines[["top","right"]].set_visible(False)

    for row, ds in enumerate(PAPER_DATASETS[5:]):
        X_tr, y_tr, X_te, y_te, info = load_uci_split(ds)
        y_raw = np.concatenate([y_tr, y_te])
        X_raw = np.vstack([X_tr, X_te])

        pos = (y_raw == 1).sum(); neg = (y_raw == -1).sum()

        ax = axes[row, 2]
        ax.bar(["Pos", "Neg"], [pos, neg], color=[C0, C1], edgecolor="w", width=0.5)
        ax.set_title(f"{ds.capitalize()}\nn={info['n_samples']}, p={info['n_features']}",
                     fontsize=8, fontweight="bold")
        ax.set_ylabel("Count", fontsize=7); ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)

        ax = axes[row, 3]
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X_raw)
        var = pca.explained_variance_ratio_ * 100
        mask = y_raw == 1
        ax.scatter(Xp[mask,0], Xp[mask,1], c=C0, s=5, alpha=0.45, edgecolors="none")
        ax.scatter(Xp[~mask,0], Xp[~mask,1], c=C1, s=5, alpha=0.45, edgecolors="none")
        ax.set_xlabel(f"PC1 {var[0]:.0f}%", fontsize=6)
        ax.set_ylabel(f"PC2 {var[1]:.0f}%", fontsize=6)
        ax.set_title("PCA", fontsize=8); ax.tick_params(labelsize=5)
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Dataset Overview — Class Balance and 2D PCA Projections\n"
                 "Blue = positive class  ·  Orange = negative class",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = str(FIGURES_DIR / "dataset_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview → {out}")


if __name__ == "__main__":
    print("[dataset_plots] Generating dataset visualizations ...")
    plot_all_datasets()
    plot_overview()
    print("[dataset_plots] Done.")
