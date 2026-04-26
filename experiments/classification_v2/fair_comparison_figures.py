"""
Fair Kernel Comparison: all methods use KRR as the downstream classifier.

This isolates the kernel quality — any accuracy difference is attributable
purely to the kernel representation, not to the choice of classifier.

Reads existing benchmark_results.json and kan_benchmark_results.json.
No re-training needed.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)

DATASET_ORDER = [
    "iris", "wine", "breastcancer", "ionosphere",
    "spambase", "banknote", "heart", "haberman",
    "mammographic", "parkinsons",
]

# Only KRR-based methods — same classifier, different kernel
KRR_METHODS = [
    ("KAN+KRR",        "KAN (rank-1 Gram)",        "#e6550d", "kan"),
    ("Ours+RBF",       "KN Per-feature RBF",        "#2ca02c", "kn"),
    ("Ours+Linear",    "KN Per-feature Linear",     "#98df8a", "kn"),
    ("Ours+Mixed",     "KN Per-feature RBF+Linear", "#17becf", "kn"),
    ("KRR-RBF(γ=0.5)", "Global RBF (γ=0.5)",       "#d62728", "kn"),
    ("KRR-Linear",     "Global Linear",             "#1f77b4", "kn"),
    ("KRR-Poly3",      "Global Poly-3",             "#9467bd", "kn"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def get_acc(res, dataset, method):
    ds = res.get(dataset, {})
    if method in ds and isinstance(ds[method], dict):
        return ds[method].get("accuracy_pct", np.nan)
    return np.nan


def build_matrix(kn_res, kan_res):
    """Returns (n_methods × n_datasets) accuracy matrix."""
    n_m, n_d = len(KRR_METHODS), len(DATASET_ORDER)
    mat = np.full((n_m, n_d), np.nan)
    for mi, (method, _, _, src) in enumerate(KRR_METHODS):
        res = kan_res if src == "kan" else kn_res
        for di, ds in enumerate(DATASET_ORDER):
            mat[mi, di] = get_acc(res, ds, method)
    return mat


# ── Figure 1: Grouped bar chart (KRR-only) ────────────────────────────────────

def plot_fair_bars(mat, save_path=None):
    n_m, n_d = mat.shape
    width = 0.82 / n_m
    offsets = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * width
    x = np.arange(n_d)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#fafafa")

    for mi, ((method, label, color, _), offset) in enumerate(zip(KRR_METHODS, offsets)):
        vals = mat[mi]
        bars = ax.bar(x + offset, vals, width*0.93,
                      color=color, alpha=0.88, label=label,
                      edgecolor="white", linewidth=0.3, zorder=3)
        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isnan(val):
                continue
            if val == np.nanmax(mat[:, bi]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                        f"{val:.0f}", ha="center", va="bottom",
                        fontsize=5.5, fontweight="bold", color="#111")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DATASET_ORDER], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_ylim(20, 108)
    ax.set_title(
        "Fair Kernel Comparison — All Methods Use Kernel Ridge Regression\n"
        "Differences reflect kernel quality only, not classifier choice",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, ncol=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Bracket + label for our models vs global kernels
    ax.axvline(3.5, color="#aaa", lw=1, ls=":", alpha=0.7)
    ax.text(1.75, 106, "← Simple problems", ha="center", fontsize=8, color="#888")
    ax.text(6.5,  106, "Heterogeneous / noisy problems →", ha="center", fontsize=8, color="#888")

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fair_fig1_krr_only_bars.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure 2: Heatmap — all methods × all datasets ────────────────────────────

def plot_fair_heatmap(mat, save_path=None):
    labels_y = [label for _, label, _, _ in KRR_METHODS]
    labels_x = [d.capitalize() for d in DATASET_ORDER]

    # Rank within each dataset column (1=best)
    rank_mat = np.full_like(mat, np.nan)
    for di in range(mat.shape[1]):
        col = mat[:, di]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        ranks = np.argsort(np.argsort(-col[valid])).astype(float) + 1
        rank_mat[valid, di] = ranks

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5), facecolor="#fafafa")

    # Accuracy heatmap
    im1 = ax1.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=40, vmax=100)
    plt.colorbar(im1, ax=ax1, label="Test Accuracy (%)", fraction=0.03)
    ax1.set_xticks(range(len(DATASET_ORDER))); ax1.set_xticklabels(labels_x, rotation=35, ha="right", fontsize=9)
    ax1.set_yticks(range(len(KRR_METHODS))); ax1.set_yticklabels(labels_y, fontsize=9)
    ax1.set_title("Accuracy (%) — Same Classifier (KRR) for All", fontsize=11, fontweight="bold")
    for mi in range(mat.shape[0]):
        for di in range(mat.shape[1]):
            v = mat[mi, di]
            if not np.isnan(v):
                ax1.text(di, mi, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                         color="white" if v < 60 else "black")

    # Rank heatmap (1=best, 7=worst)
    im2 = ax2.imshow(rank_mat, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(KRR_METHODS))
    plt.colorbar(im2, ax=ax2, label="Rank (1=best)", fraction=0.03)
    ax2.set_xticks(range(len(DATASET_ORDER))); ax2.set_xticklabels(labels_x, rotation=35, ha="right", fontsize=9)
    ax2.set_yticks(range(len(KRR_METHODS))); ax2.set_yticklabels(labels_y, fontsize=9)
    ax2.set_title("Rank per Dataset (1 = best kernel on that dataset)", fontsize=11, fontweight="bold")
    for mi in range(rank_mat.shape[0]):
        for di in range(rank_mat.shape[1]):
            v = rank_mat[mi, di]
            if not np.isnan(v):
                ax2.text(di, mi, f"{int(v)}", ha="center", va="center", fontsize=9,
                         color="white" if v >= len(KRR_METHODS)-1 else "black")

    fig.suptitle("Fair Kernel Comparison Heatmaps — KRR Classifier Fixed",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fair_fig2_heatmap_and_rank.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure 3: Mean rank summary bar ──────────────────────────────────────────

def plot_mean_rank(mat, save_path=None):
    rank_mat = np.full_like(mat, np.nan)
    for di in range(mat.shape[1]):
        col = mat[:, di]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        ranks = np.argsort(np.argsort(-col[valid])).astype(float) + 1
        rank_mat[valid, di] = ranks

    mean_rank = np.nanmean(rank_mat, axis=1)
    order = np.argsort(mean_rank)
    colors = [KRR_METHODS[i][2] for i in order]
    labels = [KRR_METHODS[i][1] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fafafa")
    y = np.arange(len(KRR_METHODS))
    ax.barh(y, mean_rank[order], color=colors, edgecolor="white", lw=0.4, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean Rank across 10 UCI Datasets (lower = better)", fontsize=10)
    ax.set_title("Fair Comparison — Mean Rank (KRR classifier fixed)\n"
                 "Isolates kernel representation quality",
                 fontsize=11, fontweight="bold")
    ax.axvline(1, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    for i, (mi, mr) in enumerate(zip(order, mean_rank[order])):
        ax.text(mr + 0.05, i, f"{mr:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fair_fig3_mean_rank.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex(mat):
    method_names = [m for m, _, _, _ in KRR_METHODS]
    header = " & ".join(["Dataset"] + method_names) + " \\\\"
    print("\n% Fair comparison table — all methods use KRR")
    print("\\begin{tabular}{l" + "r"*len(KRR_METHODS) + "}")
    print("\\hline"); print(header); print("\\hline")
    for di, ds in enumerate(DATASET_ORDER):
        col = mat[:, di]
        best = np.nanmax(col)
        cells = []
        for v in col:
            if np.isnan(v):
                cells.append("--")
            elif abs(v - best) < 0.05:
                cells.append(f"\\textbf{{{v:.1f}}}")
            else:
                cells.append(f"{v:.1f}")
        print(ds.capitalize() + " & " + " & ".join(cells) + " \\\\")
    print("\\hline\n\\end{tabular}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    kn_path  = RESULTS_DIR / "benchmark_results.json"
    kan_path = RESULTS_DIR / "kan_benchmark_results.json"

    if not kn_path.exists():
        print(f"ERROR: {kn_path} not found"); return
    if not kan_path.exists():
        print(f"ERROR: {kan_path} not found"); return

    kn_res  = load(kn_path)
    kan_res = load(kan_path)
    mat = build_matrix(kn_res, kan_res)

    print("[fair_comparison] Generating figures ...")
    plot_fair_bars(mat)
    plot_fair_heatmap(mat)
    plot_mean_rank(mat)
    print_latex(mat)
    print("[fair_comparison] Done.")


if __name__ == "__main__":
    run()
