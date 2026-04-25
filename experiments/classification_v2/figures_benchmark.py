"""
Visualization generator for EXP-1 benchmark results.

Reads results/benchmark_results.json and produces:
  fig1a — Grouped accuracy bar chart (all methods, all datasets)
  fig1b — Accuracy vs. Baseline: scatter + delta heatmap
  fig1c — Normalized rank chart
  fig1d — Alpha sparsity analysis across datasets
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────

PALETTE = {
    # Our models
    "Ours+RBF":    "#2ca02c",
    "Ours+Linear": "#98df8a",
    "Ours+Mixed":  "#17becf",
    # Paper MKL
    "EasyMKL":     "#aec7e8",
    "AverageMKL":  "#c5b0d5",
    "CKA":         "#c49c94",
    "SMKL":        "#ffbb78",
    # Sklearn baselines
    "SVM-RBF(C=10)":   "#d62728",
    "SVM-Linear(C=1)": "#9467bd",
    "SVM-Poly3(C=1)":  "#e377c2",
    "KRR-RBF(γ=auto)": "#8c564b",
    "KRR-RBF(γ=0.5)":  "#bcbd22",
    "KRR-Linear":      "#1f77b4",
    "KRR-Poly3":       "#7f7f7f",
}

DATASET_ORDER = [
    "iris", "wine", "breastcancer", "ionosphere",
    "spambase", "banknote", "heart", "haberman",
    "mammographic", "parkinsons",
]

METHODS_ORDERED = [
    "Ours+RBF", "Ours+Linear", "Ours+Mixed",
    "SVM-RBF(C=10)", "SVM-Linear(C=1)", "SVM-Poly3(C=1)",
    "KRR-RBF(γ=auto)", "KRR-RBF(γ=0.5)", "KRR-Linear", "KRR-Poly3",
    "EasyMKL", "AverageMKL", "CKA", "SMKL",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(json_path: Path = None) -> dict:
    path = json_path or RESULTS_DIR / "benchmark_results.json"
    with open(path) as f:
        return json.load(f)


def extract_accuracy_matrix(results: dict) -> tuple:
    """Returns (methods, datasets, matrix[n_methods, n_datasets])."""
    present_methods = []
    # Determine which methods are present
    for ds in DATASET_ORDER:
        if ds in results:
            for m in METHODS_ORDERED:
                if m in results[ds] and m not in present_methods:
                    present_methods.append(m)
            # Paper baselines
            for m in ["EasyMKL", "AverageMKL", "CKA", "SMKL"]:
                if m in results[ds].get("_paper_baselines", {}) and m not in present_methods:
                    present_methods.append(m)

    n_methods = len(present_methods)
    n_ds = len(DATASET_ORDER)
    matrix = np.full((n_methods, n_ds), np.nan)

    for di, ds in enumerate(DATASET_ORDER):
        if ds not in results:
            continue
        ds_res = results[ds]
        paper = ds_res.get("_paper_baselines", {})
        for mi, m in enumerate(present_methods):
            if m in ds_res and isinstance(ds_res[m], dict):
                matrix[mi, di] = ds_res[m].get("accuracy_pct", np.nan)
            elif m in paper:
                matrix[mi, di] = float(paper[m])

    return present_methods, DATASET_ORDER, matrix


# ── Figure 1a: Grouped bar chart ──────────────────────────────────────────────

def plot_grouped_bars(results: dict, save_path: str = None):
    methods, datasets, matrix = extract_accuracy_matrix(results)

    # Only show a clean subset for the main figure
    show_methods = [m for m in [
        "Ours+RBF", "Ours+Mixed",
        "SVM-RBF(C=10)", "KRR-Linear",
        "EasyMKL", "SMKL",
    ] if m in methods]

    row_idx = [methods.index(m) for m in show_methods]
    sub_matrix = matrix[row_idx, :]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#fafafa")
    n_show = len(show_methods)
    n_ds = len(datasets)
    x = np.arange(n_ds)
    width = 0.85 / n_show
    offsets = np.linspace(-(n_show - 1) / 2, (n_show - 1) / 2, n_show) * width

    for mi, (m, offset) in enumerate(zip(show_methods, offsets)):
        vals = sub_matrix[mi, :]
        bars = ax.bar(x + offset, vals, width=width * 0.95,
                      color=PALETTE.get(m, "#999"), alpha=0.9,
                      label=m, edgecolor="white", linewidth=0.3, zorder=3)
        # Annotate top value per dataset with bold if it's ours
        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isnan(val):
                continue
            if val == np.nanmax(sub_matrix[:, bi]):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                        f"{val:.0f}", ha="center", va="bottom",
                        fontsize=6, fontweight="bold", color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_ylim(55, 105)
    ax.set_title(
        "EXP-1: UCI Benchmark — Accuracy Comparison\n"
        "Our Per-Feature Kernel vs. Global Kernel Methods vs. Published MKL Baselines",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=3)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fig1a_grouped_bars.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure 1b: Delta heatmap (ours vs. each baseline) ─────────────────────────

def plot_delta_heatmap(results: dict, save_path: str = None):
    methods, datasets, matrix = extract_accuracy_matrix(results)

    competitors = [m for m in [
        "SVM-RBF(C=10)", "SVM-Linear(C=1)", "KRR-Linear",
        "EasyMKL", "AverageMKL", "SMKL",
    ] if m in methods]

    # Best of "ours" for each dataset
    our_methods = [m for m in ["Ours+RBF", "Ours+Linear", "Ours+Mixed"] if m in methods]
    our_idx = [methods.index(m) for m in our_methods]
    best_ours = np.nanmax(matrix[our_idx, :], axis=0)  # (n_datasets,)

    delta_matrix = np.zeros((len(competitors), len(datasets)))
    for ci, comp in enumerate(competitors):
        ci_full = methods.index(comp)
        delta_matrix[ci, :] = best_ours - matrix[ci_full, :]

    vmax = max(np.abs(delta_matrix[np.isfinite(delta_matrix)]).max(), 1.0)

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="#fafafa")
    im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Δ Accuracy (pp)\nOurs(best) − Competitor",
                 fraction=0.025, pad=0.02)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.capitalize() for d in datasets], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(competitors)))
    ax.set_yticklabels(competitors, fontsize=9)
    ax.set_title(
        "EXP-1: Delta Heatmap — Best(Ours) minus Competitor\n"
        "Green = our model wins, Red = competitor wins",
        fontsize=11, fontweight="bold",
    )

    for ci in range(len(competitors)):
        for di in range(len(datasets)):
            v = delta_matrix[ci, di]
            if np.isfinite(v):
                ax.text(di, ci, f"{v:+.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(v) > vmax * 0.6 else "black")

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fig1b_delta_heatmap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure 1c: Rank chart ─────────────────────────────────────────────────────

def plot_rank_chart(results: dict, save_path: str = None):
    methods, datasets, matrix = extract_accuracy_matrix(results)

    # Rank methods on each dataset (1=best), NaN → last
    n_m, n_d = matrix.shape
    rank_matrix = np.full_like(matrix, np.nan)
    for di in range(n_d):
        col = matrix[:, di]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        ranks = np.argsort(np.argsort(-col[valid]))  # 0-indexed
        rank_matrix[valid, di] = ranks.astype(float) + 1.0

    mean_rank = np.nanmean(rank_matrix, axis=1)
    order = np.argsort(mean_rank)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fafafa")
    y = np.arange(len(methods))
    colors = [PALETTE.get(methods[i], "#aaa") for i in order]
    ax.barh(y, mean_rank[order], color=colors, edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([methods[i] for i in order], fontsize=9)
    ax.set_xlabel("Mean Rank (lower = better)", fontsize=10)
    ax.set_title("EXP-1: Mean Rank Across 10 UCI Datasets\n(1st = best on that dataset)", fontsize=11, fontweight="bold")
    ax.axvline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    for i, (mi, mr) in enumerate(zip(order, mean_rank[order])):
        ax.text(mr + 0.05, i, f"{mr:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fig1c_rank_chart.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure 1d: Alpha sparsity ─────────────────────────────────────────────────

def plot_alpha_sparsity(results: dict, save_path: str = None):
    """Show top-5 alpha weights per dataset for Ours+RBF."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), facecolor="#fafafa")
    axes = axes.ravel()

    for ax, ds in zip(axes, DATASET_ORDER):
        if ds not in results:
            ax.set_visible(False)
            continue
        ds_res = results[ds]
        if "Ours+RBF" not in ds_res or not isinstance(ds_res["Ours+RBF"], dict):
            ax.set_visible(False)
            continue
        alphas = ds_res["Ours+RBF"].get("alphas", [])
        if not alphas:
            ax.set_visible(False)
            continue

        alphas_np = np.array(alphas)
        p = len(alphas_np)
        x = np.arange(p)
        top5 = np.argsort(-alphas_np)[:5]

        bars = ax.bar(x, alphas_np, color="#4e79a7", alpha=0.75,
                      edgecolor="none", zorder=3)
        for i in top5:
            bars[i].set_color("#f28e2b")
            bars[i].set_alpha(1.0)

        acc_ours = ds_res["Ours+RBF"].get("accuracy_pct", "?")
        ax.set_title(f"{ds.capitalize()}\nOurs+RBF acc={acc_ours:.1f}%", fontsize=8, fontweight="bold")
        ax.set_xlabel("Feature", fontsize=7)
        ax.set_ylabel("α", fontsize=7)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "EXP-1: Learned Alpha Profiles (Ours+RBF)\n"
        "Orange bars = top-5 most important features per dataset",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "fig1d_alpha_sparsity.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Publication table ─────────────────────────────────────────────────────────

def print_latex_table(results: dict):
    methods, datasets, matrix = extract_accuracy_matrix(results)
    show = [m for m in [
        "Ours+RBF", "Ours+Mixed",
        "SVM-RBF(C=10)", "KRR-Linear",
        "EasyMKL", "AverageMKL", "SMKL",
    ] if m in methods]
    show_idx = [methods.index(m) for m in show]

    header = " & ".join(["Dataset"] + show) + " \\\\"
    print("\\begin{tabular}{l" + "r" * len(show) + "}")
    print("\\hline")
    print(header)
    print("\\hline")
    for di, ds in enumerate(datasets):
        col_vals = [matrix[mi, di] for mi in show_idx]
        best = max(v for v in col_vals if np.isfinite(v))
        cells = []
        for v in col_vals:
            if np.isnan(v):
                cells.append("--")
            elif abs(v - best) < 0.05:
                cells.append(f"\\textbf{{{v:.1f}}}")
            else:
                cells.append(f"{v:.1f}")
        print(ds.capitalize() + " & " + " & ".join(cells) + " \\\\")
    print("\\hline")
    print("\\end{tabular}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(json_path: str = None):
    print("[Figures] Generating benchmark figures ...")
    results = load_results(Path(json_path) if json_path else None)

    plot_grouped_bars(results)
    plot_delta_heatmap(results)
    plot_rank_chart(results)
    plot_alpha_sparsity(results)

    print("\n[Table] LaTeX accuracy table:")
    print_latex_table(results)
    print("\n[Figures] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    run(args.json)
