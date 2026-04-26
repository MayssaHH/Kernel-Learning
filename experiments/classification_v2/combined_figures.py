"""
Combined Figures: KAN_Kernel_NX vs KernelNetwork vs Baselines

Reads both benchmark JSONs and produces side-by-side publication figures.
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

DATASET_ORDER = [
    "iris", "wine", "breastcancer", "ionosphere",
    "spambase", "banknote", "heart", "haberman",
    "mammographic", "parkinsons",
]

PALETTE = {
    "KAN+KRR":         "#e6550d",
    "Ours+RBF":        "#2ca02c",
    "Ours+Linear":     "#98df8a",
    "Ours+Mixed":      "#17becf",
    "SVM-RBF(C=10)":   "#d62728",
    "SVM-Linear(C=1)": "#9467bd",
    "KRR-Linear":      "#1f77b4",
    "EasyMKL":         "#aec7e8",
    "SMKL":            "#ffbb78",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def acc(res, method):
    """Extract accuracy_pct from result dict, handling paper baseline dict."""
    if method in res and isinstance(res[method], dict):
        return res[method].get("accuracy_pct", np.nan)
    paper = res.get("_paper_baselines", {})
    if method in paper:
        return float(paper[method])
    return np.nan


# ── Figure C1: Combined grouped bar chart ─────────────────────────────────────

def plot_combined_bars(kn_res, kan_res, save_path=None):
    show_methods = ["KAN+KRR", "Ours+RBF", "Ours+Mixed",
                    "SVM-RBF(C=10)", "KRR-Linear", "EasyMKL", "SMKL"]

    n_ds = len(DATASET_ORDER)
    n_m  = len(show_methods)
    width = 0.85 / n_m
    offsets = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * width
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#fafafa")
    for mi, (method, offset) in enumerate(zip(show_methods, offsets)):
        vals = []
        for ds in DATASET_ORDER:
            if method == "KAN+KRR":
                v = acc(kan_res.get(ds, {}), method)
            else:
                v = acc(kn_res.get(ds, {}), method)
            vals.append(v)
        vals = np.array(vals)

        bars = ax.bar(x + offset, vals, width*0.95, color=PALETTE.get(method, "#aaa"),
                      alpha=0.9, label=method, edgecolor="white", lw=0.3, zorder=3)

        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isnan(val): continue
            col_vals = []
            for m2 in show_methods:
                if m2 == "KAN+KRR":
                    col_vals.append(acc(kan_res.get(DATASET_ORDER[bi], {}), m2))
                else:
                    col_vals.append(acc(kn_res.get(DATASET_ORDER[bi], {}), m2))
            if val == np.nanmax(col_vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.3,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DATASET_ORDER], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_ylim(55, 108)
    ax.set_title("Combined Benchmark: KAN_Kernel_NX vs KernelNetwork vs Global Kernel Baselines\n"
                 "10 UCI Datasets — paper_strict protocol (seed=123, 80/20 split)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=4)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "combined_fig1_benchmark_bars.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure C2: Head-to-head scatter: KAN vs KernelNetwork ─────────────────────

def plot_head_to_head(kn_res, kan_res, save_path=None):
    """Scatter plot: KAN accuracy vs KernelNetwork accuracy per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#fafafa")

    kn_methods  = ["Ours+RBF", "Ours+Mixed", "Ours+Linear"]
    kan_method  = "KAN+KRR"
    titles = ["vs Ours+RBF", "vs Ours+Mixed", "vs Ours+Linear"]

    for ax, kn_m, title in zip(axes, kn_methods, titles):
        x_vals, y_vals, labels = [], [], []
        for ds in DATASET_ORDER:
            kn_v  = acc(kn_res.get(ds,  {}), kn_m)
            kan_v = acc(kan_res.get(ds, {}), kan_method)
            if np.isnan(kn_v) or np.isnan(kan_v): continue
            x_vals.append(kn_v); y_vals.append(kan_v); labels.append(ds)

        x_vals, y_vals = np.array(x_vals), np.array(y_vals)
        lo, hi = min(x_vals.min(), y_vals.min())-2, max(x_vals.max(), y_vals.max())+2
        ax.plot([lo,hi],[lo,hi], "k--", lw=1, alpha=0.4, label="Equal")
        sc = ax.scatter(x_vals, y_vals, c=y_vals-x_vals, cmap="RdYlGn",
                        vmin=-15, vmax=15, s=80, edgecolors="k", lw=0.5, zorder=4)
        for xi, yi, lb in zip(x_vals, y_vals, labels):
            ax.annotate(lb[:4], (xi, yi), textcoords="offset points", xytext=(4,3), fontsize=7)
        plt.colorbar(sc, ax=ax, label="KAN − KN (pp)", fraction=0.04)
        ax.set_xlabel(f"{kn_m} (%)", fontsize=10)
        ax.set_ylabel("KAN+KRR (%)", fontsize=10)
        ax.set_title(f"KAN+KRR {title}", fontsize=10, fontweight="bold")
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.grid(alpha=0.2); ax.spines[["top","right"]].set_visible(False)

        n_kan_wins = (y_vals > x_vals).sum()
        ax.text(0.05, 0.95, f"KAN wins: {n_kan_wins}/{len(x_vals)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="#fff3cd", alpha=0.8))

    fig.suptitle("Head-to-Head: KAN_Kernel_NX vs KernelNetwork\n"
                 "Points above diagonal = KAN wins; below = KernelNetwork wins",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "combined_fig2_head_to_head.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure C3: Delta heatmap (both architectures vs baselines) ────────────────

def plot_dual_delta(kn_res, kan_res, save_path=None):
    competitors = ["SVM-RBF(C=10)", "SVM-Linear(C=1)", "KRR-Linear", "EasyMKL", "SMKL"]
    n_ds, n_comp = len(DATASET_ORDER), len(competitors)

    def best_ours_row(res, methods):
        row = np.full(n_ds, np.nan)
        for di, ds in enumerate(DATASET_ORDER):
            vals = [acc(res.get(ds,{}), m) for m in methods]
            valid = [v for v in vals if not np.isnan(v)]
            if valid: row[di] = max(valid)
        return row

    best_kn  = best_ours_row(kn_res,  ["Ours+RBF","Ours+Mixed","Ours+Linear"])
    best_kan = best_ours_row(kan_res, ["KAN+KRR"])

    def delta_matrix(best_row, res_for_paper):
        mat = np.zeros((n_comp, n_ds))
        for ci, comp in enumerate(competitors):
            for di, ds in enumerate(DATASET_ORDER):
                cv = acc(res_for_paper.get(ds,{}), comp)
                mat[ci, di] = best_row[di] - cv
        return mat

    delta_kn  = delta_matrix(best_kn,  kn_res)
    delta_kan = delta_matrix(best_kan, kn_res)  # baselines same for both

    vmax = max(np.abs(delta_kn[np.isfinite(delta_kn)]).max(),
               np.abs(delta_kan[np.isfinite(delta_kan)]).max(), 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), facecolor="#fafafa")

    for ax, mat, title in [
        (ax1, delta_kn,  "KernelNetwork (Best of 3 archs) − Competitor"),
        (ax2, delta_kan, "KAN+KRR − Competitor"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="Δ Accuracy (pp)", fraction=0.03)
        ax.set_xticks(range(n_ds))
        ax.set_xticklabels([d.capitalize() for d in DATASET_ORDER], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(n_comp))
        ax.set_yticklabels(competitors, fontsize=9)
        ax.set_title(f"{title}\nGreen = we win", fontsize=10, fontweight="bold")
        for ci in range(n_comp):
            for di in range(n_ds):
                v = mat[ci, di]
                if np.isfinite(v):
                    ax.text(di, ci, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v)>vmax*0.6 else "black")

    fig.suptitle("Combined Delta Heatmap: Our Architectures vs Baselines\n"
                 "Green = our model wins, Red = baseline wins",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "combined_fig3_delta_heatmap.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Figure C4: Mean rank chart (both architectures) ───────────────────────────

def plot_combined_rank(kn_res, kan_res, save_path=None):
    all_methods = ["KAN+KRR", "Ours+RBF", "Ours+Mixed",
                   "SVM-RBF(C=10)", "SVM-Linear(C=1)", "KRR-Linear", "EasyMKL", "SMKL"]

    n_m = len(all_methods); n_ds = len(DATASET_ORDER)
    matrix = np.full((n_m, n_ds), np.nan)

    for di, ds in enumerate(DATASET_ORDER):
        for mi, m in enumerate(all_methods):
            if m == "KAN+KRR":
                matrix[mi, di] = acc(kan_res.get(ds,{}), m)
            else:
                matrix[mi, di] = acc(kn_res.get(ds,{}), m)

    rank_matrix = np.full_like(matrix, np.nan)
    for di in range(n_ds):
        col = matrix[:, di]; valid = ~np.isnan(col)
        if valid.sum() == 0: continue
        ranks = np.argsort(np.argsort(-col[valid])).astype(float) + 1
        rank_matrix[valid, di] = ranks

    mean_rank = np.nanmean(rank_matrix, axis=1)
    order = np.argsort(mean_rank)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#fafafa")
    y = np.arange(n_m)
    colors = [PALETTE.get(all_methods[i], "#aaa") for i in order]
    ax.barh(y, mean_rank[order], color=colors, edgecolor="white", lw=0.4, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([all_methods[i] for i in order], fontsize=10)
    ax.set_xlabel("Mean Rank (lower = better)", fontsize=10)
    ax.set_title("Combined Mean Rank — Both Architectures vs All Baselines\n"
                 "(10 UCI datasets, paper_strict protocol)",
                 fontsize=11, fontweight="bold")
    ax.axvline(1, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    ax.invert_yaxis()
    for i, (mi, mr) in enumerate(zip(order, mean_rank[order])):
        ax.text(mr+0.05, i, f"{mr:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "combined_fig4_rank_chart.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_combined_latex(kn_res, kan_res):
    show = ["KAN+KRR", "Ours+RBF", "Ours+Mixed",
            "SVM-RBF(C=10)", "KRR-Linear", "EasyMKL", "SMKL"]
    header = " & ".join(["Dataset"] + show) + " \\\\"
    print("\n\\begin{tabular}{l" + "r"*len(show) + "}")
    print("\\hline"); print(header); print("\\hline")
    for ds in DATASET_ORDER:
        vals = []
        for m in show:
            if m == "KAN+KRR":
                vals.append(acc(kan_res.get(ds,{}), m))
            else:
                vals.append(acc(kn_res.get(ds,{}), m))
        best = max(v for v in vals if not np.isnan(v))
        cells = []
        for v in vals:
            if np.isnan(v): cells.append("--")
            elif abs(v-best)<0.05: cells.append(f"\\textbf{{{v:.1f}}}")
            else: cells.append(f"{v:.1f}")
        print(ds.capitalize() + " & " + " & ".join(cells) + " \\\\")
    print("\\hline\n\\end{tabular}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    kn_path  = RESULTS_DIR / "benchmark_results.json"
    kan_path = RESULTS_DIR / "kan_benchmark_results.json"

    if not kn_path.exists():
        print(f"[combined] ERROR: {kn_path} not found. Run benchmark.py first."); return
    if not kan_path.exists():
        print(f"[combined] ERROR: {kan_path} not found. Run kan_benchmark.py first."); return

    kn_res  = load_json(kn_path)
    kan_res = load_json(kan_path)

    print("[combined] Generating combined figures ...")
    plot_combined_bars(kn_res, kan_res)
    plot_head_to_head(kn_res, kan_res)
    plot_dual_delta(kn_res, kan_res)
    plot_combined_rank(kn_res, kan_res)
    print_combined_latex(kn_res, kan_res)
    print("[combined] Done.")


if __name__ == "__main__":
    run()
