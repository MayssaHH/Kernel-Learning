"""
Clean KRR and SVM comparison figures — one set per classifier.

Reads:
  results/benchmark_results.json          (KN models + global KRR baselines)
  results/kan_benchmark_results.json      (KAN + global KRR baselines)
  results/svm/benchmark_svm_results.json  (all models + global SVM baselines)

Produces:
  figures/krr/  — all figures for the KRR comparison
  figures/svm/  — all figures for the SVM comparison
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = Path(__file__).parent / "results"
BASE_FIGS = Path(__file__).parent / "figures"
KRR_FIGS = BASE_FIGS / "krr"
SVM_FIGS = BASE_FIGS / "svm"
KRR_FIGS.mkdir(parents=True, exist_ok=True)
SVM_FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ["iris","wine","breastcancer","ionosphere","spambase",
            "banknote","heart","haberman","mammographic","parkinsons"]
DS_SHORT  = ["Iris","Wine","Breast\nCancer","Ionosphere","Spambase",
             "Banknote","Heart","Haberman","Mammo-\ngraphic","Parkinsons"]

# ── Colour palette ─────────────────────────────────────────────────────────────

COLORS = {
    # Our proposed
    "KN+RBF":    "#2ca02c",
    "KN+Linear": "#98df8a",
    "KN+Mixed":  "#17becf",
    "KAN":       "#e6550d",
    # Global baselines
    "RBF":       "#d62728",
    "Linear":    "#1f77b4",
    "Poly3":     "#9467bd",
    # Paper MKL
    "EasyMKL":   "#aec7e8",
    "SMKL":      "#ffbb78",
}


def load_json(p):
    if not Path(p).exists():
        return {}
    with open(p) as f:
        return json.load(f)


def acc(res, ds, key):
    d = res.get(ds, {})
    if key in d and isinstance(d[key], dict):
        return d[key].get("accuracy_pct", np.nan)
    return float(d.get("_paper_baselines", {}).get(key, np.nan))


def build_krr_matrix(kn_res, kan_res):
    """Return ordered list of (label, color, values[10]) for KRR comparison."""
    methods = [
        ("KN+RBF",  COLORS["KN+RBF"],  [acc(kn_res, d, "Ours+RBF")    for d in DATASETS]),
        ("KN+Linear",COLORS["KN+Linear"],[acc(kn_res, d, "Ours+Linear") for d in DATASETS]),
        ("KN+Mixed", COLORS["KN+Mixed"], [acc(kn_res, d, "Ours+Mixed")  for d in DATASETS]),
        ("KAN",      COLORS["KAN"],      [acc(kan_res, d, "KAN+KRR")    for d in DATASETS]),
        ("Global RBF",COLORS["RBF"],     [acc(kn_res, d, "KRR-RBF(γ=0.5)") for d in DATASETS]),
        ("Global Lin",COLORS["Linear"],  [acc(kn_res, d, "KRR-Linear")  for d in DATASETS]),
        ("Global Poly3",COLORS["Poly3"], [acc(kn_res, d, "KRR-Poly3")   for d in DATASETS]),
        ("SMKL",     COLORS["SMKL"],     [acc(kn_res, d, "SMKL")        for d in DATASETS]),
    ]
    return methods


def build_svm_matrix(svm_res, kn_res):
    methods = [
        ("KN+RBF",   COLORS["KN+RBF"],  [acc(svm_res, d, "KN+RBF+SVM")    for d in DATASETS]),
        ("KN+Linear",COLORS["KN+Linear"],[acc(svm_res, d, "KN+Linear+SVM") for d in DATASETS]),
        ("KN+Mixed", COLORS["KN+Mixed"], [acc(svm_res, d, "KN+Mixed+SVM")  for d in DATASETS]),
        ("KAN",      COLORS["KAN"],      [acc(svm_res, d, "KAN+SVM")       for d in DATASETS]),
        ("Global RBF",COLORS["RBF"],     [acc(svm_res, d, "SVM-RBF")       for d in DATASETS]),
        ("Global Lin",COLORS["Linear"],  [acc(svm_res, d, "SVM-Linear")     for d in DATASETS]),
        ("Global Poly3",COLORS["Poly3"], [acc(svm_res, d, "SVM-Poly3")     for d in DATASETS]),
        ("SMKL",     COLORS["SMKL"],     [acc(kn_res, d, "SMKL")           for d in DATASETS]),
    ]
    return methods


# ── Plot helpers ───────────────────────────────────────────────────────────────

def grouped_bar(methods, title, out_path, classifier_label):
    n_m = len(methods); n_d = len(DATASETS)
    w = 0.82 / n_m
    offsets = np.linspace(-(n_m-1)/2, (n_m-1)/2, n_m) * w
    x = np.arange(n_d)

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
    for (label, color, vals), off in zip(methods, offsets):
        vals = np.array(vals, dtype=float)
        bars = ax.bar(x + off, vals, w*0.93, color=color, alpha=0.88,
                      label=label, edgecolor="white", lw=0.3, zorder=3)
        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isnan(val): continue
            col_vals = np.array([m[2][bi] for m in methods], dtype=float)
            if val >= np.nanmax(col_vals) - 0.05:
                ax.text(bar.get_x()+bar.get_width()/2, val+0.4,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(DS_SHORT, fontsize=9.5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_ylim(20, 112)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, ncol=2)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

    # Divider: label our methods vs baselines
    ax.axvline(3.5, color="#ccc", lw=1, ls=":")
    ax.text(1.5, 110, "← Simpler datasets", ha="center", fontsize=8, color="#999")
    ax.text(6.5, 110, "Harder / heterogeneous datasets →", ha="center", fontsize=8, color="#999")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


def scorecard(methods, title, out_path):
    """Traffic-light table: green=best, red=worst per row."""
    n_m = len(methods); n_d = len(DATASETS)
    mat = np.array([np.array(m[2], dtype=float) for m in methods]).T  # (n_d, n_m)

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.set_xlim(-0.5, n_m-0.5); ax.set_ylim(-0.5, n_d-0.5); ax.invert_yaxis()

    for di in range(n_d):
        row = mat[di]; best = np.nanmax(row); worst = np.nanmin(row)
        for mi in range(n_m):
            v = row[mi]
            t = (v - worst) / (best - worst + 1e-6) if not np.isnan(v) else 0
            r = 0.85 - 0.55*t; g = 0.30 + 0.60*t; b = 0.25
            fc = (r, g, b) if not np.isnan(v) else "#eee"
            rect = mpatches.FancyBboxPatch((mi-0.44, di-0.41), 0.88, 0.82,
                                           boxstyle="round,pad=0.03",
                                           fc=fc, ec="white", lw=1.2)
            ax.add_patch(rect)
            if not np.isnan(v):
                bold = "bold" if abs(v-best)<0.05 else "normal"
                ax.text(mi, di, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.5, fontweight=bold,
                        color="white" if t < 0.35 else "#111")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([m[0] for m in methods], fontsize=9, fontweight="bold")
    ax.set_yticks(range(n_d))
    ax.set_yticklabels(DS_SHORT, fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title+"\nGreen = best on that dataset  ·  Red = worst  ·  Bold = within 0.05% of best",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


def mean_rank_chart(methods, title, out_path):
    n_m = len(methods); n_d = len(DATASETS)
    mat = np.array([np.array(m[2], dtype=float) for m in methods]).T

    rank_mat = np.full_like(mat, np.nan)
    for di in range(n_d):
        col = mat[di]; valid = ~np.isnan(col)
        if valid.sum() == 0: continue
        ranks = np.argsort(np.argsort(-col[valid])).astype(float)+1
        rank_mat[di, valid] = ranks

    mean_r = np.nanmean(rank_mat, axis=0)
    order  = np.argsort(mean_r)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    y = np.arange(n_m)
    colors_o = [methods[i][1] for i in order]
    labels_o = [methods[i][0] for i in order]
    ax.barh(y, mean_r[order], color=colors_o, edgecolor="white", lw=0.4, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels_o, fontsize=10)
    ax.set_xlabel("Mean Rank across 10 datasets (lower = better)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axvline(1, color="#bbb", lw=1, ls="--")
    ax.grid(axis="x", alpha=0.2); ax.spines[["top","right"]].set_visible(False)
    ax.invert_yaxis()
    for i, mr in enumerate(mean_r[order]):
        ax.text(mr+0.05, i, f"{mr:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


def our_vs_best(methods, title, out_path, our_labels=("KN+RBF","KN+Mixed","KAN")):
    """Simple: our best vs best global baseline per dataset."""
    our_best = np.full(len(DATASETS), np.nan)
    glob_best = np.full(len(DATASETS), np.nan)

    for label, _, vals in methods:
        vals = np.array(vals, dtype=float)
        if label in our_labels:
            our_best = np.fmax(our_best, vals)
        elif label not in ("SMKL", "EasyMKL"):
            glob_best = np.fmax(glob_best, vals)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    y = np.arange(len(DATASETS)); h = 0.35

    ax.barh(y+h/2, our_best,  h, color="#2ca02c", alpha=0.88, label="Our best (KN or KAN)", zorder=3)
    ax.barh(y-h/2, glob_best, h, color="#d62728", alpha=0.88, label="Best global baseline",  zorder=3)

    for i, (o, g) in enumerate(zip(our_best, glob_best)):
        if not np.isnan(o):
            ax.text(o+0.3,  i+h/2, f"{o:.0f}%",  va="center", fontsize=8.5, fontweight="bold")
        if not np.isnan(g):
            ax.text(g+0.3,  i-h/2, f"{g:.0f}%",  va="center", fontsize=8.5)
        gap = o - g
        color = "#2ca02c" if gap > 0 else "#d62728"
        symbol = f"+{gap:.0f}pp" if gap > 0 else f"{gap:.0f}pp"
        ax.text(102, i, symbol, va="center", ha="left", fontsize=8, color=color, fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(DS_SHORT, fontsize=10)
    ax.set_xlabel("Test Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 113)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.tick_params(bottom=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    kn_res  = load_json(RESULTS / "benchmark_results.json")
    kan_res = load_json(RESULTS / "kan_benchmark_results.json")
    svm_res = load_json(RESULTS / "svm" / "benchmark_svm_results.json")

    print("[comparison_figures] KRR figures ...")
    krr_methods = build_krr_matrix(kn_res, kan_res)
    grouped_bar(krr_methods,
                "All Kernels + KRR Classifier — 10 UCI Datasets\nOur proposed kernels (KN, KAN) vs global baselines",
                KRR_FIGS / "krr_grouped_bars.png", "KRR")
    scorecard(krr_methods,
              "KRR Classifier — Accuracy Scorecard",
              KRR_FIGS / "krr_scorecard.png")
    mean_rank_chart(krr_methods,
                    "Mean Rank across 10 Datasets — KRR Classifier\n(Lower = better kernel)",
                    KRR_FIGS / "krr_mean_rank.png")
    our_vs_best(krr_methods,
                "Our Best Kernel vs Best Global Baseline — KRR Classifier\n+/- = accuracy gap",
                KRR_FIGS / "krr_our_vs_best.png")

    if svm_res:
        print("[comparison_figures] SVM figures ...")
        svm_methods = build_svm_matrix(svm_res, kn_res)
        grouped_bar(svm_methods,
                    "All Kernels + SVM Classifier — 10 UCI Datasets\nOur proposed kernels (KN, KAN) vs global baselines",
                    SVM_FIGS / "svm_grouped_bars.png", "SVM")
        scorecard(svm_methods,
                  "SVM Classifier — Accuracy Scorecard",
                  SVM_FIGS / "svm_scorecard.png")
        mean_rank_chart(svm_methods,
                        "Mean Rank across 10 Datasets — SVM Classifier\n(Lower = better kernel)",
                        SVM_FIGS / "svm_mean_rank.png")
        our_vs_best(svm_methods,
                    "Our Best Kernel vs Best Global Baseline — SVM Classifier\n+/- = accuracy gap",
                    SVM_FIGS / "svm_our_vs_best.png",
                    our_labels=("KN+RBF","KN+Mixed","KAN"))

        # KRR vs SVM side-by-side for our models
        print("[comparison_figures] KRR vs SVM comparison ...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
        for ax, methods, clf in [(axes[0], krr_methods, "KRR"), (axes[1], svm_methods, "SVM")]:
            ours = [m for m in methods if m[0] in ("KN+RBF","KN+Mixed","KAN")]
            n_m = len(ours); x = np.arange(len(DATASETS))
            w = 0.8/n_m; offsets = np.linspace(-(n_m-1)/2,(n_m-1)/2,n_m)*w
            for (label, color, vals), off in zip(ours, offsets):
                ax.bar(x+off, vals, w*0.93, color=color, alpha=0.88,
                       label=label, edgecolor="white", lw=0.3, zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(DS_SHORT, fontsize=8)
            ax.set_ylabel("Test Accuracy (%)", fontsize=10)
            ax.set_ylim(20, 112); ax.set_title(f"Our Kernels + {clf}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2); ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("KRR vs SVM: Does the Choice of Downstream Classifier Matter?\n"
                     "Our learned kernels evaluated with both classifiers",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = BASE_FIGS / "krr_vs_svm_comparison.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"  Saved → {p}")
        plt.close(fig)
    else:
        print("[comparison_figures] SVM results not available yet — skipping SVM figures.")

    print("[comparison_figures] Done.")


if __name__ == "__main__":
    run()
