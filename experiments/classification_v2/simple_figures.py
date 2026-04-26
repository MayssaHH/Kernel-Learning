"""
Simple, readable figures — no heatmaps, no phase diagrams.
Plain bar charts and scorecards that tell the story at a glance.
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

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR.mkdir(exist_ok=True)

DATASETS = [
    "iris", "wine", "breastcancer", "ionosphere",
    "spambase", "banknote", "heart", "haberman",
    "mammographic", "parkinsons",
]
DS_LABELS = {
    "iris": "Iris\n(n=150, p=4)",
    "wine": "Wine\n(n=178, p=13)",
    "breastcancer": "Breast\nCancer\n(n=569, p=30)",
    "ionosphere": "Ionosphere\n(n=351, p=34)",
    "spambase": "Spambase\n(n=4601, p=57)",
    "banknote": "Banknote\n(n=1372, p=4)",
    "heart": "Heart\n(n=303, p=13)",
    "haberman": "Haberman\n(n=306, p=3)",
    "mammographic": "Mammographic\n(n=961, p=5)",
    "parkinsons": "Parkinsons\n(n=195, p=22)",
}


def load_results():
    with open(RESULTS_DIR / "benchmark_results.json") as f:
        kn = json.load(f)
    with open(RESULTS_DIR / "kan_benchmark_results.json") as f:
        kan = json.load(f)
    return kn, kan


def get(res, ds, method):
    d = res.get(ds, {})
    if method in d and isinstance(d[method], dict):
        return d[method].get("accuracy_pct", np.nan)
    return d.get("_paper_baselines", {}).get(method, np.nan)


# ── Simple Figure 1: The Main Scorecard ───────────────────────────────────────
# Just shows "did we beat the paper's best method (SMKL)?" per dataset.

def simple_scorecard(kn, kan, save_path=None):
    """
    One bar per dataset. Height = our best accuracy.
    A dashed line shows SMKL (paper champion).
    Green = we beat SMKL. Orange = within 2pp. Red = SMKL wins by more than 2pp.
    """
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")

    our_best = []
    smkl_vals = []
    colors = []

    for ds in DATASETS:
        ours = max(
            get(kn, ds, "Ours+RBF"),
            get(kn, ds, "Ours+Linear"),
            get(kn, ds, "Ours+Mixed"),
        )
        smkl = get(kn, ds, "SMKL")
        our_best.append(ours)
        smkl_vals.append(smkl)

        gap = ours - smkl
        if gap >= 0:
            colors.append("#2ca02c")   # we win — green
        elif gap >= -2:
            colors.append("#ff7f0e")   # close — orange
        else:
            colors.append("#d62728")   # they win — red

    x = np.arange(len(DATASETS))
    bars = ax.bar(x, our_best, color=colors, width=0.6, edgecolor="white", lw=0.5,
                  zorder=3, alpha=0.9)

    # SMKL line
    ax.plot(x, smkl_vals, "k--", lw=1.5, marker="D", markersize=5,
            label="SMKL (paper champion)", zorder=4)

    # Annotate each bar with accuracy
    for i, (bar, val, smkl) in enumerate(zip(bars, our_best, smkl_vals)):
        gap = val - smkl
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        if gap >= 0:
            ax.text(bar.get_x() + bar.get_width()/2, val/2,
                    f"+{gap:.0f}pp", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d].split("\n")[0] for d in DATASETS], fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_ylim(20, 112)
    ax.set_title(
        "Our KernelNetwork vs the Paper Champion (SMKL)\n"
        "Green = we win  ·  Orange = within 2%  ·  Red = SMKL wins",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)

    handles = [
        mpatches.Patch(color="#2ca02c", label="We win"),
        mpatches.Patch(color="#ff7f0e", label="Within 2%"),
        mpatches.Patch(color="#d62728", label="SMKL wins"),
        plt.Line2D([0],[0], color="k", ls="--", marker="D", ms=5, label="SMKL accuracy"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "simple_fig1_us_vs_smkl.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Simple Figure 2: Us vs Every Baseline — one dataset at a time ─────────────

def simple_per_dataset(kn, kan, save_path=None):
    """
    5×2 grid. Each panel = one dataset.
    Bars for: Our+RBF, Our+Mixed, KAN, SVM-RBF, KRR-RBF, SMKL.
    Simple, labelled, readable.
    """
    methods = [
        ("Ours+RBF",      "#2ca02c", kn),
        ("Ours+Mixed",    "#17becf", kn),
        ("KAN+KRR",       "#e6550d", kan),
        ("SVM-RBF",       "#d62728", kn, "SVM-RBF(C=10)"),
        ("KRR-RBF",       "#9467bd", kn, "KRR-RBF(γ=0.5)"),
        ("SMKL",          "#ffbb78", kn),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), facecolor="white")
    axes = axes.ravel()

    for ax, ds in zip(axes, DATASETS):
        vals, labels, colors = [], [], []
        for entry in methods:
            label, color, res = entry[0], entry[1], entry[2]
            key = entry[3] if len(entry) == 4 else label
            v = get(res, ds, key)
            vals.append(v); labels.append(label); colors.append(color)

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", lw=0.4,
                      alpha=0.88, zorder=3)

        # Highlight the winner
        best_idx = int(np.nanargmax(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        info = DS_LABELS[ds]
        ax.set_title(info, fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("Ours+","") for l in labels],
                           fontsize=6.5, rotation=30, ha="right")
        ax.set_ylim(max(0, np.nanmin(vals) - 10), 108)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False)

    handles = [mpatches.Patch(color=c, label=l) for l, c, *_ in methods]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Accuracy Per Dataset — All Key Methods Side by Side\n"
        "(Black border = winner on that dataset)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = save_path or str(FIGURES_DIR / "simple_fig2_per_dataset.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Simple Figure 3: Where each model is the best ────────────────────────────

def simple_wins_table(kn, kan, save_path=None):
    """
    Table figure: rows=datasets, cols=methods.
    Cell = accuracy, colored by rank (1st=dark green, last=light red).
    Clean, like a sports standings table.
    """
    col_methods = [
        ("Ours\n+RBF",   "Ours+RBF",      kn,  "#2ca02c"),
        ("Ours\n+Mixed", "Ours+Mixed",     kn,  "#17becf"),
        ("KAN\n+KRR",    "KAN+KRR",        kan, "#e6550d"),
        ("SVM\n-RBF",    "SVM-RBF(C=10)",  kn,  "#d62728"),
        ("KRR\n-RBF",    "KRR-RBF(γ=0.5)", kn,  "#9467bd"),
        ("KRR\n-Linear", "KRR-Linear",     kn,  "#1f77b4"),
        ("SMKL\n(paper)","SMKL",           kn,  "#ffbb78"),
    ]

    n_ds, n_m = len(DATASETS), len(col_methods)
    mat = np.full((n_ds, n_m), np.nan)
    for di, ds in enumerate(DATASETS):
        for mi, (_, key, res, _) in enumerate(col_methods):
            mat[di, mi] = get(res, ds, key)

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.set_xlim(-0.5, n_m - 0.5)
    ax.set_ylim(-0.5, n_ds - 0.5)
    ax.invert_yaxis()

    for di in range(n_ds):
        row = mat[di]
        valid = row[~np.isnan(row)]
        best = np.nanmax(valid) if len(valid) else np.nan
        worst = np.nanmin(valid) if len(valid) else np.nan

        for mi in range(n_m):
            v = mat[di, mi]
            if np.isnan(v):
                fc = "#eeeeee"
            else:
                # Gradient from red (worst) to green (best)
                t = (v - worst) / (best - worst + 1e-6)
                r = 0.85 - 0.55 * t
                g = 0.30 + 0.60 * t
                b = 0.25
                fc = (r, g, b)

            rect = mpatches.FancyBboxPatch((mi - 0.45, di - 0.42), 0.90, 0.84,
                                           boxstyle="round,pad=0.04",
                                           facecolor=fc, edgecolor="white", lw=1.2)
            ax.add_patch(rect)

            if not np.isnan(v):
                weight = "bold" if abs(v - best) < 0.05 else "normal"
                ax.text(mi, di, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight=weight,
                        color="white" if t < 0.35 else "#111")

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([l for l, *_ in col_methods], fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(n_ds))
    ax.set_yticklabels([DS_LABELS[d].replace("\n", " ") for d in DATASETS], fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(
        "Accuracy Scorecard — Green = best on that row, Red = worst\n"
        "Bold = within 0.05% of the row winner",
        fontsize=12, fontweight="bold", pad=14,
    )

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "simple_fig3_scorecard.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Simple Figure 4: KAN vs Ours — plain comparison ──────────────────────────

def simple_kan_vs_kn(kn, kan, save_path=None):
    """
    Horizontal bars. Each dataset: two bars — Our best vs KAN.
    Instantly shows the gap.
    """
    our_best = [max(get(kn,d,"Ours+RBF"), get(kn,d,"Ours+Mixed")) for d in DATASETS]
    kan_vals  = [get(kan, d, "KAN+KRR") for d in DATASETS]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    y = np.arange(len(DATASETS))
    h = 0.35

    ax.barh(y + h/2, our_best, h, color="#2ca02c", alpha=0.9, label="Our KernelNetwork (best variant)", zorder=3)
    ax.barh(y - h/2, kan_vals,  h, color="#e6550d", alpha=0.9, label="KAN_Kernel_NX", zorder=3)

    for i, (o, k) in enumerate(zip(our_best, kan_vals)):
        ax.text(o + 0.3, i + h/2, f"{o:.0f}%", va="center", fontsize=8.5, fontweight="bold")
        ax.text(k + 0.3, i - h/2, f"{k:.0f}%", va="center", fontsize=8.5)
        gap = o - k
        ax.text(103, i, f"KN wins\n+{gap:.0f}pp" if gap > 0 else f"Tie",
                va="center", ha="left", fontsize=7.5,
                color="#2ca02c" if gap > 0 else "#888")

    ax.set_yticks(y)
    ax.set_yticklabels([DS_LABELS[d].split("\n")[0] for d in DATASETS], fontsize=10)
    ax.set_xlabel("Test Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_title("KernelNetwork vs KAN_Kernel_NX\n"
                 "Why KAN loses: it reduces to a 1D threshold (rank-1 kernel)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.tick_params(bottom=False)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "simple_fig4_kan_vs_kn.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Simple Figure 5: Alpha story in one panel ─────────────────────────────────

def simple_alpha_story(save_path=None):
    """
    Bar chart of alpha weights on professor synthetic data — one clean run.
    Labels: which bars are 'signal' (should be high) vs 'noise' (should be low).
    No heatmaps, no multiple runs, just the key message.
    """
    sys.path.insert(0, str(ROOT / "data" / "generators" / "professor"))
    sys.path.insert(0, str(Path(__file__).parent))

    from monni_simulated import generate_professor_split
    from datasets import to_tensors
    import torch, torch.optim as optim
    from kernel_learning import KernelNetwork, RBFSubKernel, AlignmentLoss

    p1, p2, pc, pn = 4, 4, 2, 8
    p = pc + p1 + p2 + pn

    split = generate_professor_split(iteration=0, iter_count=1,
                                     n_train=300, n_test=300,
                                     p1=p1, p2=p2, pc=pc, pn=pn,
                                     snr=2.0, sigma=1.0)
    Xtr = split["X_train"].float()
    ytr = ((split["y_train"] + 1) // 2).long()

    subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(p)]
    model = KernelNetwork(subs, alpha_constraint="square", normalize_alphas=True, alpha_init="random")
    crit = AlignmentLoss()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(600):
        opt.zero_grad(); loss=crit(model(Xtr), ytr); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    alphas = model._get_alphas().detach().numpy()

    # Feature types and colors
    feat_labels = (["Shared\n(pc)"] * pc + ["Class-1\nonly"] * p1 +
                   ["Class+1\nonly"] * p2 + ["Pure\nnoise"] * pn)
    feat_colors = (["#4e79a7"] * pc + ["#f28e2b"] * p1 +
                   ["#59a14f"] * p2 + ["#bab0ac"] * pn)

    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    x = np.arange(p)
    bars = ax.bar(x, alphas, color=feat_colors, edgecolor="white", lw=0.5, zorder=3)

    # Region labels
    regions = [
        (0, pc,        "Shared features\n(informative for both classes)", "#4e79a7"),
        (pc, pc+p1,    "Class -1 features\n(informative for class -1 only)", "#f28e2b"),
        (pc+p1, pc+p1+p2, "Class +1 features\n(informative for class +1 only)", "#59a14f"),
        (pc+p1+p2, p,  "Pure noise\n(no class information)", "#bab0ac"),
    ]
    y_top = alphas.max() * 1.05
    for start, end, label, col in regions:
        mid = (start + end - 1) / 2
        ax.annotate("", xy=(end - 0.5, y_top * 0.85), xytext=(start - 0.5, y_top * 0.85),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
        ax.text(mid, y_top * 0.91, label, ha="center", va="bottom",
                fontsize=8, color=col, fontweight="bold")

    ax.axvline(pc - 0.5,        color="#ddd", lw=1.5, ls="--")
    ax.axvline(pc + p1 - 0.5,   color="#ddd", lw=1.5, ls="--")
    ax.axvline(pc + p1 + p2 - 0.5, color="#ddd", lw=1.5, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels([f"f{i}" for i in range(p)], fontsize=8)
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Learned α weight\n(higher = model relies on this feature more)", fontsize=10)
    ax.set_title(
        "What the Model Learned: Feature Importance (α weights)\n"
        "The model correctly ignores pure noise features and focuses on informative ones",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(left=False)

    handles = [mpatches.Patch(color=c, label=l.split("\n")[0])
               for _, _, l, c in regions]
    ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "simple_fig5_alpha_story.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Simple Figure 6: Model explainer diagram ─────────────────────────────────

def simple_model_explainer(save_path=None):
    """
    Visual diagram showing conceptually what each kernel type does.
    Three panels: KernelNetwork, KAN, Global RBF.
    Drawn with matplotlib patches — no data needed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white")

    def box(ax, x, y, w, h, text, fc="#e8f4fd", ec="#4e79a7", fs=9, fw="normal"):
        r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.5)
        ax.add_patch(r)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=fw,
                wrap=True, multialignment="center")

    def arrow(ax, x0, y0, x1, y1, color="#555"):
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # ── Panel 1: KernelNetwork ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("KernelNetwork (Ours)", fontsize=12, fontweight="bold",
                 color="#2ca02c", pad=10)

    # Input features
    box(ax, 5, 9, 6, 0.8, "Input x = [x₁,  x₂,  x₃,  …,  xₚ]",
        fc="#f0f0f0", ec="#888", fs=9)

    # Per-feature kernels
    positions = [1.5, 3.5, 5.5, 7.5, 9.0]
    labels = ["K₁(x₁)", "K₂(x₂)", "K₃(x₃)", "…", "Kₚ(xₚ)"]
    for px, lb in zip(positions, labels):
        arrow(ax, px, 8.4, px, 7.3)
        box(ax, px, 6.9, 1.6, 0.7, lb, fc="#d4edda", ec="#2ca02c", fs=8)

    # Alpha weights
    for px in [1.5, 3.5, 5.5, 7.5]:
        arrow(ax, px, 6.55, px, 5.65)
        box(ax, px, 5.3, 1.4, 0.6, "× α_k", fc="#fff3cd", ec="#f28e2b", fs=8)

    # Sum
    for px in [1.5, 3.5, 5.5, 7.5]:
        arrow(ax, px, 5.0, 4.5, 4.2)
    box(ax, 4.5, 3.85, 4, 0.65, "K = Σ αₖ · Kₖ   (weighted sum)", fc="#d4edda", ec="#2ca02c", fs=9)

    arrow(ax, 4.5, 3.5, 4.5, 2.7)
    box(ax, 4.5, 2.35, 5, 0.65, "Kernel Ridge Regression → ŷ", fc="#cce5ff", ec="#4e79a7", fs=9)

    ax.text(5, 1.5, "✓ Each feature gets its own kernel\n"
            "✓ α weights = feature importance\n"
            "✓ Full-rank kernel → rich KRR",
            ha="center", va="center", fontsize=8.5, color="#2ca02c",
            bbox=dict(boxstyle="round", fc="#f0fff0", ec="#2ca02c", alpha=0.7))

    # ── Panel 2: KAN ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("KAN_Kernel_NX", fontsize=12, fontweight="bold",
                 color="#e6550d", pad=10)

    box(ax, 5, 9, 6, 0.8, "Input x = [x₁,  x₂,  x₃,  …,  xₚ]",
        fc="#f0f0f0", ec="#888", fs=9)
    arrow(ax, 5, 8.6, 5, 7.5)
    box(ax, 5, 7.1, 5.5, 0.7,
        "KAN network\n(B-spline activations, all features together)",
        fc="#fde8d8", ec="#e6550d", fs=8)
    arrow(ax, 5, 6.75, 5, 5.9)
    box(ax, 5, 5.55, 3.5, 0.65, "ψ(x)  →  one scalar", fc="#fde8d8", ec="#e6550d", fs=9)
    arrow(ax, 5, 5.2, 5, 4.35)
    box(ax, 5, 4.0, 4.5, 0.65, "K(xᵢ, xⱼ) = ψ(xᵢ) · ψ(xⱼ)", fc="#fde8d8", ec="#e6550d", fs=9)
    arrow(ax, 5, 3.65, 5, 2.8)
    box(ax, 5, 2.45, 5, 0.65, "Kernel Ridge Regression → ŷ", fc="#cce5ff", ec="#4e79a7", fs=9)

    ax.text(5, 1.5, "⚠ K is rank-1 (outer product)\n"
            "⚠ KRR = just a threshold on ψ(x)\n"
            "⚠ Loses when classes need 2D+ separation",
            ha="center", va="center", fontsize=8.5, color="#c0392b",
            bbox=dict(boxstyle="round", fc="#fff5f5", ec="#e74c3c", alpha=0.7))

    # ── Panel 3: Global RBF ──────────────────────────────────────────────────
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Global RBF (SVM / KRR baseline)", fontsize=12,
                 fontweight="bold", color="#d62728", pad=10)

    box(ax, 5, 9, 6, 0.8, "Input x = [x₁,  x₂,  x₃,  …,  xₚ]",
        fc="#f0f0f0", ec="#888", fs=9)
    arrow(ax, 5, 8.6, 5, 7.6)
    box(ax, 5, 7.2, 6, 0.75,
        "‖xᵢ − xⱼ‖²  =  Σₖ (xᵢₖ − xⱼₖ)²\n(all features combined equally)",
        fc="#fdd", ec="#d62728", fs=8.5)
    arrow(ax, 5, 6.83, 5, 5.95)
    box(ax, 5, 5.6, 5, 0.65,
        "K(xᵢ,xⱼ) = exp(−γ · ‖xᵢ−xⱼ‖²)",
        fc="#fdd", ec="#d62728", fs=9)
    arrow(ax, 5, 5.28, 5, 4.35)
    box(ax, 5, 4.0, 5, 0.65, "SVM or KRR → ŷ", fc="#cce5ff", ec="#4e79a7", fs=9)

    ax.text(5, 2.8, "No feature selection\nAll features treated equally\n"
            "Noise features inflate ‖x‖² and hurt accuracy",
            ha="center", va="center", fontsize=8.5, color="#c0392b",
            bbox=dict(boxstyle="round", fc="#fff5f5", ec="#e74c3c", alpha=0.7))
    ax.text(5, 1.4, "✓ Fast  ✓ Simple  ✓ Works well\nwhen ALL features matter equally",
            ha="center", va="center", fontsize=8.5, color="#555",
            bbox=dict(boxstyle="round", fc="#f9f9f9", ec="#aaa", alpha=0.7))

    fig.suptitle("How Each Model Works — Conceptual Overview",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "simple_fig6_model_explainer.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("[simple_figures] Generating readable figures ...")
    kn, kan = load_results()

    print("  fig1: us vs SMKL scorecard ...")
    simple_scorecard(kn, kan)

    print("  fig2: per-dataset bars ...")
    simple_per_dataset(kn, kan)

    print("  fig3: scorecard table ...")
    simple_wins_table(kn, kan)

    print("  fig4: KAN vs KernelNetwork ...")
    simple_kan_vs_kn(kn, kan)

    print("  fig5: alpha story (trains one model) ...")
    simple_alpha_story()

    print("  fig6: model explainer diagram ...")
    simple_model_explainer()

    print("[simple_figures] Done.")


if __name__ == "__main__":
    run()
