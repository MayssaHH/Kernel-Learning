"""
EXP-3: Alpha Weight Analysis — Feature Selection by the Model

Hypothesis: The learned α_k weights correctly concentrate on informative
features and suppress pure noise, demonstrating that the per-feature kernel
decomposition performs implicit feature selection. On the professor synthetic
data, the known ground-truth structure lets us verify this precisely:
  - α_k → high for class-specific (p1, p2) and shared (pc) features
  - α_k → 0  for pure noise (pn) features

Three scenarios:
  A. Professor data: known feature blocks
  B. UCI breastcancer: 30 features, alpha profile across runs
  C. UCI spambase: 57 features, sparse alpha profile
"""

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    LinearSubKernel,
    RBFSubKernel,
)
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier

from datasets import load_uci_split, to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── Training helper ────────────────────────────��──────────────────────────────

def build_and_train(X_tr: torch.Tensor, y_tr: torch.Tensor,
                    arch="all_rbf", epochs=800, lr=3e-3) -> KernelNetwork:
    p = X_tr.shape[1]
    if arch == "all_rbf":
        subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(p)]
    elif arch == "all_linear":
        subs = [LinearSubKernel() for _ in range(p)]
    else:
        n_rbf = (p + 1) // 2
        subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(n_rbf)]
        subs += [LinearSubKernel() for _ in range(p - n_rbf)]

    model = KernelNetwork(sub_kernels=subs, alpha_constraint="square",
                          normalize_alphas=True, alpha_init="random").to(DEVICE)
    X_t = X_tr.to(DEVICE)
    y_t = y_tr.to(DEVICE)
    criterion = AlignmentLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.cpu()


# ── Scenario A: Professor data ────────────────────────────────────────────────

def run_professor_scenario(n_runs=8, n=300, p1=4, p2=4, pc=2, pn=8, snr=2.0):
    """
    Train on professor synthetic data. Return alpha arrays and feature-type labels.
    """
    sys.path.insert(0, str(ROOT / "data" / "generators" / "professor"))
    from monni_simulated import generate_professor_split

    all_alphas = []
    all_accs = []
    p = p1 + p2 + pc + pn

    for i in range(n_runs):
        split = generate_professor_split(
            iteration=i, iter_count=n_runs,
            n_train=n, n_test=n,
            p1=p1, p2=p2, pc=pc, pn=pn,
            snr=snr, sigma=1.0,
        )
        X_tr = split["X_train"].float()
        y_tr = ((split["y_train"] + 1) // 2).long()   # {-1,1} → {0,1}
        X_te = split["X_test"].float()
        y_te = ((split["y_test"] + 1) // 2).long()

        model = build_and_train(X_tr, y_tr, arch="all_rbf", epochs=600, lr=3e-3)
        alphas = model._get_alphas().detach().cpu().numpy()
        all_alphas.append(alphas)

        clf = KernelRidgeClassifier(kernel=model, lambda_ridge=1e-4)
        clf.fit(X_tr, y_tr)
        acc = float(clf.predict(X_te).eq(y_te).float().mean())
        all_accs.append(acc)
        print(f"    Prof run {i+1}/{n_runs}  acc={acc*100:.1f}%")

    alpha_matrix = np.array(all_alphas)
    # Feature type labels: pc shared | p1 class-1 | p2 class+1 | pn noise
    feature_types = (
        ["shared"] * pc +
        ["class−1"] * p1 +
        ["class+1"] * p2 +
        ["noise"] * pn
    )
    return alpha_matrix, feature_types, all_accs


# ── Scenario B/C: UCI datasets ──────────────────────────────��─────────────────

def run_uci_alpha(dataset_name: str, n_runs=5, epochs=800) -> np.ndarray:
    all_alphas = []
    for run in range(n_runs):
        X_tr_np, y_tr_np, X_te_np, y_te_np, info = load_uci_split(dataset_name, seed=123 + run * 7)
        X_tr, y_tr, X_te, y_te = to_tensors(X_tr_np, y_tr_np, X_te_np, y_te_np)
        model = build_and_train(X_tr, y_tr, arch="all_rbf", epochs=epochs, lr=3e-3)
        all_alphas.append(model._get_alphas().detach().cpu().numpy())
        print(f"    {dataset_name} run {run+1}/{n_runs} done")
    return np.array(all_alphas)


# ── Plots ─────────────────────────���──────────────────────────────���────────────

TYPE_COLORS = {
    "shared":  "#4e79a7",
    "class−1": "#f28e2b",
    "class+1": "#59a14f",
    "noise":   "#bab0ac",
}

def plot_professor_alphas(alpha_matrix, feature_types, accs, ax_bar, ax_heat):
    p = alpha_matrix.shape[1]
    mean_alpha = alpha_matrix.mean(axis=0)
    std_alpha = alpha_matrix.std(axis=0)
    x = np.arange(p)
    colors = [TYPE_COLORS[ft] for ft in feature_types]

    ax_bar.bar(x, mean_alpha, yerr=std_alpha, color=colors, edgecolor="white",
               linewidth=0.5, capsize=2.5, alpha=0.92, zorder=3)
    ax_bar.set_xlabel("Feature Index", fontsize=9)
    ax_bar.set_ylabel("Mean α (±1σ)", fontsize=9)
    ax_bar.set_title(f"Alpha Profile — Professor Synthetic Data\n"
                     f"Mean acc={np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%",
                     fontsize=10, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.grid(axis="y", alpha=0.25, zorder=0)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l) for l, c in TYPE_COLORS.items()]
    ax_bar.legend(handles=legend_handles, fontsize=8, loc="upper right",
                  framealpha=0.9, ncol=2)

    # Heatmap
    im = ax_heat.imshow(alpha_matrix, aspect="auto", cmap="viridis",
                        interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, label="α", fraction=0.02, pad=0.02)
    ax_heat.set_xlabel("Feature Index", fontsize=9)
    ax_heat.set_ylabel("Run", fontsize=9)
    ax_heat.set_title("Alpha Heatmap Across Runs (Professor)", fontsize=10, fontweight="bold")
    ax_heat.set_xticks(x)

    # Draw vertical separators for feature blocks (positions determined by feature_types)
    pass  # separators drawn in the main run() function where p1/p2/pc/pn are known


def plot_uci_alphas(alpha_matrix, dataset_name, ax):
    p = alpha_matrix.shape[1]
    mean_alpha = alpha_matrix.mean(axis=0)
    std_alpha = alpha_matrix.std(axis=0)
    x = np.arange(p)
    top5 = np.argsort(-mean_alpha)[:5]

    bars = ax.bar(x, mean_alpha, yerr=std_alpha, color="#4e79a7", edgecolor="white",
                  linewidth=0.4, capsize=2, alpha=0.85, zorder=3)
    for i in top5:
        bars[i].set_color("#f28e2b")
        bars[i].set_alpha(1.0)
    ax.set_xlabel("Feature Index", fontsize=9)
    ax.set_ylabel("Mean α (±1σ)", fontsize=9)
    ax.set_title(f"{dataset_name.title()} — Alpha Profile", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_xticks(x if p <= 30 else x[::5])


# ── Main ─────────────────────────────────────────────────────────��────────────

def run(save_path: str = None):
    print("[EXP-3] Alpha Weight Analysis")

    # ── A: Professor data ────────────────────���────────────────────────────────
    print("\n  [A] Professor synthetic data ...")
    p1, p2, pc, pn = 4, 4, 2, 8
    alpha_prof, feat_types, prof_accs = run_professor_scenario(
        n_runs=8, n=300, p1=p1, p2=p2, pc=pc, pn=pn, snr=2.0
    )

    # ── B: UCI breastcancer ───────────────────────────────────���───────────────
    print("\n  [B] UCI breastcancer ...")
    alpha_bc = run_uci_alpha("breastcancer", n_runs=5, epochs=800)

    # ── C: UCI spambase ───────────────────────��───────────────────────────────
    print("\n  [C] UCI spambase ...")
    alpha_spam = run_uci_alpha("spambase", n_runs=3, epochs=600)

    # ── Figure ──────────────────────��──────────────────────────────��─────────
    fig = plt.figure(figsize=(16, 14), facecolor="#fafafa")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax_bar  = fig.add_subplot(gs[0, :])
    ax_heat = fig.add_subplot(gs[1, :])
    ax_bc   = fig.add_subplot(gs[2, 0])
    ax_spam = fig.add_subplot(gs[2, 1])

    # Recompute block separators
    p_shared = pc
    seps = [pc - 0.5, pc + p1 - 0.5, pc + p1 + p2 - 0.5]
    p_total = pc + p1 + p2 + pn
    x_all = np.arange(p_total)
    colors_bar = [TYPE_COLORS[ft] for ft in feat_types]
    mean_a = alpha_prof.mean(axis=0)
    std_a = alpha_prof.std(axis=0)

    ax_bar.bar(x_all, mean_a, yerr=std_a, color=colors_bar, edgecolor="white",
               linewidth=0.5, capsize=2.5, alpha=0.92, zorder=3)
    for sx in seps:
        ax_bar.axvline(sx, color="#e15759", linewidth=1.5, linestyle="--", alpha=0.7)
    ax_bar.set_xlabel("Feature Index", fontsize=10)
    ax_bar.set_ylabel("Mean α (±1σ)", fontsize=10)
    mean_acc_p = np.mean(prof_accs) * 100
    std_acc_p  = np.std(prof_accs) * 100
    ax_bar.set_title(
        f"A. Alpha Profile — Professor Synthetic Data  "
        f"(p1={p1}, p2={p2}, pc={pc}, pn={pn}, SNR=2.0)\n"
        f"Mean test accuracy: {mean_acc_p:.1f}% ± {std_acc_p:.1f}%",
        fontsize=11, fontweight="bold",
    )
    ax_bar.grid(axis="y", alpha=0.25, zorder=0)
    ax_bar.set_xticks(x_all)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=l.replace("−", "-"))
                      for l, c in TYPE_COLORS.items()]
    ax_bar.legend(handles=legend_handles, fontsize=9, loc="upper right", ncol=2)

    # Block labels
    midpoints = [
        pc // 2 - 0.5,
        pc + p1 // 2 - 0.5,
        pc + p1 + p2 // 2 - 0.5,
        pc + p1 + p2 + pn // 2 - 0.5,
    ]
    labels_b = ["Shared\n(pc=2)", "Class-1\n(p1=4)", "Class+1\n(p2=4)", f"Noise\n(pn={pn})"]
    y_top = mean_a.max() * 1.15
    for mx, lb in zip(midpoints, labels_b):
        ax_bar.text(mx, y_top, lb, ha="center", fontsize=8, color="#555")

    # Heatmap
    im = ax_heat.imshow(alpha_prof, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
    plt.colorbar(im, ax=ax_heat, label="α value", fraction=0.015, pad=0.01)
    for sx in seps:
        ax_heat.axvline(sx, color="#e15759", linewidth=1.5, linestyle="--", alpha=0.8)
    ax_heat.set_xlabel("Feature Index", fontsize=10)
    ax_heat.set_ylabel("Run index", fontsize=10)
    ax_heat.set_title("Alpha Heatmap Across Runs — Professor Data", fontsize=11, fontweight="bold")
    ax_heat.set_xticks(x_all)
    ax_heat.set_yticks(range(len(prof_accs)))

    plot_uci_alphas(alpha_bc, "breastcancer (p=30)", ax_bc)
    plot_uci_alphas(alpha_spam, "spambase (p=57)", ax_spam)

    fig.suptitle(
        "EXP-3: Learned α Weights — Implicit Feature Selection",
        fontsize=14, fontweight="bold", y=1.01,
    )

    out = save_path or str(FIGURES_DIR / "fig3_alpha_profiles.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\n  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
