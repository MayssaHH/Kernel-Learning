"""
EXP-4: Robustness to Noise — SNR × pn Phase Diagram

Hypothesis: Our per-feature learnable kernel degrades more gracefully as
noise features are added (increasing pn) and as the signal weakens
(decreasing SNR) compared to global kernel baselines. This is because
the learned α_k weights can suppress noise features, whereas global kernels
treat all features equally.

Experiment design:
  - Sweep SNR ∈ {0.5, 1.0, 1.5, 2.0, 3.0} and pn ∈ {0, 2, 5, 10, 20}
  - Fixed: p1=4, p2=4, pc=2, n_train=300, n_test=300, 5 runs per cell
  - Models: Ours+RBF vs SVM-RBF vs KRR-RBF(global) vs KRR-Linear(global)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    RBFSubKernel,
)
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier

from baselines import (
    KRR_Global,
    make_svm_rbf,
    rbf_kernel_global,
    linear_kernel_global,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_professor():
    sys.path.insert(0, str(ROOT / "data" / "generators" / "professor"))
    from monni_simulated import generate_professor_split
    return generate_professor_split


generate_professor_split = None  # lazy import


def train_kernel_network(X_tr, y_tr, arch="all_rbf", epochs=400, lr=3e-3):
    p = X_tr.shape[1]
    subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(p)]
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


def run_cell(snr: float, pn: int, n_runs: int = 5,
             p1=4, p2=4, pc=2, n=300) -> Dict[str, float]:
    global generate_professor_split
    if generate_professor_split is None:
        generate_professor_split = load_professor()

    results = {m: [] for m in ["Ours+RBF", "SVM-RBF", "KRR-RBF", "KRR-Linear"]}

    for i in range(n_runs):
        split = generate_professor_split(
            iteration=i, iter_count=n_runs,
            n_train=n, n_test=n,
            p1=p1, p2=p2, pc=pc, pn=pn,
            snr=snr, sigma=1.0, startseed=57474 + pn * 1000 + int(snr * 100),
        )
        X_tr = split["X_train"].float()
        y_tr = ((split["y_train"] + 1) // 2).long()
        X_te = split["X_test"].float()
        y_te = ((split["y_test"] + 1) // 2).long()

        X_tr_np = X_tr.numpy()
        y_tr_np = split["y_train"].numpy()  # keep {-1,1} for sklearn
        X_te_np = X_te.numpy()
        y_te_np = split["y_test"].numpy()
        p = X_tr_np.shape[1]

        # Ours+RBF
        model = train_kernel_network(X_tr, y_tr, epochs=400, lr=3e-3)
        clf = KernelRidgeClassifier(kernel=model, lambda_ridge=1e-4)
        clf.fit(X_tr, y_tr)
        acc = float(clf.predict(X_te).eq(y_te).float().mean())
        results["Ours+RBF"].append(acc)

        # SVM-RBF
        svm = make_svm_rbf(C=10.0, gamma="scale")
        svm.fit(X_tr_np, y_tr_np)
        results["SVM-RBF"].append(svm.score(X_te_np, y_te_np))

        # KRR-RBF global
        gamma = 1.0 / (p * max(X_tr_np.var(), 1e-8))
        krr_rbf = KRR_Global(rbf_kernel_global(gamma=gamma), lambda_ridge=1e-4)
        krr_rbf.fit(X_tr_np, y_tr_np)
        results["KRR-RBF"].append(krr_rbf.score(X_te_np, y_te_np))

        # KRR-Linear global
        krr_lin = KRR_Global(linear_kernel_global(), lambda_ridge=1e-4)
        krr_lin.fit(X_tr_np, y_tr_np)
        results["KRR-Linear"].append(krr_lin.score(X_te_np, y_te_np))

    return {m: float(np.mean(v)) for m, v in results.items()}


def run_sweep(
    snr_values: List[float] = None,
    pn_values: List[int] = None,
    n_runs: int = 5,
    save_path: str = None,
):
    snr_values = snr_values or [0.5, 1.0, 1.5, 2.0, 3.0]
    pn_values  = pn_values  or [0, 2, 5, 10, 20]
    models = ["Ours+RBF", "SVM-RBF", "KRR-RBF", "KRR-Linear"]

    # Accumulate results: model -> (n_snr, n_pn) matrix
    grid = {m: np.zeros((len(snr_values), len(pn_values))) for m in models}

    total = len(snr_values) * len(pn_values)
    done = 0
    for si, snr in enumerate(snr_values):
        for pi, pn in enumerate(pn_values):
            print(f"  Cell ({done+1}/{total})  SNR={snr}  pn={pn} ...")
            cell = run_cell(snr, pn, n_runs=n_runs)
            for m in models:
                grid[m][si, pi] = cell[m]
            done += 1

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#fafafa")
    axes = axes.ravel()

    colors_snr = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(snr_values)))
    ours_color = "#2ca02c"
    model_colors = {
        "Ours+RBF":  "#2ca02c",
        "SVM-RBF":   "#d62728",
        "KRR-RBF":   "#9467bd",
        "KRR-Linear":"#1f77b4",
    }

    for ax, m in zip(axes, models):
        for si, snr in enumerate(snr_values):
            ax.plot(pn_values, grid[m][si, :] * 100,
                    marker="o", markersize=5,
                    color=colors_snr[si], linewidth=1.8,
                    label=f"SNR={snr}")
        ax.set_title(m, fontsize=11, fontweight="bold", color=model_colors[m])
        ax.set_xlabel("# Noise Features (pn)", fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_ylim(45, 100)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="SNR", title_fontsize=8, loc="lower left")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f%%"))

    fig.suptitle(
        "EXP-4: Accuracy vs. Noise (SNR × pn Phase Diagram)\n"
        "Professor Synthetic Data — p1=4, p2=4, pc=2, n=300",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "fig4_snr_phase_diagram.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)

    # ── Advantage map: Ours minus best competitor ─────────────────────────────
    best_competitor = np.maximum(grid["SVM-RBF"], np.maximum(grid["KRR-RBF"], grid["KRR-Linear"]))
    advantage = (grid["Ours+RBF"] - best_competitor) * 100

    fig2, ax2 = plt.subplots(figsize=(8, 5.5), facecolor="#fafafa")
    vmax = max(abs(advantage).max(), 2.0)
    im = ax2.imshow(advantage, aspect="auto", cmap="RdYlGn",
                    vmin=-vmax, vmax=vmax, origin="lower",
                    extent=[-0.5, len(pn_values)-0.5, -0.5, len(snr_values)-0.5])
    plt.colorbar(im, ax=ax2, label="Ours+RBF advantage (pp)", fraction=0.025)
    ax2.set_xticks(range(len(pn_values)))
    ax2.set_xticklabels(pn_values)
    ax2.set_yticks(range(len(snr_values)))
    ax2.set_yticklabels(snr_values)
    ax2.set_xlabel("# Noise Features (pn)", fontsize=10)
    ax2.set_ylabel("SNR", fontsize=10)
    ax2.set_title(
        "Ours+RBF Advantage over Best Competitor\n(green = we win, red = competitor wins)",
        fontsize=11, fontweight="bold",
    )
    for si in range(len(snr_values)):
        for pi in range(len(pn_values)):
            ax2.text(pi, si, f"{advantage[si, pi]:+.1f}",
                     ha="center", va="center", fontsize=8,
                     color="black" if abs(advantage[si, pi]) < vmax * 0.6 else "white")
    plt.tight_layout()
    adv_path = str(FIGURES_DIR / "fig4b_advantage_map.png")
    fig2.savefig(adv_path, dpi=160, bbox_inches="tight")
    print(f"  Saved → {adv_path}")
    plt.close(fig2)

    return grid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Reduced grid for quick test")
    args = parser.parse_args()

    snr_vals = [1.0, 2.0, 3.0] if args.fast else [0.5, 1.0, 1.5, 2.0, 3.0]
    pn_vals  = [0, 5, 10]       if args.fast else [0, 2, 5, 10, 20]

    print("[EXP-4] SNR Phase Diagram")
    run_sweep(snr_values=snr_vals, pn_values=pn_vals, n_runs=args.n_runs)
