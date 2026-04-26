"""
KAN_Kernel_NX — EXP-3: Feature Importance via Input Gradient Sensitivity

The KAN has no explicit α weights. Instead, we compute input gradient sensitivity:
    I_k = (1/n) Σ_i |∂ψ(x_i)/∂x_{i,k}|
This measures how much each feature affects the KAN's scalar output ψ,
averaged over training samples. It is the natural analog of α for the KAN.

Scenarios:
  A. Professor synthetic (known ground truth)
  B. UCI breastcancer (p=30)
  C. UCI spambase (p=57)

Also runs the same scenarios for KernelNetwork (loads α from the existing
alpha_analysis run if available, else re-computes) for direct comparison.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernel_learning.kernel_network.KAN_Kernel import KAN_Kernel_NX
from kernel_learning.kernel_network.network import KernelNetwork
from kernel_learning.sub_kernels.rbf import RBFSubKernel
from kernel_learning.losses.alignment import AlignmentLoss
from datasets import load_uci_split, to_tensors

DEVICE = torch.device("cpu")   # professor + breastcancer small; spambase on GPU
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TYPE_COLORS = {"shared":"#4e79a7","class−1":"#f28e2b","class+1":"#59a14f","noise":"#bab0ac"}


# ── Training helpers ──────────────────────────────────────────────────────────

def train_kan(X_tr, y_tr, p, epochs=500, device=DEVICE):
    model = KAN_Kernel_NX(p=p, hidden_dims=[p], grid=3, k=3).to(device)
    Xt = X_tr.to(device); yt = y_tr.to(device)
    crit = AlignmentLoss(); opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss=crit(model(Xt), yt); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.cpu()


def train_kn(X_tr, y_tr, p, epochs=500, device=DEVICE):
    subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(p)]
    model = KernelNetwork(sub_kernels=subs, alpha_constraint="square",
                          normalize_alphas=True, alpha_init="random").to(device)
    Xt = X_tr.to(device); yt = y_tr.to(device)
    crit = AlignmentLoss(); opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss=crit(model(Xt), yt); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.cpu()


def kan_sensitivity(model, X_tr, device=DEVICE):
    """Compute |∂ψ(x)/∂x_k| averaged over samples."""
    model = model.to(device)
    Xt = X_tr.to(device).requires_grad_(True)
    psi = model._psi(Xt)        # (n,)
    grad = torch.autograd.grad(psi.sum(), Xt)[0]  # (n, p)
    imp = grad.abs().mean(dim=0).detach().cpu().numpy()  # (p,)
    model = model.cpu()
    return imp / (imp.sum() + 1e-8)   # normalise to sum=1 for comparison with α


# ── Scenario A: Professor synthetic ──────────────────────────────────────────

def run_professor(n_runs=6, n=300, p1=4, p2=4, pc=2, pn=8, snr=2.0):
    sys.path.insert(0, str(ROOT / "data" / "generators" / "professor"))
    from monni_simulated import generate_professor_split

    p = pc + p1 + p2 + pn
    kan_imps, kn_alphas = [], []

    for i in range(n_runs):
        split = generate_professor_split(iteration=i, iter_count=n_runs,
                                         n_train=n, n_test=n,
                                         p1=p1, p2=p2, pc=pc, pn=pn,
                                         snr=snr, sigma=1.0)
        Xtr = split["X_train"].float()
        ytr = ((split["y_train"]+1)//2).long()

        kan = train_kan(Xtr, ytr, p=p, epochs=400)
        kan_imps.append(kan_sensitivity(kan, Xtr))

        kn = train_kn(Xtr, ytr, p=p, epochs=500)
        kn_alphas.append(kn._get_alphas().detach().cpu().numpy())

        print(f"    Professor run {i+1}/{n_runs}")

    feat_types = ["shared"]*pc + ["class−1"]*p1 + ["class+1"]*p2 + ["noise"]*pn
    return np.array(kan_imps), np.array(kn_alphas), feat_types


# ── Scenario B/C: UCI ─────────────────────────────────────────────────────────

def run_uci(dataset_name, n_runs=3, epochs=500, large_n=False):
    dev = torch.device("cuda") if (large_n and torch.cuda.is_available()) else DEVICE
    kan_imps, kn_alphas = [], []
    for run in range(n_runs):
        Xtr_np, ytr_np, _, _, _ = load_uci_split(dataset_name, seed=123+run*7)
        Xtr, ytr, _, _ = to_tensors(Xtr_np, ytr_np, Xtr_np, ytr_np)
        p = Xtr.shape[1]

        kan = train_kan(Xtr, ytr, p=p, epochs=epochs, device=dev)
        kan_imps.append(kan_sensitivity(kan, Xtr, device=dev))
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        kn = train_kn(Xtr, ytr, p=p, epochs=epochs, device=dev)
        kn_alphas.append(kn._get_alphas().detach().cpu().numpy())
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"    {dataset_name} run {run+1}/{n_runs}")

    return np.array(kan_imps), np.array(kn_alphas)


# ── Plotting ──────────────────────────────────────────────────────────────────

def bar_pair(ax, x, mean_kan, std_kan, mean_kn, std_kn, colors=None, title=""):
    w = 0.38
    bars_kan = ax.bar(x-w/2, mean_kan, w, yerr=std_kan, color="#e6550d", alpha=0.85,
                      label="KAN sensitivity", capsize=2, edgecolor="white", lw=0.4, zorder=3)
    bars_kn  = ax.bar(x+w/2, mean_kn,  w, yerr=std_kn,  color="#2ca02c", alpha=0.85,
                      label="KN α weight",   capsize=2, edgecolor="white", lw=0.4, zorder=3)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    return bars_kan, bars_kn


def run(save_path=None):
    print("[KAN EXP-3] Feature Importance Analysis")

    print("\n  [A] Professor synthetic ...")
    kan_imp_prof, kn_alp_prof, feat_types = run_professor()

    print("\n  [B] UCI breastcancer ...")
    kan_imp_bc, kn_alp_bc = run_uci("breastcancer", n_runs=3, epochs=500)

    print("\n  [C] UCI spambase ...")
    kan_imp_sp, kn_alp_sp = run_uci("spambase", n_runs=2, epochs=200, large_n=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), facecolor="#fafafa")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax_prof_bar  = fig.add_subplot(gs[0, :])
    ax_prof_heat = fig.add_subplot(gs[1, 0])
    ax_kn_heat   = fig.add_subplot(gs[1, 1])
    ax_bc        = fig.add_subplot(gs[2, 0])
    ax_sp        = fig.add_subplot(gs[2, 1])

    # A: Professor bar
    p_tot = kan_imp_prof.shape[1]
    x = np.arange(p_tot)
    pc, p1, p2, pn = 2, 4, 4, 8
    seps = [pc-0.5, pc+p1-0.5, pc+p1+p2-0.5]
    colors = [TYPE_COLORS[ft] for ft in feat_types]

    bk, bn = bar_pair(ax_prof_bar, x,
                      kan_imp_prof.mean(0), kan_imp_prof.std(0),
                      kn_alp_prof.mean(0),  kn_alp_prof.std(0),
                      title=f"A. Professor Synthetic — Feature Importance (pc={pc},p1={p1},p2={p2},pn={pn})")
    for sx in seps:
        ax_prof_bar.axvline(sx, color="#e15759", lw=1.5, ls="--", alpha=0.7)
    ax_prof_bar.legend(fontsize=9, ncol=2)
    ax_prof_bar.set_xlabel("Feature Index", fontsize=9)
    ax_prof_bar.set_ylabel("Importance (normalised)", fontsize=9)

    # Block labels
    tops = [pc//2-0.5, pc+p1//2-0.5, pc+p1+p2//2-0.5, pc+p1+p2+pn//2-0.5]
    lbls = ["Shared\n(pc=2)","Class-1\n(p1=4)","Class+1\n(p2=4)",f"Noise\n(pn={pn})"]
    y_top = max(kan_imp_prof.mean(0).max(), kn_alp_prof.mean(0).max()) * 1.18
    for mx, lb in zip(tops, lbls):
        ax_prof_bar.text(mx, y_top, lb, ha="center", fontsize=8, color="#555")

    # Heat: KAN sensitivity
    im1 = ax_prof_heat.imshow(kan_imp_prof, aspect="auto", cmap="Oranges", interpolation="nearest")
    plt.colorbar(im1, ax=ax_prof_heat, label="sensitivity", fraction=0.02)
    for sx in seps: ax_prof_heat.axvline(sx, color="#e15759", lw=1.5, ls="--", alpha=0.8)
    ax_prof_heat.set_title("KAN Sensitivity Heatmap (runs)", fontsize=9, fontweight="bold")
    ax_prof_heat.set_xlabel("Feature Index"); ax_prof_heat.set_ylabel("Run")
    ax_prof_heat.set_xticks(x); ax_prof_heat.set_yticks(range(kan_imp_prof.shape[0]))

    # Heat: KN alpha
    im2 = ax_kn_heat.imshow(kn_alp_prof, aspect="auto", cmap="Greens", interpolation="nearest")
    plt.colorbar(im2, ax=ax_kn_heat, label="α value", fraction=0.02)
    for sx in seps: ax_kn_heat.axvline(sx, color="#e15759", lw=1.5, ls="--", alpha=0.8)
    ax_kn_heat.set_title("KernelNetwork α Heatmap (runs)", fontsize=9, fontweight="bold")
    ax_kn_heat.set_xlabel("Feature Index"); ax_kn_heat.set_ylabel("Run")
    ax_kn_heat.set_xticks(x); ax_kn_heat.set_yticks(range(kn_alp_prof.shape[0]))

    # B: Breastcancer
    xbc = np.arange(kan_imp_bc.shape[1])
    bar_pair(ax_bc, xbc, kan_imp_bc.mean(0), kan_imp_bc.std(0),
             kn_alp_bc.mean(0), kn_alp_bc.std(0),
             title="B. Breastcancer (p=30)")
    ax_bc.set_xlabel("Feature Index", fontsize=9)
    ax_bc.set_ylabel("Importance", fontsize=9)
    top5_kan = np.argsort(-kan_imp_bc.mean(0))[:5]
    top5_kn  = np.argsort(-kn_alp_bc.mean(0))[:5]
    ax_bc.legend(fontsize=8)

    # C: Spambase
    xsp = np.arange(kan_imp_sp.shape[1])
    bar_pair(ax_sp, xsp, kan_imp_sp.mean(0), kan_imp_sp.std(0),
             kn_alp_sp.mean(0), kn_alp_sp.std(0),
             title="C. Spambase (p=57)")
    ax_sp.set_xlabel("Feature Index", fontsize=9)
    ax_sp.set_ylabel("Importance", fontsize=9)
    ax_sp.set_xticks(xsp[::5])
    ax_sp.legend(fontsize=8)

    fig.suptitle(
        "EXP-3: Feature Importance — KAN Input Sensitivity vs KernelNetwork α Weights\n"
        "Orange = KAN gradient sensitivity  |  Green = KernelNetwork α",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out = save_path or str(FIGURES_DIR / "kan_fig3_feature_importance.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\n  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
