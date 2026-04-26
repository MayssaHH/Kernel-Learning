"""
KAN_Kernel_NX — EXP-5: Training Convergence Curves

Compares KAN+KRR vs Ours+RBF training dynamics on breastcancer.
Tracks alignment loss and test accuracy every 25 epochs.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier
from datasets import load_uci_split, to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def track(model, X_tr, y_tr, X_te, y_te, epochs=600, lr=3e-3, lam=1e-4, record_every=25, label=""):
    crit = AlignmentLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    model = model.to(DEVICE)
    Xd, yd = X_tr.to(DEVICE), y_tr.to(DEVICE)

    steps, losses, accs = [], [], []

    for ep in range(epochs+1):
        if ep % record_every == 0:
            with torch.no_grad():
                lv = crit(model(Xd), yd).item()
            mc = model.cpu()
            clf = KernelRidgeClassifier(kernel=mc, lambda_ridge=lam)
            clf.fit(X_tr, y_tr)
            acc = float(clf.predict(X_te).eq(y_te).float().mean())
            steps.append(ep); losses.append(lv); accs.append(acc*100)
            model.to(DEVICE)
            if ep % 150 == 0:
                print(f"  {label:12s}  ep={ep:4d}  loss={lv:.5f}  acc={acc*100:.1f}%")
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if ep < epochs:
            opt.zero_grad()
            K = model(Xd)
            loss = crit(K, yd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.cpu()
    return np.array(steps), np.array(losses), np.array(accs)


def run(save_path=None):
    print(f"[KAN EXP-5] Convergence Curves  device={DEVICE}")
    X_tr_np, y_tr_np, X_te_np, y_te_np, info = load_uci_split("breastcancer")
    X_tr, y_tr, X_te, y_te = to_tensors(X_tr_np, y_tr_np, X_te_np, y_te_np)
    p = X_tr.shape[1]
    print(f"  n_train={info['train_size']}  n_test={info['test_size']}  p={p}")

    # KAN+KRR
    print("\n  [KAN+KRR]")
    kan = KAN_Kernel_NX(p=p, hidden_dims=[p], grid=3, k=3)
    s_kan, l_kan, a_kan = track(kan, X_tr, y_tr, X_te, y_te, epochs=600, label="KAN+KRR")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Ours+RBF
    print("\n  [Ours+RBF]")
    subs = [RBFSubKernel(0.5) for _ in range(p)]
    kn = KernelNetwork(subs, alpha_constraint="square", normalize_alphas=True, alpha_init="random")
    s_kn, l_kn, a_kn = track(kn, X_tr, y_tr, X_te, y_te, epochs=600, label="Ours+RBF")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#fafafa")

    ax_l.plot(s_kan, -l_kan, color="#e6550d", lw=2, label="KAN+KRR",  marker="o", ms=2.5)
    ax_l.plot(s_kn,  -l_kn,  color="#2ca02c", lw=2, label="Ours+RBF", marker="s", ms=2.5, ls="--")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Kernel Alignment (−loss)")
    ax_l.set_title("A. Alignment Loss Convergence\n(Breastcancer, n=455, p=30)", fontsize=11, fontweight="bold")
    ax_l.legend(); ax_l.grid(alpha=0.25); ax_l.spines[["top","right"]].set_visible(False)

    ax_a.plot(s_kan, a_kan, color="#e6550d", lw=2, label="KAN+KRR",  marker="o", ms=2.5)
    ax_a.plot(s_kn,  a_kn,  color="#2ca02c", lw=2, label="Ours+RBF", marker="s", ms=2.5, ls="--")
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Test Accuracy (%)")
    ax_a.set_title("B. Test Accuracy During Training\n(KRR evaluated every 25 epochs)", fontsize=11, fontweight="bold")
    ax_a.legend(); ax_a.grid(alpha=0.25); ax_a.spines[["top","right"]].set_visible(False)

    fig.suptitle("EXP-5: Training Convergence — KAN+KRR vs KernelNetwork (Breastcancer)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = save_path or str(FIGURES_DIR / "kan_fig5_convergence.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); print(f"\n  Saved → {out}"); plt.close(fig)


if __name__ == "__main__":
    run()
