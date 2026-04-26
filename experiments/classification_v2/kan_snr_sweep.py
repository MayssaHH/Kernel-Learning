"""
KAN_Kernel_NX — EXP-4: Noise Robustness (SNR × pn Phase Diagram)

Same sweep as snr_sweep.py but adds KAN+KRR alongside Ours+RBF, SVM-RBF, KRR-RBF.
Produces combined phase diagrams showing all four models.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernel_learning.kernel_network.KAN_Kernel import KAN_Kernel_NX
from kernel_learning.kernel_network.network import KernelNetwork
from kernel_learning.sub_kernels.rbf import RBFSubKernel
from kernel_learning.losses.alignment import AlignmentLoss
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier
from baselines import KRR_Global, make_svm_rbf, rbf_kernel_global, linear_kernel_global

DEVICE = torch.device("cpu")
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

_gensplit = None

def _load():
    global _gensplit
    sys.path.insert(0, str(ROOT / "data" / "generators" / "professor"))
    from monni_simulated import generate_professor_split
    _gensplit = generate_professor_split


def run_cell(snr, pn, n_runs=4, p1=4, p2=4, pc=2, n=300):
    if _gensplit is None:
        _load()
    p = pc + p1 + p2 + pn
    accs = {"KAN+KRR": [], "Ours+RBF": [], "SVM-RBF": [], "KRR-RBF": []}
    crit = AlignmentLoss()

    for i in range(n_runs):
        split = _gensplit(iteration=i, iter_count=n_runs, n_train=n, n_test=n,
                          p1=p1, p2=p2, pc=pc, pn=pn, snr=snr, sigma=1.0,
                          startseed=57474+pn*1000+int(snr*100))
        Xtr = split["X_train"].float()
        ytr = ((split["y_train"]+1)//2).long()
        Xte = split["X_test"].float()
        yte = ((split["y_test"]+1)//2).long()
        Xtr_np, ytr_np = Xtr.numpy(), split["y_train"].numpy()
        Xte_np, yte_np = Xte.numpy(), split["y_test"].numpy()

        # KAN+KRR
        kan = KAN_Kernel_NX(p=p, hidden_dims=[p], grid=3, k=3).to(DEVICE)
        opt = optim.Adam(kan.parameters(), lr=3e-3)
        for _ in range(300):
            opt.zero_grad(); loss=crit(kan(Xtr.to(DEVICE)), ytr.to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(kan.parameters(),1.0); opt.step()
        kan_cpu = kan.cpu()
        clf = KernelRidgeClassifier(kernel=kan_cpu, lambda_ridge=1e-4)
        clf.fit(Xtr, ytr)
        accs["KAN+KRR"].append(float(clf.predict(Xte).eq(yte).float().mean()))

        # Ours+RBF (KernelNetwork)
        subs = [RBFSubKernel(0.5) for _ in range(p)]
        kn = KernelNetwork(subs, alpha_constraint="square", normalize_alphas=True, alpha_init="random").to(DEVICE)
        opt2 = optim.Adam(kn.parameters(), lr=3e-3)
        for _ in range(300):
            opt2.zero_grad(); loss=crit(kn(Xtr.to(DEVICE)), ytr.to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(kn.parameters(),1.0); opt2.step()
        kn_cpu = kn.cpu()
        clf2 = KernelRidgeClassifier(kernel=kn_cpu, lambda_ridge=1e-4)
        clf2.fit(Xtr, ytr)
        accs["Ours+RBF"].append(float(clf2.predict(Xte).eq(yte).float().mean()))

        # SVM-RBF
        svm = make_svm_rbf(C=10.0, gamma="scale")
        svm.fit(Xtr_np, ytr_np); accs["SVM-RBF"].append(svm.score(Xte_np, yte_np))

        # KRR-RBF
        gm = 1.0/(p*max(Xtr_np.var(),1e-8))
        krr = KRR_Global(rbf_kernel_global(gamma=gm), lambda_ridge=1e-4)
        krr.fit(Xtr_np, ytr_np); accs["KRR-RBF"].append(krr.score(Xte_np, yte_np))

    return {m: float(np.mean(v)) for m, v in accs.items()}


def run_sweep(snr_values=None, pn_values=None, n_runs=4, save_path=None):
    snr_values = snr_values or [0.5, 1.0, 1.5, 2.0, 3.0]
    pn_values  = pn_values  or [0, 2, 5, 10, 20]
    models = ["KAN+KRR", "Ours+RBF", "SVM-RBF", "KRR-RBF"]
    grid = {m: np.zeros((len(snr_values), len(pn_values))) for m in models}

    total = len(snr_values)*len(pn_values); done = 0
    for si, snr in enumerate(snr_values):
        for pi, pn in enumerate(pn_values):
            print(f"  Cell ({done+1}/{total})  SNR={snr}  pn={pn}")
            cell = run_cell(snr, pn, n_runs=n_runs)
            for m in models:
                grid[m][si, pi] = cell[m]
            done += 1

    model_colors = {"KAN+KRR":"#e6550d","Ours+RBF":"#2ca02c","SVM-RBF":"#d62728","KRR-RBF":"#9467bd"}
    colors_snr = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(snr_values)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#fafafa")
    for ax, m in zip(axes.ravel(), models):
        for si, snr in enumerate(snr_values):
            ax.plot(pn_values, grid[m][si,:]*100, marker="o", markersize=5,
                    color=colors_snr[si], lw=1.8, label=f"SNR={snr}")
        ax.set_title(m, fontsize=11, fontweight="bold", color=model_colors[m])
        ax.set_xlabel("# Noise Features (pn)", fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_ylim(40, 100); ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="SNR", loc="lower left")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f%%"))

    fig.suptitle("EXP-4: Noise Robustness — KAN+KRR vs Ours+RBF vs Baselines\n"
                 "Professor Synthetic (p1=4,p2=4,pc=2,n=300)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "kan_fig4_snr_phase_diagram.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); print(f"  Saved → {out}"); plt.close(fig)

    # Advantage map: KAN vs best competitor
    best_comp = np.maximum(grid["Ours+RBF"], np.maximum(grid["SVM-RBF"], grid["KRR-RBF"]))
    adv = (grid["KAN+KRR"] - best_comp) * 100
    fig2, ax2 = plt.subplots(figsize=(8, 5.5), facecolor="#fafafa")
    vmax = max(abs(adv).max(), 2.0)
    im = ax2.imshow(adv, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax, origin="lower",
                    extent=[-0.5,len(pn_values)-0.5,-0.5,len(snr_values)-0.5])
    plt.colorbar(im, ax=ax2, label="KAN+KRR advantage (pp)", fraction=0.025)
    ax2.set_xticks(range(len(pn_values))); ax2.set_xticklabels(pn_values)
    ax2.set_yticks(range(len(snr_values))); ax2.set_yticklabels(snr_values)
    ax2.set_xlabel("# Noise Features (pn)"); ax2.set_ylabel("SNR")
    ax2.set_title("KAN+KRR Advantage over Best Competitor\n(green = KAN wins)", fontsize=11, fontweight="bold")
    for si in range(len(snr_values)):
        for pi in range(len(pn_values)):
            ax2.text(pi, si, f"{adv[si,pi]:+.1f}", ha="center", va="center", fontsize=8,
                     color="black" if abs(adv[si,pi])<vmax*0.6 else "white")
    plt.tight_layout()
    adv_out = str(FIGURES_DIR / "kan_fig4b_advantage_map.png")
    fig2.savefig(adv_out, dpi=160, bbox_inches="tight"); print(f"  Saved → {adv_out}"); plt.close(fig2)
    return grid


if __name__ == "__main__":
    print("[KAN EXP-4] SNR Phase Diagram")
    run_sweep()
