"""
KAN_Kernel_NX — EXP-2: Decision Boundary Visualization

Same synthetic datasets as decision_boundary.py, comparing:
  KAN+KRR  vs  Ours+RBF  vs  SVM-RBF  vs  SVM-Linear  vs  KRR-RBF
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernel_learning.kernel_network.KAN_Kernel import KAN_Kernel_NX
from kernel_learning.kernel_network.network import KernelNetwork
from kernel_learning.sub_kernels.rbf import RBFSubKernel
from kernel_learning.losses.alignment import AlignmentLoss
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier
from baselines import KRR_Global, make_svm_rbf, make_svm_linear, rbf_kernel_global

DEVICE = torch.device("cpu")
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

C0, C1 = "#4e79a7", "#f28e2b"
CMAP_BG = ListedColormap([(*[c/255 for c in (200,220,240)], 0.6),
                           (*[c/255 for c in (255,225,185)], 0.6)])
CMAP_SC  = ListedColormap([C0, C1])


def two_moons(n=200, noise=0.12, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, np.pi, n//2)
    x1 = np.c_[np.cos(t), np.sin(t)] + np.random.randn(n//2, 2)*noise
    x2 = np.c_[1-np.cos(t), -np.sin(t)+0.5] + np.random.randn(n//2, 2)*noise
    X = np.vstack([x1, x2]).astype(np.float32)
    y = np.array([0]*(n//2)+[1]*(n//2), dtype=np.int64)
    return X, y

def concentric_circles(n=200, r_inner=0.5, r_outer=1.2, noise=0.06, seed=42):
    np.random.seed(seed)
    angles = np.random.rand(n)*2*np.pi
    radii = np.where(np.arange(n)<n//2, r_inner, r_outer)
    X = np.c_[radii*np.cos(angles), radii*np.sin(angles)].astype(np.float32)
    X += (np.random.randn(n,2)*noise).astype(np.float32)
    y = np.array([0]*(n//2)+[1]*(n//2), dtype=np.int64)
    return X, y

def gaussian_mix_noisy(n=200, seed=42):
    np.random.seed(seed)
    half = n//2
    Xs = np.vstack([np.random.randn(half,2)+[-2,0], np.random.randn(half,2)+[2,0]]).astype(np.float32)
    Xn = (np.random.randn(n,2)*3).astype(np.float32)
    X  = np.hstack([Xs, Xn])
    y  = np.array([0]*half+[1]*half, dtype=np.int64)
    perm = np.random.permutation(n); return X[perm], y[perm]

DATASETS = {
    "Two Moons":         two_moons,
    "Concentric Circles":concentric_circles,
    "Gaussians + Noise": gaussian_mix_noisy,
}


def train_kan(X, y, epochs=500):
    p = X.shape[1]
    model = KAN_Kernel_NX(p=p, hidden_dims=[p], grid=3, k=3).to(DEVICE)
    Xt = torch.from_numpy(X).float().to(DEVICE)
    yt = torch.from_numpy(y).long().to(DEVICE)
    crit = AlignmentLoss(); opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss=crit(model(Xt), yt); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.cpu()


def train_kn_rbf(X, y, epochs=500):
    p = X.shape[1]
    subs = [RBFSubKernel(initial_gamma=1.0) for _ in range(p)]
    model = KernelNetwork(sub_kernels=subs, alpha_constraint="square",
                          normalize_alphas=True, alpha_init="random").to(DEVICE)
    Xt = torch.from_numpy(X).float().to(DEVICE)
    yt = torch.from_numpy(y).long().to(DEVICE)
    crit = AlignmentLoss(); opt = optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss=crit(model(Xt), yt); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.cpu()


def predict_kernel_model(model, X_train, y_train, X_grid, lam=1e-3):
    clf = KernelRidgeClassifier(kernel=model, lambda_ridge=lam)
    clf.fit(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    return clf.predict(torch.from_numpy(X_grid).float()).numpy()


def plot_row(ax_row, ds_name, X, y):
    margin = 0.4
    x0, x1 = X[:,0].min()-margin, X[:,0].max()+margin
    y0, y1 = X[:,1].min()-margin, X[:,1].max()+margin
    xx, yy = np.meshgrid(np.linspace(x0,x1,180), np.linspace(y0,y1,180))
    grid2d = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    gridfull = np.hstack([grid2d, np.zeros((len(grid2d), X.shape[1]-2), np.float32)]) if X.shape[1]>2 else grid2d

    configs = [
        ("KAN+KRR",    "#e6550d", lambda: train_kan(X, y)),
        ("Ours+RBF",   "#2ca02c", lambda: train_kn_rbf(X, y)),
        ("SVM-RBF",    "#d62728", None),
        ("SVM-Linear", "#1f77b4", None),
        ("KRR-RBF",    "#9467bd", None),
    ]

    for ax, (label, color, build_fn) in zip(ax_row, configs):
        if label in ("KAN+KRR", "Ours+RBF"):
            model = build_fn()
            Z = predict_kernel_model(model, X, y, gridfull).reshape(xx.shape).astype(float)
            clf2 = KernelRidgeClassifier(kernel=model, lambda_ridge=1e-3)
            clf2.fit(torch.from_numpy(X).float(), torch.from_numpy(y).long())
            tr_acc = float(clf2.predict(torch.from_numpy(X).float()).numpy()==y).mean() if False else \
                     float((clf2.predict(torch.from_numpy(X).float()).numpy()==y).mean())
        elif label == "KRR-RBF":
            gamma = 1.0 / (X.shape[1]*max(X.var(),1e-8))
            krr = KRR_Global(rbf_kernel_global(gamma=gamma), lambda_ridge=1e-3)
            krr.fit(X, y); Z=krr.predict(gridfull).reshape(xx.shape).astype(float)
            tr_acc = krr.score(X, y)
        elif label == "SVM-RBF":
            clf = make_svm_rbf(C=10.0, gamma="scale"); clf.fit(X[:,:2], y)
            Z=clf.predict(grid2d).reshape(xx.shape).astype(float); tr_acc=clf.score(X[:,:2], y)
        else:
            clf = make_svm_linear(C=1.0); clf.fit(X[:,:2], y)
            Z=clf.predict(grid2d).reshape(xx.shape).astype(float); tr_acc=clf.score(X[:,:2], y)

        ax.contourf(xx, yy, Z, alpha=0.35, cmap=CMAP_BG, levels=[-0.5,0.5,1.5])
        ax.contour(xx, yy, Z, colors=[color], linewidths=1.4, levels=[0.5])
        ax.scatter(X[:,0], X[:,1], c=y, cmap=CMAP_SC, edgecolors="k", s=22, lw=0.4, zorder=4)
        ax.set_title(f"{label}\nacc={tr_acc*100:.1f}%", fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    ax_row[0].set_ylabel(ds_name, fontsize=10, fontweight="bold", labelpad=8)


def run(save_path=None):
    print("[KAN EXP-2] Decision Boundary Visualization")
    n_rows = len(DATASETS)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3.5*n_rows), facecolor="#fafafa")

    for ri, (name, fn) in enumerate(DATASETS.items()):
        print(f"  Dataset: {name}")
        X, y = fn()
        plot_row(axes[ri], name, X, y)

    fig.suptitle(
        "KAN_Kernel_NX vs KernelNetwork — Decision Boundaries\n"
        "KAN+KRR  |  Ours+RBF  |  SVM-RBF  |  SVM-Linear  |  KRR-RBF",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = save_path or str(FIGURES_DIR / "kan_fig2_decision_boundaries.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    run()
