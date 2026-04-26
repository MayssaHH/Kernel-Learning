"""
EXP-2: Decision Boundary Visualization

Hypothesis: Our per-feature learnable kernel learns more adaptive, data-driven
decision boundaries than fixed global kernels, especially on non-linearly
separable synthetic datasets.

Datasets tested:
  - Two Moons (canonical non-linear challenge)
  - Concentric Circles (radially symmetric — favors RBF but tests adaptability)
  - Mixture of Gaussians with noise (heterogeneous features)
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

from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    LinearSubKernel,
    RBFSubKernel,
)
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier

from baselines import KRR_Global, make_svm_rbf, make_svm_linear, rbf_kernel_global

DEVICE = torch.device("cpu")  # small n — CPU is faster due to less overhead
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────

C0, C1 = "#4e79a7", "#f28e2b"
CMAP_BG = ListedColormap([(*[c/255 for c in (200, 220, 240)], 0.6),
                           (*[c/255 for c in (255, 225, 185)], 0.6)])
CMAP_SCATTER = ListedColormap([C0, C1])


# ── Dataset generators ────────────────────────────────────────────────────────

def two_moons(n=200, noise=0.12, seed=42):
    np.random.seed(seed)
    theta = np.linspace(0, np.pi, n // 2)
    x1 = np.c_[np.cos(theta), np.sin(theta)] + np.random.randn(n // 2, 2) * noise
    x2 = np.c_[1 - np.cos(theta), -np.sin(theta) + 0.5] + np.random.randn(n // 2, 2) * noise
    X = np.vstack([x1, x2]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    return X, y


def concentric_circles(n=200, r_inner=0.5, r_outer=1.2, noise=0.06, seed=42):
    np.random.seed(seed)
    angles = np.random.rand(n) * 2 * np.pi
    radii = np.where(np.arange(n) < n // 2, r_inner, r_outer)
    X = np.c_[radii * np.cos(angles), radii * np.sin(angles)].astype(np.float32)
    X += np.random.randn(n, 2).astype(np.float32) * noise
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    return X, y


def gaussian_mix_noisy(n=200, seed=42):
    """Two Gaussians + 2 noise features that carry no label signal."""
    np.random.seed(seed)
    half = n // 2
    X_signal = np.vstack([
        np.random.randn(half, 2) + np.array([-2, 0]),
        np.random.randn(half, 2) + np.array([2, 0]),
    ]).astype(np.float32)
    X_noise = (np.random.randn(n, 2) * 3).astype(np.float32)
    X = np.hstack([X_signal, X_noise])
    y = np.array([0] * half + [1] * half, dtype=np.int64)
    perm = np.random.permutation(n)
    return X[perm], y[perm]


DATASETS = {
    "Two Moons":            (two_moons,),
    "Concentric Circles":   (concentric_circles,),
    "Gaussians + Noise":    (gaussian_mix_noisy,),
}


# ── Train KernelNetwork ───────────────────────────────────────────────────────

def train_kernel_network(X: np.ndarray, y: np.ndarray, arch="all_rbf",
                         epochs=600, lr=3e-3) -> KernelNetwork:
    p = X.shape[1]
    if arch == "all_rbf":
        subs = [RBFSubKernel(initial_gamma=1.0) for _ in range(p)]
    elif arch == "all_linear":
        subs = [LinearSubKernel() for _ in range(p)]
    elif arch == "half_rbf_half_linear":
        n_rbf = (p + 1) // 2
        subs = [RBFSubKernel(initial_gamma=1.0) for _ in range(n_rbf)]
        subs += [LinearSubKernel() for _ in range(p - n_rbf)]
    else:
        raise ValueError(arch)

    model = KernelNetwork(sub_kernels=subs, alpha_constraint="square",
                          normalize_alphas=True, alpha_init="random").to(DEVICE)
    X_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).long().to(DEVICE)
    criterion = AlignmentLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        K = model(X_t)
        loss = criterion(K, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model.cpu()


def kernel_network_predict(model: KernelNetwork, X_train, y_train,
                           X_grid, lambda_ridge=1e-3) -> np.ndarray:
    X_tr_t = torch.from_numpy(X_train).float()
    y_tr_t = torch.from_numpy(y_train.astype(np.int64)).long()
    X_gr_t = torch.from_numpy(X_grid).float()

    clf = KernelRidgeClassifier(kernel=model, lambda_ridge=lambda_ridge)
    clf.fit(X_tr_t, y_tr_t)
    preds = clf.predict(X_gr_t)
    return preds.numpy()


# ── Single-dataset plotting ───────────────────────────────────────────────────

def plot_dataset_boundaries(dataset_name: str, X: np.ndarray, y: np.ndarray,
                             axes_row, lambda_ridge=1e-3):
    margin = 0.4
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_2d = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # For noisy dataset: extend grid with zero noise features
    if X.shape[1] > 2:
        extra = np.zeros((grid_2d.shape[0], X.shape[1] - 2), dtype=np.float32)
        grid_full = np.hstack([grid_2d, extra])
    else:
        grid_full = grid_2d

    configs = [
        ("Ours+RBF",     "#2ca02c", lambda: train_kernel_network(X, y, "all_rbf")),
        ("Ours+Mixed",   "#9467bd", lambda: train_kernel_network(X, y, "half_rbf_half_linear")),
        ("SVM-RBF",      "#d62728", lambda: make_svm_rbf(C=10.0, gamma="scale")),
        ("SVM-Linear",   "#1f77b4", lambda: make_svm_linear(C=1.0)),
        ("KRR-RBF",      "#e377c2", lambda: None),
    ]

    for ax, (label, color, build_fn) in zip(axes_row, configs):
        if label.startswith("Ours"):
            model = build_fn()
            Z = kernel_network_predict(model, X, y, grid_full, lambda_ridge)
        elif label == "KRR-RBF":
            gamma = 1.0 / (X.shape[1] * max(X.var(), 1e-8))
            krr = KRR_Global(rbf_kernel_global(gamma=gamma), lambda_ridge=lambda_ridge)
            krr.fit(X, y)
            Z = krr.predict(grid_full)
        else:
            clf = build_fn()
            clf.fit(X[:, :2], y)
            Z = clf.predict(grid_2d)

        Z = Z.reshape(xx.shape).astype(float)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=CMAP_BG, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, colors=[color], linewidths=1.2, levels=[0.5])
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_SCATTER, edgecolors="k",
                   linewidths=0.5, s=28, zorder=4)

        train_acc = float((Z.ravel()[:len(y)] == y).mean())  # approx
        # Proper train accuracy
        if label.startswith("Ours"):
            X_tr_t = torch.from_numpy(X).float()
            y_tr_t = torch.from_numpy(y.astype(np.int64)).long()
            clf2 = KernelRidgeClassifier(kernel=model, lambda_ridge=lambda_ridge)
            clf2.fit(X_tr_t, y_tr_t)
            preds = clf2.predict(X_tr_t).numpy()
            train_acc = float((preds == y).mean())
        elif label == "KRR-RBF":
            gamma2 = 1.0 / (X.shape[1] * max(X.var(), 1e-8))
            krr2 = KRR_Global(rbf_kernel_global(gamma=gamma2), lambda_ridge=lambda_ridge)
            krr2.fit(X, y)
            train_acc = krr2.score(X, y)
        else:
            if label == "SVM-RBF":
                clf3 = make_svm_rbf(C=10.0, gamma="scale")
                clf3.fit(X[:, :2], y)
                train_acc = clf3.score(X[:, :2], y)
            else:
                clf3 = make_svm_linear(C=1.0)
                clf3.fit(X[:, :2], y)
                train_acc = clf3.score(X[:, :2], y)

        ax.set_title(f"{label}\nTrain acc: {train_acc*100:.1f}%", fontsize=9, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])

    axes_row[0].set_ylabel(dataset_name, fontsize=10, fontweight="bold", labelpad=8)


# ── Main ────────────────────────────────────────────────────────────────��─────

def run(save_path: str = None):
    n_rows = len(DATASETS)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    fig.patch.set_facecolor("#fafafa")

    for row_idx, (ds_name, (gen_fn,)) in enumerate(DATASETS.items()):
        print(f"  Plotting: {ds_name}")
        X, y = gen_fn()
        plot_dataset_boundaries(ds_name, X, y, axes[row_idx] if n_rows > 1 else axes)

    fig.suptitle(
        "Decision Boundary Comparison\n"
        "Our Per-Feature Learnable Kernel vs. Global Kernel Methods",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 1])

    out = save_path or str(FIGURES_DIR / "fig2_decision_boundaries.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("[EXP-2] Decision Boundary Visualization")
    run()
