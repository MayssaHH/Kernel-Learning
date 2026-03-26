from kernel_learning import *
import torch
import numpy as np
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from sklearn.svm import SVC

choice_of_kernel = "linear"
our = "kan"

print("=" * 80)
print("KAN KERNEL NETWORK - EXPERIMENT")
print("=" * 80)

# Set random seed for reproducibility
torch.manual_seed(0)

# Load data
i = 5
data_dir = Path(f"experiments/KAN_KERNEL_EXP/data_{i}")
X_path = data_dir / f"X_{i}_2.txt"
Y_path = data_dir / f"Y_{i}_2.txt"

X_np = np.loadtxt(X_path, dtype=np.float64)
Y_np = np.loadtxt(Y_path, dtype=np.float64)

X = torch.from_numpy(X_np).float()
Y = torch.from_numpy(Y_np.reshape(-1)).int()

p = X.shape[1]
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Number of features (p): {p}")
print(f"KAN architecture: [{p}, {2*p+1}, 1]")

# ============================================================================
# KAN Kernel Model
# ============================================================================

model = KAN_Kernel_NX(
    p=p,
    grid=12,
    hidden_dims=[2 * p + 1, 16, 8, 4],
    k=3,
    psd_epsilon=1e-4,
)

criterion = AlignmentLossWithoutNormalization()
trainer = ManualGradientTrainer(lr=0.001)

# Training loop
epochs = 5000
for epoch in range(epochs):
    loss = trainer.train_epoch(
        model=model,
        X=X,
        criterion=criterion,
        lambda_lasso=0.0,
        y=Y
    )

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

print("=" * 80)

# ============================================================================
# Comparison with sklearn SVM
# ============================================================================

X_np = X.numpy()
Y_np = Y.numpy()

# --- Train both classifiers ---
clf_linear = SVC(kernel=choice_of_kernel)
clf_kan = SVC(kernel=model.inference_kernel)

clf_linear.fit(X_np, Y_np)
clf_kan.fit(X_np, Y_np)

# --- Build mesh grid ---
margin = 0.5
x_min, x_max = X_np[:, 0].min() - margin, X_np[:, 0].max() + margin
y_min, y_max = X_np[:, 1].min() - margin, X_np[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
clfs   = [clf_linear, clf_kan]
titles = [f"{choice_of_kernel.upper()} Kernel", "KAN Kernel"]

for ax, clf, title in zip(axes, clfs, titles):
    Z = clf.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.contour(xx, yy, Z, colors="k", linewidths=0.8)
    ax.scatter(*X_np.T, c=Y_np, cmap="coolwarm", edgecolors="k", s=40, zorder=3)

    acc = clf.score(X_np, Y_np)
    ax.set_title(f"{title}\nTrain acc: {acc:.3f}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle("SVM Decision Boundary Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"experiments/KAN_KERNEL_EXP/our_{our}_vs_{choice_of_kernel}.png", dpi=300)
plt.show()