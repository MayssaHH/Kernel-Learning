from kernel_learning import *
import torch
import numpy as np
from pathlib import Path

choice_of_kernel = "linear"  
our = "rbf"
class AlignmentLossWithoutNormalization(BaseLoss):
    """
    Kernel alignment loss:
    A(theta) = sum_{i,j} K_pred[i,j] * y[i] * y[j]
    """

    def forward(self, K_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # y: shape (n,)
        # K_pred: shape (n, n)

        # Ensure y is float for multiplication
        y = y.float()

        # Outer product y y^T
        yyT = torch.outer(y, y)  # shape (n, n)

        # Alignment score
        alignment_before_norm = torch.sum(K_pred * yyT)

        # adding normalization term to avoid it is sqrt(frobenius norm of K_pred ) + sqrt(frobenius norm of yyT)
        norm_term = torch.sqrt(torch.sum(K_pred ** 2)) + torch.sqrt(torch.sum(yyT ** 2))
        alignment = alignment_before_norm / norm_term
        # We maximize alignment, so return negative for minimization
        return -alignment



print("=" * 80)
print("MODULAR KERNEL NETWORK - EXAMPLE USAGE Uniform kernels (all linear) ")
print("=" * 80)

# Set random seed for reproducibility
torch.manual_seed(0)

# Problem setup
# n = 200  # number of samples
# p = 10   # number of features

# Load data
i = 5
data_dir = Path(f"experiments/uniform_sub_kernels/data_{i}")
X_path = data_dir / f"X_{i}_2.txt"
Y_path = data_dir / f"Y_{i}_2.txt"

X_np = np.loadtxt(X_path, dtype=np.float64)
Y_np = np.loadtxt(Y_path, dtype=np.float64)

# if X_np.shape != (n, p):
#     raise ValueError(f"Expected X shape {(n, p)}, got {X_np.shape}")
# if Y_np.shape not in {(n,), (n, 1)}:
#     raise ValueError(f"Expected Y shape {(n,)} or {(n, 1)}, got {Y_np.shape}")

X = torch.from_numpy(X_np).float()
Y = torch.from_numpy(Y_np.reshape(-1)).int()  # Ensure Y is shape (n,) and integer type for classification



print(X.shape)
print(Y.shape)



model1 = KernelNetwork.from_uniform_kernels(
    kernel_class=RBFSubKernel,
    num_features=X.shape[1],
    # kernel_params={'initial_gamma': 1.0},
    alpha_constraint='square',
    normalize_alphas=True,
    alpha_init='ones'
)

criterion = AlignmentLossWithoutNormalization()

trainer = ManualGradientTrainer(lr=0.001)

# print(f"Initial alphas: {model1._get_alphas().detach().tolist()}")

# Training loop
epochs = 150
for epoch in range(epochs):
    loss = trainer.train_epoch(
        model=model1,
        X=X,
        criterion=criterion,
        lambda_lasso=0.0,
        y=Y
    )
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

# learned_sigmas = [sk.sigma.item() for sk in model1.sub_kernels] #NOTE: for now I set these sigmas fixed, because it is a linear kernel so we can just keep the alphas to learn
# print(f"Learned sigmas: {learned_sigmas}")
print("=" * 80)
# print alphas only 
print(f"Final alphas: {model1._get_alphas().detach().tolist()}")

def my_kernel(X_1, X_2, alphas, gammas):
    """
    Parameters:
        X_1 : array of shape (n_samples_X, n_features)
        X_2 : array of shape (n_samples_Y, n_features)
    Returns:
        K : array of shape (n_samples_X, n_samples_Y)
    """
    K = np.zeros((X_1.shape[0], X_2.shape[0]))
    for k in range(X_1.shape[1]):
        diff = X_1[:, k:k+1] - X_2[:, k:k+1].T  # shape (n_X, n_Y)
        K += alphas[k] * np.exp(-gammas[k] * diff ** 2)
    return K
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

alphas = model1._get_alphas().detach().numpy()
gammas = [sk.gamma.item() for sk in model1.sub_kernels]  
X_np = X.numpy()
Y_np = Y.numpy()

# --- Train both classifiers ---\

clf_linear = SVC(kernel=choice_of_kernel)
clf_custom = SVC(kernel=partial(my_kernel, alphas=alphas, gammas=gammas)) 

clf_linear.fit(X_np, Y_np)
clf_custom.fit(X_np, Y_np)

# --- Build mesh grid ---
margin = 0.5
x_min, x_max = X_np[:, 0].min() - margin, X_np[:, 0].max() + margin
y_min, y_max = X_np[:, 1].min() - margin, X_np[:, 1].max() + margin
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
clfs   = [clf_linear, clf_custom]
titles = [f"{choice_of_kernel.upper()} Kernel", "Custom Kernel"]

for ax, clf, title in zip(axes, clfs, titles):
    Z = clf.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.contour(xx, yy,  Z, colors="k", linewidths=0.8)
    ax.scatter(*X_np.T, c=Y_np, cmap="coolwarm", edgecolors="k", s=40, zorder=3)

    acc = clf.score(X_np, Y_np)
    ax.set_title(f"{title}\nTrain acc: {acc:.3f}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.suptitle("SVM Decision Boundary Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"experiments/uniform_sub_kernels/our_{our}_vs_{choice_of_kernel}.png", dpi=300)
plt.show()
    