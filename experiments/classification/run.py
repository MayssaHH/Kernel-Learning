import copy
import torch
import matplotlib.pyplot as plt

from kernel_learning import *
from kernel_learning.methods.KernelRidgeClassifier import KernelRidgeClassifier
from data.generators.mixture_of_gaussians import generate_mixture_of_gaussians


def full_rbf_kernel(X_a: torch.Tensor, X_b: torch.Tensor, gamma: float) -> torch.Tensor:
    sq_dists = torch.cdist(X_a, X_b, p=2) ** 2
    return torch.exp(-gamma * sq_dists)


def full_rbf_krr_accuracy(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    gamma: float = 0.5,
    lambda_ridge: float = 1.0,
) -> float:
    classes, y_indices = torch.unique(y_train, sorted=True, return_inverse=True)
    K_train = full_rbf_kernel(X_train, X_train, gamma=gamma)
    Y_train = torch.nn.functional.one_hot(
        y_indices, num_classes=classes.numel()
    ).to(device=K_train.device, dtype=K_train.dtype)
    eye = torch.eye(K_train.shape[0], device=K_train.device, dtype=K_train.dtype)
    alpha = torch.linalg.solve(K_train + lambda_ridge * eye, Y_train)
    K_test_train = full_rbf_kernel(X_test, X_train, gamma=gamma)
    pred_indices = torch.argmax(K_test_train @ alpha, dim=1)
    pred_labels = classes[pred_indices]
    return pred_labels.eq(y_test).float().mean().item()


p = 2
num_classes = 3

# Use distinct class means; if means are identical, no method can separate classes well.
class_means = [torch.randn(p) * 2 + i * 4 for i in range(num_classes)]
class_covariances = [torch.eye(p)*5 for _ in range(num_classes)]
sub_kernelss=[RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(p)]
## make the kernels RBF LINEAR RBF LINEAR
## so just modify the odd indexed sub-kernels to be linear instead of RBF
for i in range(p):
    if i % 2 == 1:
        sub_kernelss[i] = LinearSubKernel(random=True)
model = KernelNetwork(
    sub_kernels=sub_kernelss,
    alpha_constraint="square",
    normalize_alphas=True,
    alpha_init="random",
)

# Keep exact same initialization for a fair "trained vs untrained" comparison.
untrained_model = copy.deepcopy(model)

initial_alphas = model._get_alphas().detach().cpu().clone()
initial_raw_alphas = model.raw_alphas.detach().cpu().clone()
initial_gammas = [sk.gamma.detach().item() for sk in model.sub_kernels if isinstance(sk, RBFSubKernel)]
initial_sigmas = [sk.sigma.detach().item() for sk in model.sub_kernels if isinstance(sk, LinearSubKernel)]

X, Y = generate_mixture_of_gaussians(
    number_of_classes=num_classes,
    samples_per_class=100,
    dimension=p,
    list_of_means=class_means,
    list_of_covariances=class_covariances,
)
X_test, Y_test = generate_mixture_of_gaussians(
    number_of_classes=num_classes,
    samples_per_class=50,
    dimension=p,
    list_of_means=class_means,
    list_of_covariances=class_covariances,
)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Class means:", [m.tolist() for m in class_means])

criterion = AlignmentLoss()
trainer = ManualGradientTrainer(lr=0.01)

epochs = 10000
for epoch in range(epochs):
    loss = trainer.train_epoch(
        model=model,
        X=X,
        criterion=criterion,
        lambda_lasso=0.0,
        y=Y,
    )
    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

final_alphas = model._get_alphas().detach().cpu().clone()
final_raw_alphas = model.raw_alphas.detach().cpu().clone()
final_sigmas = [sk.sigma.detach().item() for sk in model.sub_kernels if isinstance(sk, LinearSubKernel)]
final_gammas = [sk.gamma.detach().item() for sk in model.sub_kernels if isinstance(sk, RBFSubKernel)]

print("Initial alphas:", initial_alphas.tolist())
print("Final alphas:", final_alphas.tolist())
print("Alpha L1 delta:", torch.sum(torch.abs(final_alphas - initial_alphas)).item())
print("Initial raw alphas:", initial_raw_alphas.tolist())
print("Final raw alphas:", final_raw_alphas.tolist())
print("Initial gammas:", initial_gammas)
print("Final gammas:", final_gammas)
print(
    "Gamma L1 delta:",
    sum(abs(f - i) for f, i in zip(final_gammas, initial_gammas)),
)

# Learned kernel
learned_krr = KernelRidgeClassifier(learnt_kernel=model, lambda_ridge=1.0)
learned_krr.fit(X, Y)
learned_accuracy = learned_krr.predict(X_test).eq(Y_test).float().mean().item()
print(f"Learned-kernel Test Accuracy: {learned_accuracy:.4f}")

# Same architecture, but untrained kernel params (fair control)
untrained_krr = KernelRidgeClassifier(learnt_kernel=untrained_model, lambda_ridge=1.0)
untrained_krr.fit(X, Y)
untrained_accuracy = untrained_krr.predict(X_test).eq(Y_test).float().mean().item()
print(f"Untrained-same-architecture Test Accuracy: {untrained_accuracy:.4f}")

# Same architecture, fixed-gamma additive RBF baseline
additive_fixed_kernel = KernelNetwork(
    sub_kernels=[RBFSubKernel(initial_gamma=0.5, random=False) for _ in range(p)],
    alpha_constraint="square",
    normalize_alphas=True,
    alpha_init="ones",
)
additive_fixed_krr = KernelRidgeClassifier(
    learnt_kernel=additive_fixed_kernel, lambda_ridge=1.0
)
additive_fixed_krr.fit(X, Y)
additive_fixed_accuracy = (
    additive_fixed_krr.predict(X_test).eq(Y_test).float().mean().item()
)
print(f"Fixed-gamma additive-kernel Test Accuracy: {additive_fixed_accuracy:.4f}")

# True p-dimensional RBF baseline (not additive across features)
full_rbf_accuracy = full_rbf_krr_accuracy(
    X_train=X, y_train=Y, X_test=X_test, y_test=Y_test, gamma=0.5, lambda_ridge=1.0
)
print(f"Full p-dimensional RBF Test Accuracy: {full_rbf_accuracy:.4f}")

# Visualize train data (only meaningful for p=2)
if p == 2:
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis")
    plt.title("Mixture of Gaussians - Training Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
