# from kernel_learning.all_components import *
# or even you can do: 
from kernel_learning import *
import torch 

print("=" * 80)
print("MODULAR KERNEL NETWORK - EXAMPLE USAGE Uniform kernels (all linear) ")
print("=" * 80)

# Set random seed for reproducibility
torch.manual_seed(0)

# Problem setup (matching your original code)
n = 100  # number of samples
p = 5    # number of features

# Generate synthetic data
X = torch.randn(n, p)

# Generate true kernel (for demonstration)
true_sigmas = torch.tensor([1.5, 0.8, 1.2, 0.5, 1.0])
true_alphas = torch.ones(p)  # ones for simplicity
K_TRUE = torch.zeros(n, n)
for k in range(p):
    K_TRUE += true_alphas[k] * true_sigmas[k]**2 * torch.outer(X[:, k], X[:, k])
K_TRUE = 0.5 * (K_TRUE + K_TRUE.T)  # ensure symmetry

print(f"\nData: X shape = {X.shape}")
print(f"True kernel: K shape = {K_TRUE.shape}")
print(f"True sigmas: {true_sigmas.tolist()}")


print("\n" + "-" * 80)
print("EXAMPLE 1: Uniform Linear Kernels")
print("-" * 80)

model1 = KernelNetwork.from_uniform_kernels(
    kernel_class=LinearSubKernel,
    num_features=p,
    kernel_params={'initial_sigma': 1.0},
    alpha_constraint='square',
    normalize_alphas=False,
    alpha_init='ones'
)

criterion = MSELoss()
trainer = ManualGradientTrainer(lr=0.01)

print(f"Initial alphas: {model1._get_alphas().detach().tolist()}")

# Training loop
epochs = 100
for epoch in range(epochs):
    loss = trainer.train_epoch(
        model=model1,
        X=X,
        criterion=criterion,
        lambda_lasso=0.0,
        K_true=K_TRUE
    )
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

print(f"Final alphas: {model1._get_alphas().detach().tolist()}")
learned_sigmas = [sk.sigma.item() for sk in model1.sub_kernels]
print(f"Learned sigmas: {learned_sigmas}")
print("=" * 80)
print("Final Coefficients (alphas * sigmas^2):", [a * s**2 for a, s in zip(model1._get_alphas().detach().tolist(), learned_sigmas)])
print("=" * 80)
print("True Coefficients (true_alphas * true_sigmas^2):", [a * s**2 for a, s in zip(true_alphas.tolist(), true_sigmas.tolist())])
