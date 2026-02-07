import torch
from kernel_learning import *
from kernel_learning.methods.KernelRidgeClassifier import KernelRidgeClassifier
from data.generators.mixture_of_gaussians import generate_mixture_of_gaussians
import copy
import matplotlib.pyplot as plt


'''
This file will compare ( all kernels will be used in the same method)

Our architecture on these setups:
1) RBF sub-kernels with learnable gammas (randomly initialized)
2) Linear sub-kernels with learnable sigmas (randomly initialized)
3) Mixed RBF and Linear sub-kernels with learnable gammas and sigmas (randomly initialized)
4) Mixed RBF and Linear and Polynomial sub-kernels with learnable gammas, sigmas, and degrees (randomly initialized)

and 
1) A full RBF kernel ridge regression classifier with a fixed gamma (not learnable) and a fixed regularization parameter (not learnable)
2) A full linear kernel ridge regression classifier with a fixed regularization parameter (not learnable)
'''

def evaluate_model(model, X_test, Y_test):
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        pred_labels = model.predict(X_test)
        acc = pred_labels.eq(Y_test).float().mean().item()
    return acc
def run_experiement(
    num_classes: int = 3,
    samples_per_class: int = 100,
    dimension: int = 2,
    epochs: int = 1000,
    sub_kernels: list = None,
    class_means: list = None,
    class_covariances: list = None,
):
    ## generate data
    X, Y = generate_mixture_of_gaussians(
        number_of_classes=num_classes,
        samples_per_class=samples_per_class,
        dimension=dimension,
        list_of_means=class_means,
        list_of_covariances=class_covariances,
    )
    if X.shape[1] != dimension:
        raise ValueError(
            f"Generated data has {X.shape[1]} features, but dimension={dimension}. "
            "Make sure class_means and class_covariances match the same dimension."
        )
    ## visualize data
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis")
    plt.title("Mixture of Gaussians Data")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    ## split into train and test
    n = X.shape[0]
    indices = torch.randperm(n)
    train_indices = indices[: int(0.8 * n)]
    test_indices = indices[int(0.8 * n) :]
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    ## initialize models
    architecture_labels = [
        "Archi + All RBF",
        "Archi + All Linear",
        "Archi + Alternating RBF/Linear",
        "Archi + Alternating RBF/Linear/Poly",
    ]
    model1 = KernelNetwork(
        sub_kernels=[RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(dimension)],
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )
    model2 = KernelNetwork(
        sub_kernels=[LinearSubKernel(initial_sigma=0.5, random=True) for _ in range(dimension)],
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )
    sub_kernelss=[RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(dimension)]
    ## make the kernels RBF LINEAR RBF LINEAR
    ## so just modify the odd indexed sub-kernels to be linear instead of RBF
    for i in range(dimension):
        if i % 2 == 1:
            sub_kernelss[i] = LinearSubKernel(random=True)
    model3 = KernelNetwork(
        sub_kernels=sub_kernelss,
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )
    ## make the kernels RBF LINEAR POLY RBF LINEAR POLY
    sub_kernelss2=[RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(dimension)]
    for i in range(dimension):
        if i % 3 == 1:
            sub_kernelss2[i] = LinearSubKernel(random=True)
        elif i % 3 == 2:
            sub_kernelss2[i] = PolynomialSubKernel(degree=3, random=True)
    model4 = KernelNetwork(
        sub_kernels=sub_kernelss2,
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )
    ## initialize untrained models for comparison
    untrained_model1 = copy.deepcopy(model1)
    untrained_model2 = copy.deepcopy(model2)
    untrained_model3 = copy.deepcopy(model3)
    untrained_model4 = copy.deepcopy(model4)
    ## train models
    criterion = AlignmentLoss()
    trainer = ManualGradientTrainer(lr=0.01)
    loss1_list = []
    loss2_list = []
    loss3_list = []
    loss4_list = []
    for epoch in range(epochs):
        loss1 = trainer.train_epoch(
            model=model1,
            X=X_train,
            criterion=criterion,
            lambda_lasso=0.0,
            y=Y_train,
        )
        loss2 = trainer.train_epoch(
            model=model2,
            X=X_train,
            criterion=criterion,
            lambda_lasso=0.0,
            y=Y_train,
        )
        loss3 = trainer.train_epoch(
            model=model3,
            X=X_train,
            criterion=criterion,
            lambda_lasso=0.0,
            y=Y_train,
        )
        loss4 = trainer.train_epoch(
            model=model4,
            X=X_train,
            criterion=criterion,
            lambda_lasso=0.0,
            y=Y_train,
        )
        ## save losses for plotting later
        loss1_list.append(loss1)
        loss2_list.append(loss2)
        loss3_list.append(loss3)
        loss4_list.append(loss4)
        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"{architecture_labels[0]}: {loss1:.6f}, "
                f"{architecture_labels[1]}: {loss2:.6f}, "
                f"{architecture_labels[2]}: {loss3:.6f}, "
                f"{architecture_labels[3]}: {loss4:.6f}"
            )
    ## evaluate models
    clf1 = KernelRidgeClassifier(learnt_kernel=model1, lambda_ridge=1.0)
    clf2 = KernelRidgeClassifier(learnt_kernel=model2, lambda_ridge=1.0)
    clf3 = KernelRidgeClassifier(learnt_kernel=model3, lambda_ridge=1.0)
    clf4 = KernelRidgeClassifier(learnt_kernel=model4, lambda_ridge=1.0)
    clf_untrained1 = KernelRidgeClassifier(learnt_kernel=untrained_model1, lambda_ridge=1.0)
    clf_untrained2 = KernelRidgeClassifier(learnt_kernel=untrained_model2, lambda_ridge=1.0)
    clf_untrained3 = KernelRidgeClassifier(learnt_kernel=untrained_model3, lambda_ridge=1.0)
    clf_untrained4 = KernelRidgeClassifier(learnt_kernel=untrained_model4, lambda_ridge=1.0)

    clf1.fit(X_train, Y_train)
    clf2.fit(X_train, Y_train)
    clf3.fit(X_train, Y_train)
    clf4.fit(X_train, Y_train)
    clf_untrained1.fit(X_train, Y_train)
    clf_untrained2.fit(X_train, Y_train)
    clf_untrained3.fit(X_train, Y_train)
    clf_untrained4.fit(X_train, Y_train)

    acc1 = evaluate_model(clf1, X_test, Y_test)
    acc2 = evaluate_model(clf2, X_test, Y_test)
    acc3 = evaluate_model(clf3, X_test, Y_test)
    acc4 = evaluate_model(clf4, X_test, Y_test)
    acc_untrained1 = evaluate_model(clf_untrained1, X_test, Y_test)
    acc_untrained2 = evaluate_model(clf_untrained2, X_test, Y_test)
    acc_untrained3 = evaluate_model(clf_untrained3, X_test, Y_test)
    acc_untrained4 = evaluate_model(clf_untrained4, X_test, Y_test)
    print(
        f"{architecture_labels[0]} | Trained: {acc1:.4f}, Untrained: {acc_untrained1:.4f}"
    )
    print(
        f"{architecture_labels[1]} | Trained: {acc2:.4f}, Untrained: {acc_untrained2:.4f}"
    )
    print(
        f"{architecture_labels[2]} | Trained: {acc3:.4f}, Untrained: {acc_untrained3:.4f}"
    )
    print(
        f"{architecture_labels[3]} | Trained: {acc4:.4f}, Untrained: {acc_untrained4:.4f}"
    )

    ## now lets do krr with fixed (non-learned) additive kernels as baselines
    fixed_rbf_kernel = KernelNetwork(
        sub_kernels=[RBFSubKernel(initial_gamma=0.5, random=False) for _ in range(dimension)],
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="ones",
    )
    fixed_rbf_krr = KernelRidgeClassifier(learnt_kernel=fixed_rbf_kernel, lambda_ridge=1.0)
    fixed_rbf_krr.fit(X_train, Y_train)
    fixed_rbf_acc = evaluate_model(fixed_rbf_krr, X_test, Y_test)
    print(f"Fixed Additive RBF Baseline | Accuracy: {fixed_rbf_acc:.4f}")

    fixed_linear_kernel = KernelNetwork(
        sub_kernels=[LinearSubKernel(initial_sigma=0.5, random=False) for _ in range(dimension)],
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="ones",
    )
    fixed_linear_krr = KernelRidgeClassifier(learnt_kernel=fixed_linear_kernel, lambda_ridge=1.0)
    fixed_linear_krr.fit(X_train, Y_train)
    fixed_linear_acc = evaluate_model(fixed_linear_krr, X_test, Y_test)
    print(f"Fixed Additive Linear Baseline | Accuracy: {fixed_linear_acc:.4f}")

    ## plot losses
    plt.figure(figsize=(12, 8))
    plt.plot(loss1_list, label=architecture_labels[0])
    plt.plot(loss2_list, label=architecture_labels[1])
    plt.plot(loss3_list, label=architecture_labels[2])
    plt.plot(loss4_list, label=architecture_labels[3])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.show()
    ## grouped bar plot of accuracies (side-by-side, no overlap)
    plt.figure(figsize=(11, 6))
    model_labels = architecture_labels
    trained_scores = [acc1, acc2, acc3, acc4]
    untrained_scores = [acc_untrained1, acc_untrained2, acc_untrained3, acc_untrained4]
    x = torch.arange(len(model_labels), dtype=torch.float32)
    width = 0.35

    plt.bar((x - width / 2).tolist(), trained_scores, width=width, alpha=0.8, label="Trained")
    plt.bar((x + width / 2).tolist(), untrained_scores, width=width, alpha=0.8, label="Untrained")

    baseline_labels = ["Fixed Additive RBF", "Fixed Additive Linear"]
    baseline_scores = [fixed_rbf_acc, fixed_linear_acc]
    x_base = (torch.arange(len(baseline_labels), dtype=torch.float32) + len(model_labels) + 0.7)
    plt.bar(x_base.tolist(), baseline_scores, width=0.45, alpha=0.8, label="Fixed Baselines")

    xticks = torch.cat([x, x_base]).tolist()
    plt.xticks(xticks, model_labels + baseline_labels)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracies (Trained vs Untrained + Fixed Baselines)")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
    ## plot change of alphas, gammas, and sigmas for the trained models
    ## plot change of alphas
    plt.figure(figsize=(12, 8))
    plt.plot(model1._get_alphas().detach().cpu().numpy(), label=architecture_labels[0])
    plt.plot(model2._get_alphas().detach().cpu().numpy(), label=architecture_labels[1])
    plt.plot(model3._get_alphas().detach().cpu().numpy(), label=architecture_labels[2])
    plt.plot(model4._get_alphas().detach().cpu().numpy(), label=architecture_labels[3])
    plt.xlabel("Sub-kernel Index")
    plt.ylabel("Alpha Value")
    plt.title("Learned Alpha Values for Each Sub-kernel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_classes = 3
    samples_per_class = 1000
    p = 2
    class_means = [torch.randn(p) * 2 + i * 4 for i in range(num_classes)]
    class_covariances = [torch.eye(p) * 5 for _ in range(num_classes)]

    run_experiement(
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        dimension=p,
        epochs=1000,
        class_means=class_means,
        class_covariances=class_covariances,
    )

    
