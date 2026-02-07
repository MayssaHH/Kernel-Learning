import copy
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch

from data.generators.mixture_of_gaussians import generate_mixture_of_gaussians
from kernel_learning import *
from kernel_learning.methods.KernelRidgeClassifier import KernelRidgeClassifier


def make_gaussian_params(
    num_classes: int,
    dimension: int,
    mean_radius: float = 2.4,
    extra_dim_jitter: float = 0.20,
    covariance_mode: str = "diagonal",  # "diagonal" or "correlated"
    min_var: float = 1.4,
    max_var: float = 2.2,
    variance_scale: float = 3.0,
    min_corr: float = 0.20,
    max_corr: float = 0.45,
):
    """
    Configurable Gaussian parameter generator.
    Use `covariance_mode="diagonal"` for uncorrelated data (recommended default),
    or `covariance_mode="correlated"` for harder correlated data.
    """
    if covariance_mode not in {"diagonal", "correlated"}:
        raise ValueError(f"Unsupported covariance_mode: {covariance_mode}")

    means = []
    covariances = []
    identity = torch.eye(dimension)
    ones = torch.ones(dimension, dimension)

    for cls_idx in range(num_classes):
        mu = torch.zeros(dimension)
        if dimension == 1:
            mu[0] = (cls_idx - (num_classes - 1) / 2.0) * mean_radius
        else:
            angle = 2.0 * math.pi * cls_idx / max(num_classes, 1)
            mu[0] = mean_radius * math.cos(angle)
            mu[1] = mean_radius * math.sin(angle)
            if dimension > 2:
                mu[2:] = extra_dim_jitter * torch.randn(dimension - 2)
        means.append(mu)

        if covariance_mode == "diagonal":
            diag_vars = min_var + (max_var - min_var) * torch.rand(dimension)
            cov = torch.diag(diag_vars)
        else:
            corr = min_corr + (max_corr - min_corr) * torch.rand(1).item()
            base_cov = variance_scale * ((1.0 - corr) * identity + corr * ones)
            q, _ = torch.linalg.qr(torch.randn(dimension, dimension))
            cov = q @ base_cov @ q.T
            cov = 0.5 * (cov + cov.T)

        covariances.append(cov)

    return means, covariances


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


def full_vector_rbf_kernel(X_a: torch.Tensor, X_b: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    sq_dist = torch.cdist(X_a, X_b, p=2) ** 2
    return torch.exp(-gamma * sq_dist)


def full_vector_linear_kernel(X_a: torch.Tensor, X_b: torch.Tensor) -> torch.Tensor:
    return X_a @ X_b.T


def evaluate_full_vector_krr(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    kernel_fn,
    lambda_ridge: float = 1.0,
) -> float:
    classes, y_indices = torch.unique(Y_train, sorted=True, return_inverse=True)
    K_train = kernel_fn(X_train, X_train)
    Y_train_one_hot = torch.nn.functional.one_hot(
        y_indices, num_classes=classes.numel()
    ).to(device=K_train.device, dtype=K_train.dtype)
    eye = torch.eye(K_train.shape[0], device=K_train.device, dtype=K_train.dtype)
    alpha = torch.linalg.solve(K_train + lambda_ridge * eye, Y_train_one_hot)

    K_test_train = kernel_fn(X_test, X_train)
    pred_indices = torch.argmax(K_test_train @ alpha, dim=1)
    pred_labels = classes[pred_indices]
    return pred_labels.eq(Y_test).float().mean().item()


def run_experiement(
    num_classes: int = 3,
    samples_per_class: int = 100,
    dimension: int = 2,
    epochs: int = 1000,
    sub_kernels: list = None,
    class_means: list = None,
    class_covariances: list = None,
    dataset_name: str = "mixture_of_gaussians",
    data_generator: Optional[Callable] = None,
    data_generator_kwargs: Optional[dict] = None,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
):
    run_uuid = run_uuid or str(uuid.uuid4())
    results_dir = Path(results_root) / f"{run_uuid}_modular_experiment"
    if save_results:
        results_dir.mkdir(parents=True, exist_ok=True)
    saved_plot_paths = []

    def finalize_plot(plot_name: str):
        if save_results:
            plot_path = results_dir / f"{plot_name}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            saved_plot_paths.append(str(plot_path))
        if show_plots:
            plt.show()
        plt.close()

    ## generate data
    if data_generator is None:
        X, Y = generate_mixture_of_gaussians(
            number_of_classes=num_classes,
            samples_per_class=samples_per_class,
            dimension=dimension,
            list_of_means=class_means,
            list_of_covariances=class_covariances,
        )
    else:
        generator_kwargs = data_generator_kwargs or {}
        X, Y = data_generator(**generator_kwargs)

    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("data_generator must return (X, Y) as torch.Tensor objects.")
    if X.ndim != 2:
        raise ValueError(f"Expected X to have shape (n, p), got shape {tuple(X.shape)}.")
    if Y.ndim != 1:
        raise ValueError(f"Expected Y to have shape (n,), got shape {tuple(Y.shape)}.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples. Got {X.shape[0]} and {Y.shape[0]}."
        )
    if X.shape[1] != dimension:
        raise ValueError(
            f"Generated data has {X.shape[1]} features, but dimension={dimension}. "
            "Make sure class_means and class_covariances match the same dimension."
        )
    ## visualize data
    if dimension >= 2:
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis")
        plt.title(f"{dataset_name} data (first 2 dimensions)")
        plt.xlabel("X1")
        plt.ylabel("X2")
        finalize_plot("data_scatter_first2dims")

    ## split into train and test
    n = X.shape[0]
    indices = torch.randperm(n)
    train_indices = indices[: int(0.8 * n)]
    test_indices = indices[int(0.8 * n) :]
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    def evaluate_kernel_model_with_krr(kernel_model: KernelNetwork) -> float:
        clf = KernelRidgeClassifier(learnt_kernel=kernel_model, lambda_ridge=1.0)
        clf.fit(X_train, Y_train)
        return evaluate_model(clf, X_test, Y_test)

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
    acc1_list = []
    acc2_list = []
    acc3_list = []
    acc4_list = []
    acc_eval_epochs = []
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
            acc1 = evaluate_kernel_model_with_krr(model1)
            acc2 = evaluate_kernel_model_with_krr(model2)
            acc3 = evaluate_kernel_model_with_krr(model3)
            acc4 = evaluate_kernel_model_with_krr(model4)
            acc1_list.append(acc1)
            acc2_list.append(acc2)
            acc3_list.append(acc3)
            acc4_list.append(acc4)
            acc_eval_epochs.append(epoch)
            print(
                f"Test Accuracies at Epoch {epoch:3d} | "
                f"{architecture_labels[0]}: {acc1:.4f}, "
                f"{architecture_labels[1]}: {acc2:.4f}, "
                f"{architecture_labels[2]}: {acc3:.4f}, "
                f"{architecture_labels[3]}: {acc4:.4f}"
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

    ## true full-vector baselines (not additive across features)
    full_vector_rbf_acc = evaluate_full_vector_krr(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        kernel_fn=lambda Xa, Xb: full_vector_rbf_kernel(Xa, Xb, gamma=0.5),
        lambda_ridge=1.0,
    )
    print(f"Full-Vector RBF KRR Baseline (fixed gamma=0.5) | Accuracy: {full_vector_rbf_acc:.4f}")

    full_vector_linear_acc = evaluate_full_vector_krr(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        kernel_fn=full_vector_linear_kernel,
        lambda_ridge=1.0,
    )
    print(f"Full-Vector Linear KRR Baseline | Accuracy: {full_vector_linear_acc:.4f}")

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
    finalize_plot("loss_curves")
    ## plot accuracies
    plt.figure(figsize=(12, 8))
    plt.plot(acc_eval_epochs, acc1_list, label=architecture_labels[0])
    plt.plot(acc_eval_epochs, acc2_list, label=architecture_labels[1])
    plt.plot(acc_eval_epochs, acc3_list, label=architecture_labels[2])
    plt.plot(acc_eval_epochs, acc4_list, label=architecture_labels[3])
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Curves")
    plt.legend()
    finalize_plot("test_accuracy_curves")
    
    ## grouped bar plot of accuracies (side-by-side, no overlap)
    plt.figure(figsize=(11, 6))
    model_labels = architecture_labels
    trained_scores = [acc1, acc2, acc3, acc4]
    untrained_scores = [acc_untrained1, acc_untrained2, acc_untrained3, acc_untrained4]
    x = torch.arange(len(model_labels), dtype=torch.float32)
    width = 0.35

    plt.bar((x - width / 2).tolist(), trained_scores, width=width, alpha=0.8, label="Trained")
    plt.bar((x + width / 2).tolist(), untrained_scores, width=width, alpha=0.8, label="Untrained")

    baseline_labels = ["Full-Vector RBF", "Full-Vector Linear"]
    baseline_scores = [full_vector_rbf_acc, full_vector_linear_acc]
    x_base = (torch.arange(len(baseline_labels), dtype=torch.float32) + len(model_labels) + 0.7)
    plt.bar(x_base.tolist(), baseline_scores, width=0.45, alpha=0.8, label="Full-Vector Baselines")

    xticks = torch.cat([x, x_base]).tolist()
    plt.xticks(xticks, model_labels + baseline_labels)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracies (Archi Trained/Untrained + Full-Vector Baselines)")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    finalize_plot("final_accuracy_bars")
    ## plot change of alphas, gammas, and sigmas for the trained models
    ## plot change of alphas
    plt.figure(figsize=(12, 8))
    alpha_indices = torch.arange(dimension).tolist()
    plt.scatter(
        alpha_indices,
        model1._get_alphas().detach().cpu().numpy(),
        label=architecture_labels[0],
        alpha=0.85,
    )
    plt.scatter(
        alpha_indices,
        model2._get_alphas().detach().cpu().numpy(),
        label=architecture_labels[1],
        alpha=0.85,
    )
    plt.scatter(
        alpha_indices,
        model3._get_alphas().detach().cpu().numpy(),
        label=architecture_labels[2],
        alpha=0.85,
    )
    plt.scatter(
        alpha_indices,
        model4._get_alphas().detach().cpu().numpy(),
        label=architecture_labels[3],
        alpha=0.85,
    )
    plt.xlabel("Sub-kernel Index (integer)")
    plt.ylabel("Alpha Value")
    plt.title("Learned Alpha Values for Each Sub-kernel")
    if dimension <= 30:
        plt.xticks(alpha_indices)
    plt.legend()
    finalize_plot("final_alphas")

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_name": dataset_name,
            "data_generator": data_generator.__name__ if data_generator is not None else "generate_mixture_of_gaussians",
            "data_generator_kwargs": data_generator_kwargs,
            "num_classes": num_classes,
            "samples_per_class": samples_per_class,
            "dimension": dimension,
            "epochs": epochs,
            "results_root": results_root,
            "save_results": save_results,
            "show_plots": show_plots,
            "class_means": [m.tolist() for m in class_means] if class_means is not None else None,
            "class_covariances": [c.tolist() for c in class_covariances] if class_covariances is not None else None,
        },
        "metrics": {
            "trained_accuracy": {
                architecture_labels[0]: acc1,
                architecture_labels[1]: acc2,
                architecture_labels[2]: acc3,
                architecture_labels[3]: acc4,
            },
            "untrained_accuracy": {
                architecture_labels[0]: acc_untrained1,
                architecture_labels[1]: acc_untrained2,
                architecture_labels[2]: acc_untrained3,
                architecture_labels[3]: acc_untrained4,
            },
            "full_vector_baselines": {
                "Full-Vector RBF (gamma=0.5)": full_vector_rbf_acc,
                "Full-Vector Linear": full_vector_linear_acc,
            },
            "loss_curves": {
                architecture_labels[0]: loss1_list,
                architecture_labels[1]: loss2_list,
                architecture_labels[2]: loss3_list,
                architecture_labels[3]: loss4_list,
            },
            "accuracy_curve_epochs": acc_eval_epochs,
            "accuracy_curves": {
                architecture_labels[0]: acc1_list,
                architecture_labels[1]: acc2_list,
                architecture_labels[2]: acc3_list,
                architecture_labels[3]: acc4_list,
            },
            "final_alphas": {
                architecture_labels[0]: model1._get_alphas().detach().cpu().tolist(),
                architecture_labels[1]: model2._get_alphas().detach().cpu().tolist(),
                architecture_labels[2]: model3._get_alphas().detach().cpu().tolist(),
                architecture_labels[3]: model4._get_alphas().detach().cpu().tolist(),
            },
        },
        "artifacts": {
            "plots": saved_plot_paths,
        },
    }

    if save_results:
        manifest_path = results_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved results to: {results_dir}")

    return manifest


if __name__ == "__main__":
    num_classes = 4
    samples_per_class = 1000
    p = 5
    class_means, class_covariances = make_gaussian_params(
        num_classes=num_classes,
        dimension=p,
        mean_radius=2.4,
        covariance_mode="diagonal",
        min_var=1.4,
        max_var=2.2,
        extra_dim_jitter=0.20,
    )

    run_experiement(
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        dimension=p,
        epochs=1000,
        class_means=class_means,
        class_covariances=class_covariances,
        run_uuid=str(uuid.uuid4()),
        save_results=True,
        show_plots=True,
    )

    
