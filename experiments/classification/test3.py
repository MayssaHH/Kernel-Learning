import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from data.generators.two_moons import generate_two_moons
from experiments.classification.modular_experiment import (
    evaluate_full_vector_krr,
    evaluate_model,
    full_vector_linear_kernel,
    full_vector_rbf_kernel,
)
from kernel_learning import AlignmentLoss, KernelNetwork, ManualGradientTrainer, RBFSubKernel
from kernel_learning.methods.KernelRidgeClassifier import KernelRidgeClassifier


"""
Test 3: Feature-selection stress test with useless features.

Core idea:
1) Start from a signal dataset (default: two moons).
2) Add m useless Gaussian features (independent from labels).
3) Re-train/evaluate for each m in noise_feature_counts.

Compared models:
- Archi + All RBF (our additive architecture with per-feature alpha weights)
- Full-Vector RBF KRR baseline
- Full-Vector Linear KRR baseline

What we inspect:
- Accuracy vs number of useless features
- Whether alpha mass stays concentrated on the true signal dimensions
"""


def append_useless_features(
    X_signal: torch.Tensor,
    num_useless_features: int,
    useless_feature_std: float = 1.0,
) -> torch.Tensor:
    if num_useless_features <= 0:
        return X_signal
    noise = useless_feature_std * torch.randn(X_signal.shape[0], num_useless_features)
    return torch.cat([X_signal, noise], dim=1)


def expand_signal_features(
    X_signal: torch.Tensor,
    target_num_signal_features: int,
    mode: str = "mixed",
    expansion_noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Expand informative signal dimensions when target_num_signal_features > current dims.

    The extra features are deterministic transforms of the original signal (plus tiny jitter),
    so they still carry label-relevant structure instead of being useless noise features.
    """
    n, p = X_signal.shape
    if target_num_signal_features <= p:
        return X_signal[:, :target_num_signal_features]

    if mode not in {"mixed", "projection"}:
        raise ValueError(f"Unsupported signal_expansion_mode: {mode}")

    extra_count = target_num_signal_features - p
    generated_features = []

    for idx in range(extra_count):
        w = torch.randn(p, device=X_signal.device, dtype=X_signal.dtype)
        w = w / (torch.norm(w) + 1e-8)
        proj = X_signal @ w  # (n,)

        if mode == "projection":
            feat = proj
        else:
            pattern = idx % 5
            if pattern == 0:
                feat = proj
            elif pattern == 1:
                feat = proj ** 2
            elif pattern == 2:
                feat = torch.sin(proj)
            elif pattern == 3:
                feat = torch.cos(proj)
            else:
                feat = torch.tanh(proj)

        if expansion_noise_std > 0:
            feat = feat + expansion_noise_std * torch.randn_like(feat)

        generated_features.append(feat)

    X_extra = torch.stack(generated_features, dim=1)  # (n, extra_count)
    return torch.cat([X_signal, X_extra], dim=1)


def evaluate_kernel_model_with_krr(
    kernel_model: KernelNetwork,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    lambda_ridge: float = 1.0,
) -> float:
    clf = KernelRidgeClassifier(learnt_kernel=kernel_model, lambda_ridge=lambda_ridge)
    clf.fit(X_train, Y_train)
    return evaluate_model(clf, X_test, Y_test)


def run_experiement(
    noise_feature_counts: list = None,
    dataset_name: str = "two_moons",
    signal_generator: Callable = generate_two_moons,
    signal_generator_kwargs: Optional[dict] = None,
    num_signal_features: Optional[int] = None,
    signal_expansion_mode: str = "mixed",
    signal_expansion_noise_std: float = 0.01,
    allow_signal_expansion: bool = True,
    samples_per_class: int = 300,
    two_moons_noise_std: float = 0.10,
    useless_feature_std: float = 1.0,
    epochs: int = 500,
    lr: float = 0.01,
    lambda_ridge: float = 1.0,
    full_vector_rbf_gamma: float = 0.5,
    seed: int = 42,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    noise_feature_counts = noise_feature_counts or [0, 2, 5, 10, 20, 50]
    run_uuid = run_uuid or str(uuid.uuid4())

    results_dir = Path(results_root) / f"{run_uuid}_test3_useless_features"
    if save_results:
        results_dir.mkdir(parents=True, exist_ok=True)
    saved_plot_paths = []
    deferred_plot_figures = []

    def finalize_plot(plot_name: str):
        fig = plt.gcf()
        if save_results:
            plot_path = results_dir / f"{plot_name}.png"
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200)
            saved_plot_paths.append(str(plot_path))
        if show_plots and show_plots_at_end:
            deferred_plot_figures.append(fig)
        elif show_plots and not show_plots_at_end:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)

    torch.manual_seed(seed)

    # Build default signal-generator kwargs to preserve current two-moons behavior.
    if signal_generator_kwargs is None:
        signal_generator_kwargs = {
            "samples_per_class": samples_per_class,
            "noise_std": two_moons_noise_std,
            "radius": 1.0,
            "shuffle": True,
        }

    # Generate one shared signal dataset so only the added useless features change across runs.
    X_signal, Y = signal_generator(**signal_generator_kwargs)

    if not isinstance(X_signal, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("signal_generator must return (X, Y) as torch.Tensor objects.")
    if X_signal.ndim != 2:
        raise ValueError(
            f"Expected signal data X to have shape (n, p), got {tuple(X_signal.shape)}."
        )
    if Y.ndim != 1:
        raise ValueError(f"Expected labels Y to have shape (n,), got {tuple(Y.shape)}.")
    if X_signal.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples. Got {X_signal.shape[0]} and {Y.shape[0]}."
        )

    original_signal_features = int(X_signal.shape[1])
    if num_signal_features is None:
        num_signal_features = original_signal_features
    elif num_signal_features < 1:
        raise ValueError(
            f"num_signal_features must be >= 1. Got {num_signal_features}."
        )

    if num_signal_features > original_signal_features:
        if allow_signal_expansion:
            X_signal = expand_signal_features(
                X_signal=X_signal,
                target_num_signal_features=num_signal_features,
                mode=signal_expansion_mode,
                expansion_noise_std=signal_expansion_noise_std,
            )
        else:
            warnings.warn(
                "num_signal_features is larger than the number of columns returned by "
                f"signal_generator ({original_signal_features}). "
                f"Clamping num_signal_features from {num_signal_features} to "
                f"{original_signal_features} because allow_signal_expansion=False.",
                stacklevel=2,
            )
            num_signal_features = original_signal_features
            X_signal = X_signal[:, :num_signal_features]
    else:
        # Keep only the first num_signal_features as true signal dimensions.
        X_signal = X_signal[:, :num_signal_features]

    # Shared split for all m values (fair comparison across noise levels).
    n = X_signal.shape[0]
    indices = torch.randperm(n)
    train_indices = indices[: int(0.8 * n)]
    test_indices = indices[int(0.8 * n) :]

    # Visualize signal dimensions.
    if num_signal_features >= 2:
        plt.figure(figsize=(7, 6))
        plt.scatter(X_signal[:, 0], X_signal[:, 1], c=Y, cmap="viridis", s=18, alpha=0.85)
        plt.title(f"Signal Dataset: {dataset_name} (first 2 signal dims)")
        plt.xlabel("Signal Feature 0")
        plt.ylabel("Signal Feature 1")
        finalize_plot("signal_data_first2dims")
    else:
        plt.figure(figsize=(8, 4))
        x_values = X_signal[:, 0]
        y_dummy = torch.zeros_like(x_values)
        plt.scatter(x_values, y_dummy, c=Y, cmap="viridis", s=18, alpha=0.85)
        plt.title(f"Signal Dataset: {dataset_name} (1D signal)")
        plt.xlabel("Signal Feature 0")
        plt.yticks([])
        finalize_plot("signal_data_1d")

    criterion = AlignmentLoss()

    archi_acc_list = []
    full_rbf_acc_list = []
    full_linear_acc_list = []
    signal_mass_list = []
    noise_mass_list = []
    run_rows = []

    for m in noise_feature_counts:
        X_aug = append_useless_features(
            X_signal=X_signal,
            num_useless_features=int(m),
            useless_feature_std=useless_feature_std,
        )
        dimension = X_aug.shape[1]

        X_train = X_aug[train_indices]
        Y_train = Y[train_indices]
        X_test = X_aug[test_indices]
        Y_test = Y[test_indices]

        # Model under test: our architecture with one RBF sub-kernel per feature.
        model_archi_rbf = KernelNetwork(
            sub_kernels=[
                RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(dimension)
            ],
            alpha_constraint="square",
            normalize_alphas=True,
            alpha_init="random",
        )
        trainer = ManualGradientTrainer(lr=lr)

        for epoch in range(epochs):
            loss = trainer.train_epoch(
                model=model_archi_rbf,
                X=X_train,
                criterion=criterion,
                lambda_lasso=0.0,
                y=Y_train,
            )
            if epoch % 50 == 0:
                print(
                    f"[m={m:>3}] Epoch {epoch:>3} | "
                    f"Archi + All RBF alignment loss: {loss:.6f}"
                )

        archi_acc = evaluate_kernel_model_with_krr(
            kernel_model=model_archi_rbf,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            lambda_ridge=lambda_ridge,
        )
        full_rbf_acc = evaluate_full_vector_krr(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            kernel_fn=lambda Xa, Xb: full_vector_rbf_kernel(
                Xa, Xb, gamma=full_vector_rbf_gamma
            ),
            lambda_ridge=lambda_ridge,
        )
        full_linear_acc = evaluate_full_vector_krr(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            kernel_fn=full_vector_linear_kernel,
            lambda_ridge=lambda_ridge,
        )

        alphas = model_archi_rbf._get_alphas().detach().cpu()
        signal_alphas = alphas[:num_signal_features]
        noise_alphas = (
            alphas[num_signal_features:]
            if dimension > num_signal_features
            else torch.empty(0)
        )
        signal_mass = signal_alphas.sum().item()
        noise_mass = noise_alphas.sum().item() if noise_alphas.numel() > 0 else 0.0

        archi_acc_list.append(archi_acc)
        full_rbf_acc_list.append(full_rbf_acc)
        full_linear_acc_list.append(full_linear_acc)
        signal_mass_list.append(signal_mass)
        noise_mass_list.append(noise_mass)

        run_rows.append(
            {
                "useless_features": int(m),
                "dimension": int(dimension),
                "archi_all_rbf_accuracy": archi_acc,
                "full_vector_rbf_accuracy": full_rbf_acc,
                "full_vector_linear_accuracy": full_linear_acc,
                "signal_alpha_mass": signal_mass,
                "noise_alpha_mass": noise_mass,
                "alphas": alphas.tolist(),
            }
        )

        print(
            f"[m={m:>3}] Final | "
            f"Archi + All RBF: {archi_acc:.4f}, "
            f"Full-Vector RBF: {full_rbf_acc:.4f}, "
            f"Full-Vector Linear: {full_linear_acc:.4f}, "
            f"Signal alpha mass: {signal_mass:.4f}, "
            f"Noise alpha mass: {noise_mass:.4f}"
        )

        # Alpha plot: markers only (no connecting lines), integer feature indices.
        plt.figure(figsize=(10, 5))
        x_idx = torch.arange(dimension).tolist()
        alpha_vals = alphas.tolist()
        colors = ["tab:blue"] * num_signal_features + ["tab:gray"] * (
            dimension - num_signal_features
        )
        plt.scatter(x_idx, alpha_vals, c=colors, s=42, alpha=0.9)
        if dimension > num_signal_features:
            plt.axvline(
                num_signal_features - 0.5,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
        plt.xlabel("Sub-kernel Index (integer)")
        plt.ylabel("Alpha Value")
        plt.title(f"Alpha Values After Training | Useless Features m={m}")
        if dimension <= 30:
            plt.xticks(list(range(dimension)))
        plt.grid(axis="y", alpha=0.25)
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="tab:blue",
                markersize=8,
                label=f"Signal dims (0..{num_signal_features - 1})",
            )
        ]
        if dimension > num_signal_features:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="tab:gray",
                    markersize=8,
                    label=f"Useless dims ({num_signal_features}..p-1)",
                )
            )
        plt.legend(handles=legend_handles)
        finalize_plot(f"alphas_m{m}")

    # Accuracy summary across noise-feature counts.
    plt.figure(figsize=(10, 6))
    x = [int(v) for v in noise_feature_counts]
    plt.plot(x, archi_acc_list, marker="o", label="Archi + All RBF")
    plt.plot(x, full_rbf_acc_list, marker="o", label="Full-Vector RBF")
    plt.plot(x, full_linear_acc_list, marker="o", label="Full-Vector Linear")
    plt.xlabel("Number of Useless Features (m)")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Number of Useless Features")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("accuracy_vs_useless_features")

    # Alpha mass summary: how much total alpha remains on true signal dimensions.
    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        signal_mass_list,
        marker="o",
        label=f"Signal alpha mass (dims 0..{num_signal_features - 1})",
    )
    plt.plot(
        x,
        noise_mass_list,
        marker="o",
        label=f"Noise alpha mass (dims {num_signal_features}..p-1)",
    )
    plt.xlabel("Number of Useless Features (m)")
    plt.ylabel("Total Alpha Mass")
    plt.title("Alpha Mass Allocation vs Useless Features")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("alpha_mass_vs_useless_features")

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_name": "test3_useless_feature_selection",
            "dataset_name": dataset_name,
            "signal_generator": signal_generator.__name__,
            "signal_generator_kwargs": signal_generator_kwargs,
            "num_signal_features": num_signal_features,
            "original_signal_features": original_signal_features,
            "allow_signal_expansion": allow_signal_expansion,
            "signal_expansion_mode": signal_expansion_mode,
            "signal_expansion_noise_std": signal_expansion_noise_std,
            "noise_feature_counts": [int(v) for v in noise_feature_counts],
            "samples_per_class": samples_per_class,
            "two_moons_noise_std": two_moons_noise_std,
            "useless_feature_std": useless_feature_std,
            "epochs": epochs,
            "lr": lr,
            "lambda_ridge": lambda_ridge,
            "full_vector_rbf_gamma": full_vector_rbf_gamma,
            "seed": seed,
            "results_root": results_root,
            "save_results": save_results,
            "show_plots": show_plots,
        },
        "results_per_m": run_rows,
        "artifacts": {
            "plots": saved_plot_paths,
        },
    }

    if save_results:
        manifest_path = results_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved results to: {results_dir}")

    # Show all collected figures once at the very end (optional).
    if show_plots and show_plots_at_end and deferred_plot_figures:
        plt.show()
        plt.close("all")

    return manifest


if __name__ == "__main__":
    run_experiement(
        dataset_name="two_moons",
        signal_generator=generate_two_moons,
        signal_generator_kwargs={
            "samples_per_class": 300,
            "noise_std": 0.10,
            "radius": 1.0,
            "shuffle": True,
        },
        num_signal_features=50,
        allow_signal_expansion=True,
        signal_expansion_mode="mixed",
        signal_expansion_noise_std=0.01,
        noise_feature_counts=[0, 2, 5, 10, 20, 50,100,200],
        samples_per_class=300,
        two_moons_noise_std=0.10,
        useless_feature_std=1.0,
        epochs=500,
        lr=0.01,
        lambda_ridge=1.0,
        full_vector_rbf_gamma=0.5,
        seed=42,
        run_uuid=str(uuid.uuid4()),
        save_results=True,
        show_plots=True,
    )
