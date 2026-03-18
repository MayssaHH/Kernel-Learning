import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from data.generators.professor.monni_simulated_cauchy import (
    generate_professor_split_cauchy,
)
from experiments.classification.modular_experiment import (
    evaluate_full_vector_krr,
    evaluate_model,
    full_vector_linear_kernel,
    full_vector_rbf_kernel,
)
from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    LinearSubKernel,
    ManualGradientTrainer,
    RBFSubKernel,
)
from kernel_learning.methods.KernelRidgeClassifier import KernelRidgeClassifier


FEATURE_TYPE_COLORS = {
    "shared": "tab:blue",
    "class_minus_only": "tab:orange",
    "class_plus_only": "tab:green",
    "noise": "tab:gray",
}

FEATURE_TYPE_LABELS = {
    "shared": "Shared Features",
    "class_minus_only": "Class -1 Specific Features",
    "class_plus_only": "Class +1 Specific Features",
    "noise": "Pure Noise Features",
}

KERNEL_TYPE_MARKERS = {
    "rbf": "o",
    "linear": "x",
}

KERNEL_TYPE_LABELS = {
    "rbf": "RBF Sub-kernel",
    "linear": "Linear Sub-kernel",
}


def evaluate_kernel_model_with_krr(
    kernel_model: KernelNetwork,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lambda_ridge: float = 1.0,
) -> float:
    clf = KernelRidgeClassifier(learnt_kernel=kernel_model, lambda_ridge=lambda_ridge)
    clf.fit(X_train, y_train)
    return evaluate_model(clf, X_test, y_test)


def build_feature_types(pc: int, p1: int, p2: int, pn: int) -> list:
    return (
        ["shared"] * int(pc)
        + ["class_minus_only"] * int(p1)
        + ["class_plus_only"] * int(p2)
        + ["noise"] * int(pn)
    )


def build_kernel_types_for_dimension(dimension: int) -> list:
    """
    Half RBF, half Linear (if odd, one extra RBF).
    """
    num_rbf = (dimension + 1) // 2
    num_linear = dimension - num_rbf
    return (["rbf"] * num_rbf) + (["linear"] * num_linear)


def instantiate_sub_kernels(kernel_types_by_index: list) -> list:
    sub_kernels = []
    for kernel_type in kernel_types_by_index:
        if kernel_type == "rbf":
            sub_kernels.append(RBFSubKernel(initial_gamma=0.5, random=True))
        elif kernel_type == "linear":
            sub_kernels.append(LinearSubKernel(initial_sigma=0.5, random=True))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    return sub_kernels


def run_experiement(
    pn_values: list = None,
    iter_count: int = 10,
    n_train: int = 400,
    n_test: int = 400,
    p1: int = 8,
    p2: int = 8,
    pc: int = 6,
    snr: float = 2.0,
    sigma: float = 1.0,
    startseed: int = 57474,
    include_validation: bool = False,
    n_validation: int = 400,
    epochs: int = 500,
    lr: float = 0.01,
    lambda_ridge: float = 1.0,
    full_vector_rbf_gamma: float = 0.5,
    cauchy_clip: Optional[float] = 25.0,
    shuffle_features: bool = True,
    shuffle_seed_offset: int = 100000,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    """
    Test 12: Cauchy professor protocol with mixed architecture (half RBF / half Linear).

    - Same data protocol as test11.
    - Same shuffled-feature tracking as test9/test11.
    - Alpha plot includes:
      color = feature type (shared/p1/p2/noise)
      marker shape = kernel type (RBF/Linear)
    """
    pn_values = pn_values or [5, 10, 20, 50, 100]
    run_uuid = run_uuid or str(uuid.uuid4())

    results_dir = Path(results_root) / f"{run_uuid}_test12_professor_data_cauchy_mixed_shuffled_types"
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

    criterion = AlignmentLoss()
    total_signal_count = pc + p1 + p2

    summary_rows = []
    all_runs = []
    first_split_plotted = False
    feature_layout_per_pn = {}

    for pn in pn_values:
        base_feature_types = build_feature_types(pc=pc, p1=p1, p2=p2, pn=pn)
        dimension = len(base_feature_types)
        kernel_types_by_index = build_kernel_types_for_dimension(dimension)

        if shuffle_features:
            shuffle_seed = int(startseed + shuffle_seed_offset + pn)
            generator = torch.Generator().manual_seed(shuffle_seed)
            permutation = torch.randperm(dimension, generator=generator)
        else:
            shuffle_seed = None
            permutation = torch.arange(dimension)

        shuffled_feature_types = [base_feature_types[idx] for idx in permutation.tolist()]
        signal_mask = torch.tensor(
            [feature_type != "noise" for feature_type in shuffled_feature_types],
            dtype=torch.bool,
        )
        noise_mask = ~signal_mask
        shared_mask = torch.tensor(
            [feature_type == "shared" for feature_type in shuffled_feature_types],
            dtype=torch.bool,
        )
        minus_mask = torch.tensor(
            [feature_type == "class_minus_only" for feature_type in shuffled_feature_types],
            dtype=torch.bool,
        )
        plus_mask = torch.tensor(
            [feature_type == "class_plus_only" for feature_type in shuffled_feature_types],
            dtype=torch.bool,
        )
        rbf_mask = torch.tensor([kt == "rbf" for kt in kernel_types_by_index], dtype=torch.bool)
        linear_mask = ~rbf_mask

        feature_layout_per_pn[int(pn)] = {
            "shuffle_seed": shuffle_seed,
            "permutation": permutation.tolist(),
            "feature_types_after_shuffle": shuffled_feature_types,
            "kernel_types_by_index": kernel_types_by_index,
            "counts": {
                "shared": int(shared_mask.sum().item()),
                "class_minus_only": int(minus_mask.sum().item()),
                "class_plus_only": int(plus_mask.sum().item()),
                "noise": int(noise_mask.sum().item()),
                "rbf": int(rbf_mask.sum().item()),
                "linear": int(linear_mask.sum().item()),
            },
        }

        archi_acc_runs = []
        full_rbf_acc_runs = []
        full_linear_acc_runs = []
        signal_mass_runs = []
        noise_mass_runs = []
        shared_mass_runs = []
        minus_mass_runs = []
        plus_mass_runs = []
        rbf_mass_runs = []
        linear_mass_runs = []
        snr_true_runs = []
        alpha_vectors_runs = []

        for i in range(iter_count):
            split = generate_professor_split_cauchy(
                iteration=i,
                iter_count=iter_count,
                n_train=n_train,
                n_test=n_test,
                n_val=n_validation,
                p1=p1,
                p2=p2,
                pc=pc,
                pn=pn,
                snr=snr,
                sigma=sigma,
                startseed=startseed,
                include_validation=include_validation,
                output_prefix=None,
                cauchy_clip=cauchy_clip,
            )

            X_train = split["X_train"][:, permutation]
            y_train = split["y_train"]
            X_test = split["X_test"][:, permutation]
            y_test = split["y_test"]
            snr_true = split["snr_true"]

            if not first_split_plotted:
                if dimension >= 2:
                    plt.figure(figsize=(7, 6))
                    plt.scatter(
                        X_train[:, 0],
                        X_train[:, 1],
                        c=y_train,
                        cmap="viridis",
                        s=18,
                        alpha=0.85,
                    )
                    plt.title("Professor Cauchy Data (train split, first 2 dims, shuffled)")
                    plt.xlabel("Feature 0")
                    plt.ylabel("Feature 1")
                    finalize_plot("signal_data_first2dims_shuffled")
                else:
                    plt.figure(figsize=(8, 4))
                    x_values = X_train[:, 0]
                    y_dummy = torch.zeros_like(x_values)
                    plt.scatter(
                        x_values,
                        y_dummy,
                        c=y_train,
                        cmap="viridis",
                        s=18,
                        alpha=0.85,
                    )
                    plt.title("Professor Cauchy Data (train split, 1D, shuffled)")
                    plt.xlabel("Feature 0")
                    plt.yticks([])
                    finalize_plot("signal_data_1d_shuffled")
                first_split_plotted = True

            model_archi_mixed = KernelNetwork(
                sub_kernels=instantiate_sub_kernels(kernel_types_by_index),
                alpha_constraint="square",
                normalize_alphas=True,
                alpha_init="random",
            )
            trainer = ManualGradientTrainer(lr=lr)

            for epoch in range(epochs):
                loss = trainer.train_epoch(
                    model=model_archi_mixed,
                    X=X_train,
                    criterion=criterion,
                    lambda_lasso=0.0,
                    y=y_train,
                )
                if epoch % 100 == 0:
                    print(
                        f"[pn={pn:>3}] [iter={i + 1:>2}/{iter_count}] "
                        f"Epoch {epoch:>3} | Archi + Half RBF/Half Linear alignment loss: {loss:.6f}"
                    )

            archi_acc = evaluate_kernel_model_with_krr(
                kernel_model=model_archi_mixed,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                lambda_ridge=lambda_ridge,
            )
            full_rbf_acc = evaluate_full_vector_krr(
                X_train=X_train,
                Y_train=y_train,
                X_test=X_test,
                Y_test=y_test,
                kernel_fn=lambda Xa, Xb: full_vector_rbf_kernel(Xa, Xb, gamma=full_vector_rbf_gamma),
                lambda_ridge=lambda_ridge,
            )
            full_linear_acc = evaluate_full_vector_krr(
                X_train=X_train,
                Y_train=y_train,
                X_test=X_test,
                Y_test=y_test,
                kernel_fn=full_vector_linear_kernel,
                lambda_ridge=lambda_ridge,
            )

            alphas = model_archi_mixed._get_alphas().detach().cpu()
            signal_mass = alphas[signal_mask].sum().item()
            noise_mass = alphas[noise_mask].sum().item()
            shared_mass = alphas[shared_mask].sum().item()
            minus_mass = alphas[minus_mask].sum().item()
            plus_mass = alphas[plus_mask].sum().item()
            rbf_mass = alphas[rbf_mask].sum().item()
            linear_mass = alphas[linear_mask].sum().item()

            archi_acc_runs.append(archi_acc)
            full_rbf_acc_runs.append(full_rbf_acc)
            full_linear_acc_runs.append(full_linear_acc)
            signal_mass_runs.append(signal_mass)
            noise_mass_runs.append(noise_mass)
            shared_mass_runs.append(shared_mass)
            minus_mass_runs.append(minus_mass)
            plus_mass_runs.append(plus_mass)
            rbf_mass_runs.append(rbf_mass)
            linear_mass_runs.append(linear_mass)
            snr_true_runs.append(snr_true)
            alpha_vectors_runs.append(alphas.tolist())

            all_runs.append(
                {
                    "pn": int(pn),
                    "iteration": int(i + 1),
                    "dimension": int(dimension),
                    "seed_train": int(split["seed_train"]),
                    "seed_test": int(split["seed_test"]),
                    "snr_true": float(snr_true),
                    "shuffle_seed": shuffle_seed,
                    "permutation": permutation.tolist(),
                    "feature_types_after_shuffle": shuffled_feature_types,
                    "kernel_types_by_index": kernel_types_by_index,
                    "archi_half_rbf_half_linear_accuracy": float(archi_acc),
                    "full_vector_rbf_accuracy": float(full_rbf_acc),
                    "full_vector_linear_accuracy": float(full_linear_acc),
                    "signal_alpha_mass": float(signal_mass),
                    "noise_alpha_mass": float(noise_mass),
                    "shared_alpha_mass": float(shared_mass),
                    "class_minus_only_alpha_mass": float(minus_mass),
                    "class_plus_only_alpha_mass": float(plus_mass),
                    "rbf_alpha_mass": float(rbf_mass),
                    "linear_alpha_mass": float(linear_mass),
                    "alphas": alphas.tolist(),
                }
            )

            print(
                f"[pn={pn:>3}] [iter={i + 1:>2}/{iter_count}] Final | "
                f"Archi + Half RBF/Half Linear: {archi_acc:.4f}, "
                f"Full-Vector RBF: {full_rbf_acc:.4f}, "
                f"Full-Vector Linear: {full_linear_acc:.4f}, "
                f"Signal alpha mass: {signal_mass:.4f}, "
                f"Noise alpha mass: {noise_mass:.4f}, "
                f"Shared/P1/P2 alpha mass: {shared_mass:.4f}/{minus_mass:.4f}/{plus_mass:.4f}, "
                f"RBF/Linear alpha mass: {rbf_mass:.4f}/{linear_mass:.4f}, "
                f"SNR(true): {snr_true:.4f}"
            )

        summary_rows.append(
            {
                "pn": int(pn),
                "mean_archi_half_rbf_half_linear_accuracy": float(torch.tensor(archi_acc_runs).mean().item()),
                "std_archi_half_rbf_half_linear_accuracy": float(torch.tensor(archi_acc_runs).std(unbiased=False).item()),
                "mean_full_vector_rbf_accuracy": float(torch.tensor(full_rbf_acc_runs).mean().item()),
                "std_full_vector_rbf_accuracy": float(torch.tensor(full_rbf_acc_runs).std(unbiased=False).item()),
                "mean_full_vector_linear_accuracy": float(torch.tensor(full_linear_acc_runs).mean().item()),
                "std_full_vector_linear_accuracy": float(torch.tensor(full_linear_acc_runs).std(unbiased=False).item()),
                "mean_signal_alpha_mass": float(torch.tensor(signal_mass_runs).mean().item()),
                "std_signal_alpha_mass": float(torch.tensor(signal_mass_runs).std(unbiased=False).item()),
                "mean_noise_alpha_mass": float(torch.tensor(noise_mass_runs).mean().item()),
                "std_noise_alpha_mass": float(torch.tensor(noise_mass_runs).std(unbiased=False).item()),
                "mean_shared_alpha_mass": float(torch.tensor(shared_mass_runs).mean().item()),
                "mean_class_minus_only_alpha_mass": float(torch.tensor(minus_mass_runs).mean().item()),
                "mean_class_plus_only_alpha_mass": float(torch.tensor(plus_mass_runs).mean().item()),
                "mean_rbf_alpha_mass": float(torch.tensor(rbf_mass_runs).mean().item()),
                "mean_linear_alpha_mass": float(torch.tensor(linear_mass_runs).mean().item()),
                "mean_snr_true": float(torch.tensor(snr_true_runs).mean().item()),
            }
        )

        alpha_mean = torch.tensor(alpha_vectors_runs, dtype=torch.float32).mean(dim=0)
        x_idx = torch.arange(dimension).tolist()

        plt.figure(figsize=(11, 5))
        for kernel_type in ["rbf", "linear"]:
            indices = [idx for idx, kt in enumerate(kernel_types_by_index) if kt == kernel_type]
            if not indices:
                continue
            x_vals = [x_idx[idx] for idx in indices]
            y_vals = [alpha_mean[idx].item() for idx in indices]
            colors = [FEATURE_TYPE_COLORS[shuffled_feature_types[idx]] for idx in indices]
            marker = KERNEL_TYPE_MARKERS[kernel_type]
            if marker == "x":
                plt.scatter(
                    x_vals,
                    y_vals,
                    c=colors,
                    marker=marker,
                    s=58,
                    alpha=0.95,
                    linewidths=1.4,
                )
            else:
                plt.scatter(
                    x_vals,
                    y_vals,
                    c=colors,
                    marker=marker,
                    s=42,
                    alpha=0.9,
                )

        plt.xlabel("Sub-kernel Index (integer, shuffled order)")
        plt.ylabel("Alpha Value")
        plt.title(f"Alpha Values After Training (mean over iters) | pn={pn}")
        if dimension <= 40:
            plt.xticks(list(range(dimension)))
        plt.grid(axis="y", alpha=0.25)

        legend_handles = []
        for feature_type in ["shared", "class_minus_only", "class_plus_only", "noise"]:
            if feature_type in shuffled_feature_types:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=FEATURE_TYPE_COLORS[feature_type],
                        markersize=8,
                        label=FEATURE_TYPE_LABELS[feature_type],
                    )
                )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=7,
                label=KERNEL_TYPE_LABELS["rbf"],
            )
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                linestyle="None",
                markersize=8,
                label=KERNEL_TYPE_LABELS["linear"],
            )
        )
        plt.legend(handles=legend_handles)
        finalize_plot(f"alphas_pn{pn}")

    x = [row["pn"] for row in summary_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        [row["mean_archi_half_rbf_half_linear_accuracy"] for row in summary_rows],
        marker="o",
        label="Archi + Half RBF/Half Linear",
    )
    plt.plot(x, [row["mean_full_vector_rbf_accuracy"] for row in summary_rows], marker="o", label="Full-Vector RBF")
    plt.plot(x, [row["mean_full_vector_linear_accuracy"] for row in summary_rows], marker="o", label="Full-Vector Linear")
    plt.xlabel("Number of Pure Noise Predictors (pn)")
    plt.ylabel("Mean Test Accuracy")
    plt.title("Test 12 (Professor Cauchy Data, Shuffled Features): Accuracy vs Pure Noise Predictors")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("accuracy_vs_pn")

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        [row["mean_signal_alpha_mass"] for row in summary_rows],
        marker="o",
        label="Signal alpha mass (shared + p1 + p2)",
    )
    plt.plot(
        x,
        [row["mean_noise_alpha_mass"] for row in summary_rows],
        marker="o",
        label="Noise alpha mass",
    )
    plt.xlabel("Number of Pure Noise Predictors (pn)")
    plt.ylabel("Mean Alpha Mass")
    plt.title("Test 12: Signal/Noise Alpha Mass vs Pure Noise Predictors")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("alpha_mass_vs_pn")

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        [row["mean_shared_alpha_mass"] for row in summary_rows],
        marker="o",
        label="Shared features",
    )
    plt.plot(
        x,
        [row["mean_class_minus_only_alpha_mass"] for row in summary_rows],
        marker="o",
        label="Class -1 specific features",
    )
    plt.plot(
        x,
        [row["mean_class_plus_only_alpha_mass"] for row in summary_rows],
        marker="o",
        label="Class +1 specific features",
    )
    plt.plot(
        x,
        [row["mean_noise_alpha_mass"] for row in summary_rows],
        marker="o",
        label="Noise features",
    )
    plt.xlabel("Number of Pure Noise Predictors (pn)")
    plt.ylabel("Mean Alpha Mass")
    plt.title("Test 12: Alpha Mass by Feature Type vs Pure Noise Predictors")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("alpha_mass_by_feature_type_vs_pn")

    plt.figure(figsize=(10, 6))
    plt.plot(x, [row["mean_rbf_alpha_mass"] for row in summary_rows], marker="o", label="RBF alpha mass")
    plt.plot(x, [row["mean_linear_alpha_mass"] for row in summary_rows], marker="o", label="Linear alpha mass")
    plt.xlabel("Number of Pure Noise Predictors (pn)")
    plt.ylabel("Mean Alpha Mass")
    plt.title("Test 12: Alpha Mass by Kernel Type vs Pure Noise Predictors")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("alpha_mass_by_kernel_type_vs_pn")

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_name": "test12_professor_data_cauchy_mixed_shuffled_types",
            "dataset_name": "professor_simulated_cauchy",
            "architecture_under_test": "half_rbf_half_linear",
            "pn_values": [int(v) for v in pn_values],
            "iter_count": iter_count,
            "n_train": n_train,
            "n_test": n_test,
            "n_validation": n_validation,
            "include_validation": include_validation,
            "p1": p1,
            "p2": p2,
            "pc": pc,
            "signal_feature_count": total_signal_count,
            "snr_target": snr,
            "sigma": sigma,
            "cauchy_clip": cauchy_clip,
            "startseed": startseed,
            "shuffle_features": shuffle_features,
            "shuffle_seed_offset": shuffle_seed_offset,
            "epochs": epochs,
            "lr": lr,
            "lambda_ridge": lambda_ridge,
            "full_vector_rbf_gamma": full_vector_rbf_gamma,
            "results_root": results_root,
            "save_results": save_results,
            "show_plots": show_plots,
            "show_plots_at_end": show_plots_at_end,
        },
        "feature_layout_per_pn": feature_layout_per_pn,
        "summary_per_pn": summary_rows,
        "runs": all_runs,
        "artifacts": {
            "plots": saved_plot_paths,
        },
    }

    if save_results:
        manifest_path = results_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved results to: {results_dir}")

    if show_plots and show_plots_at_end and deferred_plot_figures:
        plt.show()
        plt.close("all")

    return manifest


if __name__ == "__main__":
    run_experiement(
        pn_values=[5, 10, 20, 50, 100],
        iter_count=10,
        n_train=400,
        n_test=400,
        p1=8,
        p2=8,
        pc=6,
        snr=2.0,
        sigma=1.0,
        cauchy_clip=25.0,
        startseed=57474,
        include_validation=False,
        n_validation=400,
        epochs=500,
        lr=0.01,
        lambda_ridge=1.0,
        full_vector_rbf_gamma=0.5,
        shuffle_features=True,
        shuffle_seed_offset=100000,
        run_uuid=str(uuid.uuid4()),
        save_results=True,
        show_plots=True,
        show_plots_at_end=True,
    )
