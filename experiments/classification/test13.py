import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from data.generators.professor.monni_simulated import generate_professor_split
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

MODEL_ORDER = [
    "archi_all_rbf",
    "archi_half_rbf_half_linear",
    "archi_all_linear",
    "full_vector_rbf",
    "full_vector_linear",
]

MODEL_LABELS = {
    "archi_all_rbf": "Archi + All RBF",
    "archi_half_rbf_half_linear": "Archi + Half RBF/Half Linear",
    "archi_all_linear": "Archi + All Linear",
    "full_vector_rbf": "Full-Vector RBF",
    "full_vector_linear": "Full-Vector Linear",
}

ARCHITECTURE_KEYS = [
    "archi_all_rbf",
    "archi_half_rbf_half_linear",
    "archi_all_linear",
]


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


def build_feature_types(pc: int, p1: int, p2: int, pn: int) -> List[str]:
    return (
        ["shared"] * int(pc)
        + ["class_minus_only"] * int(p1)
        + ["class_plus_only"] * int(p2)
        + ["noise"] * int(pn)
    )


def build_kernel_types_for_dimension(dimension: int, architecture_key: str) -> List[str]:
    if architecture_key == "archi_all_rbf":
        return ["rbf"] * int(dimension)
    if architecture_key == "archi_all_linear":
        return ["linear"] * int(dimension)
    if architecture_key == "archi_half_rbf_half_linear":
        num_rbf = (int(dimension) + 1) // 2
        num_linear = int(dimension) - num_rbf
        return (["rbf"] * num_rbf) + (["linear"] * num_linear)
    raise ValueError(f"Unknown architecture_key: {architecture_key}")


def instantiate_sub_kernels(kernel_types_by_index: List[str]):
    sub_kernels = []
    for kernel_type in kernel_types_by_index:
        if kernel_type == "rbf":
            sub_kernels.append(RBFSubKernel(initial_gamma=0.5, random=True))
        elif kernel_type == "linear":
            sub_kernels.append(LinearSubKernel(initial_sigma=0.5, random=True))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    return sub_kernels


def mean_and_std(values: List[float]):
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def snr_tag(snr: float) -> str:
    return str(snr).replace("-", "m").replace(".", "p")


def run_experiement(
    snr_values: list = None,
    pn_values: list = None,
    iter_count: int = 5,
    n_train: int = 400,
    n_test: int = 400,
    p1: int = 8,
    p2: int = 8,
    pc: int = 6,
    sigma: float = 1.0,
    startseed: int = 57474,
    include_validation: bool = False,
    n_validation: int = 400,
    epochs: int = 300,
    lr: float = 0.01,
    lambda_ridge: float = 1.0,
    full_vector_rbf_gamma: float = 0.5,
    shuffle_features: bool = True,
    shuffle_seed_offset: int = 100000,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    """
    Test 13: SNR x PN phase-diagram experiment.

    Compared models:
    - Archi + All RBF
    - Archi + Half RBF/Half Linear
    - Archi + All Linear
    - Full-Vector RBF baseline
    - Full-Vector Linear baseline

    For each (snr, pn):
    - run iter_count independent splits,
    - train all 3 architecture variants,
    - evaluate all 5 models,
    - track alpha mass behavior for architecture variants.
    """
    snr_values = snr_values or [0.5, 1.0, 2.0, 4.0]
    pn_values = pn_values or [5, 10, 20, 50]
    run_uuid = run_uuid or str(uuid.uuid4())

    results_dir = Path(results_root) / f"{run_uuid}_test13_snr_pn_phase_diagram"
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
    all_runs = []
    summary_rows = []
    feature_layout_per_snr_pn = {}
    first_split_plotted = False

    for snr in snr_values:
        for pn in pn_values:
            base_feature_types = build_feature_types(pc=pc, p1=p1, p2=p2, pn=pn)
            dimension = len(base_feature_types)

            if shuffle_features:
                seed_for_shuffle = int(startseed + shuffle_seed_offset + int(pn * 100) + int(snr * 1000))
                generator = torch.Generator().manual_seed(seed_for_shuffle)
                permutation = torch.randperm(dimension, generator=generator)
            else:
                seed_for_shuffle = None
                permutation = torch.arange(dimension)

            shuffled_feature_types = [base_feature_types[idx] for idx in permutation.tolist()]
            feature_type_mask = {
                "shared": torch.tensor(
                    [ft == "shared" for ft in shuffled_feature_types], dtype=torch.bool
                ),
                "class_minus_only": torch.tensor(
                    [ft == "class_minus_only" for ft in shuffled_feature_types], dtype=torch.bool
                ),
                "class_plus_only": torch.tensor(
                    [ft == "class_plus_only" for ft in shuffled_feature_types], dtype=torch.bool
                ),
                "noise": torch.tensor(
                    [ft == "noise" for ft in shuffled_feature_types], dtype=torch.bool
                ),
            }
            signal_mask = ~feature_type_mask["noise"]
            noise_mask = feature_type_mask["noise"]

            layout_key = f"snr_{snr}_pn_{pn}"
            feature_layout_per_snr_pn[layout_key] = {
                "snr": float(snr),
                "pn": int(pn),
                "shuffle_seed": seed_for_shuffle,
                "permutation": permutation.tolist(),
                "feature_types_after_shuffle": shuffled_feature_types,
                "counts": {
                    "shared": int(feature_type_mask["shared"].sum().item()),
                    "class_minus_only": int(feature_type_mask["class_minus_only"].sum().item()),
                    "class_plus_only": int(feature_type_mask["class_plus_only"].sum().item()),
                    "noise": int(feature_type_mask["noise"].sum().item()),
                },
            }

            accuracy_runs: Dict[str, List[float]] = {k: [] for k in MODEL_ORDER}
            alpha_vectors_runs: Dict[str, List[List[float]]] = {k: [] for k in ARCHITECTURE_KEYS}
            alpha_mass_runs: Dict[str, Dict[str, List[float]]] = {
                k: {
                    "signal": [],
                    "noise": [],
                    "shared": [],
                    "class_minus_only": [],
                    "class_plus_only": [],
                    "rbf": [],
                    "linear": [],
                }
                for k in ARCHITECTURE_KEYS
            }
            snr_true_runs: List[float] = []
            final_loss_runs: Dict[str, List[float]] = {k: [] for k in ARCHITECTURE_KEYS}

            for i in range(iter_count):
                split = generate_professor_split(
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
                )

                X_train = split["X_train"][:, permutation]
                y_train = split["y_train"]
                X_test = split["X_test"][:, permutation]
                y_test = split["y_test"]
                snr_true = float(split["snr_true"])
                snr_true_runs.append(snr_true)

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
                        plt.title("Professor Data (train split, first 2 dims, shuffled)")
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
                        plt.title("Professor Data (train split, 1D, shuffled)")
                        plt.xlabel("Feature 0")
                        plt.yticks([])
                        finalize_plot("signal_data_1d_shuffled")
                    first_split_plotted = True

                archi_iteration_cache = {}
                for architecture_key in ARCHITECTURE_KEYS:
                    kernel_types = build_kernel_types_for_dimension(dimension, architecture_key)
                    model = KernelNetwork(
                        sub_kernels=instantiate_sub_kernels(kernel_types),
                        alpha_constraint="square",
                        normalize_alphas=True,
                        alpha_init="random",
                    )
                    trainer = ManualGradientTrainer(lr=lr)

                    final_loss = None
                    for epoch in range(epochs):
                        final_loss = trainer.train_epoch(
                            model=model,
                            X=X_train,
                            criterion=criterion,
                            lambda_lasso=0.0,
                            y=y_train,
                        )
                        if epoch % 100 == 0:
                            print(
                                f"[snr={snr:>4}] [pn={pn:>3}] [iter={i + 1:>2}/{iter_count}] "
                                f"Epoch {epoch:>3} | {MODEL_LABELS[architecture_key]} alignment loss: {final_loss:.6f}"
                            )

                    acc = evaluate_kernel_model_with_krr(
                        kernel_model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        lambda_ridge=lambda_ridge,
                    )
                    alphas = model._get_alphas().detach().cpu()
                    kernel_type_by_index = build_kernel_types_for_dimension(dimension, architecture_key)
                    rbf_mask = torch.tensor([kt == "rbf" for kt in kernel_type_by_index], dtype=torch.bool)
                    linear_mask = ~rbf_mask

                    accuracy_runs[architecture_key].append(float(acc))
                    alpha_vectors_runs[architecture_key].append(alphas.tolist())
                    final_loss_runs[architecture_key].append(float(final_loss))

                    alpha_mass_runs[architecture_key]["signal"].append(float(alphas[signal_mask].sum().item()))
                    alpha_mass_runs[architecture_key]["noise"].append(float(alphas[noise_mask].sum().item()))
                    alpha_mass_runs[architecture_key]["shared"].append(float(alphas[feature_type_mask["shared"]].sum().item()))
                    alpha_mass_runs[architecture_key]["class_minus_only"].append(
                        float(alphas[feature_type_mask["class_minus_only"]].sum().item())
                    )
                    alpha_mass_runs[architecture_key]["class_plus_only"].append(
                        float(alphas[feature_type_mask["class_plus_only"]].sum().item())
                    )
                    alpha_mass_runs[architecture_key]["rbf"].append(float(alphas[rbf_mask].sum().item()))
                    alpha_mass_runs[architecture_key]["linear"].append(float(alphas[linear_mask].sum().item()))

                    archi_iteration_cache[architecture_key] = {
                        "accuracy": float(acc),
                        "alphas": alphas.tolist(),
                        "kernel_types_by_index": kernel_type_by_index,
                        "final_alignment_loss": float(final_loss),
                        "signal_alpha_mass": float(alphas[signal_mask].sum().item()),
                        "noise_alpha_mass": float(alphas[noise_mask].sum().item()),
                        "shared_alpha_mass": float(alphas[feature_type_mask["shared"]].sum().item()),
                        "class_minus_only_alpha_mass": float(alphas[feature_type_mask["class_minus_only"]].sum().item()),
                        "class_plus_only_alpha_mass": float(alphas[feature_type_mask["class_plus_only"]].sum().item()),
                        "rbf_alpha_mass": float(alphas[rbf_mask].sum().item()),
                        "linear_alpha_mass": float(alphas[linear_mask].sum().item()),
                    }

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
                accuracy_runs["full_vector_rbf"].append(float(full_rbf_acc))
                accuracy_runs["full_vector_linear"].append(float(full_linear_acc))

                all_runs.append(
                    {
                        "snr": float(snr),
                        "pn": int(pn),
                        "iteration": int(i + 1),
                        "dimension": int(dimension),
                        "seed_train": int(split["seed_train"]),
                        "seed_test": int(split["seed_test"]),
                        "snr_true": snr_true,
                        "shuffle_seed": seed_for_shuffle,
                        "permutation": permutation.tolist(),
                        "feature_types_after_shuffle": shuffled_feature_types,
                        "models": {
                            "archi_all_rbf": archi_iteration_cache["archi_all_rbf"],
                            "archi_half_rbf_half_linear": archi_iteration_cache["archi_half_rbf_half_linear"],
                            "archi_all_linear": archi_iteration_cache["archi_all_linear"],
                            "full_vector_rbf": {"accuracy": float(full_rbf_acc)},
                            "full_vector_linear": {"accuracy": float(full_linear_acc)},
                        },
                    }
                )

                print(
                    f"[snr={snr:>4}] [pn={pn:>3}] [iter={i + 1:>2}/{iter_count}] Final | "
                    f"Archi+RBF: {archi_iteration_cache['archi_all_rbf']['accuracy']:.4f}, "
                    f"Archi+Mixed: {archi_iteration_cache['archi_half_rbf_half_linear']['accuracy']:.4f}, "
                    f"Archi+Linear: {archi_iteration_cache['archi_all_linear']['accuracy']:.4f}, "
                    f"Full-RBF: {full_rbf_acc:.4f}, Full-Linear: {full_linear_acc:.4f}, "
                    f"SNR(true): {snr_true:.4f}"
                )

            summary_row = {
                "snr": float(snr),
                "pn": int(pn),
                "dimension": int(dimension),
                "mean_snr_true": mean_and_std(snr_true_runs)[0],
            }

            for model_key in MODEL_ORDER:
                mean_acc, std_acc = mean_and_std(accuracy_runs[model_key])
                summary_row[f"mean_{model_key}_accuracy"] = mean_acc
                summary_row[f"std_{model_key}_accuracy"] = std_acc

            for architecture_key in ARCHITECTURE_KEYS:
                mean_loss, std_loss = mean_and_std(final_loss_runs[architecture_key])
                summary_row[f"mean_{architecture_key}_final_alignment_loss"] = mean_loss
                summary_row[f"std_{architecture_key}_final_alignment_loss"] = std_loss
                for mass_name in [
                    "signal",
                    "noise",
                    "shared",
                    "class_minus_only",
                    "class_plus_only",
                    "rbf",
                    "linear",
                ]:
                    mean_mass, std_mass = mean_and_std(alpha_mass_runs[architecture_key][mass_name])
                    summary_row[f"mean_{architecture_key}_{mass_name}_alpha_mass"] = mean_mass
                    summary_row[f"std_{architecture_key}_{mass_name}_alpha_mass"] = std_mass

            summary_rows.append(summary_row)

            snr_part = snr_tag(snr)
            for architecture_key in ARCHITECTURE_KEYS:
                alpha_mean = torch.tensor(alpha_vectors_runs[architecture_key], dtype=torch.float32).mean(dim=0)
                kernel_type_by_index = build_kernel_types_for_dimension(dimension, architecture_key)
                x_idx = torch.arange(dimension).tolist()

                plt.figure(figsize=(11, 5))
                if architecture_key == "archi_half_rbf_half_linear":
                    for kernel_type in ["rbf", "linear"]:
                        indices = [idx for idx, kt in enumerate(kernel_type_by_index) if kt == kernel_type]
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
                else:
                    colors = [FEATURE_TYPE_COLORS[feature_type] for feature_type in shuffled_feature_types]
                    plt.scatter(x_idx, alpha_mean.tolist(), c=colors, s=42, alpha=0.9, marker="o")

                plt.xlabel("Sub-kernel Index (integer, shuffled order)")
                plt.ylabel("Alpha Value")
                plt.title(
                    f"Alpha Values (mean over iters) | {MODEL_LABELS[architecture_key]} | snr={snr}, pn={pn}"
                )
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
                if architecture_key == "archi_half_rbf_half_linear":
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="black",
                            linestyle="None",
                            markersize=7,
                            label="RBF Sub-kernel",
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
                            label="Linear Sub-kernel",
                        )
                    )
                plt.legend(handles=legend_handles)
                finalize_plot(f"alphas_{architecture_key}_snr{snr_part}_pn{pn}")

    # Line plots per SNR (same style as earlier tests, but now repeated for each SNR slice).
    for snr in snr_values:
        snr_part = snr_tag(snr)
        rows = [r for r in summary_rows if float(r["snr"]) == float(snr)]
        rows = sorted(rows, key=lambda r: int(r["pn"]))
        x = [int(r["pn"]) for r in rows]

        plt.figure(figsize=(10, 6))
        for model_key in MODEL_ORDER:
            plt.plot(
                x,
                [r[f"mean_{model_key}_accuracy"] for r in rows],
                marker="o",
                label=MODEL_LABELS[model_key],
            )
        plt.xlabel("Number of Pure Noise Predictors (pn)")
        plt.ylabel("Mean Test Accuracy")
        plt.title(f"Test 13: Accuracy vs PN | snr={snr}")
        plt.ylim(0.0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        finalize_plot(f"accuracy_vs_pn_snr{snr_part}")

        for architecture_key in ARCHITECTURE_KEYS:
            plt.figure(figsize=(10, 6))
            plt.plot(
                x,
                [r[f"mean_{architecture_key}_signal_alpha_mass"] for r in rows],
                marker="o",
                label=f"{MODEL_LABELS[architecture_key]} signal alpha mass",
            )
            plt.plot(
                x,
                [r[f"mean_{architecture_key}_noise_alpha_mass"] for r in rows],
                marker="o",
                label=f"{MODEL_LABELS[architecture_key]} noise alpha mass",
            )
            plt.xlabel("Number of Pure Noise Predictors (pn)")
            plt.ylabel("Mean Alpha Mass")
            plt.title(f"Test 13: Signal/Noise Alpha Mass vs PN | snr={snr} | {MODEL_LABELS[architecture_key]}")
            plt.ylim(0.0, 1.05)
            plt.grid(alpha=0.3)
            plt.legend()
            finalize_plot(f"alpha_mass_vs_pn_{architecture_key}_snr{snr_part}")

            plt.figure(figsize=(10, 6))
            plt.plot(
                x,
                [r[f"mean_{architecture_key}_shared_alpha_mass"] for r in rows],
                marker="o",
                label="Shared features",
            )
            plt.plot(
                x,
                [r[f"mean_{architecture_key}_class_minus_only_alpha_mass"] for r in rows],
                marker="o",
                label="Class -1 specific features",
            )
            plt.plot(
                x,
                [r[f"mean_{architecture_key}_class_plus_only_alpha_mass"] for r in rows],
                marker="o",
                label="Class +1 specific features",
            )
            plt.plot(
                x,
                [r[f"mean_{architecture_key}_noise_alpha_mass"] for r in rows],
                marker="o",
                label="Noise features",
            )
            plt.xlabel("Number of Pure Noise Predictors (pn)")
            plt.ylabel("Mean Alpha Mass")
            plt.title(f"Test 13: Alpha Mass by Feature Type vs PN | snr={snr} | {MODEL_LABELS[architecture_key]}")
            plt.ylim(0.0, 1.05)
            plt.grid(alpha=0.3)
            plt.legend()
            finalize_plot(f"alpha_mass_by_feature_type_vs_pn_{architecture_key}_snr{snr_part}")

    # Heatmaps over the SNR x PN grid for all 5 compared models.
    pn_ticks = [int(v) for v in pn_values]
    snr_ticks = [float(v) for v in snr_values]

    def matrix_from_summary(metric_key_prefix: str):
        mat = torch.zeros(len(snr_ticks), len(pn_ticks), dtype=torch.float32)
        for i_snr, snr in enumerate(snr_ticks):
            for i_pn, pn in enumerate(pn_ticks):
                row = next(r for r in summary_rows if float(r["snr"]) == float(snr) and int(r["pn"]) == int(pn))
                mat[i_snr, i_pn] = row[f"mean_{metric_key_prefix}_accuracy"]
        return mat

    accuracy_mats = {model_key: matrix_from_summary(model_key) for model_key in MODEL_ORDER}

    for model_key in MODEL_ORDER:
        mat = accuracy_mats[model_key]
        plt.figure(figsize=(9, 5))
        im = plt.imshow(mat.numpy(), aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        plt.colorbar(im, label="Mean Test Accuracy")
        plt.xticks(list(range(len(pn_ticks))), [str(v) for v in pn_ticks])
        plt.yticks(list(range(len(snr_ticks))), [str(v) for v in snr_ticks])
        plt.xlabel("pn (pure noise predictors)")
        plt.ylabel("snr")
        plt.title(f"Test 13 Heatmap: {MODEL_LABELS[model_key]}")
        finalize_plot(f"accuracy_heatmap_{model_key}")

    baseline_best_mat = torch.maximum(
        accuracy_mats["full_vector_rbf"], accuracy_mats["full_vector_linear"]
    )
    for architecture_key in ARCHITECTURE_KEYS:
        diff_mat = accuracy_mats[architecture_key] - baseline_best_mat
        vmax = float(torch.max(torch.abs(diff_mat)).item()) + 1e-8
        plt.figure(figsize=(9, 5))
        im = plt.imshow(
            diff_mat.numpy(),
            aspect="auto",
            origin="lower",
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        plt.colorbar(im, label="Accuracy Advantage vs Best Full-Vector Baseline")
        plt.xticks(list(range(len(pn_ticks))), [str(v) for v in pn_ticks])
        plt.yticks(list(range(len(snr_ticks))), [str(v) for v in snr_ticks])
        plt.xlabel("pn (pure noise predictors)")
        plt.ylabel("snr")
        plt.title(f"Test 13 Advantage Heatmap: {MODEL_LABELS[architecture_key]} - max(Full-RBF, Full-Linear)")
        finalize_plot(f"advantage_heatmap_{architecture_key}")

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_name": "test13_snr_pn_phase_diagram",
            "dataset_name": "professor_simulated",
            "snr_values": [float(v) for v in snr_values],
            "pn_values": [int(v) for v in pn_values],
            "iter_count": iter_count,
            "n_train": n_train,
            "n_test": n_test,
            "n_validation": n_validation,
            "include_validation": include_validation,
            "p1": p1,
            "p2": p2,
            "pc": pc,
            "sigma": sigma,
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
            "model_order": MODEL_ORDER,
            "model_labels": MODEL_LABELS,
        },
        "feature_layout_per_snr_pn": feature_layout_per_snr_pn,
        "summary_per_snr_pn": summary_rows,
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
        snr_values=[0.5, 1.0, 2.0, 4.0],
        pn_values=[5, 10, 20, 50],
        iter_count=5,
        n_train=400,
        n_test=400,
        p1=8,
        p2=8,
        pc=6,
        sigma=1.0,
        startseed=57474,
        include_validation=False,
        n_validation=400,
        epochs=300,
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
