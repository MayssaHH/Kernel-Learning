import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


def build_half_rbf_half_linear_kernel_network(dimension: int) -> KernelNetwork:
    """
    Build architecture with half RBF and half Linear sub-kernels.
    If dimension is odd, one extra RBF is used.
    """
    num_rbf = (dimension + 1) // 2
    num_linear = dimension - num_rbf

    sub_kernels = [RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(num_rbf)]
    sub_kernels += [LinearSubKernel(initial_sigma=0.5, random=True) for _ in range(num_linear)]

    return KernelNetwork(
        sub_kernels=sub_kernels,
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )


def run_experiement(
    pn_values: list = None,
    iter_count: int = 10,
    n_train: int = 400,
    n_test: int = 400,
    p1: int = 4,
    p2: int = 4,
    pc: int = 2,
    snr: float = 2.0,
    sigma: float = 1.0,
    startseed: int = 57474,
    include_validation: bool = False,
    n_validation: int = 400,
    epochs: int = 500,
    lr: float = 0.01,
    lambda_ridge: float = 1.0,
    full_vector_rbf_gamma: float = 0.5,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    """
    Test 8: same protocol as test7, but architecture under test is Half RBF / Half Linear.

    Plot style intentionally matches the other tests:
    - One signal-data preview plot
    - One alpha scatter plot per pn
    - Accuracy-vs-pn summary plot
    - Alpha-mass-vs-pn summary plot
    """
    pn_values = pn_values or [5, 10, 20, 50, 100]
    run_uuid = run_uuid or str(uuid.uuid4())

    results_dir = Path(results_root) / f"{run_uuid}_test8_professor_data_half_rbf_half_linear"
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
    signal_feature_count = pc + p1 + p2

    summary_rows = []
    all_runs = []
    first_split_plotted = False

    for pn in pn_values:
        archi_acc_runs = []
        full_rbf_acc_runs = []
        full_linear_acc_runs = []
        signal_mass_runs = []
        noise_mass_runs = []
        snr_true_runs = []
        alpha_vectors_runs = []

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

            X_train = split["X_train"]
            y_train = split["y_train"]
            X_test = split["X_test"]
            y_test = split["y_test"]
            snr_true = split["snr_true"]

            dimension = X_train.shape[1]

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
                    plt.title("Professor Data (train split, first 2 dims)")
                    plt.xlabel("Feature 0")
                    plt.ylabel("Feature 1")
                    finalize_plot("signal_data_first2dims")
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
                    plt.title("Professor Data (train split, 1D)")
                    plt.xlabel("Feature 0")
                    plt.yticks([])
                    finalize_plot("signal_data_1d")
                first_split_plotted = True

            model_archi_mixed = build_half_rbf_half_linear_kernel_network(dimension=dimension)
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
            signal_mass = alphas[:signal_feature_count].sum().item()
            noise_mass = alphas[signal_feature_count:].sum().item() if pn > 0 else 0.0

            archi_acc_runs.append(archi_acc)
            full_rbf_acc_runs.append(full_rbf_acc)
            full_linear_acc_runs.append(full_linear_acc)
            signal_mass_runs.append(signal_mass)
            noise_mass_runs.append(noise_mass)
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
                    "archi_half_rbf_half_linear_accuracy": float(archi_acc),
                    "full_vector_rbf_accuracy": float(full_rbf_acc),
                    "full_vector_linear_accuracy": float(full_linear_acc),
                    "signal_alpha_mass": float(signal_mass),
                    "noise_alpha_mass": float(noise_mass),
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
                "mean_snr_true": float(torch.tensor(snr_true_runs).mean().item()),
            }
        )

        # Same style as the earlier tests: one alpha scatter per stress level.
        alpha_mean = torch.tensor(alpha_vectors_runs, dtype=torch.float32).mean(dim=0)
        dimension = int(alpha_mean.shape[0])
        x_idx = torch.arange(dimension).tolist()
        colors = ["tab:blue"] * min(signal_feature_count, dimension) + ["tab:gray"] * max(
            0, dimension - signal_feature_count
        )

        plt.figure(figsize=(10, 5))
        plt.scatter(x_idx, alpha_mean.tolist(), c=colors, s=42, alpha=0.9)
        if dimension > signal_feature_count:
            plt.axvline(
                signal_feature_count - 0.5,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
        plt.xlabel("Sub-kernel Index (integer)")
        plt.ylabel("Alpha Value")
        plt.title(f"Alpha Values After Training (mean over iters) | pn={pn}")
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
                label=f"Signal dims (0..{signal_feature_count - 1})",
            )
        ]
        if dimension > signal_feature_count:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="tab:gray",
                    markersize=8,
                    label=f"Noise dims ({signal_feature_count}..p-1)",
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
    plt.plot(
        x,
        [row["mean_full_vector_linear_accuracy"] for row in summary_rows],
        marker="o",
        label="Full-Vector Linear",
    )
    plt.xlabel("Number of Pure Noise Predictors (pn)")
    plt.ylabel("Mean Test Accuracy")
    plt.title("Test 8 (Professor Data): Accuracy vs Pure Noise Predictors")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("accuracy_vs_pn")

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        [row["mean_signal_alpha_mass"] for row in summary_rows],
        marker="o",
        label=f"Signal alpha mass (dims 0..{signal_feature_count - 1})",
    )
    plt.plot(
        x,
        [row["mean_noise_alpha_mass"] for row in summary_rows],
        marker="o",
        label=f"Noise alpha mass (dims {signal_feature_count}..p-1)",
    )
    plt.xlabel("Number of Pure Noise Predictors (pn)")
    plt.ylabel("Mean Alpha Mass")
    plt.title("Test 8 (Professor Data): Alpha Mass vs Pure Noise Predictors")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("alpha_mass_vs_pn")

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_name": "test8_professor_data_half_rbf_half_linear",
            "dataset_name": "professor_simulated",
            "pn_values": [int(v) for v in pn_values],
            "iter_count": iter_count,
            "n_train": n_train,
            "n_test": n_test,
            "n_validation": n_validation,
            "include_validation": include_validation,
            "p1": p1,
            "p2": p2,
            "pc": pc,
            "snr_target": snr,
            "sigma": sigma,
            "startseed": startseed,
            "signal_feature_count": signal_feature_count,
            "architecture_under_test": "half_rbf_half_linear",
            "epochs": epochs,
            "lr": lr,
            "lambda_ridge": lambda_ridge,
            "full_vector_rbf_gamma": full_vector_rbf_gamma,
            "results_root": results_root,
            "save_results": save_results,
            "show_plots": show_plots,
            "show_plots_at_end": show_plots_at_end,
        },
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
        p1=4,
        p2=4,
        pc=2,
        snr=2.0,
        sigma=1.0,
        startseed=57474,
        include_validation=False,
        n_validation=400,
        epochs=500,
        lr=0.01,
        lambda_ridge=1.0,
        full_vector_rbf_gamma=0.5,
        run_uuid=str(uuid.uuid4()),
        save_results=True,
        show_plots=True,
        show_plots_at_end=True,
    )
