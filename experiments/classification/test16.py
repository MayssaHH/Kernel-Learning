import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from MKLpy.algorithms import AverageMKL, CKA, EasyMKL
    from MKLpy.metrics.pairwise import homogeneous_polynomial_kernel
except ImportError as exc:
    raise ImportError(
        "test16 requires MKLpy. Install with: .venv\\Scripts\\python.exe -m pip install MKLpy"
    ) from exc

from experiments.classification.modular_experiment import (
    evaluate_full_vector_krr,
    full_vector_linear_kernel,
    full_vector_rbf_kernel,
)
from experiments.classification.test14 import (
    ARCHITECTURE_LABELS,
    PAPER_DATASETS,
    PROTOCOL_CHOICES,
    build_model,
    evaluate_model,
    load_paper_dataset_full,
    load_paper_dataset_strict_split,
)
from kernel_learning import AlignmentLoss, ManualGradientTrainer


REFERENCE_RESULTS_JSON = Path("experiments/classification/paper_smkl_2025_results.json")

ARCH_TO_MODEL_KEY = {
    "all_rbf": "archi_all_rbf",
    "all_linear": "archi_all_linear",
    "half_rbf_half_linear": "archi_half_rbf_half_linear",
}

MODEL_LABELS = {
    "archi_all_rbf": "Archi + All RBF",
    "archi_all_linear": "Archi + All Linear",
    "archi_half_rbf_half_linear": "Archi + Half RBF/Half Linear",
    "full_vector_rbf": "Full-Vector RBF",
    "full_vector_linear": "Full-Vector Linear",
    "mkl_average": "AverageMKL (MKLpy)",
    "mkl_easy": "EasyMKL (MKLpy)",
    "mkl_cka": "CKA (MKLpy)",
}

ALPHA_SOURCE_CHOICES = [
    "all_rbf",
    "all_linear",
    "half_rbf_half_linear",
    "mean_all_architectures",
]


def _load_paper_accuracy_reference(dataset_name: str) -> Dict[str, float]:
    if not REFERENCE_RESULTS_JSON.exists():
        return {}
    try:
        payload = json.loads(REFERENCE_RESULTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("accuracy_percent_table2", [])
    for row in rows:
        if row.get("dataset") == dataset_name:
            return {
                "EasyMKL": float(row.get("EasyMKL")),
                "AverageMKL": float(row.get("AverageMKL")),
                "CKA": float(row.get("CKA")),
                "Algorithm1_SMKL": float(row.get("Algorithm1_SMKL")),
            }
    return {}


def _build_split_payloads(
    dataset_name: str,
    protocol: str,
    n_splits: int,
    test_size: float,
    random_seed: int,
    paper_seed: int,
    frac: float,
    train_ratio: float,
    force_download: bool,
    data_cache_dir: str,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Dict, int]:
    split_payloads = []
    if protocol == "paper_strict":
        if n_splits != 1:
            print(f"[test16] protocol=paper_strict uses exactly 1 split. Ignoring n_splits={n_splits}.")
        X_train_np, y_train_np, X_test_np, y_test_np, dataset_info = load_paper_dataset_strict_split(
            dataset_name=dataset_name,
            frac=frac,
            train_ratio=train_ratio,
            seed=paper_seed,
            force_download=force_download,
            data_cache_dir=data_cache_dir,
        )
        split_payloads.append((1, X_train_np, X_test_np, y_train_np, y_test_np))
        effective_n_splits = 1
    else:
        X_all, y_all, dataset_info = load_paper_dataset_full(
            dataset_name=dataset_name,
            frac=frac,
            seed=paper_seed,
            force_download=force_download,
            data_cache_dir=data_cache_dir,
        )
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_seed,
        )
        for split_idx, (train_idx, test_idx) in enumerate(splitter.split(X_all, y_all), start=1):
            X_train_raw = X_all[train_idx]
            X_test_raw = X_all[test_idx]
            y_train_np = y_all[train_idx]
            y_test_np = y_all[test_idx]

            scaler = StandardScaler()
            X_train_np = scaler.fit_transform(X_train_raw)
            X_test_np = scaler.transform(X_test_raw)
            split_payloads.append((split_idx, X_train_np, X_test_np, y_train_np, y_test_np))
        effective_n_splits = n_splits
    return split_payloads, dataset_info, effective_n_splits


def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _train_architecture(
    architecture: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    criterion: AlignmentLoss,
    epochs: int,
    lr: float,
    lambda_ridge: float,
    split_idx: int,
    effective_n_splits: int,
    seed: Optional[int] = None,
) -> Dict:
    _set_seed(seed)
    model = build_model(architecture=architecture, dimension=X_train.shape[1])
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
                f"[test16] [split={split_idx}/{effective_n_splits}] "
                f"Epoch {epoch:>3} | {ARCHITECTURE_LABELS[architecture]} alignment loss: {final_loss:.6f}"
            )

    accuracy = evaluate_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        lambda_ridge=lambda_ridge,
    )
    alphas = model._get_alphas().detach().cpu().numpy().astype(np.float64)
    return {
        "model": model,
        "accuracy": float(accuracy),
        "final_alignment_loss": float(final_loss),
        "alphas": alphas,
    }


def _evaluate_full_vector_baselines(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lambda_ridge: float,
    full_vector_rbf_gamma: float,
) -> Dict[str, float]:
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
    return {
        "full_vector_rbf": float(full_rbf_acc),
        "full_vector_linear": float(full_linear_acc),
    }


def _build_hpk_kernel_lists(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    degrees: List[int],
):
    KL_train = [homogeneous_polynomial_kernel(X_train, X_train, degree=d) for d in degrees]
    KL_test = [homogeneous_polynomial_kernel(X_test, X_train, degree=d) for d in degrees]
    return KL_train, KL_test


def _evaluate_mklpy_baselines(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    mkl_degrees: List[int],
    easy_mkl_lam: float,
    mkl_max_iter: int,
    mkl_tolerance: float,
) -> Dict[str, float]:
    KL_train, KL_test = _build_hpk_kernel_lists(X_train=X_train, X_test=X_test, degrees=mkl_degrees)
    y_train_np = y_train.detach().cpu().numpy().reshape(-1)
    y_test_np = y_test.detach().cpu().numpy().reshape(-1)

    out = {}
    candidates = {
        "mkl_average": AverageMKL(max_iter=int(mkl_max_iter), tolerance=float(mkl_tolerance)),
        "mkl_easy": EasyMKL(
            lam=float(easy_mkl_lam),
            max_iter=int(mkl_max_iter),
            tolerance=float(mkl_tolerance),
        ),
        "mkl_cka": CKA(max_iter=int(mkl_max_iter), tolerance=float(mkl_tolerance)),
    }

    for key, model in candidates.items():
        model.fit(KL_train, y_train_np)
        y_pred = np.asarray(model.predict(KL_test), dtype=np.float64).reshape(-1)
        out[key] = float(np.mean(y_pred == y_test_np))
    return out


def _select_features_from_alphas(
    alphas: np.ndarray,
    alpha_keep_mass: float,
    min_features_keep: int,
    max_features_keep: Optional[int],
) -> Dict:
    if alphas.ndim != 1:
        raise ValueError(f"Expected 1D alpha vector. Got shape={alphas.shape}")
    if alphas.size == 0:
        raise ValueError("Alpha vector is empty.")

    a = np.array(alphas, dtype=np.float64)
    a = np.clip(a, a_min=0.0, a_max=None)
    s = float(a.sum())
    if s <= 0.0:
        a = np.ones_like(a) / float(len(a))
    else:
        a = a / s

    order_desc = np.argsort(-a)
    cumsum = np.cumsum(a[order_desc])
    k = int(np.searchsorted(cumsum, alpha_keep_mass, side="left") + 1)
    k = max(int(min_features_keep), k)
    if max_features_keep is not None:
        k = min(k, int(max_features_keep))
    k = max(1, min(k, len(a)))

    selected_desc = order_desc[:k]
    selected_sorted = np.sort(selected_desc)
    selected_mask = np.zeros(len(a), dtype=bool)
    selected_mask[selected_sorted] = True

    threshold_alpha = float(a[selected_desc[-1]])
    selected_mass = float(a[selected_desc].sum())

    return {
        "normalized_alphas": a,
        "selected_indices_sorted": selected_sorted.tolist(),
        "selected_mask": selected_mask.tolist(),
        "num_selected": int(k),
        "selected_mass": selected_mass,
        "threshold_alpha": threshold_alpha,
    }


def run_experiement(
    dataset_name: str = "breastcancer",
    architectures: Optional[List[str]] = None,
    alpha_source: str = "all_rbf",
    alpha_keep_mass: float = 0.95,
    min_features_keep: int = 1,
    max_features_keep: Optional[int] = None,
    protocol: str = "paper_strict",
    n_splits: int = 1,
    test_size: float = 0.2,
    random_seed: int = 42,
    paper_seed: int = 123,
    frac: float = 1.0,
    train_ratio: float = 0.8,
    force_download: bool = False,
    data_cache_dir: str = "experiments/classification/_paper_data_cache",
    epochs: int = 500,
    lr: float = 0.01,
    lambda_ridge: float = 1e-4,
    full_vector_rbf_gamma: float = 0.5,
    mkl_degrees: Optional[List[int]] = None,
    easy_mkl_lam: float = 0.1,
    mkl_max_iter: int = 500,
    mkl_tolerance: float = 1e-5,
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    mkl_degrees = mkl_degrees or list(range(1, 11))
    architectures = architectures or ["all_rbf", "half_rbf_half_linear", "all_linear"]
    for arch in architectures:
        if arch not in ARCHITECTURE_LABELS:
            raise ValueError(f"Unknown architecture '{arch}'. Choices: {list(ARCHITECTURE_LABELS.keys())}")
    if protocol not in PROTOCOL_CHOICES:
        raise ValueError(f"Unknown protocol '{protocol}'. Choices: {PROTOCOL_CHOICES}")
    if alpha_source not in ALPHA_SOURCE_CHOICES:
        raise ValueError(f"Unknown alpha_source '{alpha_source}'. Choices: {ALPHA_SOURCE_CHOICES}")
    if alpha_source != "mean_all_architectures" and alpha_source not in architectures:
        raise ValueError(
            f"alpha_source='{alpha_source}' requires that architecture in --architectures. "
            f"Current architectures={architectures}"
        )
    if not (0.0 < alpha_keep_mass <= 1.0):
        raise ValueError(f"alpha_keep_mass must be in (0, 1], got {alpha_keep_mass}")
    if int(min_features_keep) < 1:
        raise ValueError(f"min_features_keep must be >= 1, got {min_features_keep}")
    if not mkl_degrees:
        raise ValueError("mkl_degrees must contain at least one degree.")
    if any(int(d) < 1 for d in mkl_degrees):
        raise ValueError(f"All mkl_degrees must be >= 1, got {mkl_degrees}")
    if int(mkl_max_iter) < 1:
        raise ValueError(f"mkl_max_iter must be >= 1, got {mkl_max_iter}")
    if float(mkl_tolerance) <= 0.0:
        raise ValueError(f"mkl_tolerance must be > 0, got {mkl_tolerance}")

    run_uuid = run_uuid or str(uuid.uuid4())
    results_dir = Path(results_root) / f"{run_uuid}_test16_alpha_pruning_mklpy_{dataset_name}"
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

    split_payloads, dataset_info, effective_n_splits = _build_split_payloads(
        dataset_name=dataset_name,
        protocol=protocol,
        n_splits=n_splits,
        test_size=test_size,
        random_seed=random_seed,
        paper_seed=paper_seed,
        frac=frac,
        train_ratio=train_ratio,
        force_download=force_download,
        data_cache_dir=data_cache_dir,
    )
    paper_reference = _load_paper_accuracy_reference(dataset_name=dataset_name)

    print(
        f"[test16] dataset={dataset_name} | protocol={protocol} | "
        f"samples={dataset_info['n_samples']} | features={dataset_info['n_features']} | "
        f"positive={dataset_info['positive_count']} | negative={dataset_info['negative_count']}"
    )

    local_model_keys = [ARCH_TO_MODEL_KEY[a] for a in architectures] + [
        "full_vector_rbf",
        "full_vector_linear",
        "mkl_average",
        "mkl_easy",
        "mkl_cka",
    ]
    pass1_acc = {k: [] for k in local_model_keys}
    pass2_acc = {k: [] for k in local_model_keys}
    pass1_loss = {ARCH_TO_MODEL_KEY[a]: [] for a in architectures}
    pass2_loss = {ARCH_TO_MODEL_KEY[a]: [] for a in architectures}

    source_alpha_vectors = []
    selected_masks = []
    selected_counts = []
    selected_masses = []
    selected_thresholds = []

    split_rows = []
    criterion = AlignmentLoss()
    first_split_plotted = False

    for split_idx, X_train_np, X_test_np, y_train_np, y_test_np in split_payloads:
        X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_np, dtype=torch.long)
        y_test_t = torch.tensor(y_test_np, dtype=torch.long)

        if not first_split_plotted and X_train_t.shape[1] >= 2:
            plt.figure(figsize=(7, 6))
            plt.scatter(
                X_train_t[:, 0].numpy(),
                X_train_t[:, 1].numpy(),
                c=y_train_t.numpy(),
                cmap="viridis",
                s=18,
                alpha=0.85,
            )
            plt.title(f"{dataset_name} (first split, first 2 standardized features)")
            plt.xlabel("Feature 0 (standardized)")
            plt.ylabel("Feature 1 (standardized)")
            finalize_plot("first_split_data_preview")
            first_split_plotted = True

        split_seed_base = int(random_seed + split_idx * 10000)

        pass1_arch = {}
        for arch_idx, arch in enumerate(architectures):
            result = _train_architecture(
                architecture=arch,
                X_train=X_train_t,
                y_train=y_train_t,
                X_test=X_test_t,
                y_test=y_test_t,
                criterion=criterion,
                epochs=epochs,
                lr=lr,
                lambda_ridge=lambda_ridge,
                split_idx=split_idx,
                effective_n_splits=effective_n_splits,
                seed=split_seed_base + arch_idx,
            )
            model_key = ARCH_TO_MODEL_KEY[arch]
            pass1_arch[arch] = result
            pass1_acc[model_key].append(float(result["accuracy"]))
            pass1_loss[model_key].append(float(result["final_alignment_loss"]))

        pass1_full = _evaluate_full_vector_baselines(
            X_train=X_train_t,
            y_train=y_train_t,
            X_test=X_test_t,
            y_test=y_test_t,
            lambda_ridge=lambda_ridge,
            full_vector_rbf_gamma=full_vector_rbf_gamma,
        )
        pass1_acc["full_vector_rbf"].append(float(pass1_full["full_vector_rbf"]))
        pass1_acc["full_vector_linear"].append(float(pass1_full["full_vector_linear"]))
        pass1_mkl = _evaluate_mklpy_baselines(
            X_train=X_train_t,
            y_train=y_train_t,
            X_test=X_test_t,
            y_test=y_test_t,
            mkl_degrees=mkl_degrees,
            easy_mkl_lam=easy_mkl_lam,
            mkl_max_iter=mkl_max_iter,
            mkl_tolerance=mkl_tolerance,
        )
        pass1_acc["mkl_average"].append(float(pass1_mkl["mkl_average"]))
        pass1_acc["mkl_easy"].append(float(pass1_mkl["mkl_easy"]))
        pass1_acc["mkl_cka"].append(float(pass1_mkl["mkl_cka"]))

        if alpha_source == "mean_all_architectures":
            source_alpha = np.mean(
                np.stack([pass1_arch[a]["alphas"] for a in architectures], axis=0),
                axis=0,
            )
        else:
            source_alpha = np.array(pass1_arch[alpha_source]["alphas"], dtype=np.float64)

        select_info = _select_features_from_alphas(
            alphas=source_alpha,
            alpha_keep_mass=alpha_keep_mass,
            min_features_keep=min_features_keep,
            max_features_keep=max_features_keep,
        )
        selected_indices = select_info["selected_indices_sorted"]
        selected_mask = np.array(select_info["selected_mask"], dtype=bool)

        source_alpha_vectors.append(select_info["normalized_alphas"])
        selected_masks.append(selected_mask.astype(np.float64))
        selected_counts.append(int(select_info["num_selected"]))
        selected_masses.append(float(select_info["selected_mass"]))
        selected_thresholds.append(float(select_info["threshold_alpha"]))

        X_train_pruned = X_train_t[:, selected_indices]
        X_test_pruned = X_test_t[:, selected_indices]

        pass2_arch = {}
        for arch_idx, arch in enumerate(architectures):
            result = _train_architecture(
                architecture=arch,
                X_train=X_train_pruned,
                y_train=y_train_t,
                X_test=X_test_pruned,
                y_test=y_test_t,
                criterion=criterion,
                epochs=epochs,
                lr=lr,
                lambda_ridge=lambda_ridge,
                split_idx=split_idx,
                effective_n_splits=effective_n_splits,
                seed=split_seed_base + 500 + arch_idx,
            )
            model_key = ARCH_TO_MODEL_KEY[arch]
            pass2_arch[arch] = result
            pass2_acc[model_key].append(float(result["accuracy"]))
            pass2_loss[model_key].append(float(result["final_alignment_loss"]))

        pass2_full = _evaluate_full_vector_baselines(
            X_train=X_train_pruned,
            y_train=y_train_t,
            X_test=X_test_pruned,
            y_test=y_test_t,
            lambda_ridge=lambda_ridge,
            full_vector_rbf_gamma=full_vector_rbf_gamma,
        )
        pass2_acc["full_vector_rbf"].append(float(pass2_full["full_vector_rbf"]))
        pass2_acc["full_vector_linear"].append(float(pass2_full["full_vector_linear"]))
        pass2_mkl = _evaluate_mklpy_baselines(
            X_train=X_train_pruned,
            y_train=y_train_t,
            X_test=X_test_pruned,
            y_test=y_test_t,
            mkl_degrees=mkl_degrees,
            easy_mkl_lam=easy_mkl_lam,
            mkl_max_iter=mkl_max_iter,
            mkl_tolerance=mkl_tolerance,
        )
        pass2_acc["mkl_average"].append(float(pass2_mkl["mkl_average"]))
        pass2_acc["mkl_easy"].append(float(pass2_mkl["mkl_easy"]))
        pass2_acc["mkl_cka"].append(float(pass2_mkl["mkl_cka"]))

        split_row = {
            "split": int(split_idx),
            "train_size": int(X_train_t.shape[0]),
            "test_size": int(X_test_t.shape[0]),
            "dimension_before": int(X_train_t.shape[1]),
            "dimension_after": int(X_train_pruned.shape[1]),
            "selected_feature_indices": [int(v) for v in selected_indices],
            "selected_feature_mass": float(select_info["selected_mass"]),
            "selected_feature_threshold_alpha": float(select_info["threshold_alpha"]),
            "alpha_source": alpha_source,
            "pass1": {
                "architectures": {
                    arch: {
                        "label": ARCHITECTURE_LABELS[arch],
                        "accuracy": float(pass1_arch[arch]["accuracy"]),
                        "final_alignment_loss": float(pass1_arch[arch]["final_alignment_loss"]),
                        "alphas": pass1_arch[arch]["alphas"].tolist(),
                    }
                    for arch in architectures
                },
                "full_vector_rbf_accuracy": float(pass1_full["full_vector_rbf"]),
                "full_vector_linear_accuracy": float(pass1_full["full_vector_linear"]),
                "mkl_average_accuracy": float(pass1_mkl["mkl_average"]),
                "mkl_easy_accuracy": float(pass1_mkl["mkl_easy"]),
                "mkl_cka_accuracy": float(pass1_mkl["mkl_cka"]),
            },
            "pass2": {
                "architectures": {
                    arch: {
                        "label": ARCHITECTURE_LABELS[arch],
                        "accuracy": float(pass2_arch[arch]["accuracy"]),
                        "final_alignment_loss": float(pass2_arch[arch]["final_alignment_loss"]),
                        "alphas": pass2_arch[arch]["alphas"].tolist(),
                    }
                    for arch in architectures
                },
                "full_vector_rbf_accuracy": float(pass2_full["full_vector_rbf"]),
                "full_vector_linear_accuracy": float(pass2_full["full_vector_linear"]),
                "mkl_average_accuracy": float(pass2_mkl["mkl_average"]),
                "mkl_easy_accuracy": float(pass2_mkl["mkl_easy"]),
                "mkl_cka_accuracy": float(pass2_mkl["mkl_cka"]),
            },
        }
        split_rows.append(split_row)

        pass1_report = ", ".join(
            [f"{ARCHITECTURE_LABELS[a]}={pass1_arch[a]['accuracy']:.4f}" for a in architectures]
            + [
                f"{MODEL_LABELS['full_vector_rbf']}={pass1_full['full_vector_rbf']:.4f}",
                f"{MODEL_LABELS['full_vector_linear']}={pass1_full['full_vector_linear']:.4f}",
                f"{MODEL_LABELS['mkl_average']}={pass1_mkl['mkl_average']:.4f}",
                f"{MODEL_LABELS['mkl_easy']}={pass1_mkl['mkl_easy']:.4f}",
                f"{MODEL_LABELS['mkl_cka']}={pass1_mkl['mkl_cka']:.4f}",
            ]
        )
        pass2_report = ", ".join(
            [f"{ARCHITECTURE_LABELS[a]}={pass2_arch[a]['accuracy']:.4f}" for a in architectures]
            + [
                f"{MODEL_LABELS['full_vector_rbf']}={pass2_full['full_vector_rbf']:.4f}",
                f"{MODEL_LABELS['full_vector_linear']}={pass2_full['full_vector_linear']:.4f}",
                f"{MODEL_LABELS['mkl_average']}={pass2_mkl['mkl_average']:.4f}",
                f"{MODEL_LABELS['mkl_easy']}={pass2_mkl['mkl_easy']:.4f}",
                f"{MODEL_LABELS['mkl_cka']}={pass2_mkl['mkl_cka']:.4f}",
            ]
        )
        print(
            f"[test16] [split={split_idx}/{effective_n_splits}] "
            f"kept={len(selected_indices)}/{X_train_t.shape[1]} (mass={select_info['selected_mass']:.4f})"
        )
        print(f"[test16] pass1 | {pass1_report}")
        print(f"[test16] pass2 | {pass2_report}")

    summary_models = {}
    for model_key in local_model_keys:
        m1, s1 = _mean_std(pass1_acc[model_key])
        m2, s2 = _mean_std(pass2_acc[model_key])
        summary_models[model_key] = {
            "label": MODEL_LABELS[model_key],
            "pass1_mean_accuracy": float(m1),
            "pass1_std_accuracy": float(s1),
            "pass1_accuracy_percent_raw": float(m1 * 100.0),
            "pass1_accuracy_percent_1dp": float(np.round(m1 * 100.0, 1)),
            "pass2_mean_accuracy": float(m2),
            "pass2_std_accuracy": float(s2),
            "pass2_accuracy_percent_raw": float(m2 * 100.0),
            "pass2_accuracy_percent_1dp": float(np.round(m2 * 100.0, 1)),
            "delta_mean_accuracy": float(m2 - m1),
            "delta_percent_points": float((m2 - m1) * 100.0),
        }
        if model_key in pass1_loss:
            l1_mean, l1_std = _mean_std(pass1_loss[model_key])
            l2_mean, l2_std = _mean_std(pass2_loss[model_key])
            summary_models[model_key]["pass1_mean_alignment_loss"] = float(l1_mean)
            summary_models[model_key]["pass1_std_alignment_loss"] = float(l1_std)
            summary_models[model_key]["pass2_mean_alignment_loss"] = float(l2_mean)
            summary_models[model_key]["pass2_std_alignment_loss"] = float(l2_std)

    source_alpha_matrix = np.array(source_alpha_vectors, dtype=np.float64)
    selected_mask_matrix = np.array(selected_masks, dtype=np.float64)
    mean_source_alpha = source_alpha_matrix.mean(axis=0)
    keep_frequency = selected_mask_matrix.mean(axis=0)
    p_full = mean_source_alpha.shape[0]
    top_k = min(20, p_full)
    top_idx = np.argsort(-mean_source_alpha)[:top_k].tolist()

    x_model = np.arange(len(local_model_keys))
    pass1_percent = [summary_models[k]["pass1_accuracy_percent_raw"] for k in local_model_keys]
    pass2_percent = [summary_models[k]["pass2_accuracy_percent_raw"] for k in local_model_keys]

    plt.figure(figsize=(12, 6))
    width = 0.38
    plt.bar(x_model - width / 2, pass1_percent, width=width, label="Pass1 (all features)")
    plt.bar(x_model + width / 2, pass2_percent, width=width, label="Pass2 (alpha-pruned features)")
    plt.xticks(x_model, [MODEL_LABELS[k] for k in local_model_keys], rotation=20, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0.0, 100.0)
    plt.title(f"Test16 Before/After Pruning Accuracy | {dataset_name}")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    finalize_plot("accuracy_before_after_by_model")

    plt.figure(figsize=(12, 5))
    deltas = [summary_models[k]["delta_percent_points"] for k in local_model_keys]
    colors = ["tab:green" if d >= 0 else "tab:red" for d in deltas]
    plt.bar(np.arange(len(local_model_keys)), deltas, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(np.arange(len(local_model_keys)), [MODEL_LABELS[k] for k in local_model_keys], rotation=20, ha="right")
    plt.ylabel("Delta Accuracy (percentage points)")
    plt.title(f"Test16 Pass2 - Pass1 Delta | {dataset_name}")
    plt.grid(axis="y", alpha=0.25)
    finalize_plot("accuracy_delta_by_model")

    plt.figure(figsize=(11, 5))
    x_idx = np.arange(p_full)
    plt.scatter(x_idx, mean_source_alpha, s=24, alpha=0.9, label="Mean source alpha")
    plt.scatter(
        x_idx[keep_frequency > 0.5],
        mean_source_alpha[keep_frequency > 0.5],
        s=36,
        alpha=0.9,
        label="Kept > 50% of splits",
    )
    plt.xlabel("Feature Index (integer)")
    plt.ylabel("Mean Source Alpha")
    plt.title(
        f"Source Alpha Profile | {dataset_name} | source={alpha_source} | keep_mass={alpha_keep_mass:.2f}"
    )
    if p_full <= 80:
        plt.xticks(list(range(p_full)))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    finalize_plot("source_alpha_profile")

    plt.figure(figsize=(11, 4))
    plt.bar(np.arange(p_full), keep_frequency, color="tab:purple", alpha=0.85)
    plt.xlabel("Feature Index (integer)")
    plt.ylabel("Keep Frequency Across Splits")
    plt.ylim(0.0, 1.05)
    plt.title(f"Feature Keep Frequency | {dataset_name}")
    if p_full <= 80:
        plt.xticks(list(range(p_full)))
    plt.grid(axis="y", alpha=0.25)
    finalize_plot("feature_keep_frequency")

    plt.figure(figsize=(8, 4))
    plt.plot(
        list(range(1, effective_n_splits + 1)),
        selected_counts,
        marker="o",
        linestyle="-",
    )
    plt.xlabel("Split")
    plt.ylabel("Number of Selected Features")
    plt.title(f"Selected Feature Count by Split | {dataset_name}")
    plt.grid(alpha=0.3)
    finalize_plot("selected_feature_count_by_split")

    summary = {
        "models": summary_models,
        "alpha_source": {
            "mode": alpha_source,
            "alpha_keep_mass": float(alpha_keep_mass),
            "mean_source_alpha": mean_source_alpha.tolist(),
            "top_feature_indices_by_mean_source_alpha": [int(v) for v in top_idx],
            "top_feature_alphas_by_mean_source_alpha": [float(mean_source_alpha[idx]) for idx in top_idx],
            "keep_frequency_by_feature": keep_frequency.tolist(),
        },
        "feature_pruning": {
            "mean_selected_feature_count": float(np.mean(selected_counts)),
            "std_selected_feature_count": float(np.std(selected_counts)),
            "mean_selected_feature_ratio": float(np.mean(selected_counts) / max(1, p_full)),
            "mean_selected_feature_mass": float(np.mean(selected_masses)),
            "mean_threshold_alpha": float(np.mean(selected_thresholds)),
        },
        "paper_reference_accuracy_percent": paper_reference,
        "notes": {
            "paper_models_after_pruning": (
                "EasyMKL/AverageMKL/CKA are retrained locally with MKLpy in both passes. "
                "Algorithm1_SMKL remains a paper reference only."
            )
        },
    }

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_name": "test16_alpha_pruning_mklpy_two_pass",
            "dataset_name": dataset_name,
            "architectures": architectures,
            "alpha_source": alpha_source,
            "alpha_keep_mass": float(alpha_keep_mass),
            "min_features_keep": int(min_features_keep),
            "max_features_keep": None if max_features_keep is None else int(max_features_keep),
            "protocol": protocol,
            "n_splits": int(n_splits),
            "effective_n_splits": int(effective_n_splits),
            "test_size": float(test_size),
            "random_seed": int(random_seed),
            "paper_seed": int(paper_seed),
            "frac": float(frac),
            "train_ratio": float(train_ratio),
            "force_download": bool(force_download),
            "data_cache_dir": data_cache_dir,
            "epochs": int(epochs),
            "lr": float(lr),
            "lambda_ridge": float(lambda_ridge),
            "full_vector_rbf_gamma": float(full_vector_rbf_gamma),
            "mkl_degrees": [int(v) for v in mkl_degrees],
            "easy_mkl_lam": float(easy_mkl_lam),
            "mkl_max_iter": int(mkl_max_iter),
            "mkl_tolerance": float(mkl_tolerance),
            "save_results": bool(save_results),
            "show_plots": bool(show_plots),
            "show_plots_at_end": bool(show_plots_at_end),
            "results_root": results_root,
            "dataset_shape": {
                "n_samples": int(dataset_info["n_samples"]),
                "n_features": int(dataset_info["n_features"]),
            },
            "class_balance": {
                "positive_count": int(dataset_info["positive_count"]),
                "negative_count": int(dataset_info["negative_count"]),
            },
            "data_source_url": dataset_info["source_url"],
            "data_local_path": dataset_info["local_path"],
            "numeric_feature_count": int(dataset_info["numeric_feature_count"]),
            "categorical_feature_count": int(dataset_info["categorical_feature_count"]),
        },
        "split_results": split_rows,
        "summary": summary,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test16: two-pass alpha-guided feature pruning with MKLpy baselines."
    )
    parser.add_argument("--dataset", type=str, default="breastcancer", choices=PAPER_DATASETS)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["all_rbf", "half_rbf_half_linear", "all_linear"],
        choices=list(ARCHITECTURE_LABELS.keys()),
    )
    parser.add_argument(
        "--alpha-source",
        type=str,
        default="all_rbf",
        choices=ALPHA_SOURCE_CHOICES,
        help="Source of alpha weights for pruning mask.",
    )
    parser.add_argument("--alpha-keep-mass", type=float, default=0.95)
    parser.add_argument("--min-features-keep", type=int, default=1)
    parser.add_argument("--max-features-keep", type=int, default=None)
    parser.add_argument("--protocol", type=str, default="paper_strict", choices=PROTOCOL_CHOICES)
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--paper-seed", type=int, default=123)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--data-cache-dir", type=str, default="experiments/classification/_paper_data_cache")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-ridge", type=float, default=1e-4)
    parser.add_argument("--full-vector-rbf-gamma", type=float, default=0.5)
    parser.add_argument(
        "--mkl-degrees",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Degrees for homogeneous polynomial base kernels used by MKLpy methods.",
    )
    parser.add_argument("--easy-mkl-lam", type=float, default=0.1)
    parser.add_argument("--mkl-max-iter", type=int, default=500)
    parser.add_argument("--mkl-tolerance", type=float, default=1e-5)
    parser.add_argument("--run-uuid", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="experiments/classification/results")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--show-plots-at-end", action="store_true")
    parser.add_argument("--no-save-results", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiement(
        dataset_name=args.dataset,
        architectures=args.architectures,
        alpha_source=args.alpha_source,
        alpha_keep_mass=args.alpha_keep_mass,
        min_features_keep=args.min_features_keep,
        max_features_keep=args.max_features_keep,
        protocol=args.protocol,
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_seed=args.random_seed,
        paper_seed=args.paper_seed,
        frac=args.frac,
        train_ratio=args.train_ratio,
        force_download=args.force_download,
        data_cache_dir=args.data_cache_dir,
        epochs=args.epochs,
        lr=args.lr,
        lambda_ridge=args.lambda_ridge,
        full_vector_rbf_gamma=args.full_vector_rbf_gamma,
        mkl_degrees=args.mkl_degrees,
        easy_mkl_lam=args.easy_mkl_lam,
        mkl_max_iter=args.mkl_max_iter,
        mkl_tolerance=args.mkl_tolerance,
        run_uuid=args.run_uuid,
        results_root=args.results_root,
        save_results=not args.no_save_results,
        show_plots=args.show_plots,
        show_plots_at_end=args.show_plots_at_end,
    )
