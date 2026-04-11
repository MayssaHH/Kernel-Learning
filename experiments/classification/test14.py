import argparse
import json
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    LinearSubKernel,
    ManualGradientTrainer,
    RBFSubKernel,
)
from kernel_learning.methods.KernelRidgeClassifier import KernelRidgeClassifier

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("test14 requires pandas. Install with: pip install pandas") from exc


PAPER_DATASETS = [
    "breastcancer",
    "ionosphere",
    "spambase",
    "banknote",
    "haberman",
    "mammographic",
    "parkinsons",
    "wine",
    "iris",
    "heart",
]

ARCHITECTURE_LABELS = {
    "all_rbf": "Archi + All RBF",
    "half_rbf_half_linear": "Archi + Half RBF/Half Linear",
    "all_linear": "Archi + All Linear",
}

PROTOCOL_CHOICES = ["paper_strict", "stratified_repeated"]


def _paper_configs() -> Dict[str, Dict]:
    breast_names = ["id", "diagnosis"] + [f"feat{i}" for i in range(1, 31)]
    iono_names = [f"feat{i}" for i in range(1, 35)] + ["label"]
    spam_names = [f"feat{i}" for i in range(1, 58)] + ["is_spam"]
    heart_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    parkinson_names = [
        "name",
        "mdvp_fo",
        "mdvp_fhi",
        "mdvp_flo",
        "mdvp_jitter_percent",
        "mdvp_jitter_abs",
        "mdvp_rap",
        "mdvp_ppq",
        "jitter_ddp",
        "mdvp_shimmer",
        "mdvp_shimmer_dB",
        "shimmer_apq3",
        "shimmer_apq5",
        "mdvp_apq",
        "shimmer_dda",
        "nhr",
        "hnr",
        "status",
        "rpde",
        "dfa",
        "spread1",
        "spread2",
        "d2",
        "ppe",
    ]
    return {
        "iris": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "local_filename": "iris.csv",
            "has_header": False,
            "rename_map": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
            "numeric_cols": ["sepal_length", "sepal_width"],
            "categorical_cols": [],
            "label_column": "species",
            "label_positive_value": "Iris-setosa",
            "label_positive_if_equal": True,
        },
        "wine": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
            "local_filename": "wine.csv",
            "has_header": False,
            "rename_map": [
                "class",
                "alcohol",
                "malic_acid",
                "ash",
                "ash_alcalinity",
                "magnesium",
                "total_phenols",
                "flavanoids",
                "nonflav_phenols",
                "proanthocyanins",
                "color_intensity",
                "hue",
                "od280_od315",
                "proline",
            ],
            "numeric_cols": [
                "alcohol",
                "malic_acid",
                "ash",
                "ash_alcalinity",
                "magnesium",
                "total_phenols",
                "flavanoids",
                "nonflav_phenols",
                "proanthocyanins",
                "color_intensity",
                "hue",
                "od280_od315",
                "proline",
            ],
            "categorical_cols": [],
            "label_column": "class",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "breastcancer": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            "local_filename": "breastcancer.csv",
            "has_header": False,
            "rename_map": breast_names,
            "numeric_cols": [f"feat{i}" for i in range(1, 31)],
            "categorical_cols": [],
            "label_column": "diagnosis",
            "label_positive_value": "M",
            "label_positive_if_equal": True,
        },
        "ionosphere": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
            "local_filename": "ionosphere.csv",
            "has_header": False,
            "rename_map": iono_names,
            "numeric_cols": [f"feat{i}" for i in range(1, 35)],
            "categorical_cols": [],
            "label_column": "label",
            "label_positive_value": "g",
            "label_positive_if_equal": True,
        },
        "spambase": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
            "local_filename": "spambase.csv",
            "has_header": False,
            "rename_map": spam_names,
            "numeric_cols": [f"feat{i}" for i in range(1, 58)],
            "categorical_cols": [],
            "label_column": "is_spam",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "banknote": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
            "local_filename": "banknote.csv",
            "has_header": False,
            "rename_map": ["variance", "skewness", "curtosis", "entropy", "class"],
            "numeric_cols": ["variance", "skewness", "curtosis", "entropy"],
            "categorical_cols": [],
            "label_column": "class",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "heart": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "local_filename": "heart.csv",
            "has_header": False,
            "rename_map": heart_names,
            "numeric_cols": heart_names[:-1],
            "categorical_cols": [],
            "label_column": "num",
            "label_positive_value": 0,
            "label_positive_if_equal": False,
        },
        "haberman": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",
            "local_filename": "haberman.csv",
            "has_header": False,
            "rename_map": ["age", "operation_year", "positive_axillary_nodes", "survival_status"],
            "numeric_cols": ["age", "operation_year", "positive_axillary_nodes"],
            "categorical_cols": [],
            "label_column": "survival_status",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "mammographic": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
            "local_filename": "mammographic.csv",
            "has_header": False,
            "rename_map": ["BI_RADS", "age", "shape", "margin", "density", "severity"],
            "numeric_cols": ["BI_RADS", "age", "shape", "margin", "density"],
            "categorical_cols": [],
            "label_column": "severity",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "parkinsons": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
            "local_filename": "parkinsons.csv",
            "has_header": True,
            "rename_map": parkinson_names,
            "numeric_cols": [c for c in parkinson_names if c not in ("name", "status")],
            "categorical_cols": [],
            "label_column": "status",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
    }


PAPER_DATASET_CONFIGS = _paper_configs()


def _download_csv_if_needed(dataset_name: str, cfg: Dict, data_cache_dir: str, force_download: bool = False) -> Path:
    cache_dir = Path(data_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / cfg["local_filename"]
    if force_download or not csv_path.exists():
        print(f"[test14] downloading {dataset_name} from {cfg['url']}")
        urllib.request.urlretrieve(cfg["url"], csv_path)
    return csv_path


def _load_raw_dataframe(
    dataset_name: str,
    force_download: bool = False,
    data_cache_dir: str = "experiments/classification/_paper_data_cache",
) -> Tuple[pd.DataFrame, Dict, str]:
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in PAPER_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choices: {PAPER_DATASETS}")
    cfg = PAPER_DATASET_CONFIGS[dataset_name]
    csv_path = _download_csv_if_needed(dataset_name, cfg, data_cache_dir, force_download=force_download)
    header = 0 if cfg["has_header"] else None
    df = pd.read_csv(csv_path, header=header)
    rename_map = cfg["rename_map"]
    if len(rename_map) == df.shape[1]:
        df.columns = rename_map
    else:
        print(
            f"[test14] warning: {dataset_name} rename_map length={len(rename_map)} "
            f"!= n_columns={df.shape[1]}"
        )
    return df, cfg, str(csv_path)


def _coerce_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        numeric = pd.to_numeric(out[col].astype(str).str.strip(), errors="coerce")
        # Matches Julia parser behavior in get_data.jl: parse failures -> 0.0
        out[col] = numeric.fillna(0.0).astype(np.float64)
    return out


def _build_binary_labels(df: pd.DataFrame, label_column: str, positive_value, positive_if_equal: bool) -> np.ndarray:
    raw = df[label_column]
    if isinstance(positive_value, str):
        equal_mask = raw.astype(str).str.strip() == str(positive_value)
    else:
        equal_mask = pd.to_numeric(raw, errors="coerce") == float(positive_value)
    if positive_if_equal:
        y = np.where(equal_mask.to_numpy(), 1, -1)
    else:
        y = np.where(equal_mask.to_numpy(), -1, 1)
    return y.astype(np.int64)


def _build_feature_matrix(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[np.ndarray, int]:
    X_num = df[numeric_cols].to_numpy(dtype=np.float64, copy=True)
    X_num = np.nan_to_num(X_num, nan=0.0)
    num_dim = X_num.shape[1]
    if categorical_cols:
        X_cat = pd.get_dummies(df[categorical_cols], drop_first=True).to_numpy(dtype=np.float64, copy=True)
    else:
        X_cat = np.zeros((X_num.shape[0], 0), dtype=np.float64)
    X = X_cat if X_num.shape[1] == 0 else np.hstack([X_num, X_cat])
    return X, num_dim


def load_paper_dataset_full(
    dataset_name: str,
    frac: float = 1.0,
    seed: int = 123,
    force_download: bool = False,
    data_cache_dir: str = "experiments/classification/_paper_data_cache",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    df, cfg, csv_path = _load_raw_dataframe(dataset_name, force_download=force_download, data_cache_dir=data_cache_dir)
    rng = np.random.RandomState(seed)
    n_rows = df.shape[0]
    keep_size = int(np.floor(n_rows * frac))
    if keep_size < n_rows:
        chosen = rng.choice(np.arange(n_rows), size=keep_size, replace=False)
        df = df.iloc[chosen].reset_index(drop=True)
    df = _coerce_numeric_columns(df, cfg["numeric_cols"])
    y = _build_binary_labels(df, cfg["label_column"], cfg["label_positive_value"], cfg["label_positive_if_equal"])
    X, num_dim = _build_feature_matrix(df, cfg["numeric_cols"], cfg["categorical_cols"])
    info = {
        "dataset_name": dataset_name,
        "source_url": cfg["url"],
        "local_path": csv_path,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "numeric_feature_count": int(num_dim),
        "categorical_feature_count": int(X.shape[1] - num_dim),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == -1).sum()),
        "frac": float(frac),
        "seed_for_row_sampling": int(seed),
    }
    return X, y, info


def load_paper_dataset_strict_split(
    dataset_name: str,
    frac: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 123,
    force_download: bool = False,
    data_cache_dir: str = "experiments/classification/_paper_data_cache",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Mirrors _tmp_paper_repo/data/get_data.jl:
    - same source csv files
    - same numeric coercion (invalid/missing -> 0)
    - same one-shot random permutation split (seed=123 by default)
    - same standardization rule on numeric columns only with sample std (ddof=1)
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    df, cfg, csv_path = _load_raw_dataframe(dataset_name, force_download=force_download, data_cache_dir=data_cache_dir)
    rng = np.random.RandomState(seed)
    n_rows = df.shape[0]
    keep_size = int(np.floor(n_rows * frac))
    if keep_size < n_rows:
        chosen = rng.choice(np.arange(n_rows), size=keep_size, replace=False)
        df = df.iloc[chosen].reset_index(drop=True)

    df = _coerce_numeric_columns(df, cfg["numeric_cols"])
    y = _build_binary_labels(df, cfg["label_column"], cfg["label_positive_value"], cfg["label_positive_if_equal"])
    X, num_dim = _build_feature_matrix(df, cfg["numeric_cols"], cfg["categorical_cols"])

    indices = rng.permutation(X.shape[0])
    train_size = int(np.floor(X.shape[0] * train_ratio))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    X_train = X[train_idx].copy()
    X_test = X[test_idx].copy()
    y_train = y[train_idx].copy()
    y_test = y[test_idx].copy()

    if num_dim > 0:
        train_mean = X_train[:, :num_dim].mean(axis=0)
        train_std = X_train[:, :num_dim].std(axis=0, ddof=1)
        train_std = np.where(np.isfinite(train_std) & (train_std > 0.0), train_std, 1e-8)
        X_train[:, :num_dim] = (X_train[:, :num_dim] - train_mean) / train_std
        X_test[:, :num_dim] = (X_test[:, :num_dim] - train_mean) / train_std

    info = {
        "dataset_name": dataset_name,
        "source_url": cfg["url"],
        "local_path": csv_path,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "numeric_feature_count": int(num_dim),
        "categorical_feature_count": int(X.shape[1] - num_dim),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == -1).sum()),
        "frac": float(frac),
        "train_ratio": float(train_ratio),
        "seed_for_sampling_and_split": int(seed),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    return X_train, y_train, X_test, y_test, info


def build_model(architecture: str, dimension: int) -> KernelNetwork:
    if architecture == "all_rbf":
        sub_kernels = [RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(dimension)]
    elif architecture == "all_linear":
        sub_kernels = [LinearSubKernel(initial_sigma=0.5, random=True) for _ in range(dimension)]
    elif architecture == "half_rbf_half_linear":
        num_rbf = (dimension + 1) // 2
        num_linear = dimension - num_rbf
        sub_kernels = [RBFSubKernel(initial_gamma=0.5, random=True) for _ in range(num_rbf)]
        sub_kernels += [LinearSubKernel(initial_sigma=0.5, random=True) for _ in range(num_linear)]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return KernelNetwork(
        sub_kernels=sub_kernels,
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )


def evaluate_model(
    model: KernelNetwork,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lambda_ridge: float,
) -> float:
    clf = KernelRidgeClassifier(learnt_kernel=model, lambda_ridge=lambda_ridge)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred.eq(y_test).float().mean().item()


def _collect_historical_test14_scores(
    results_root: str,
    dataset_name: str,
    current_run_uuid: str,
) -> Dict[str, List[float]]:
    """
    Collect previous test14 table-accuracy (%) values for the same dataset.
    Returns: {architecture_key: [accuracy_percent, ...]}.
    """
    root = Path(results_root)
    if not root.exists():
        return {}

    out: Dict[str, List[float]] = {}
    for run_dir in root.glob(f"*_test14_uci_{dataset_name}"):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if str(manifest.get("run_uuid", "")) == str(current_run_uuid):
            continue
        config = manifest.get("config", {})
        if config.get("dataset_name") != dataset_name:
            continue
        if config.get("experiment_name") != "test14_external_benchmark_dataset_runner":
            continue

        summary = manifest.get("summary", {})
        if not isinstance(summary, dict):
            continue
        for arch, stats in summary.items():
            if not isinstance(stats, dict):
                continue
            val = stats.get("table_accuracy_percent_raw")
            if val is None and stats.get("mean_accuracy") is not None:
                val = float(stats["mean_accuracy"]) * 100.0
            if val is None:
                continue
            try:
                val_f = float(val)
            except Exception:
                continue
            if np.isfinite(val_f):
                out.setdefault(arch, []).append(val_f)

    return out


def run_experiement(
    dataset_name: str = "breastcancer",
    architectures: Optional[List[str]] = None,
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
    run_uuid: Optional[str] = None,
    results_root: str = "experiments/classification/results",
    save_results: bool = True,
    show_plots: bool = True,
    show_plots_at_end: bool = True,
):
    architectures = architectures or ["all_rbf"]
    for arch in architectures:
        if arch not in ARCHITECTURE_LABELS:
            raise ValueError(f"Unknown architecture '{arch}'. Choices: {list(ARCHITECTURE_LABELS.keys())}")
    if protocol not in PROTOCOL_CHOICES:
        raise ValueError(f"Unknown protocol '{protocol}'. Choices: {PROTOCOL_CHOICES}")

    run_uuid = run_uuid or str(uuid.uuid4())
    results_dir = Path(results_root) / f"{run_uuid}_test14_uci_{dataset_name}"
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

    split_payloads = []
    if protocol == "paper_strict":
        if n_splits != 1:
            print(f"[test14] protocol=paper_strict uses exactly 1 split. Ignoring n_splits={n_splits}.")
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

    print(
        f"[test14] dataset={dataset_name} | protocol={protocol} | "
        f"samples={dataset_info['n_samples']} | features={dataset_info['n_features']} | "
        f"positive={dataset_info['positive_count']} | negative={dataset_info['negative_count']}"
    )

    criterion = AlignmentLoss()
    split_rows = []
    per_arch_accuracies = {arch: [] for arch in architectures}
    per_arch_alpha_vectors = {arch: [] for arch in architectures}

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

        split_row = {
            "split": int(split_idx),
            "train_size": int(X_train_t.shape[0]),
            "test_size": int(X_test_t.shape[0]),
            "architectures": {},
        }

        for arch in architectures:
            model = build_model(architecture=arch, dimension=X_train_t.shape[1])
            trainer = ManualGradientTrainer(lr=lr)
            final_loss = None
            for epoch in range(epochs):
                final_loss = trainer.train_epoch(
                    model=model,
                    X=X_train_t,
                    criterion=criterion,
                    lambda_lasso=0.0,
                    y=y_train_t,
                )
                if epoch % 100 == 0:
                    print(
                        f"[test14] [{dataset_name}] [split={split_idx}/{effective_n_splits}] "
                        f"Epoch {epoch:>3} | {ARCHITECTURE_LABELS[arch]} alignment loss: {final_loss:.6f}"
                    )

            accuracy = evaluate_model(
                model=model,
                X_train=X_train_t,
                y_train=y_train_t,
                X_test=X_test_t,
                y_test=y_test_t,
                lambda_ridge=lambda_ridge,
            )
            alphas = model._get_alphas().detach().cpu().numpy()
            per_arch_accuracies[arch].append(float(accuracy))
            per_arch_alpha_vectors[arch].append(alphas.tolist())
            top_k = min(10, len(alphas))
            top_indices = np.argsort(-alphas)[:top_k].tolist()
            split_row["architectures"][arch] = {
                "label": ARCHITECTURE_LABELS[arch],
                "accuracy": float(accuracy),
                "final_alignment_loss": float(final_loss),
                "alphas": alphas.tolist(),
                "top_feature_indices": top_indices,
                "top_feature_alphas": [float(alphas[idx]) for idx in top_indices],
            }
            print(
                f"[test14] [{dataset_name}] [split={split_idx}/{effective_n_splits}] Final | "
                f"{ARCHITECTURE_LABELS[arch]} accuracy: {accuracy:.4f}"
            )

        split_rows.append(split_row)

    plt.figure(figsize=(10, 6))
    split_axis = list(range(1, effective_n_splits + 1))
    for arch in architectures:
        plt.plot(split_axis, per_arch_accuracies[arch], marker="o", label=ARCHITECTURE_LABELS[arch])
    plt.xlabel("Split")
    plt.ylabel("Test Accuracy")
    plt.title(f"Test14 Accuracy by Split | {dataset_name}")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    finalize_plot("accuracy_by_split")

    plt.figure(figsize=(9, 6))
    box_data = [per_arch_accuracies[arch] for arch in architectures]
    box_labels = [ARCHITECTURE_LABELS[arch] for arch in architectures]
    plt.boxplot(box_data, tick_labels=box_labels)
    plt.ylabel("Test Accuracy")
    plt.title(f"Test14 Accuracy Distribution | {dataset_name}")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    finalize_plot("accuracy_boxplot")

    for arch in architectures:
        alpha_matrix = np.array(per_arch_alpha_vectors[arch], dtype=np.float64)
        mean_alpha = alpha_matrix.mean(axis=0)
        p = mean_alpha.shape[0]
        x_idx = np.arange(p)

        plt.figure(figsize=(11, 5))
        plt.scatter(x_idx, mean_alpha, s=28, alpha=0.9)
        plt.xlabel("Feature Index (integer)")
        plt.ylabel("Mean Alpha (across splits)")
        plt.title(f"Mean Alpha by Feature | {dataset_name} | {ARCHITECTURE_LABELS[arch]}")
        if p <= 60:
            plt.xticks(list(range(p)))
        plt.grid(axis="y", alpha=0.25)
        finalize_plot(f"mean_alphas_{arch}")

        if p > 120:
            top_idx = np.argsort(-mean_alpha)[:120]
            heat = alpha_matrix[:, top_idx]
            xlabel = "Top 120 features by mean alpha"
        else:
            heat = alpha_matrix
            xlabel = "Feature Index"
        plt.figure(figsize=(12, 4))
        im = plt.imshow(heat, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Alpha Value")
        plt.xlabel(xlabel)
        plt.ylabel("Split")
        plt.title(f"Alpha Heatmap | {dataset_name} | {ARCHITECTURE_LABELS[arch]}")
        finalize_plot(f"alpha_heatmap_{arch}")

    summary = {}
    for arch in architectures:
        vals = np.array(per_arch_accuracies[arch], dtype=np.float64)
        mean_alpha = np.array(per_arch_alpha_vectors[arch], dtype=np.float64).mean(axis=0)
        top_k = min(15, mean_alpha.shape[0])
        top_indices = np.argsort(-mean_alpha)[:top_k].tolist()
        table_percent = float(vals.mean() * 100.0)
        summary[arch] = {
            "label": ARCHITECTURE_LABELS[arch],
            "mean_accuracy": float(vals.mean()),
            "std_accuracy": float(vals.std()),
            "best_split_accuracy": float(vals.max()),
            "worst_split_accuracy": float(vals.min()),
            "table_accuracy_percent_raw": table_percent,
            "table_accuracy_percent_1dp": float(np.round(table_percent, 1)),
            "mean_alphas": mean_alpha.tolist(),
            "top_feature_indices_by_mean_alpha": top_indices,
            "top_feature_alphas_by_mean_alpha": [float(mean_alpha[idx]) for idx in top_indices],
        }

    # Compare this run against all previous test14 runs on the same dataset.
    historical_scores = _collect_historical_test14_scores(
        results_root=results_root,
        dataset_name=dataset_name,
        current_run_uuid=run_uuid,
    )
    if architectures:
        n_arch = len(architectures)
        n_cols = 2 if n_arch > 1 else 1
        n_rows = int(np.ceil(n_arch / n_cols))
        plt.figure(figsize=(7 * n_cols, 4.5 * n_rows))
        for idx, arch in enumerate(architectures, start=1):
            plt.subplot(n_rows, n_cols, idx)
            peers = historical_scores.get(arch, [])
            current = float(summary[arch]["table_accuracy_percent_raw"])

            if peers:
                bins = max(3, min(12, int(np.sqrt(len(peers))) + 1))
                plt.hist(
                    peers,
                    bins=bins,
                    color="#4C78A8",
                    alpha=0.75,
                    edgecolor="white",
                    label=f"Other runs (n={len(peers)})",
                )
            else:
                plt.text(
                    0.5,
                    0.55,
                    "No previous runs\nfor this architecture",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
            plt.axvline(
                current,
                color="#E45756",
                linewidth=2.0,
                linestyle="--",
                label=f"This run: {current:.2f}%",
            )
            plt.xlim(0.0, 100.0)
            plt.xlabel("Accuracy (%)")
            plt.ylabel("Count")
            plt.title(f"{ARCHITECTURE_LABELS.get(arch, arch)} | {dataset_name}")
            plt.grid(axis="y", alpha=0.25)
            plt.legend()
        plt.suptitle(f"Historical Accuracy Histogram vs Other Runs | {dataset_name}", y=1.02)
        finalize_plot("accuracy_histogram_vs_other_runs")

    manifest = {
        "run_uuid": run_uuid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_name": "test14_external_benchmark_dataset_runner",
            "paper_reference": "https://arxiv.org/html/2511.21890",
            "paper_repo_reference_loader": "_tmp_paper_repo/data/get_data.jl",
            "dataset_name": dataset_name,
            "architectures": architectures,
            "protocol": protocol,
            "n_splits": n_splits,
            "effective_n_splits": effective_n_splits,
            "test_size": test_size,
            "random_seed": random_seed,
            "paper_seed": paper_seed,
            "frac": frac,
            "train_ratio": train_ratio,
            "force_download": force_download,
            "data_cache_dir": data_cache_dir,
            "epochs": epochs,
            "lr": lr,
            "lambda_ridge": lambda_ridge,
            "save_results": save_results,
            "show_plots": show_plots,
            "show_plots_at_end": show_plots_at_end,
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
        "historical_context": {
            arch: {"num_other_runs_same_dataset": int(len(historical_scores.get(arch, [])))}
            for arch in architectures
        },
        "artifacts": {"plots": saved_plot_paths},
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
        description="Test14: Run our architecture(s) on one selected UCI paper dataset."
    )
    parser.add_argument("--dataset", type=str, default="breastcancer", choices=PAPER_DATASETS)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["all_rbf"],
        choices=list(ARCHITECTURE_LABELS.keys()),
    )
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
        run_uuid=args.run_uuid,
        results_root=args.results_root,
        save_results=not args.no_save_results,
        show_plots=args.show_plots,
        show_plots_at_end=args.show_plots_at_end,
    )
