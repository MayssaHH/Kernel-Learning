"""
Dataset loading utilities for classification-v2.

Mirrors the data-loading protocol from the reference paper
(Bertsimas et al., TMLR 2025, https://arxiv.org/abs/2511.21890).

Protocol:
- One-shot 80/20 random permutation split, seed=123
- Standardize numeric columns using train statistics (ddof=1)
- Missing / unparseable values coerced to 0 (Julia parser parity)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

DATA_CACHE_DIR = Path(__file__).parent / "data_cache"

PAPER_DATASETS = [
    "breastcancer", "ionosphere", "spambase", "banknote",
    "haberman", "mammographic", "parkinsons", "wine", "iris", "heart",
]

# ── Dataset-specific column definitions ──────────────────────────────────────

def _paper_configs() -> Dict[str, Dict]:
    breast_names = ["id", "diagnosis"] + [f"feat{i}" for i in range(1, 31)]
    iono_names = [f"feat{i}" for i in range(1, 35)] + ["label"]
    spam_names = [f"feat{i}" for i in range(1, 58)] + ["is_spam"]
    heart_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    parkinson_names = [
        "name", "mdvp_fo", "mdvp_fhi", "mdvp_flo",
        "mdvp_jitter_percent", "mdvp_jitter_abs", "mdvp_rap", "mdvp_ppq",
        "jitter_ddp", "mdvp_shimmer", "mdvp_shimmer_dB", "shimmer_apq3",
        "shimmer_apq5", "mdvp_apq", "shimmer_dda", "nhr", "hnr", "status",
        "rpde", "dfa", "spread1", "spread2", "d2", "ppe",
    ]
    return {
        "iris": {
            "local_filename": "iris.csv",
            "has_header": False,
            "rename_map": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
            "numeric_cols": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "label_column": "species",
            "label_positive_value": "Iris-setosa",
            "label_positive_if_equal": True,
        },
        "wine": {
            "local_filename": "wine.csv",
            "has_header": False,
            "rename_map": [
                "class", "alcohol", "malic_acid", "ash", "ash_alcalinity",
                "magnesium", "total_phenols", "flavanoids", "nonflav_phenols",
                "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline",
            ],
            "numeric_cols": [
                "alcohol", "malic_acid", "ash", "ash_alcalinity", "magnesium",
                "total_phenols", "flavanoids", "nonflav_phenols", "proanthocyanins",
                "color_intensity", "hue", "od280_od315", "proline",
            ],
            "label_column": "class",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "breastcancer": {
            "local_filename": "breastcancer.csv",
            "has_header": False,
            "rename_map": breast_names,
            "numeric_cols": [f"feat{i}" for i in range(1, 31)],
            "label_column": "diagnosis",
            "label_positive_value": "M",
            "label_positive_if_equal": True,
        },
        "ionosphere": {
            "local_filename": "ionosphere.csv",
            "has_header": False,
            "rename_map": iono_names,
            "numeric_cols": [f"feat{i}" for i in range(1, 35)],
            "label_column": "label",
            "label_positive_value": "g",
            "label_positive_if_equal": True,
        },
        "spambase": {
            "local_filename": "spambase.csv",
            "has_header": False,
            "rename_map": spam_names,
            "numeric_cols": [f"feat{i}" for i in range(1, 58)],
            "label_column": "is_spam",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "banknote": {
            "local_filename": "banknote.csv",
            "has_header": False,
            "rename_map": ["variance", "skewness", "curtosis", "entropy", "class"],
            "numeric_cols": ["variance", "skewness", "curtosis", "entropy"],
            "label_column": "class",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "heart": {
            "local_filename": "heart.csv",
            "has_header": False,
            "rename_map": heart_names,
            "numeric_cols": heart_names[:-1],
            "label_column": "num",
            "label_positive_value": 0,
            "label_positive_if_equal": False,
        },
        "haberman": {
            "local_filename": "haberman.csv",
            "has_header": False,
            "rename_map": ["age", "operation_year", "positive_axillary_nodes", "survival_status"],
            "numeric_cols": ["age", "operation_year", "positive_axillary_nodes"],
            "label_column": "survival_status",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "mammographic": {
            "local_filename": "mammographic.csv",
            "has_header": False,
            "rename_map": ["BI_RADS", "age", "shape", "margin", "density", "severity"],
            "numeric_cols": ["BI_RADS", "age", "shape", "margin", "density"],
            "label_column": "severity",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
        "parkinsons": {
            "local_filename": "parkinsons.csv",
            "has_header": True,
            "rename_map": parkinson_names,
            "numeric_cols": [c for c in parkinson_names if c not in ("name", "status")],
            "label_column": "status",
            "label_positive_value": 1,
            "label_positive_if_equal": True,
        },
    }


PAPER_DATASET_CONFIGS = _paper_configs()


def _build_binary_labels(df, label_column, positive_value, positive_if_equal) -> np.ndarray:
    raw = df[label_column]
    if isinstance(positive_value, str):
        equal_mask = raw.astype(str).str.strip() == str(positive_value)
    else:
        equal_mask = pd.to_numeric(raw, errors="coerce") == float(positive_value)
    y = np.where(equal_mask.to_numpy(), 1, -1) if positive_if_equal else np.where(equal_mask.to_numpy(), -1, 1)
    return y.astype(np.int64)


def load_uci_split(
    dataset_name: str,
    train_ratio: float = 0.8,
    seed: int = 123,
    data_cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load a UCI dataset and return a reproducible paper-protocol split.

    Returns (X_train, y_train, X_test, y_test, info).
    """
    cache_dir = Path(data_cache_dir) if data_cache_dir else DATA_CACHE_DIR
    cfg = PAPER_DATASET_CONFIGS[dataset_name.lower()]
    csv_path = cache_dir / cfg["local_filename"]

    header = 0 if cfg["has_header"] else None
    df = pd.read_csv(csv_path, header=header)
    if len(cfg["rename_map"]) == df.shape[1]:
        df.columns = cfg["rename_map"]

    # Coerce numeric
    for col in cfg["numeric_cols"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce").fillna(0.0)

    y = _build_binary_labels(df, cfg["label_column"], cfg["label_positive_value"], cfg["label_positive_if_equal"])
    X = df[cfg["numeric_cols"]].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])
    train_size = int(np.floor(X.shape[0] * train_ratio))
    tr, te = idx[:train_size], idx[train_size:]

    X_tr, X_te = X[tr].copy(), X[te].copy()
    y_tr, y_te = y[tr].copy(), y[te].copy()

    # Standardize using train statistics (ddof=1, Julia parity)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0, ddof=1)
    std = np.where(np.isfinite(std) & (std > 0), std, 1e-8)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    info = {
        "dataset": dataset_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_size": int(X_tr.shape[0]),
        "test_size": int(X_te.shape[0]),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == -1).sum()),
    }
    return X_tr, y_tr, X_te, y_te, info


def to_tensors(X_tr, y_tr, X_te, y_te):
    """Convert numpy arrays to PyTorch tensors."""
    # Remap labels {-1, 1} -> {0, 1} for one-hot in KernelRidgeClassifier
    def remap(y):
        return torch.from_numpy(((y + 1) // 2).astype(np.int64))

    return (
        torch.from_numpy(X_tr).float(),
        remap(y_tr),
        torch.from_numpy(X_te).float(),
        remap(y_te),
    )
