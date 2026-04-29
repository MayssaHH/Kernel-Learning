"""
Standardized result logging utilities for Phase 1 experiments.

Provides:
- compute_classification_metrics: accuracy, balanced accuracy, macro-F1, confusion matrix
- make_result_record: unified result dict with schema version, git commit, timestamps
- append_jsonl: append one record per line to a JSONL file
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if git fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def compute_classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels (numpy array, list, or torch tensor).
        y_pred: Predicted labels (numpy array, list, or torch tensor).

    Returns:
        Dict with accuracy, balanced_accuracy, macro_f1 (as fractions and pct),
        and confusion_matrix.
    """
    y_true = _to_numpy_1d(y_true)
    y_pred = _to_numpy_1d(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 6),
        "accuracy_pct": round(acc * 100, 3),
        "balanced_accuracy": round(bal_acc, 6),
        "balanced_accuracy_pct": round(bal_acc * 100, 3),
        "macro_f1": round(macro_f1, 6),
        "macro_f1_pct": round(macro_f1 * 100, 3),
        "confusion_matrix": cm,
    }


def make_result_record(
    experiment_id: str,
    dataset: str,
    model: str,
    classifier: str,
    split_seed: Optional[int],
    model_seed: Optional[int],
    metrics: Dict[str, Any],
    hyperparams: Optional[Dict[str, Any]] = None,
    train_time_s: Optional[float] = None,
    final_loss: Optional[float] = None,
    alphas: Optional[Any] = None,
    gammas: Optional[Any] = None,
    top_features: Optional[List[int]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Build one standardized result record.

    All numpy/torch objects are converted to plain Python types for JSON safety.
    """
    record = {
        "schema_version": "phase1_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model": model,
        "classifier": classifier,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "hyperparams": _to_serializable(hyperparams),
        "metrics": _to_serializable(metrics),
        "train_time_s": _to_serializable(train_time_s),
        "final_loss": _to_serializable(final_loss),
        "alphas": _to_serializable(alphas),
        "gammas": _to_serializable(gammas),
        "top_features": _to_serializable(top_features),
        "diagnostics": _to_serializable(diagnostics),
        "notes": notes,
    }
    return record


def append_jsonl(record: Dict[str, Any], path) -> None:
    """Append a single JSON record as one line to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Private helpers ───────────────────────────────────────────────────────────


def _to_numpy_1d(obj) -> np.ndarray:
    """Convert input to a 1D numpy array."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(obj).ravel()
    return arr


def _to_serializable(obj):
    """Recursively convert numpy/torch objects to JSON-safe Python types."""
    if obj is None:
        return None

    # Try torch tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return _to_serializable(obj.detach().cpu().numpy())
    except ImportError:
        pass

    # Numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Containers
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]

    return obj


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

    metrics = compute_classification_metrics(y_true, y_pred)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    record = make_result_record(
        experiment_id="selftest_001",
        dataset="fake",
        model="all_rbf",
        classifier="KRR",
        split_seed=123,
        model_seed=42,
        metrics=metrics,
        hyperparams={"lr": 3e-3, "epochs": 800, "lambda_ridge": 1e-4},
        train_time_s=12.5,
        final_loss=-0.85,
        alphas=np.array([0.1, 0.3, 0.6]),
        gammas=np.array([0.5, 1.2, 0.8]),
        top_features=[2, 1, 0],
        diagnostics={"min_eigenvalue": 1e-5, "psd": True},
        notes="self-test run",
    )
    print("\nFull record:")
    print(json.dumps(record, indent=2))
