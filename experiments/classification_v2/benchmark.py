"""
EXP-1: Full UCI Benchmark

Hypothesis: Our per-feature learnable kernel (KernelNetwork) outperforms
global kernel methods (SVM-RBF, KRR-RBF, SVM-Linear) on datasets with
heterogeneous or partially informative feature sets, and matches or exceeds
published MKL baselines (EasyMKL, AverageMKL, CKA, SMKL).

Protocol: paper_strict — single 80/20 random split, seed=123,
standardized with train statistics (ddof=1).
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    LinearSubKernel,
    RBFSubKernel,
)
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier

from baselines import (
    KRR_Global,
    linear_kernel_global,
    make_svm_linear,
    make_svm_poly,
    make_svm_rbf,
    polynomial_kernel_global,
    rbf_kernel_global,
)
from datasets import PAPER_DATASETS, load_uci_split, to_tensors

_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CPU = torch.device("cpu")
print(f"[benchmark] GPU available: {torch.cuda.is_available()}")

# Datasets whose n×p kernel stack would risk OOM on 8 GB VRAM are trained on CPU.
# Threshold: n_train > 1500 (kernel stack ≈ p × n² × 4 bytes, e.g. spambase ≈ 3 GB).
_LARGE_DATASETS = {"spambase"}

def _device_for(dataset_name: str) -> torch.device:
    return _CPU if dataset_name in _LARGE_DATASETS else _GPU

# ── Paper MKL reference numbers (Bertsimas et al., TMLR 2025, Table 2) ───────

PAPER_RESULTS = {
    "iris":         {"EasyMKL": 100.0, "AverageMKL": 100.0, "CKA": 96.7,  "SMKL": 100.0},
    "wine":         {"EasyMKL": 97.2,  "AverageMKL": 97.2,  "CKA": 91.7,  "SMKL": 100.0},
    "breastcancer": {"EasyMKL": 93.0,  "AverageMKL": 92.1,  "CKA": 94.7,  "SMKL": 98.3},
    "ionosphere":   {"EasyMKL": 73.2,  "AverageMKL": 74.6,  "CKA": 85.9,  "SMKL": 93.0},
    "spambase":     {"EasyMKL": 90.4,  "AverageMKL": 87.6,  "CKA": 81.0,  "SMKL": 90.9},
    "banknote":     {"EasyMKL": 100.0, "AverageMKL": 100.0, "CKA": 85.1,  "SMKL": 100.0},
    "heart":        {"EasyMKL": 85.2,  "AverageMKL": 85.2,  "CKA": 86.9,  "SMKL": 93.4},
    "haberman":     {"EasyMKL": 61.3,  "AverageMKL": 62.9,  "CKA": 66.1,  "SMKL": 67.7},
    "mammographic": {"EasyMKL": 80.8,  "AverageMKL": 79.3,  "CKA": 75.1,  "SMKL": 84.5},
    "parkinsons":   {"EasyMKL": 82.1,  "AverageMKL": 82.1,  "CKA": 74.4,  "SMKL": 89.7},
}

OUR_ARCH_LABELS = {
    "all_rbf":              "Ours+RBF",
    "all_linear":           "Ours+Linear",
    "half_rbf_half_linear": "Ours+Mixed",
}


# ── Model construction ────────────────────────────────────────────────────────

def build_kernel_network(arch: str, p: int) -> KernelNetwork:
    if arch == "all_rbf":
        subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(p)]
    elif arch == "all_linear":
        subs = [LinearSubKernel() for _ in range(p)]
    elif arch == "half_rbf_half_linear":
        n_rbf = (p + 1) // 2
        subs = [RBFSubKernel(initial_gamma=0.5) for _ in range(n_rbf)]
        subs += [LinearSubKernel() for _ in range(p - n_rbf)]
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return KernelNetwork(
        sub_kernels=subs,
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="random",
    )


def train_and_eval_kernel_network(
    arch: str,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
    epochs: int = 800,
    lr: float = 3e-3,
    lambda_ridge: float = 1e-4,
    device: torch.device = None,
) -> Dict:
    if device is None:
        device = _GPU
    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    X_te = X_te.to(device)
    y_te = y_te.to(device)

    p = X_tr.shape[1]
    model = build_kernel_network(arch, p).to(device)
    criterion = AlignmentLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    final_loss = 0.0

    for ep in range(epochs):
        optimizer.zero_grad()
        K = model(X_tr)
        loss = criterion(K, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = loss.item()

        if ep % 200 == 0:
            print(f"    ep={ep:4d}  loss={final_loss:.5f}")

    train_time = time.time() - t0

    # Evaluate: move to CPU for the KRR solve (numpy-backed linalg)
    model_cpu = model.cpu()
    X_tr_cpu = X_tr.cpu()
    y_tr_cpu = y_tr.cpu()
    X_te_cpu = X_te.cpu()
    y_te_cpu = y_te.cpu()

    clf = KernelRidgeClassifier(kernel=model_cpu, lambda_ridge=lambda_ridge)
    clf.fit(X_tr_cpu, y_tr_cpu)
    y_pred = clf.predict(X_te_cpu)
    acc = float(y_pred.eq(y_te_cpu).float().mean().item())

    alphas = model_cpu._get_alphas().detach().cpu().numpy()
    return {
        "accuracy_pct": round(acc * 100, 3),
        "final_loss": round(float(final_loss), 6),
        "train_time_s": round(train_time, 2),
        "alphas": alphas.tolist(),
        "top5_features": np.argsort(-alphas)[:5].tolist(),
    }


# ── Per-dataset experiment ────────────────────────────────────────────────────

def run_dataset(
    dataset_name: str,
    epochs: int = 800,
    lr: float = 3e-3,
    lambda_ridge: float = 1e-4,
) -> Dict:
    print(f"\n{'─'*66}")
    print(f"  {dataset_name.upper()}")
    print(f"{'─'*66}")

    device = _device_for(dataset_name)
    print(f"  device: {device}")

    X_tr_np, y_tr_np, X_te_np, y_te_np, info = load_uci_split(dataset_name)
    X_tr_t, y_tr_t, X_te_t, y_te_t = to_tensors(X_tr_np, y_tr_np, X_te_np, y_te_np)
    p = X_tr_np.shape[1]
    print(f"  n_train={info['train_size']:4d}  n_test={info['test_size']:3d}  p={p}")

    results: Dict = {}

    # ── Our KernelNetwork architectures ───────────────────────────────────────
    for arch, label in OUR_ARCH_LABELS.items():
        print(f"\n  [Training] {label} ...")
        r = train_and_eval_kernel_network(
            arch, X_tr_t, y_tr_t, X_te_t, y_te_t,
            epochs=epochs, lr=lr, lambda_ridge=lambda_ridge,
            device=device,
        )
        results[label] = r
        print(f"  → {label:26s}  {r['accuracy_pct']:6.2f}%  loss={r['final_loss']:.4f}  {r['train_time_s']:.1f}s")

    # ── SVM baselines ─────────────────────────────────────────────────────────
    print("\n  [Baselines] SVM / KRR ...")
    for name, clf in [
        ("SVM-RBF(C=10)",   make_svm_rbf(C=10.0, gamma="scale")),
        ("SVM-Linear(C=1)", make_svm_linear(C=1.0)),
        ("SVM-Poly3(C=1)",  make_svm_poly(degree=3, C=1.0)),
    ]:
        t0 = time.time()
        clf.fit(X_tr_np, y_tr_np)
        acc = clf.score(X_te_np, y_te_np)
        elapsed = round(time.time() - t0, 3)
        results[name] = {"accuracy_pct": round(acc * 100, 3), "train_time_s": elapsed}
        print(f"  → {name:26s}  {acc*100:6.2f}%  {elapsed:.2f}s")

    # ── KRR-Global baselines ──────────────────────────────────────────────────
    gamma_auto = float(1.0 / (p * max(X_tr_np.var(), 1e-8)))
    for name, kfn in [
        ("KRR-RBF(γ=auto)",  rbf_kernel_global(gamma=gamma_auto)),
        ("KRR-RBF(γ=0.5)",   rbf_kernel_global(gamma=0.5)),
        ("KRR-Linear",       linear_kernel_global()),
        ("KRR-Poly3",        polynomial_kernel_global(degree=3)),
    ]:
        t0 = time.time()
        krr = KRR_Global(kfn, lambda_ridge=lambda_ridge)
        krr.fit(X_tr_np, y_tr_np)
        acc = krr.score(X_te_np, y_te_np)
        elapsed = round(time.time() - t0, 3)
        results[name] = {"accuracy_pct": round(acc * 100, 3), "train_time_s": elapsed}
        print(f"  → {name:26s}  {acc*100:6.2f}%  {elapsed:.2f}s")

    results["_paper_baselines"] = PAPER_RESULTS.get(dataset_name, {})
    results["_dataset_info"] = info
    return results


# ── Full benchmark run ────────────────────────────────────────────────────────

def run_all(
    datasets: Optional[List[str]] = None,
    epochs: int = 800,
    lr: float = 3e-3,
    lambda_ridge: float = 1e-4,
    output_dir: Optional[str] = None,
) -> Dict:
    datasets = datasets or PAPER_DATASETS
    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds, epochs=epochs, lr=lr, lambda_ridge=lambda_ridge)

    def _serialise(obj):
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [_serialise(x) for x in obj]
        return obj

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(_serialise(all_results), f, indent=2)
    print(f"\n[benchmark] Results saved → {json_path}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EXP-1: UCI classification benchmark")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda-ridge", type=float, default=1e-4)
    args = parser.parse_args()

    if args.dataset:
        run_dataset(args.dataset, epochs=args.epochs, lr=args.lr, lambda_ridge=args.lambda_ridge)
    else:
        run_all(epochs=args.epochs, lr=args.lr, lambda_ridge=args.lambda_ridge)
