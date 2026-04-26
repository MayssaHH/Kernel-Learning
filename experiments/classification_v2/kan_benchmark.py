"""
KAN_Kernel_NX — EXP-1: Full UCI Benchmark

Same protocol as benchmark.py but uses KAN_Kernel_NX instead of KernelNetwork.
K(u,v) = ψ(u)·ψ(v)  where ψ: R^p → R is a KAN (B-spline activations).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernel_learning.kernel_network.KAN_Kernel import KAN_Kernel_NX
from kernel_learning.losses.alignment import AlignmentLoss
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier
from baselines import KRR_Global, make_svm_rbf, make_svm_linear, make_svm_poly, rbf_kernel_global, linear_kernel_global, polynomial_kernel_global
from datasets import PAPER_DATASETS, load_uci_split, to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[kan_benchmark] device: {DEVICE}")

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

# Epoch budget per dataset (KAN is slower than KernelNetwork)
_EPOCHS = {
    "spambase":  200,
    "banknote":  300,
    "mammographic": 400,
}
def _epochs_for(ds): return _EPOCHS.get(ds, 600)


def build_kan(p: int) -> KAN_Kernel_NX:
    return KAN_Kernel_NX(p=p, hidden_dims=[p], grid=3, k=3)


def train_and_eval_kan(X_tr, y_tr, X_te, y_te, epochs, lr=3e-3, lambda_ridge=1e-4):
    p = X_tr.shape[1]
    model = build_kan(p).to(DEVICE)
    crit = AlignmentLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    X_tr_d = X_tr.to(DEVICE)
    y_tr_d = y_tr.to(DEVICE)

    t0 = time.time()
    final_loss = 0.0
    for ep in range(epochs):
        opt.zero_grad()
        K = model(X_tr_d)
        loss = crit(K, y_tr_d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = loss.item()
        if ep % max(1, epochs // 4) == 0:
            print(f"    ep={ep:4d}  loss={final_loss:.5f}")

    train_time = time.time() - t0
    model_cpu = model.cpu()
    clf = KernelRidgeClassifier(kernel=model_cpu, lambda_ridge=lambda_ridge)
    clf.fit(X_tr.cpu(), y_tr.cpu())
    acc = float(clf.predict(X_te.cpu()).eq(y_te.cpu()).float().mean())

    return {
        "accuracy_pct": round(acc * 100, 3),
        "final_loss": round(float(final_loss), 6),
        "train_time_s": round(train_time, 2),
    }


def run_dataset(dataset_name, lr=3e-3, lambda_ridge=1e-4):
    epochs = _epochs_for(dataset_name)
    print(f"\n{'─'*66}")
    print(f"  {dataset_name.upper()}  (KAN, epochs={epochs})")
    print(f"{'─'*66}")

    X_tr_np, y_tr_np, X_te_np, y_te_np, info = load_uci_split(dataset_name)
    X_tr_t, y_tr_t, X_te_t, y_te_t = to_tensors(X_tr_np, y_tr_np, X_te_np, y_te_np)
    p = X_tr_np.shape[1]
    print(f"  n_train={info['train_size']:4d}  n_test={info['test_size']:3d}  p={p}")

    results = {}

    print(f"\n  [Training] KAN+KRR ...")
    r = train_and_eval_kan(X_tr_t, y_tr_t, X_te_t, y_te_t, epochs=epochs, lr=lr, lambda_ridge=lambda_ridge)
    results["KAN+KRR"] = r
    print(f"  → KAN+KRR  {r['accuracy_pct']:6.2f}%  loss={r['final_loss']:.4f}  {r['train_time_s']:.1f}s")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  [Baselines] SVM / KRR ...")
    gamma_auto = float(1.0 / (p * max(X_tr_np.var(), 1e-8)))
    for name, clf in [
        ("SVM-RBF(C=10)",   make_svm_rbf(C=10.0, gamma="scale")),
        ("SVM-Linear(C=1)", make_svm_linear(C=1.0)),
        ("SVM-Poly3(C=1)",  make_svm_poly(degree=3, C=1.0)),
    ]:
        t0 = time.time()
        clf.fit(X_tr_np, y_tr_np)
        acc = clf.score(X_te_np, y_te_np)
        results[name] = {"accuracy_pct": round(acc * 100, 3), "train_time_s": round(time.time()-t0, 3)}
        print(f"  → {name:26s}  {acc*100:6.2f}%  {time.time()-t0:.2f}s")

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
        results[name] = {"accuracy_pct": round(acc * 100, 3), "train_time_s": round(time.time()-t0, 3)}
        print(f"  → {name:26s}  {acc*100:6.2f}%  {time.time()-t0:.2f}s")

    results["_paper_baselines"] = PAPER_RESULTS.get(dataset_name, {})
    results["_dataset_info"] = info
    return results


def run_all(output_dir=None):
    out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds in PAPER_DATASETS:
        all_results[ds] = run_dataset(ds)

    def _ser(obj):
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, list): return [_ser(x) for x in obj]
        return obj

    path = out_dir / "kan_benchmark_results.json"
    with open(path, "w") as f:
        json.dump(_ser(all_results), f, indent=2)
    print(f"\n[kan_benchmark] Saved → {path}")
    return all_results


if __name__ == "__main__":
    run_all()
