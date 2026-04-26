"""
SVM Benchmark — all kernels evaluated with SVM (precomputed kernel).

For our learned kernels (KernelNetwork, KAN): train the kernel via CKA alignment,
then pass the precomputed kernel matrix to SVC(kernel="precomputed").

For global kernels: use standard sklearn SVC with the built-in kernels.

This gives a fair SVM comparison where every method uses the same classifier.
Results saved to results/svm/benchmark_svm_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernel_learning import KernelNetwork, RBFSubKernel, LinearSubKernel, AlignmentLoss
from kernel_learning.kernel_network.KAN_Kernel import KAN_Kernel_NX
from datasets import PAPER_DATASETS, load_uci_split, to_tensors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[benchmark_svm] device: {DEVICE}")

PAPER_RESULTS = {
    "iris":         {"EasyMKL": 100.0, "AverageMKL": 100.0, "SMKL": 100.0},
    "wine":         {"EasyMKL": 97.2,  "AverageMKL": 97.2,  "SMKL": 100.0},
    "breastcancer": {"EasyMKL": 93.0,  "AverageMKL": 92.1,  "SMKL": 98.3},
    "ionosphere":   {"EasyMKL": 73.2,  "AverageMKL": 74.6,  "SMKL": 93.0},
    "spambase":     {"EasyMKL": 90.4,  "AverageMKL": 87.6,  "SMKL": 90.9},
    "banknote":     {"EasyMKL": 100.0, "AverageMKL": 100.0, "SMKL": 100.0},
    "heart":        {"EasyMKL": 85.2,  "AverageMKL": 85.2,  "SMKL": 93.4},
    "haberman":     {"EasyMKL": 61.3,  "AverageMKL": 62.9,  "SMKL": 67.7},
    "mammographic": {"EasyMKL": 80.8,  "AverageMKL": 79.3,  "SMKL": 84.5},
    "parkinsons":   {"EasyMKL": 82.1,  "AverageMKL": 82.1,  "SMKL": 89.7},
}

_EPOCHS = {"spambase": 500}
def _epochs_for(ds): return _EPOCHS.get(ds, 800)

_KAN_EPOCHS = {"spambase": 200, "banknote": 300, "mammographic": 400}
def _kan_epochs(ds): return _KAN_EPOCHS.get(ds, 600)


# ── Train our kernels, evaluate with SVM ─────────────────────────────────────

def train_kn(arch, p, X_tr, y_tr, epochs):
    """Train KernelNetwork via CKA. Returns trained model (cpu)."""
    if arch == "all_rbf":
        subs = [RBFSubKernel(0.5) for _ in range(p)]
    elif arch == "all_linear":
        subs = [LinearSubKernel() for _ in range(p)]
    else:
        n_rbf = (p+1)//2
        subs = [RBFSubKernel(0.5) for _ in range(n_rbf)] + [LinearSubKernel() for _ in range(p-n_rbf)]

    model = KernelNetwork(subs, alpha_constraint="square", normalize_alphas=True, alpha_init="random").to(DEVICE)
    crit = AlignmentLoss()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    Xd, yd = X_tr.to(DEVICE), y_tr.to(DEVICE)
    for ep in range(epochs):
        opt.zero_grad(); loss=crit(model(Xd), yd); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % max(1, epochs//4) == 0:
            print(f"    ep={ep:4d}  loss={loss.item():.5f}")
    return model.cpu()


def train_kan(p, X_tr, y_tr, epochs):
    model = KAN_Kernel_NX(p=p, hidden_dims=[p], grid=3, k=3).to(DEVICE)
    crit = AlignmentLoss()
    opt = optim.Adam(model.parameters(), lr=3e-3)
    Xd, yd = X_tr.to(DEVICE), y_tr.to(DEVICE)
    for ep in range(epochs):
        opt.zero_grad(); loss=crit(model(Xd), yd); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % max(1, epochs//4) == 0:
            print(f"    ep={ep:4d}  loss={loss.item():.5f}")
    return model.cpu()


def eval_with_svm(model, X_tr_t, y_tr_np, X_te_t, y_te_np, C=1.0):
    """Compute precomputed kernel matrices and classify with SVM."""
    with torch.no_grad():
        K_tr = model(X_tr_t).numpy()
        K_te = model(X_te_t, X_tr_t).numpy()
    svm = SVC(kernel="precomputed", C=C)
    svm.fit(K_tr, y_tr_np)
    acc = float((svm.predict(K_te) == y_te_np).mean())
    return round(acc * 100, 3)


def run_dataset(dataset_name):
    epochs = _epochs_for(dataset_name)
    kan_ep = _kan_epochs(dataset_name)
    print(f"\n{'─'*66}")
    print(f"  {dataset_name.upper()}  [SVM benchmark]")
    print(f"{'─'*66}")

    # Load data — keep labels in {-1,+1} for SVM
    X_tr_np, y_tr_np, X_te_np, y_te_np, info = load_uci_split(dataset_name)
    # to_tensors maps to {0,1} — we need {-1,+1} for SVM labels
    X_tr_t = torch.from_numpy(X_tr_np).float()
    X_te_t = torch.from_numpy(X_te_np).float()
    # For CKA training we use {0,1}
    y_tr_01 = torch.from_numpy(((y_tr_np + 1) // 2).astype(np.int64))
    p = X_tr_np.shape[1]
    print(f"  n_train={info['train_size']}  n_test={info['test_size']}  p={p}")

    results = {}

    # ── Our learned kernels → SVM ─────────────────────────────────────────────
    for arch, label in [("all_rbf","KN+RBF+SVM"), ("all_linear","KN+Linear+SVM"), ("half_rbf_half_linear","KN+Mixed+SVM")]:
        print(f"\n  [Training] {label} (epochs={epochs}) ...")
        t0 = time.time()
        model = train_kn(arch, p, X_tr_t, y_tr_01, epochs)
        acc = eval_with_svm(model, X_tr_t, y_tr_np, X_te_t, y_te_np, C=1.0)
        elapsed = round(time.time()-t0, 2)
        results[label] = {"accuracy_pct": acc, "train_time_s": elapsed}
        print(f"  → {label:22s}  {acc:.2f}%  {elapsed:.1f}s")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── KAN → SVM ────────────────────────────────────────────────────────────
    print(f"\n  [Training] KAN+SVM (epochs={kan_ep}) ...")
    t0 = time.time()
    kan = train_kan(p, X_tr_t, y_tr_01, kan_ep)
    acc_kan = eval_with_svm(kan, X_tr_t, y_tr_np, X_te_t, y_te_np, C=1.0)
    elapsed = round(time.time()-t0, 2)
    results["KAN+SVM"] = {"accuracy_pct": acc_kan, "train_time_s": elapsed}
    print(f"  → KAN+SVM                  {acc_kan:.2f}%  {elapsed:.1f}s")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Global kernels → SVM (standard sklearn) ───────────────────────────────
    print(f"\n  [Baselines] Global kernels + SVM ...")
    gamma_scale = 1.0 / (p * max(X_tr_np.var(), 1e-8))
    for name, kwargs in [
        ("SVM-RBF",    dict(kernel="rbf",    C=10.0, gamma="scale")),
        ("SVM-Linear", dict(kernel="linear", C=1.0)),
        ("SVM-Poly3",  dict(kernel="poly",   C=1.0, degree=3, coef0=1.0)),
    ]:
        t0 = time.time()
        svm = SVC(**kwargs)
        svm.fit(X_tr_np, y_tr_np)
        acc = float(svm.score(X_te_np, y_te_np)) * 100
        results[name] = {"accuracy_pct": round(acc, 3), "train_time_s": round(time.time()-t0,3)}
        print(f"  → {name:22s}  {acc:.2f}%")

    results["_paper_baselines"] = PAPER_RESULTS.get(dataset_name, {})
    results["_dataset_info"] = info
    return results


def run_all():
    out_dir = Path(__file__).parent / "results" / "svm"
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

    path = out_dir / "benchmark_svm_results.json"
    with open(path, "w") as f:
        json.dump(_ser(all_results), f, indent=2)
    print(f"\n[benchmark_svm] Saved → {path}")
    return all_results


if __name__ == "__main__":
    run_all()
