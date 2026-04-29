"""
P1-E3: Uniform Alpha Ablation

Key question: Does learning α_k improve performance compared with α_k = 1/p?

Compares:
  - Learned alpha: α_k trained via gradient descent (starts at uniform)
  - Uniform alpha: α_k frozen at 1/p throughout training

Both modes use alpha_init="ones" so the starting point is identical.
The only difference is whether raw_alphas receives gradients.

Usage:
    .venv/bin/python experiments/classification_v2/phase1_uniform_alpha_ablation.py
    .venv/bin/python experiments/classification_v2/phase1_uniform_alpha_ablation.py \
        --archs all_rbf all_linear --alpha-modes learned uniform --seeds 123 124 125
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from kernel_learning import (
    AlignmentLoss,
    KernelNetwork,
    LinearSubKernel,
    RBFSubKernel,
)
from kernel_learning.methods.kernel_ridge_classifier import KernelRidgeClassifier

from datasets import PAPER_DATASETS, load_uci_split, to_tensors
from result_schema import append_jsonl, compute_classification_metrics, make_result_record


# ── Model name mapping ────────────────────────────────────────────────────────

ARCH_MODEL_NAMES = {
    ("all_rbf", "learned"): "Additive-RBF-LearnedAlpha",
    ("all_rbf", "uniform"): "Additive-RBF-UniformAlpha",
    ("all_linear", "learned"): "Additive-Linear-LearnedAlpha",
    ("all_linear", "uniform"): "Additive-Linear-UniformAlpha",
    ("half_rbf_half_linear", "learned"): "Additive-Mixed-LearnedAlpha",
    ("half_rbf_half_linear", "uniform"): "Additive-Mixed-UniformAlpha",
}

# Override epochs for large datasets
_EPOCHS_OVERRIDE = {
    "spambase": 500,
}


# ── Determinism ───────────────────────────────────────────────────────────────

def set_model_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model construction ────────────────────────────────────────────────────────

def build_kernel_network(arch: str, p: int, alpha_mode: str) -> KernelNetwork:
    """
    Build a KernelNetwork with uniform alpha initialization.

    Args:
        arch: one of all_rbf, all_linear, half_rbf_half_linear
        p: number of features
        alpha_mode: "learned" (alphas trainable) or "uniform" (alphas frozen at 1/p)
    """
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

    model = KernelNetwork(
        sub_kernels=subs,
        alpha_constraint="square",
        normalize_alphas=True,
        alpha_init="ones",
    )

    if alpha_mode == "uniform":
        with torch.no_grad():
            model.raw_alphas.fill_(1.0)
        model.raw_alphas.requires_grad_(False)
    elif alpha_mode == "learned":
        pass  # raw_alphas remain trainable
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

    return model


# ── Training and evaluation ───────────────────────────────────────────────────

def train_eval_one(
    dataset_name: str,
    arch: str,
    alpha_mode: str,
    split_seed: int,
    model_seed: int,
    epochs: int,
    lr: float,
    lambda_ridge: float,
    device: torch.device,
) -> dict:
    """Train one model on one split and return a standardized result record."""

    set_model_seed(model_seed)

    # Load data
    X_tr_np, y_tr_np, X_te_np, y_te_np, info = load_uci_split(dataset_name, seed=split_seed)
    X_tr_t, y_tr_t, X_te_t, y_te_t = to_tensors(X_tr_np, y_tr_np, X_te_np, y_te_np)

    p = X_tr_np.shape[1]
    eff_epochs = _EPOCHS_OVERRIDE.get(dataset_name, epochs)

    # Build model
    model = build_kernel_network(arch, p, alpha_mode).to(device)
    criterion = AlignmentLoss()

    # Only optimize parameters that require grad
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    num_trainable_params = len(trainable_params)

    X_tr_d = X_tr_t.to(device)
    y_tr_d = y_tr_t.to(device)

    # Training loop
    t0 = time.time()
    final_loss = 0.0

    if num_trainable_params > 0:
        optimizer = optim.Adam(trainable_params, lr=lr)
        for ep in range(eff_epochs):
            optimizer.zero_grad()
            K = model(X_tr_d)
            loss = criterion(K, y_tr_d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            final_loss = loss.item()
    else:
        # No trainable params — just compute forward + loss once
        with torch.no_grad():
            K = model(X_tr_d)
            final_loss = criterion(K, y_tr_d).item()

    train_time_s = time.time() - t0

    # Move to CPU for evaluation
    model_cpu = model.cpu()
    X_tr_cpu = X_tr_t.cpu()
    y_tr_cpu = y_tr_t.cpu()
    X_te_cpu = X_te_t.cpu()
    y_te_cpu = y_te_t.cpu()

    # Predict
    clf = KernelRidgeClassifier(kernel=model_cpu, lambda_ridge=lambda_ridge)
    clf.fit(X_tr_cpu, y_tr_cpu)
    y_pred = clf.predict(X_te_cpu)

    # Metrics
    metrics = compute_classification_metrics(y_te_cpu, y_pred)

    # Extract alphas
    alphas = model_cpu._get_alphas().detach().cpu().numpy()
    top5_features = np.argsort(-alphas)[:5].tolist()

    # Extract gammas from RBF sub-kernels
    gammas = []
    for sk in model_cpu.sub_kernels:
        if hasattr(sk, "gamma"):
            gammas.append(float(sk.gamma.item()))
        else:
            gammas.append(None)

    # Model name
    model_name = ARCH_MODEL_NAMES[(arch, alpha_mode)]

    # Build result record
    record = make_result_record(
        experiment_id="P1-E3",
        dataset=dataset_name,
        model=model_name,
        classifier="KRR",
        split_seed=split_seed,
        model_seed=model_seed,
        metrics=metrics,
        hyperparams={
            "arch": arch,
            "alpha_mode": alpha_mode,
            "epochs": eff_epochs,
            "lr": lr,
            "lambda_ridge": lambda_ridge,
            "initial_gamma": 0.5,
            "alpha_constraint": "square",
            "normalize_alphas": True,
            "alpha_init": "ones",
            "raw_alphas_trainable": (alpha_mode == "learned"),
            "num_trainable_params": num_trainable_params,
            "has_trainable_params": (num_trainable_params > 0),
        },
        train_time_s=round(train_time_s, 2),
        final_loss=round(final_loss, 6),
        alphas=alphas,
        gammas=gammas,
        top_features=top5_features,
        diagnostics={"dataset_info": info},
        notes="",
    )
    return record


# ── Experiment loop ───────────────────────────────────────────────────────────

def run_experiment(
    datasets: List[str],
    archs: List[str],
    alpha_modes: List[str],
    split_seeds: List[int],
    model_seed: int,
    epochs: int,
    lr: float,
    lambda_ridge: float,
    output_path: Path,
    device: torch.device,
):
    """Run all (dataset, arch, alpha_mode, seed) combinations and save to JSONL."""
    total = len(datasets) * len(archs) * len(alpha_modes) * len(split_seeds)
    done = 0

    for ds in datasets:
        for arch in archs:
            for alpha_mode in alpha_modes:
                for seed in split_seeds:
                    done += 1
                    label = ARCH_MODEL_NAMES.get((arch, alpha_mode), f"{arch}/{alpha_mode}")
                    print(f"\n[{done}/{total}] {ds} | {label} | split_seed={seed}")

                    try:
                        record = train_eval_one(
                            dataset_name=ds,
                            arch=arch,
                            alpha_mode=alpha_mode,
                            split_seed=seed,
                            model_seed=model_seed,
                            epochs=epochs,
                            lr=lr,
                            lambda_ridge=lambda_ridge,
                            device=device,
                        )
                        append_jsonl(record, output_path)

                        m = record["metrics"]
                        print(
                            f"  acc={m['accuracy_pct']:.1f}%  "
                            f"bal_acc={m['balanced_accuracy_pct']:.1f}%  "
                            f"f1={m['macro_f1_pct']:.1f}%  "
                            f"time={record['train_time_s']:.1f}s"
                        )

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        traceback.print_exc()
                        error_record = make_result_record(
                            experiment_id="P1-E3",
                            dataset=ds,
                            model=ARCH_MODEL_NAMES.get((arch, alpha_mode), f"{arch}/{alpha_mode}"),
                            classifier="KRR",
                            split_seed=seed,
                            model_seed=model_seed,
                            metrics={},
                            notes=f"ERROR: {str(e)}",
                        )
                        append_jsonl(error_record, output_path)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print(f"\n{'─'*66}")
    print(f"Done. {done} runs saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="P1-E3: Uniform alpha ablation — learned vs frozen α_k = 1/p"
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--archs", nargs="+", default=["all_rbf"])
    parser.add_argument("--alpha-modes", nargs="+", default=["learned", "uniform"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--output", type=str,
        default="experiments/classification_v2/results/phase1/raw/P1_E3_uniform_alpha_ablation.jsonl",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else PAPER_DATASETS
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    output_path = Path(ROOT) / args.output

    # Handle existing output file
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
            print(f"Removed existing output file: {output_path}")
        else:
            print(
                "WARNING: output file already exists; new records will be appended. "
                "Use --overwrite to start fresh."
            )

    # Startup summary
    print("=" * 66)
    print("P1-E3: Uniform Alpha Ablation")
    print("=" * 66)
    print(f"  Device:       {device}")
    print(f"  Datasets:     {datasets}")
    print(f"  Archs:        {args.archs}")
    print(f"  Alpha modes:  {args.alpha_modes}")
    print(f"  Split seeds:  {args.seeds}")
    print(f"  Model seed:   {args.model_seed}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  LR:           {args.lr}")
    print(f"  λ_ridge:      {args.lambda_ridge}")
    print(f"  Output:       {output_path}")
    print("=" * 66)

    run_experiment(
        datasets=datasets,
        archs=args.archs,
        alpha_modes=args.alpha_modes,
        split_seeds=args.seeds,
        model_seed=args.model_seed,
        epochs=args.epochs,
        lr=args.lr,
        lambda_ridge=args.lambda_ridge,
        output_path=output_path,
        device=device,
    )


if __name__ == "__main__":
    main()
