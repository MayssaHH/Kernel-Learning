"""
Master orchestrator: runs all classification-v2 experiments in sequence.

Usage:
    python run_all.py                    # full suite
    python run_all.py --skip-benchmark   # skip EXP-1 (uses existing JSON)
    python run_all.py --fast             # reduced grid / fewer epochs for testing
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmark",  action="store_true")
    parser.add_argument("--skip-boundaries", action="store_true")
    parser.add_argument("--skip-alpha",      action="store_true")
    parser.add_argument("--skip-snr",        action="store_true")
    parser.add_argument("--fast",            action="store_true")
    args = parser.parse_args()

    epochs  = 300 if args.fast else 800
    lr      = 3e-3
    n_runs  = 3   if args.fast else 5

    # ── EXP-1: UCI Benchmark ──────────────────────────────────────────────────
    if not args.skip_benchmark:
        print("\n" + "="*70)
        print("EXP-1: UCI Benchmark")
        print("="*70)
        from benchmark import run_all
        run_all(epochs=epochs, lr=lr, lambda_ridge=1e-4)

    # ── EXP-1 Figures ─────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "benchmark_results.json"
    if json_path.exists():
        print("\n" + "="*70)
        print("Figures: Benchmark visualizations")
        print("="*70)
        from figures_benchmark import run as run_figures
        run_figures(str(json_path))

    # ── EXP-2: Decision Boundaries ────────────────────────────────────────────
    if not args.skip_boundaries:
        print("\n" + "="*70)
        print("EXP-2: Decision Boundary Visualization")
        print("="*70)
        from decision_boundary import run as run_db
        run_db()

    # ── EXP-3: Alpha Analysis ─────────────────────────────────────────────────
    if not args.skip_alpha:
        print("\n" + "="*70)
        print("EXP-3: Alpha Weight Analysis")
        print("="*70)
        from alpha_analysis import run as run_alpha
        run_alpha()

    # ── EXP-4: SNR Phase Diagram ──────────────────────────────────────────────
    if not args.skip_snr:
        print("\n" + "="*70)
        print("EXP-4: SNR Phase Diagram")
        print("="*70)
        from snr_sweep import run_sweep

        snr_vals = [1.0, 2.0, 3.0]       if args.fast else [0.5, 1.0, 1.5, 2.0, 3.0]
        pn_vals  = [0, 5, 10]            if args.fast else [0, 2, 5, 10, 20]
        run_sweep(snr_values=snr_vals, pn_values=pn_vals, n_runs=n_runs)

    print("\n" + "="*70)
    print("All experiments complete.")
    print(f"Figures in: {Path(__file__).parent / 'figures'}")
    print("="*70)


if __name__ == "__main__":
    main()
