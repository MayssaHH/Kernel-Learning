"""
Phase 1 Result Summarizer

Reads JSONL result files from Phase 1 experiments and produces:
- A flattened CSV with one row per run
- A summary CSV with mean/std grouped by (experiment_id, dataset, model, classifier)

Usage:
    .venv/bin/python experiments/classification_v2/phase1_summarize_results.py \
        --input experiments/classification_v2/results/phase1/raw/P1_E2_repeated_seed_krr.jsonl

    .venv/bin/python experiments/classification_v2/phase1_summarize_results.py \
        --input results.jsonl --summary-output summary.csv --flat-output flat.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── JSONL loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> pd.DataFrame:
    """
    Read a JSONL file and return a flattened DataFrame.

    Nested fields from metrics, hyperparams, and diagnostics are extracted
    into top-level columns.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        metrics = rec.get("metrics") or {}
        hyperparams = rec.get("hyperparams") or {}
        diagnostics = rec.get("diagnostics") or {}
        dataset_info = diagnostics.get("dataset_info") or {}

        row = {
            # Top-level fields
            "schema_version": rec.get("schema_version"),
            "timestamp_utc": rec.get("timestamp_utc"),
            "git_commit": rec.get("git_commit"),
            "experiment_id": rec.get("experiment_id"),
            "dataset": rec.get("dataset"),
            "model": rec.get("model"),
            "classifier": rec.get("classifier"),
            "split_seed": rec.get("split_seed"),
            "model_seed": rec.get("model_seed"),
            "notes": rec.get("notes"),
            # Metrics
            "accuracy_pct": metrics.get("accuracy_pct"),
            "balanced_accuracy_pct": metrics.get("balanced_accuracy_pct"),
            "macro_f1_pct": metrics.get("macro_f1_pct"),
            # Training info
            "train_time_s": rec.get("train_time_s"),
            "final_loss": rec.get("final_loss"),
            # Hyperparams
            "epochs": hyperparams.get("epochs"),
            "lr": hyperparams.get("lr"),
            "lambda_ridge": hyperparams.get("lambda_ridge"),
            # Dataset info
            "n_samples": dataset_info.get("n_samples"),
            "n_features": dataset_info.get("n_features"),
            "train_size": dataset_info.get("train_size"),
            "test_size": dataset_info.get("test_size"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (experiment_id, dataset, model, classifier) and compute
    mean/std for key metrics.
    """
    group_cols = ["experiment_id", "dataset", "model", "classifier"]
    grouped = df.groupby(group_cols, dropna=False)

    rows = []
    for keys, g in grouped:
        key_dict = dict(zip(group_cols, keys))
        row = {
            **key_dict,
            "n_runs": int(g["accuracy_pct"].notna().sum()),
            "accuracy_pct_mean": g["accuracy_pct"].mean(),
            "accuracy_pct_std": g["accuracy_pct"].std(),
            "balanced_accuracy_pct_mean": g["balanced_accuracy_pct"].mean(),
            "balanced_accuracy_pct_std": g["balanced_accuracy_pct"].std(),
            "macro_f1_pct_mean": g["macro_f1_pct"].mean(),
            "macro_f1_pct_std": g["macro_f1_pct"].std(),
            "train_time_s_mean": g["train_time_s"].mean(),
            "train_time_s_std": g["train_time_s"].std(),
            "final_loss_mean": g["final_loss"].mean(),
            "final_loss_std": g["final_loss"].std(),
            "n_samples": g["n_samples"].dropna().iloc[0] if g["n_samples"].notna().any() else np.nan,
            "n_features": g["n_features"].dropna().iloc[0] if g["n_features"].notna().any() else np.nan,
            "train_size": g["train_size"].dropna().iloc[0] if g["train_size"].notna().any() else np.nan,
            "test_size": g["test_size"].dropna().iloc[0] if g["test_size"].notna().any() else np.nan,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Round numeric columns
    numeric_cols = [
        "accuracy_pct_mean", "accuracy_pct_std",
        "balanced_accuracy_pct_mean", "balanced_accuracy_pct_std",
        "macro_f1_pct_mean", "macro_f1_pct_std",
        "train_time_s_mean", "train_time_s_std",
        "final_loss_mean", "final_loss_std",
    ]
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(3)

    return summary


# ── Ranking ───────────────────────────────────────────────────────────────────

def add_rank_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add accuracy_rank column: rank 1 = best within each dataset."""
    df = summary_df.copy()
    df["accuracy_rank"] = (
        df.groupby("dataset", dropna=False)["accuracy_pct_mean"]
        .rank(method="min", ascending=False)
    )
    return df


# ── Formatting ────────────────────────────────────────────────────────────────

def format_mean_std(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable 'mean ± std' string columns."""
    df = summary_df.copy()

    def _fmt(mean_col, std_col):
        parts = []
        for _, row in df.iterrows():
            m = row[mean_col]
            s = row[std_col]
            if pd.isna(m):
                parts.append("")
            elif pd.isna(s):
                parts.append(f"{m:.1f} ± --")
            else:
                parts.append(f"{m:.1f} ± {s:.1f}")
        return parts

    df["accuracy_mean_std"] = _fmt("accuracy_pct_mean", "accuracy_pct_std")
    df["balanced_accuracy_mean_std"] = _fmt("balanced_accuracy_pct_mean", "balanced_accuracy_pct_std")
    df["macro_f1_mean_std"] = _fmt("macro_f1_pct_mean", "macro_f1_pct_std")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Phase 1 JSONL results into CSV tables."
    )
    parser.add_argument("--input", required=True, help="Path to JSONL result file.")
    parser.add_argument("--summary-output", default=None, help="Path for summary CSV.")
    parser.add_argument("--flat-output", default=None, help="Path for flattened raw CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return

    # Default output paths
    stem = input_path.stem
    parent = input_path.parent
    flat_path = Path(args.flat_output) if args.flat_output else parent / f"{stem}_flat.csv"
    summary_path = Path(args.summary_output) if args.summary_output else parent / f"{stem}_summary.csv"

    # Load
    df = load_jsonl(input_path)
    print(f"Loaded {len(df)} raw rows from: {input_path}")

    if df.empty:
        print("No records found. Exiting.")
        return

    # Flat CSV
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flat_path, index=False)
    print(f"Flat CSV saved ({len(df)} rows): {flat_path}")

    # Summary
    summary = summarize(df)
    summary = add_rank_columns(summary)
    summary = format_mean_std(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Summary CSV saved ({len(summary)} rows): {summary_path}")

    # Display
    print(f"\n{'─'*80}")
    print("Summary (first 20 rows):")
    print("─" * 80)
    display_cols = [
        "dataset", "model", "n_runs",
        "accuracy_mean_std", "balanced_accuracy_mean_std",
        "macro_f1_mean_std", "accuracy_rank",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
