#!/usr/bin/env bash
# =============================================================================
# run_all_figures.sh — generate ALL figures + SVM benchmark in one detached run
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$REPO/experiments/classification_v2"
RES="$EXP/results"
PYTHON="$REPO/.venv/bin/python"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$EXP"
mkdir -p "$RES/svm" "$EXP/figures/krr" "$EXP/figures/svm" "$EXP/figures/datasets"

# ── 1. Simple readable figures (no training needed, uses existing JSONs) ───────
log "=== Simple figures ==="
"$PYTHON" -u simple_figures.py 2>&1 | grep -v "checkpoint\|saving\|version"

# ── 2. Dataset visualizations (no training needed) ────────────────────────────
log "=== Dataset plots ==="
"$PYTHON" -u dataset_plots.py 2>&1

# ── 3. KRR comparison figures (no training needed, uses existing JSONs) ────────
log "=== KRR comparison figures ==="
"$PYTHON" -u comparison_figures.py 2>&1

# ── 4. SVM benchmark (heavy — trains all kernels, then evaluates with SVM) ─────
log "=== SVM Benchmark (this takes ~60-90 min) ==="
"$PYTHON" -u benchmark_svm.py 2>&1 | tee "$RES/svm/benchmark_svm.log"

# ── 5. Regenerate comparison figures now that SVM results exist ────────────────
log "=== Final comparison figures (KRR + SVM) ==="
"$PYTHON" -u comparison_figures.py 2>&1

log "=== ALL DONE ==="
log "Figures:"
ls -lh "$EXP/figures/krr/" "$EXP/figures/svm/" "$EXP/figures/datasets/" 2>/dev/null || true
ls -lh "$EXP/figures/simple_"* 2>/dev/null || true
