#!/usr/bin/env bash
# =============================================================================
# run_kan_experiments.sh
# Runs all KAN_Kernel_NX experiments then generates combined figures.
#
# Usage (detached, survives terminal close):
#   nohup bash run_kan_experiments.sh > experiments/classification_v2/results/kan_master.log 2>&1 &
#   echo "Master PID: $!"
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$REPO_ROOT/experiments/classification_v2"
RESULTS="$EXP_DIR/results"
PYTHON="$REPO_ROOT/.venv/bin/python"

mkdir -p "$RESULTS"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$EXP_DIR"

# ── Phase 1: KAN Benchmark (GPU, sequential — heaviest job) ──────────────────
log "=== Phase 1: KAN UCI Benchmark ==="
"$PYTHON" -u kan_benchmark.py 2>&1 | tee "$RESULTS/kan_benchmark.log"
log "Phase 1 complete."

# ── Phase 2: Parallel independent experiments (CPU) ──────────────────────────
log "=== Phase 2: Decision Boundaries + SNR Sweep (parallel) ==="

"$PYTHON" -u kan_decision_boundary.py > "$RESULTS/kan_decision_boundary.log" 2>&1 &
DB_PID=$!
log "  Decision boundary PID=$DB_PID"

"$PYTHON" -u kan_snr_sweep.py > "$RESULTS/kan_snr_sweep.log" 2>&1 &
SNR_PID=$!
log "  SNR sweep PID=$SNR_PID"

wait $DB_PID
log "  Decision boundary done."
wait $SNR_PID
log "  SNR sweep done."

# ── Phase 3: Feature importance + Convergence (GPU, after Phase 1) ───────────
log "=== Phase 3: Feature Importance + Convergence (parallel) ==="

"$PYTHON" -u kan_feature_importance.py > "$RESULTS/kan_feature_importance.log" 2>&1 &
FI_PID=$!
log "  Feature importance PID=$FI_PID"

"$PYTHON" -u kan_convergence.py > "$RESULTS/kan_convergence.log" 2>&1 &
CONV_PID=$!
log "  Convergence PID=$CONV_PID"

wait $FI_PID
log "  Feature importance done."
wait $CONV_PID
log "  Convergence done."

# ── Phase 4: Combined figures (needs both benchmark JSONs) ────────────────────
log "=== Phase 4: Combined Figures ==="
"$PYTHON" -u combined_figures.py 2>&1 | tee "$RESULTS/combined_figures.log"
log "Phase 4 complete."

# ── Summary ───────────────────────────────────────────────────────────────────
log "==================================================================="
log "ALL DONE. Figures in: $EXP_DIR/figures/"
log "==================================================================="
ls -lh "$EXP_DIR/figures/"
