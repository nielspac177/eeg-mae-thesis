#!/usr/bin/env bash
# Phase 2 orchestrator — runs the full experiment sequence in order.
# Every step is individually resumable/idempotent (per-epoch checkpoints + cached OOF),
# so re-running this script after any interruption continues where it left off.
#
# Usage:  bash scripts/run_phase2.sh
# Logs:   runs/phase2.log  (git-ignored)
set -u
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
mkdir -p runs
LOG=runs/phase2.log

step () {  # step "label" <command...>
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else
    echo "==== [$(date '+%F %T')] FAIL ($?): $label (continuing) ====" | tee -a "$LOG"
  fi
}

echo "######## Phase 2 run started $(date '+%F %T') ########" | tee -a "$LOG"

# --- Phase A: champion-width encoder + its classification experiments (early results) ---
step "pretrain d192_h3 (90ep, snapshots 30/60/90)" \
  "$PY" -m eeg_mae.cli.pretrain_mae --epochs 90 --enc-dim 192 --enc-heads 3 \
  --snapshots 30 60 90 --num-workers 4
step "exp3 epoch sweep" "$PY" -m eeg_mae.cli.run_experiment configs/exp3_epoch_sweep.yaml --no-progress
step "exp2 head depth"  "$PY" -m eeg_mae.cli.run_experiment configs/exp2_head_depth.yaml  --no-progress

# --- Phase B: remaining encoder widths for the enc_dim sweep (exp 4) ---
step "pretrain d128_h4 (90ep)" "$PY" -m eeg_mae.cli.pretrain_mae --epochs 90 --enc-dim 128 --enc-heads 4 --snapshots 90 --num-workers 4
step "pretrain d256_h8 (90ep)" "$PY" -m eeg_mae.cli.pretrain_mae --epochs 90 --enc-dim 256 --enc-heads 8 --snapshots 90 --num-workers 4
step "pretrain d384_h6 (90ep)" "$PY" -m eeg_mae.cli.pretrain_mae --epochs 90 --enc-dim 384 --enc-heads 6 --snapshots 90 --num-workers 4
step "exp4 enc_dim sweep" "$PY" -m eeg_mae.cli.run_experiment configs/exp4_enc_dim.yaml --no-progress

# --- Phase C: figures from whatever results exist so far ---
step "make figures" "$PY" -m eeg_mae.cli.make_figures

echo "######## Phase 2 run finished $(date '+%F %T') ########" | tee -a "$LOG"
