#!/usr/bin/env bash
# Best-KL push — add a mean-pool fine-tuned MAE and a 2nd CNN, then ensemble everything.
# Each step is resumable/idempotent (OOF caches + per-fold checkpoints).
#
# Usage: bash scripts/run_push.sh   ·   Logs: ~/.cache/eeg_mae/runs/push2.log
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
LOGDIR="$HOME/.cache/eeg_mae/runs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/push2.log"

step () {
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else echo "==== [$(date '+%F %T')] FAIL ($?): $label ====" | tee -a "$LOG"; fi
}

echo "######## Push started $(date '+%F %T') ########" | tee -a "$LOG"
# 1. Mean-pool fine-tuned MAE (exp6 said mean > cls; only cls was fine-tuned in exp5).
step "MAE mean-pool fine-tune" "$PY" -m eeg_mae.cli.run_experiment configs/push_mae_mean.yaml --no-progress
# 2. Train EfficientNet-B0 (cached) + B1 (new) and ensemble with both MAE variants.
step "ensemble MAE(cls+mean) + EfficientNet(b0+b1)" \
  "$PY" -m eeg_mae.cli.best_kl_push \
  --models efficientnet_b0,efficientnet_b1 \
  --oof exp5_finetune__finetune_enc1e5,push_mae_mean --epochs 12
# 3. Figures.
step "make figures" "$PY" -m eeg_mae.cli.make_figures
echo "######## Push finished $(date '+%F %T') ########" | tee -a "$LOG"
