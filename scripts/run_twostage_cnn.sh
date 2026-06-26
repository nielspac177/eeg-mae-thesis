#!/usr/bin/env bash
# Two-stage CNNs on the >=8-vote eval, then ensemble with the two-stage MAEs.
# Multi-backbone ensemble on the high-quality eval = the competition recipe's final piece.
# Resumable/idempotent. Usage: bash scripts/run_twostage_cnn.sh
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
LOGDIR="$HOME/.cache/eeg_mae/runs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/twostage_cnn.log"

step () {
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else echo "==== [$(date '+%F %T')] FAIL ($?): $label ====" | tee -a "$LOG"; fi
}

echo "######## Two-stage CNN+ensemble started $(date '+%F %T') ########" | tee -a "$LOG"
step "two-stage EfficientNet-B0" "$PY" -m eeg_mae.cli.twostage_push --cnn efficientnet_b0 --lr-head 3e-4 --epochs-s1 4 --epochs-s2 5 --no-progress
step "two-stage EfficientNet-B1" "$PY" -m eeg_mae.cli.twostage_push --cnn efficientnet_b1 --lr-head 3e-4 --epochs-s1 4 --epochs-s2 5 --no-progress
step "ensemble all (2 MAE + 2 CNN) on >=8-vote eval" "$PY" -m eeg_mae.cli.ensemble_votes \
  --oof twostage_mae_mean_v4-8,twostage_mae_cls_v4-8,twostage_cnn_efficientnet_b0_v4-8,twostage_cnn_efficientnet_b1_v4-8
echo "######## Two-stage CNN+ensemble finished $(date '+%F %T') ########" | tee -a "$LOG"
