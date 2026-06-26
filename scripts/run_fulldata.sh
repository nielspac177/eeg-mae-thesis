#!/usr/bin/env bash
# Full-data push: stage-1 on the full >=3-vote set (100k windows) for the two best
# models (MAE mean-pool + EfficientNet-B0), then ensemble with the existing >=4-vote
# models on the >=8-vote eval. Fewer stage-1 epochs (more data) keeps compute ~constant.
# Resumable. Usage: bash scripts/run_fulldata.sh
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
LOGDIR="$HOME/.cache/eeg_mae/runs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/fulldata.log"

step () {
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else echo "==== [$(date '+%F %T')] FAIL ($?): $label ====" | tee -a "$LOG"; fi
}

echo "######## Full-data push started $(date '+%F %T') ########" | tee -a "$LOG"
step "build >=3-vote cache (100k)" "$PY" -m eeg_mae.cli.build_cache --min-votes 3 --threads 8
step "two-stage MAE mean (>=3 stage1)" "$PY" -m eeg_mae.cli.twostage_push --pooling mean \
  --min-votes-s1 3 --epochs-s1 3 --epochs-s2 6 --no-progress
step "two-stage EfficientNet-B0 (>=3 stage1)" "$PY" -m eeg_mae.cli.twostage_push --cnn efficientnet_b0 \
  --lr-head 3e-4 --min-votes-s1 3 --epochs-s1 3 --epochs-s2 6 --no-progress
step "ensemble 6 models (>=4 and >=3 stage1) on >=8-vote eval" "$PY" -m eeg_mae.cli.ensemble_votes \
  --oof twostage_mae_mean_v4-8,twostage_mae_cls_v4-8,twostage_cnn_efficientnet_b0_v4-8,twostage_cnn_efficientnet_b1_v4-8,twostage_mae_mean_v3-8,twostage_cnn_efficientnet_b0_v3-8
echo "######## Full-data push finished $(date '+%F %T') ########" | tee -a "$LOG"
