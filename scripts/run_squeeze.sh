#!/usr/bin/env bash
# Squeeze the MAE: a bigger (d384) + longer-pretrained encoder, two-stage on full data,
# added to the ensemble. MAE-centric, no new modality. Resumable.
# Usage: bash scripts/run_squeeze.sh
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
LOGDIR="$HOME/.cache/eeg_mae/runs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/squeeze.log"

step () {
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else echo "==== [$(date '+%F %T')] FAIL ($?): $label ====" | tee -a "$LOG"; fi
}

echo "######## Squeeze-MAE started $(date '+%F %T') ########" | tee -a "$LOG"
# 1. Stronger encoder: continue d384 pretrain 90 -> 150 epochs (resumes from ckpt).
step "continue-pretrain d384 -> 150ep" "$PY" -m eeg_mae.cli.pretrain_mae \
  --epochs 150 --enc-dim 384 --enc-heads 6 --snapshots 150 --run-name mae_d384_h6
# 2. Two-stage d384 MAE mean-pool on full (>=3-vote) data, extra stage-2 sharpening.
step "two-stage d384 MAE mean (full data)" "$PY" -m eeg_mae.cli.twostage_push \
  --name twostage_mae_d384_mean_v3-8 --enc-dim 384 --enc-heads 6 \
  --pretrained runs/pretrain/mae_d384_h6/snapshots/encoder_ep150.pt \
  --pooling mean --min-votes-s1 3 --epochs-s1 3 --epochs-s2 8 --no-progress
# 3. Ensemble all MAEs + CNNs (7 models) on the >=8-vote eval.
step "ensemble 7 models on >=8-vote eval" "$PY" -m eeg_mae.cli.ensemble_votes --oof \
twostage_mae_mean_v4-8,twostage_mae_cls_v4-8,twostage_cnn_efficientnet_b0_v4-8,twostage_cnn_efficientnet_b1_v4-8,twostage_mae_mean_v3-8,twostage_cnn_efficientnet_b0_v3-8,twostage_mae_d384_mean_v3-8
step "make figures" "$PY" -m eeg_mae.cli.make_figures
echo "######## Squeeze-MAE finished $(date '+%F %T') ########" | tee -a "$LOG"
