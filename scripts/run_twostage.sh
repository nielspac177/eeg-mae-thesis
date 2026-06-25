#!/usr/bin/env bash
# Two-stage full-data MAE push toward ~0.30. Resumable/idempotent at every step.
#   1. build the >=4-vote spectrogram cache (one-time, parallel I/O)
#   2. two-stage MAE, mean pooling (exp6 winner)  -> the headline number
#   3. two-stage MAE, cls pooling (thesis comparison)
#   4. figures
# Usage: bash scripts/run_twostage.sh   ·   Logs: ~/.cache/eeg_mae/runs/twostage.log
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
LOGDIR="$HOME/.cache/eeg_mae/runs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/twostage.log"

step () {
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else echo "==== [$(date '+%F %T')] FAIL ($?): $label ====" | tee -a "$LOG"; fi
}

echo "######## Two-stage push started $(date '+%F %T') ########" | tee -a "$LOG"
step "build >=4-vote cache" "$PY" -m eeg_mae.cli.build_cache --min-votes 4 --threads 8
step "two-stage MAE mean-pool" "$PY" -m eeg_mae.cli.twostage_push --pooling mean --no-progress
step "two-stage MAE cls" "$PY" -m eeg_mae.cli.twostage_push --pooling cls --no-progress
step "make figures" "$PY" -m eeg_mae.cli.make_figures
echo "######## Two-stage push finished $(date '+%F %T') ########" | tee -a "$LOG"
