#!/usr/bin/env bash
# Phase 2b orchestrator — champion experiments on d256_h8 @ ep90.
# exp5 (frozen vs fine-tune, discriminative LR), exp6 (CLS vs mean pooling),
# exp7 (latent map), then figures. Each step is resumable/idempotent.
#
# Usage: bash scripts/run_phase2b.sh   ·   Logs: runs (off-iCloud)/phase2b.log
set -u
# Re-exec under caffeinate so macOS idle-sleep can't kill a long unattended run.
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
LOGDIR="$HOME/.cache/eeg_mae/runs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/phase2b.log"

step () {
  local label="$1"; shift
  echo "==== [$(date '+%F %T')] START: $label ====" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "==== [$(date '+%F %T')] DONE:  $label ====" | tee -a "$LOG"
  else
    echo "==== [$(date '+%F %T')] FAIL ($?): $label (continuing) ====" | tee -a "$LOG"
  fi
}

echo "######## Phase 2b started $(date '+%F %T') ########" | tee -a "$LOG"
step "exp5 fine-tune vs frozen"  "$PY" -m eeg_mae.cli.run_experiment configs/exp5_finetune.yaml --no-progress
step "exp6 CLS vs mean pooling"  "$PY" -m eeg_mae.cli.run_experiment configs/exp6_pooling.yaml  --no-progress
step "exp7 latent map"           "$PY" -m eeg_mae.cli.run_experiment configs/exp7_latent.yaml
step "make figures"              "$PY" -m eeg_mae.cli.make_figures
echo "######## Phase 2b finished $(date '+%F %T') ########" | tee -a "$LOG"
