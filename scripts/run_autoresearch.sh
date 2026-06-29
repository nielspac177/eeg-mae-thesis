#!/usr/bin/env bash
# Autoresearch orchestrator: wait for the GPU (held by the squeeze run) to free up, then
# drain the MAE config queue one at a time (serialized on the single MPS GPU), logging each
# result to autoresearch.jsonl + experiments/worklog.md. Resumable: cached configs return
# instantly, so re-running is free. Caffeinated so the Mac won't idle-sleep.
set -u
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i env CAFFEINATED=1 bash "$0" "$@"
fi
cd "$(dirname "$0")/.."
LOG="$HOME/.cache/eeg_mae/runs/autoresearch.log"; mkdir -p "$(dirname "$LOG")"
say () { echo "==== [$(date '+%F %T')] $* ====" | tee -a "$LOG"; }

# 1. Wait until no other training process is running (squeeze must finish first).
say "waiting for GPU (squeeze run) to free up"
while pgrep -fl "pretrain_mae|run_squeeze.sh|cli.twostage_push|cli.ensemble_votes" \
        | grep -vE "run_autoresearch" | grep -q . ; do
  sleep 60
done
say "GPU free — starting experiment queue"

# 2. Drain the queue (d384_mean is produced by the squeeze run itself).
QUEUE=(d256_attn d256_concat d384_attn d384_concat ensemble_mae)
for CFG in "${QUEUE[@]}"; do
  say "START experiment: $CFG"
  OUT=$(AR_CONFIG="$CFG" ./autoresearch.sh 2>>"$LOG")
  KL=$(echo "$OUT" | grep -oE 'kl=[0-9.]+' | grep -oE '[0-9.]+' | tail -1)
  if [ -z "$KL" ]; then say "FAIL: $CFG (no KL parsed)"; STATUS=crash; KL=0; else STATUS=done; fi
  RUN=$(grep -c '"run":' autoresearch.jsonl)
  RUN=$((RUN + 1))
  HASH=$(git rev-parse --short=7 HEAD)
  printf '{"run":%d,"commit":"%s","metric":%s,"metrics":{},"status":"%s","description":"%s","timestamp":%s,"segment":0}\n' \
    "$RUN" "$HASH" "$KL" "$([ "$STATUS" = done ] && echo keep || echo crash)" "$CFG" "$(date +%s)" >> autoresearch.jsonl
  {
    echo ""
    echo "### Run $RUN: $CFG — kl=$KL ($STATUS)"
    echo "- Timestamp: $(date '+%F %H:%M')"
    echo "- Result: kl=$KL"
  } >> experiments/worklog.md
  say "DONE experiment: $CFG -> kl=$KL"
done
say "queue drained — autoresearch sweep complete"
