#!/usr/bin/env bash
# Autoresearch benchmark: run ONE MAE config's two-stage OOF and print `METRIC kl=<n>`.
# Usage: AR_CONFIG=<name> ./autoresearch.sh
# Configs are cached (results/oof/<name>.npz) so finished ones return instantly.
set -euo pipefail
cd "$(dirname "$0")"
PY="$HOME/venvs/hms/bin/python"
export PYTHONUNBUFFERED=1
CFG="${AR_CONFIG:-baseline}"

D256="runs/pretrain/mae_d256_h8/snapshots/encoder_ep90.pt"
D384="runs/pretrain/mae_d384_h6/snapshots/encoder_ep150.pt"
COMMON="--min-votes-s1 3 --epochs-s1 3 --epochs-s2 8 --no-progress"

run_ts () {  # run twostage_push, echo its KL
  local out kl
  out=$("$PY" -m eeg_mae.cli.twostage_push "$@" 2>&1) || { echo "$out" >&2; return 1; }
  echo "$out" >&2
  kl=$(echo "$out" | grep -oE 'OOF KL.*=[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | tail -1)
  echo "$kl"
}

case "$CFG" in
  baseline)    KL=$(run_ts --name twostage_mae_mean_v3-8  --enc-dim 256 --enc-heads 8 --pretrained "$D256" --pooling mean   $COMMON) ;;
  d256_attn)   KL=$(run_ts --name twostage_mae_attn_v3-8  --enc-dim 256 --enc-heads 8 --pretrained "$D256" --pooling attn   $COMMON) ;;
  d256_concat) KL=$(run_ts --name twostage_mae_concat_v3-8 --enc-dim 256 --enc-heads 8 --pretrained "$D256" --pooling concat $COMMON) ;;
  d384_mean)   KL=$(run_ts --name twostage_mae_d384_mean_v3-8   --enc-dim 384 --enc-heads 6 --pretrained "$D384" --pooling mean   $COMMON) ;;
  d384_attn)   KL=$(run_ts --name twostage_mae_d384_attn_v3-8   --enc-dim 384 --enc-heads 6 --pretrained "$D384" --pooling attn   $COMMON) ;;
  d384_concat) KL=$(run_ts --name twostage_mae_d384_concat_v3-8 --enc-dim 384 --enc-heads 6 --pretrained "$D384" --pooling concat $COMMON) ;;
  ensemble_mae)  # MAE-heavy ensemble of the best MAE variants (filled in as they finish)
    OOF="${AR_OOF:-twostage_mae_mean_v3-8,twostage_mae_d384_mean_v3-8,twostage_mae_attn_v3-8,twostage_mae_d384_attn_v3-8}"
    out=$("$PY" -m eeg_mae.cli.ensemble_votes --oof "$OOF" --min-votes-eval 8 2>&1) || { echo "$out" >&2; exit 1; }
    echo "$out" >&2
    KL=$(echo "$out" | grep -oE '=[[:space:]]*[0-9.]+' | grep -oE '[0-9.]+' | tail -1)
    ;;
  *) echo "unknown AR_CONFIG=$CFG" >&2; exit 2 ;;
esac

[ -n "${KL:-}" ] || { echo "no KL parsed for $CFG" >&2; exit 3; }
echo "METRIC kl=$KL"
