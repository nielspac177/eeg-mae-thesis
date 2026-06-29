# Autoresearch worklog — MAE KL minimization

**Session start:** 2026-06-29
**Goal:** lowest OOF KL (>=8-vote eval) for the spectrogram MAE on HMS. MAE is the thesis hero.
**Run:** `AR_CONFIG=<name> ./autoresearch.sh` -> `METRIC kl=<n>`. Configs cached/resumable.

## Data / setup summary
- Two-stage OOF: stage1 on >=3-vote windows (full data), stage2 sharpen on >=8-vote,
  eval = OOF KL on held-out >=8-vote patients (patient-grouped 5-fold, no leakage).
- One MPS GPU: experiments are serialized. The "squeeze" run owns it until it finishes
  (d384 pretrain ep150 -> two-stage d384 -> ensemble).
- Known prior results: single MAE-mean two-stage **0.656**; 6-model ensemble **0.544** (best).

## Baseline
- Baseline = `twostage_mae_mean_v3-8` (d256 mean, cached) = **kl 0.656**.

---

### Run 1: baseline (d256 mean, cached) — kl=0.656 (KEEP, baseline)
- Timestamp: 2026-06-29
- What changed: nothing — records the existing cached best single MAE as the baseline.
- Result: kl 0.656.
- Insight: this is the bar a single MAE must beat; ensemble bar is 0.544.
- Next: attentive/concat pooling on d256 (cheap), then d384 variants once squeeze finishes.

## Key Insights
- mean > cls everywhere; gentle fine-tune (enc_lr 1e-5) is the big single lever; recon has
  plateaued so longer pretrain ≠ lower KL. New bets: learned pooling + MAE-heavy ensemble.

## Next Ideas
- See `autoresearch.ideas.md`. Immediate queue: d256_attn, d256_concat, d384_*, ensemble_mae.
