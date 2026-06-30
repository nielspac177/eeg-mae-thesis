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

### Run 2: d256_attn — kl=0.6881 (done)
- Timestamp: 2026-06-30 09:12
- What changed: learned attentive pooling over patch tokens (new lever, extends exp6).
- Result: kl=0.6881 vs baseline 0.6564 (+4.8%) — **attentive pooling did NOT beat mean-pool**.
- Insight: the simple mean is a strong inductive bias here; a learned pool overfits the small
  >=8-vote stage-2 set. Mean-pool remains the pooling of choice.
- Next: concat pooling, then d384 variants.

### Run 3: d384_mean — kl=0.6786 (done; from squeeze run)
- Timestamp: 2026-06-30 05:31
- What changed: stronger d384 encoder pretrained to ep150 (vs d256 ep90), two-stage full data.
- Result: kl=0.6786 vs d256 baseline 0.6564 (+3.4%) — **bigger encoder did NOT beat d256**.
  Folds 0.812 / 0.713 / 0.608 / 0.609 / 0.651 (±0.077, high variance).
- Insight: confirms exp3/exp4 — recon gains (d384 best recon) do not transfer to downstream KL;
  the larger model is harder to fine-tune on limited high-quality data. Single-model ceiling ~0.66.

### Milestone: 7-model ensemble (incl. d384) — weighted kl=0.5411 (NEW BEST)
- Timestamp: 2026-06-30 05:31 (squeeze run)
- arithmetic 0.5491 · geometric 0.5665 · **weighted 0.5411** (prev best 0.544).
- Insight: d384, though weaker alone (0.679), still *improves the ensemble* via diversity. The
  payoff is in combining diverse members, not in any single stronger MAE.
- Next: build an MAE-HEAVY ensemble (mean+attn × d256+d384) for the thesis-hero narrative, and
  try Dirichlet-weighted member search. Pursue deeper single-model fine-tune levers (encoder-LR
  sweep, LLRD, epochs_s2) which we have NOT yet tried.
