# Autoresearch: minimize MAE OOF KL (HMS, >=8-vote eval)

## Objective
Push the spectrogram **Masked Autoencoder** as far as possible on the Kaggle HMS task —
lowest out-of-fold **KL divergence** on the >=8-vote, patient-grouped eval set (how the
competition's ~0.30 target is defined). MAE is the thesis hero, so prefer MAE-centric
levers (pooling, enc_dim, fine-tune schedule, MAE-heavy ensembles) over leaning on the CNN.

## Metrics
- **Primary**: `kl` — OOF KL on the >=8-vote eval set (**lower is better**).
- **Secondary**: none yet (could add fold std, #params).

## How to Run
`AR_CONFIG=<name> ./autoresearch.sh` — runs one config's two-stage OOF and prints
`METRIC kl=<number>`. Configs are cached (results/oof/<name>.npz), so a finished config
returns instantly — the loop is fully resumable and re-running is free.

## GPU serialization (IMPORTANT)
There is **one** MPS GPU. Never run two training jobs at once. Before launching an
experiment, confirm no `pretrain_mae`/`twostage_push`/`ensemble_votes` process is running
(`pgrep -fl`). The "squeeze" run (scripts/run_squeeze.sh) owns the GPU until it finishes
(d384 pretrain -> two-stage d384 -> ensemble); let it complete, then start experiments.

## Experiments are additive (no revert)
Each experiment only **adds** a gitignored `results/oof/<name>.npz` and appends a CSV row —
there is nothing to revert. So we DO NOT use the standard keep/discard `git checkout -- .`
(that would clobber the running job's CSV writes). Instead: every result is recorded; the
"best" is just the lowest-KL config so far. Commit additively, periodically.

## Files in Scope
- `src/eeg_mae/models.py` — SpecMAE, MAEClassifier, poolings (cls/mean/attn/concat).
- `src/eeg_mae/heads.py` — MLPHead (depth/width/dropout).
- `src/eeg_mae/twostage.py` — two-stage OOF (LRs, epochs, mixup, specaugment).
- `src/eeg_mae/cli/twostage_push.py` — config -> run; `--pooling/--enc-dim/--head-*`.
- `src/eeg_mae/cli/ensemble_votes.py` — ensemble aligned OOFs.
- `autoresearch.sh` — config dispatch table.

## Off Limits
- Raw-EEG / dual-modality (deliberately deferred; squeeze the MAE first).
- The running squeeze job's checkpoints under `~/.cache/eeg_mae/runs`.
- Author hygiene: commits authored by **Niels Pacheco Barrios only — NO Co-Authored-By**.

## Constraints
- Mac MPS only; long runtimes OK; every run resumable (true: OOF is cached/idempotent).
- `~/venvs/hms/bin/python`. No new heavy deps.

## What's Been Tried
Prior experiments (exp2-7, frozen/1-stage CV KL) and the two-stage push are already done.
Robust findings going in:
- **mean-pool > cls** everywhere (frozen / 1-stage / 2-stage).
- **gentle fine-tune wins**: encoder_lr 1e-5 best (exp5: 1.207 frozen -> 0.919).
- bigger encoder -> better recon; d384 best recon, d256 nearly ties at half params.
- more pretrain epochs improve recon but barely move downstream KL (recon has plateaued).
- two-stage full-data: single MAE-mean **0.656** (>=8-vote eval); 6-model ensemble **0.544**.

New levers this session (untested -> queued): attentive pooling, concat pooling, a stronger
d384 encoder (ep150), and an MAE-heavy ensemble. See `autoresearch.ideas.md`.
