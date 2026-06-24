# Phase 4 (later) — BrainBERT for epileptiform / harmful-brain-activity detection

> Parked roadmap. This is **not** being executed now — the current work is the
> spectrogram-MAE thesis experiments (Phases 1–3). Captured here so the plan is not lost.
>
> Repo: <https://github.com/czlwang/BrainBERT> · Paper: Wang et al., *BrainBERT:
> Self-supervised representation learning for intracranial recordings* (ICLR 2023).

## Why BrainBERT

The spectrogram-MAE works on the **precomputed 4-region spectrograms**. BrainBERT instead
learns from **raw neural time series** via masked spectrogram modelling, and is pretrained on
a large corpus of intracranial **sEEG**. The thesis hypothesis: a transformer pretrained on
neural signals transfers better to the *hard* classes — **epileptiform activity (LPD, GPD,
LRDA, GRDA)** — than to the easy **Seizure** class. Seizures are relatively easy; periodic /
rhythmic epileptiform patterns are where most KL error concentrates, so a stronger
representation should pay off most there.

**Key domain gap to handle:** BrainBERT is pretrained on **intracranial sEEG** (single
electrodes, high SNR). The HMS data is **scalp EEG** (10–20 montage, lower SNR, volume
conduction). Treat BrainBERT weights as an *initialisation / feature extractor*, not a
drop-in — expect to fine-tune, and benchmark honestly against the from-scratch MAE.

## Data path (already available locally)

- `hms-harmful-brain-activity-classification/train_eegs/*.parquet` — **17,300** raw EEG
  recordings, 20 columns (19 electrodes + EKG) at **200 Hz**.
- Reuse `src/hms_utils.py`: `load_eeg`, `apply_banana_montage` (→ 18 bipolar channels),
  `butter_bandpass_filter`, `preprocess_eeg_segment` (0.5–20 Hz, z-score). BrainBERT expects
  **single-channel** input → feed one montage channel at a time, or pick a region channel.
- Labels: same `train.csv` soft labels + `high_agreement` filter as the MAE work, so results
  are directly comparable on identical patient-grouped 5-fold splits and the **same KL metric**.

## Milestones

1. **Vendor + environment.** Clone BrainBERT into `external/brainbert/` (git-ignored or a
   submodule), pin its deps in a separate `[project.optional-dependencies] brainbert` group.
   Download the released pretrained checkpoint. Confirm a forward pass on a toy input on MPS
   (may need CPU fallback for unsupported ops; check fairseq/torchaudio compatibility — this is
   the main environment risk and should be smoke-tested first).
2. **STFT front-end parity.** BrainBERT consumes a **superlet / STFT spectrogram of raw
   signal**. Implement the exact transform its tokenizer expects (`data_gen` in the repo);
   wrap our preprocessed montage channels into that format. Unit-test shape + value ranges
   against the repo's expected inputs.
3. **Feature-extraction baseline (frozen).** Run frozen BrainBERT over each HMS window, mean-
   pool token embeddings per channel, concat/avg across the 18 channels (or 4 regions) →
   fixed feature vector. Train the **same soft-label KL MLP head** (`eeg_mae.heads.MLPHead`)
   with the resumable trainer. Report OOF KL, and **per-class KL** to test the epileptiform
   hypothesis. This is the cheapest signal on whether BrainBERT transfers at all.
4. **Fine-tune.** Unfreeze with **discriminative LR** (low on the transformer, higher on the
   head) — reuse the exp-5 param-group logic. Compare: (a) frozen features, (b) full fine-tune,
   (c) last-N-layers fine-tune. Same checkpoint/resume machinery (MPS, stop/restart-safe).
5. **Channel-aggregation study.** Single best channel vs region-averaged vs all-18 attention
   pooling — scalp EEG is multi-channel, BrainBERT is single-channel, so how to combine
   channels is a real design question worth an ablation.
6. **Head-to-head + fusion.** Compare BrainBERT-head vs spectrogram-MAE vs EfficientNet on
   identical folds; then **ensemble** BrainBERT into the existing soft-mean / weighted blend
   (`03b2` Stage 3) to see if raw-signal + spectrogram views are complementary. Track in
   `results/kl_progression.csv`.
7. **Latent map + interpretability.** t-SNE/UMAP of BrainBERT embeddings (color = hard label,
   alpha = soft confidence), mirroring exp-7, to qualitatively compare the two latent spaces.

## Deliverables for the thesis

- Per-class KL table (Seizure vs the four epileptiform classes) for MAE vs BrainBERT vs fusion.
- A clear statement of the **scalp-vs-intracranial domain-shift** finding (does intracranial
  pretraining help scalp EEG, and specifically epileptiform patterns?).
- Figures: BrainBERT pipeline schematic, frozen-vs-finetune bars, fusion KL progression,
  BrainBERT latent map.

## Risks / open questions

- **Env compatibility** is the top risk: BrainBERT depends on fairseq/older torchaudio; may not
  run cleanly on torch 2.11 / MPS. Mitigation: isolated venv, CPU fallback, or a one-off cloud
  GPU run just for feature extraction (then analyse on the Mac).
- **Licensing / checkpoint availability** — confirm the released weights are usable and cite.
- **Compute** — feature extraction over 17,300 recordings × 18 channels is heavy on MPS; cache
  embeddings to `data/processed/brainbert_features.npz` once (idempotent, like the MAE features)
  and never recompute.
