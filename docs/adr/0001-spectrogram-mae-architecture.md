# ADR 0001 — Spectrogram Masked Autoencoder as the representation learner

- **Status:** Accepted
- **Date:** 2026-06-24
- **Context:** Carrera de Especialización en IA (FIUBA) — final project on harmful brain
  activity classification.

## Context

The HMS task provides ~11k EEG spectrograms but reliable (high-agreement) labels for only a
subset. We want a representation that exploits all spectrograms, labelled or not, and transfers
to the 6-class soft-label problem scored by KL divergence. Options considered:

1. Supervised CNN from scratch (EfficientNet/timm) — strong but ignores unlabelled data and
   overfits the small high-agreement set.
2. Contrastive SSL (SimCLR/BYOL) — needs careful augmentation design and large batches.
3. **Masked Autoencoder (He et al., 2022)** — reconstruct masked patches; simple objective,
   no negative pairs, works at modest batch size, and yields a reusable encoder.

## Decision

Use an asymmetric ViT **Masked Autoencoder** over the 4-region `(4, 100, 300)` spectrogram,
patchified into 10×10 patches (300 tokens), 75% masking. The encoder is ViT-Tiny by default
(`enc_dim=192`, depth 6, 3 heads); the decoder is intentionally light (2 blocks, `dec_dim=96`).
Downstream, only the encoder is kept and a small MLP head is trained on soft labels with KL.

## Consequences

- One pretraining run produces an encoder reused across all classification experiments.
- `enc_dim` must be divisible by both the head count and 4 (the 2-D sin-cos positional
  embedding) — enforced in `SpecMAE.__init__`. This constrains experiment 4's grid.
- The decoder is discarded after pretraining, so its cost is "wasted" compute, but the
  asymmetric design keeps it small (the encoder sees only 25% of tokens during pretraining).
- Spectrogram preprocessing (log + per-region z-score, fixed `(4,100,300)`) is replicated
  byte-for-byte from the original notebooks so pretrained encoders stay compatible.
