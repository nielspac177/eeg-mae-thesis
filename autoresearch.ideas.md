# Autoresearch ideas backlog (MAE KL minimization)

## Queued experiments (in priority order)
1. **d256_attn** — learned attentive pooling on the existing d256 encoder. exp6 only had
   cls/mean; attentive often beats both. Cheap (encoder cached).
2. **d256_concat** — `[CLS ; mean]` pooling on d256. Free-ish gain candidate.
3. **d384_mean** — the squeeze run produces this (stronger ep150 encoder). Expect best single.
4. **d384_attn** — attentive pooling on the strong d384. Likely best single MAE.
5. **d384_concat** — concat pooling on d384.
6. **ensemble_mae** — MAE-HEAVY ensemble (d256+d384 × mean/attn). Goal: beat 0.544 with an
   ensemble whose members are mostly MAEs (thesis-hero requirement), then add CNNs on top.

## Deeper levers (try after the pooling sweep)
- **encoder_lr_s2 sweep** {3e-6, 1e-5, 3e-5} — exp5 found 1e-5 best at 1-stage; re-confirm
  for two-stage on the d384. Small, high-value.
- **Layer-wise LR decay (LLRD)** in fine-tune — canonical MAE recipe; per-block decaying LR.
  More than the single encoder_lr the two-stage uses now.
- **epochs_s2 sweep** {6, 8, 12} on d384 — does more sharpening help or overfit the >=8 set?
- **head depth/width** on the winning pooling — depth {3,4}, hidden {512,768}, dropout {0.2,0.3}.
- **mask_ratio at pretrain** {0.5, 0.6, 0.75} — lower mask -> richer features (needs a new
  pretrain, expensive; only if pooling/FT levers stall).
- **multi-snapshot / seed ensemble** of the SAME config (different seeds) — variance reduction.
- **TTA** on the best two-stage MAE (time/freq flips) — reuse tta infra.
- **Dirichlet-weighted ensemble** search over MAE members (vs plain average).

## Notes
- recon loss has plateaued (~0.567 at d384 ep90); more pretrain epochs barely move KL.
  The KL levers are: pooling, fine-tune schedule, ensembling — not longer pretrain.
