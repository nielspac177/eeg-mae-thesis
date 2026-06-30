# Autoresearch Dashboard: mae-kl-minimize

**Runs logged:** 3 (+ d256_concat running) | **Baseline:** kl 0.6564 (#1, d256 mean)
**Best single MAE:** kl **0.6564** (#1, d256 mean) — none of the new single configs beat it yet
**Best ensemble:** kl **0.5411** (7-model weighted, incl. d384 — beats prior 0.544)

| # | kl | Δ vs baseline | status | description |
|---|------|--------------|--------|-------------|
| 1 | 0.6564 | — | keep | baseline d256 mean (cached) |
| 2 | 0.6881 | +4.8% | done | d256 attentive pooling — worse than mean |
| 3 | 0.6786 | +3.4% | done | d384 mean (stronger ep150 encoder) — worse than d256 |
| — | running | — | … | d256 concat pooling |

## Ensemble milestone (squeeze run, 7 models on >=8-vote eval)
| method | kl |
|--------|------|
| arithmetic | 0.5491 |
| geometric | 0.5665 |
| **weighted** | **0.5411** ← best |

Members: mae_mean_v4-8 0.695 · mae_cls_v4-8 0.751 · effnet_b0_v4-8 0.628 ·
effnet_b1_v4-8 0.668 · mae_mean_v3-8 0.656 · effnet_b0_v3-8 0.644 · **mae_d384_mean_v3-8 0.679**

## Read so far
- **Single-MAE cheap levers exhausted:** attentive pooling (0.688) and a bigger d384 encoder
  (0.679) both *underperform* plain d256 mean-pool (0.656). The recon plateau + these results
  say the single-model ceiling is ~0.66 for this recipe.
- **The win is in the ensemble:** diversity (d384 + CNNs) drops KL to 0.5411 even when members
  are individually weaker. → The MAE-heavy ensemble (queued) and Dirichlet weighting are the
  promising paths; deeper fine-tune levers (encoder-LR / LLRD / epochs_s2) for single models.

**Queue remaining:** d256_concat (running) → d384_attn → d384_concat → ensemble_mae
