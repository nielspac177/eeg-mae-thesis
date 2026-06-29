# Autoresearch Dashboard: mae-kl-minimize

**Runs:** 1 | **Kept:** 1 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** kl: 0.6564 (#1)
**Best:** kl: 0.6564 (#1, single MAE) · ensemble bar to beat: 0.544

| # | commit | kl | status | description |
|---|--------|-----|--------|-------------|
| 1 | 63b32c8 | 0.6564 | keep | baseline d256 mean (cached) |

**Queue:** d256_attn → d256_concat → d384_mean → d384_attn → d384_concat → ensemble_mae
**GPU:** held by squeeze run (d384 pretrain → two-stage → ensemble). Experiments start when free.
