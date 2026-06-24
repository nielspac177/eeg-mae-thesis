"""Spectrogram Masked Autoencoder (ViT) and the supervised classifier wrapper.

Ported from the project's ``src/mae_models.py`` / ``03b`` notebook, with two thesis
experiments wired in as first-class options:

* **enc_dim / enc_heads are configurable** (experiment 4) — any ``enc_dim`` divisible
  by both ``enc_heads`` and 4 (the 2-D sin-cos positional embedding needs %4==0).
* **encoder pooling is switchable** (experiment 6) — ``"cls"`` uses the CLS token,
  ``"mean"`` averages the patch tokens, for the downstream classifier input.

The decoder is intentionally light (asymmetric MAE, He et al. 2022): a small decoder
is enough to drive a good encoder, and we only keep the encoder for downstream use.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .constants import PATCH_F, PATCH_T, SPEC_C, SPEC_F, SPEC_T


# --------------------------------------------------------------------------- #
# Patchify <-> unpatchify
# --------------------------------------------------------------------------- #
def patchify(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, F, T) -> (B, N_patches, C*PF*PT)`` with non-overlapping PF×PT patches."""
    B, C, _, _ = x.shape
    x = x.unfold(2, PATCH_F, PATCH_F).unfold(3, PATCH_T, PATCH_T)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    return x.view(B, -1, C * PATCH_F * PATCH_T)


def unpatchify(patches: torch.Tensor, C: int = SPEC_C, F_: int = SPEC_F, T_: int = SPEC_T) -> torch.Tensor:
    """Exact inverse of :func:`patchify`."""
    B, _, _ = patches.shape
    nF, nT = F_ // PATCH_F, T_ // PATCH_T
    x = patches.view(B, nF, nT, C, PATCH_F, PATCH_T)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, F_, T_)


def sincos_2d_pos_embed(embed_dim: int, nF: int, nT: int) -> torch.Tensor:
    """Fixed 2-D sin-cos positional embedding (``embed_dim`` must be divisible by 4)."""
    assert embed_dim % 4 == 0, f"embed_dim={embed_dim} must be divisible by 4"
    d = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(d // 2) / (d // 2)))
    f = torch.arange(nF).float()[:, None] * omega[None]
    t = torch.arange(nT).float()[:, None] * omega[None]
    pe_f = torch.cat([f.sin(), f.cos()], dim=1)
    pe_t = torch.cat([t.sin(), t.cos()], dim=1)
    pe = torch.zeros(nF * nT, embed_dim)
    for i in range(nF):
        for j in range(nT):
            pe[i * nT + j] = torch.cat([pe_f[i], pe_t[j]])
    return pe


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block (MHSA + MLP, residual)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


# --------------------------------------------------------------------------- #
# MAE
# --------------------------------------------------------------------------- #
class SpecMAE(nn.Module):
    """Masked Autoencoder over 4-region EEG spectrograms.

    Defaults reproduce the original ViT-Tiny (``enc_dim=192, depth=6, heads=3``).
    For experiment 4, vary ``enc_dim``/``enc_heads`` (keep ``enc_dim % (4*?) ``: it
    must be divisible by both ``enc_heads`` and 4).
    """

    def __init__(
        self,
        in_chans: int = SPEC_C,
        F_: int = SPEC_F,
        T_: int = SPEC_T,
        enc_dim: int = 192,
        enc_depth: int = 6,
        enc_heads: int = 3,
        dec_dim: int = 96,
        dec_depth: int = 2,
        dec_heads: int = 3,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        if enc_dim % enc_heads != 0:
            raise ValueError(f"enc_dim={enc_dim} not divisible by enc_heads={enc_heads}")
        if enc_dim % 4 != 0:
            raise ValueError(f"enc_dim={enc_dim} must be divisible by 4 (2-D pos-embed)")
        if dec_dim % dec_heads != 0:
            raise ValueError(f"dec_dim={dec_dim} not divisible by dec_heads={dec_heads}")
        if dec_dim % 4 != 0:
            raise ValueError(f"dec_dim={dec_dim} must be divisible by 4 (2-D pos-embed)")

        self.in_chans, self.F_, self.T_ = in_chans, F_, T_
        self.nF, self.nT = F_ // PATCH_F, T_ // PATCH_T
        self.n_patches = self.nF * self.nT
        self.patch_dim = in_chans * PATCH_F * PATCH_T
        self.mask_ratio = mask_ratio
        self.enc_dim = enc_dim

        # encoder
        self.patch_embed = nn.Linear(self.patch_dim, enc_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.register_buffer("enc_pos", sincos_2d_pos_embed(enc_dim, self.nF, self.nT))
        self.enc_blocks = nn.ModuleList(TransformerBlock(enc_dim, enc_heads) for _ in range(enc_depth))
        self.enc_norm = nn.LayerNorm(enc_dim)

        # decoder (asymmetric / light)
        self.dec_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_2d_pos_embed(dec_dim, self.nF, self.nT))
        self.dec_blocks = nn.ModuleList(TransformerBlock(dec_dim, dec_heads) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, self.patch_dim)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    # -- masking -----------------------------------------------------------
    def random_mask(self, x: torch.Tensor):
        """Per-sample random masking; returns kept tokens, binary mask, restore index."""
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_kept, mask, ids_restore

    # -- encode/decode -----------------------------------------------------
    def forward_encoder(self, x: torch.Tensor):
        patches = patchify(x)
        tokens = self.patch_embed(patches) + self.enc_pos
        x_kept, mask, ids_restore = self.random_mask(tokens)
        cls = self.cls_token.expand(x_kept.size(0), -1, -1)
        z = torch.cat([cls, x_kept], dim=1)
        for blk in self.enc_blocks:
            z = blk(z)
        return self.enc_norm(z), mask, ids_restore, patches

    def forward_decoder(self, x_enc: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.dec_embed(x_enc)
        cls, x_ = x[:, :1], x[:, 1:]
        B, n_keep, D = x_.shape
        n_mask = ids_restore.size(1) - n_keep
        mask_tokens = self.mask_token.expand(B, n_mask, D)
        x_full = torch.cat([x_, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x_full = x_full + self.dec_pos
        x = torch.cat([cls, x_full], dim=1)
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        return self.dec_pred(x[:, 1:])

    def forward(self, x: torch.Tensor):
        """Returns ``(loss, pred, mask)``; loss is masked, per-patch-normalised MSE."""
        x_enc, mask, ids_restore, patches = self.forward_encoder(x)
        pred = self.forward_decoder(x_enc, ids_restore)
        mu = patches.mean(-1, keepdim=True)
        var = patches.var(-1, keepdim=True)
        tgt = (patches - mu) / (var + 1e-6).sqrt()
        loss = ((pred - tgt) ** 2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask

    # -- downstream feature extraction ------------------------------------
    def encode(self, x: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
        """Encode without masking and pool to one vector per sample.

        ``pooling="cls"`` returns the CLS token; ``pooling="mean"`` averages the
        patch tokens (experiment 6). This path is differentiable so it can be used
        for both frozen probing and full fine-tuning.
        """
        z = self.forward_tokens(x)
        if pooling == "cls":
            return z[:, 0]
        if pooling == "mean":
            return z[:, 1:].mean(dim=1)
        raise ValueError(f"unknown pooling={pooling!r}; use 'cls' or 'mean'")

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """All encoder tokens (CLS + patches), no masking — shape ``(B, 1+N, enc_dim)``."""
        tokens = self.patch_embed(patchify(x)) + self.enc_pos
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        z = torch.cat([cls, tokens], dim=1)
        for blk in self.enc_blocks:
            z = blk(z)
        return self.enc_norm(z)


# --------------------------------------------------------------------------- #
# Supervised classifier wrapper
# --------------------------------------------------------------------------- #
class MAEClassifier(nn.Module):
    """MAE encoder + an :class:`~eeg_mae.heads.MLPHead`, trained with soft-label KL.

    ``pooling`` selects CLS vs mean (experiment 6). ``freeze_encoder=True`` runs the
    encoder in no-grad eval mode for a frozen linear/MLP probe (experiment 5).
    """

    def __init__(self, encoder: SpecMAE, head: nn.Module, pooling: str = "cls", freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                feats = self.encoder.encode(x, pooling=self.pooling)
        else:
            feats = self.encoder.encode(x, pooling=self.pooling)
        return self.head(feats)

    def param_groups(self, encoder_lr: float, head_lr: float):
        """Discriminative-LR parameter groups (experiment 5): low LR encoder, higher LR head."""
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]
