"""Spectrogram augmentations: SpecAugment (time/freq masking) and mixup.

Ported from ``src/mae_models.py``. Used in the supervised stage and (mildly) at
test time for TTA in the best-KL ensemble push.
"""
from __future__ import annotations

import numpy as np
import torch


class SpecAugment:
    """Zero out a few random time strips and frequency strips on a ``(B, C, F, T)`` batch."""

    def __init__(self, n_time: int = 2, n_freq: int = 2, max_time: int = 30, max_freq: int = 20):
        self.n_t, self.n_f = n_time, n_freq
        self.mt, self.mf = max_time, max_freq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F_, T_ = x.shape
        x = x.clone()
        for b in range(B):
            for _ in range(self.n_t):
                w = np.random.randint(1, self.mt + 1)
                s = np.random.randint(0, max(1, T_ - w))
                x[b, :, :, s : s + w] = 0.0
            for _ in range(self.n_f):
                w = np.random.randint(1, self.mf + 1)
                s = np.random.randint(0, max(1, F_ - w))
                x[b, :, s : s + w, :] = 0.0
        return x


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Convex-combine pairs of (input, soft-label) by a Beta(alpha, alpha) weight."""
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]
