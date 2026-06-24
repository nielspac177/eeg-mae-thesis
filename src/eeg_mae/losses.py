"""Training objectives.

The supervised stage (experiment 1) trains on **soft labels** with KL divergence —
the same objective the Kaggle competition scores — rather than hard cross-entropy.
``nn.KLDivLoss`` expects log-probabilities as input and a probability target, so we
wrap the ``log_softmax`` here to make call sites hard to get wrong.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelKLLoss(nn.Module):
    """KL(target || softmax(logits)) averaged over the batch.

    Parameters
    ----------
    logits
        Raw classifier outputs, shape ``(B, C)``.
    target
        Soft-label distributions, shape ``(B, C)``, each row summing to ~1.

    This is the differentiable training analogue of :func:`eeg_mae.metrics.kl_divergence`
    (the eval metric); both measure the same quantity up to the input convention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return self.kl(logp, target)
