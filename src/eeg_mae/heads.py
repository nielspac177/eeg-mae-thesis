"""Classifier heads on top of the pooled encoder feature.

Experiment 2 replaces the original hard-label ``LogisticRegression`` probe with a
configurable MLP. The default — ``depth=2`` — is the requested 3-layer network
(input projection -> one hidden layer -> output), i.e. two ``Linear`` layers with a
nonlinearity between them. ``depth`` is the number of ``Linear`` layers, so:

    depth=1  ->  linear probe (logistic-regression equivalent)
    depth=2  ->  input -> hidden -> output      (the requested 3-layer MLP)
    depth>=3 ->  deeper stacks ("make it deeper... we'll see")
"""
from __future__ import annotations

import torch.nn as nn


class MLPHead(nn.Module):
    """LayerNorm -> [Linear -> GELU -> Dropout] x (depth-1) -> Linear(n_classes).

    Parameters
    ----------
    in_dim
        Input feature dimension (the encoder ``enc_dim``).
    n_classes
        Number of output logits (6 for the HMS soft-label task).
    depth
        Number of ``Linear`` layers (>=1). ``depth=2`` is the default 3-layer MLP.
    hidden
        Width of the hidden layers.
    dropout
        Dropout probability after each hidden activation.
    """

    def __init__(self, in_dim: int, n_classes: int = 6, depth: int = 2, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
