"""Soft-ensemble combiners for OOF probability matrices.

Given several models' out-of-fold predictions (each ``(n, 6)``, rows aligned to the
same samples), find a blend that minimises KL against the soft labels. Arithmetic and
geometric means are parameter-free baselines; the weighted search is a tiny Dirichlet
random search (the recipe from the original ``03b2`` notebook).
"""
from __future__ import annotations

import numpy as np

from .metrics import kl_divergence


def _normalise(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


def arithmetic_mean(stack: np.ndarray) -> np.ndarray:
    """Mean of probabilities across models; ``stack`` is ``(n_models, n, 6)``."""
    return _normalise(stack.mean(axis=0))


def geometric_mean(stack: np.ndarray) -> np.ndarray:
    """Geometric mean (log-space average) — good when models are well-calibrated."""
    log = np.log(np.clip(stack, 1e-12, 1.0))
    return _normalise(np.exp(log.mean(axis=0)))


def weighted_search(stack: np.ndarray, target: np.ndarray, n_trials: int = 500, seed: int = 0):
    """Random-search Dirichlet weights minimising KL; returns ``(weights, blend, kl)``."""
    n_models = stack.shape[0]
    rng = np.random.default_rng(seed)
    best_w = np.full(n_models, 1.0 / n_models)
    best_blend = arithmetic_mean(stack)
    best_kl = kl_divergence(target, best_blend)
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(n_models))
        blend = _normalise((w[:, None, None] * stack).sum(axis=0))
        kl = kl_divergence(target, blend)
        if kl < best_kl:
            best_w, best_blend, best_kl = w, blend, kl
    return best_w, best_blend, float(best_kl)
