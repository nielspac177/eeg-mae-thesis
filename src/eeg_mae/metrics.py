"""Evaluation metrics: the competition KL divergence + helpers for the thesis.

``kl_divergence`` is the Kaggle HMS metric (lower is better). ``per_class_kl`` and
``bootstrap_kl_ci`` support the statistical reporting in the memoria (per-class
breakdown to test the "epileptiform is harder than seizures" hypothesis, and
confidence intervals on the mean KL).
"""
from __future__ import annotations

import numpy as np


def _normalise(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Mean per-sample ``sum(y_true * log(y_true / y_pred))`` (the HMS competition metric)."""
    yt = _normalise(np.asarray(y_true, dtype=np.float64), eps)
    yp = _normalise(np.asarray(y_pred, dtype=np.float64), eps)
    return float(np.mean(np.sum(yt * np.log(yt / yp), axis=1)))


def per_sample_kl(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Per-sample KL divergence, shape ``(n_samples,)`` — input to bootstrap CIs."""
    yt = _normalise(np.asarray(y_true, dtype=np.float64), eps)
    yp = _normalise(np.asarray(y_pred, dtype=np.float64), eps)
    return np.sum(yt * np.log(yt / yp), axis=1)


def per_class_kl(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Mean KL contribution grouped by the **hard** (argmax) true class, shape ``(n_classes,)``.

    Localises where error concentrates — e.g. epileptiform (LPD/GPD/LRDA/GRDA) vs Seizure.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    sample_kl = per_sample_kl(yt, y_pred, eps)
    hard = yt.argmax(axis=1)
    n_classes = yt.shape[1]
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = hard == c
        if m.any():
            out[c] = sample_kl[m].mean()
    return out


def bootstrap_kl_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean KL: returns ``(mean, lo, hi)`` at level ``1-alpha``."""
    sample_kl = per_sample_kl(y_true, y_pred)
    n = len(sample_kl)
    rng = np.random.default_rng(seed)
    boots = np.array([sample_kl[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(sample_kl.mean()), float(lo), float(hi)
