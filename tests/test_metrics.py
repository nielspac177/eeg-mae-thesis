"""KL metric must match a reference computation and behave sensibly."""
import numpy as np
from scipy.stats import entropy

from eeg_mae.metrics import bootstrap_kl_ci, kl_divergence, per_class_kl


def _rand_probs(n, c, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((n, c))
    return p / p.sum(axis=1, keepdims=True)


def test_kl_matches_scipy_entropy():
    yt, yp = _rand_probs(50, 6, 1), _rand_probs(50, 6, 2)
    # scipy.stats.entropy(pk, qk) is the per-sample KL(pk || qk); mean over samples.
    ref = np.mean([entropy(yt[i], yp[i]) for i in range(len(yt))])
    assert abs(kl_divergence(yt, yp) - ref) < 1e-6


def test_kl_zero_when_identical():
    yt = _rand_probs(20, 6, 3)
    assert kl_divergence(yt, yt) < 1e-9


def test_per_class_kl_length_and_nonneg():
    yt, yp = _rand_probs(100, 6, 4), _rand_probs(100, 6, 5)
    pc = per_class_kl(yt, yp)
    assert pc.shape == (6,)
    assert np.all(pc[~np.isnan(pc)] >= 0)


def test_bootstrap_ci_brackets_mean():
    yt, yp = _rand_probs(200, 6, 6), _rand_probs(200, 6, 7)
    mean, lo, hi = bootstrap_kl_ci(yt, yp, n_boot=500)
    assert lo <= mean <= hi
