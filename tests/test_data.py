"""Dataset + soft-label helpers, using a stub loader (no real parquet needed)."""
import numpy as np
import pandas as pd

from eeg_mae.constants import CLASSES_6, SPEC_C, SPEC_F, SPEC_T
from eeg_mae.data import SpecDataset, soft_label_matrix


def _meta(n=4):
    rng = np.random.default_rng(0)
    votes = rng.integers(0, 10, size=(n, 6)).astype(float)
    soft = votes / votes.sum(axis=1, keepdims=True)
    df = pd.DataFrame({"spectrogram_id": np.arange(n)})
    for j, c in enumerate(CLASSES_6):
        df[f"soft_label_{c}"] = soft[:, j]
    return df


def test_soft_label_matrix_rows_sum_to_one():
    y = soft_label_matrix(_meta())
    assert y.shape[1] == 6
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-6)


def test_specdataset_returns_tensor_and_normalised_label():
    df = _meta()

    def stub_load(spec_id, offset_seconds=0):
        return np.full((SPEC_C, SPEC_F, SPEC_T), float(spec_id), dtype=np.float32)

    ds = SpecDataset(df, load_fn=stub_load, with_label=True)
    x, y = ds[2]
    assert tuple(x.shape) == (SPEC_C, SPEC_F, SPEC_T)
    assert abs(float(y.sum()) - 1.0) < 1e-6


def test_specdataset_without_label_returns_only_x():
    df = _meta()
    ds = SpecDataset(df, load_fn=lambda s, offset_seconds=0: np.zeros((SPEC_C, SPEC_F, SPEC_T), np.float32))
    assert not isinstance(ds[0], tuple)
