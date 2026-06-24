"""Patchify must be an exact inverse and produce the documented token grid."""
import torch

from eeg_mae import SPEC_C, SPEC_F, SPEC_T, patchify, unpatchify
from eeg_mae.constants import PATCH_F, PATCH_T


def test_patchify_round_trip_is_exact():
    x = torch.randn(2, SPEC_C, SPEC_F, SPEC_T)
    assert torch.allclose(unpatchify(patchify(x)), x, atol=1e-6)


def test_patchify_shapes():
    x = torch.randn(3, SPEC_C, SPEC_F, SPEC_T)
    p = patchify(x)
    n_patches = (SPEC_F // PATCH_F) * (SPEC_T // PATCH_T)
    assert p.shape == (3, n_patches, SPEC_C * PATCH_F * PATCH_T)
    assert n_patches == 300
