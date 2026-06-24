"""Head depth, MAE forward, enc_dim validation, and CLS-vs-mean pooling."""
import pytest
import torch
import torch.nn as nn

from eeg_mae import SPEC_C, SPEC_F, SPEC_T, MLPHead, SpecMAE
from eeg_mae.models import MAEClassifier


def _n_linear(module):
    return sum(isinstance(m, nn.Linear) for m in module.modules())


def test_mlphead_depth_controls_linear_count():
    assert _n_linear(MLPHead(192, depth=1)) == 1  # linear probe
    assert _n_linear(MLPHead(192, depth=2)) == 2  # the requested 3-layer MLP
    assert _n_linear(MLPHead(192, depth=4)) == 4


def test_mlphead_output_shape():
    h = MLPHead(192, n_classes=6, depth=2, hidden=64)
    assert h(torch.randn(5, 192)).shape == (5, 6)


def test_mae_forward_returns_loss_pred_mask():
    mae = SpecMAE(enc_dim=64, enc_depth=2, enc_heads=4, dec_dim=64, dec_depth=1, dec_heads=4)
    x = torch.randn(2, SPEC_C, SPEC_F, SPEC_T)
    loss, pred, mask = mae(x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert pred.shape[0] == 2
    assert 0.0 < mask.float().mean().item() < 1.0


def test_enc_dim_validation():
    with pytest.raises(ValueError):
        SpecMAE(enc_dim=100, enc_heads=3)  # not divisible by heads
    with pytest.raises(ValueError):
        SpecMAE(enc_dim=66, enc_heads=3)  # divisible by heads but not by 4


def test_pooling_shapes_match_but_values_differ():
    mae = SpecMAE(enc_dim=64, enc_depth=2, enc_heads=4, dec_dim=64, dec_depth=1, dec_heads=4)
    x = torch.randn(3, SPEC_C, SPEC_F, SPEC_T)
    cls = mae.encode(x, pooling="cls")
    mean = mae.encode(x, pooling="mean")
    assert cls.shape == mean.shape == (3, 64)
    assert not torch.allclose(cls, mean)


def test_classifier_frozen_encoder_has_no_encoder_grads():
    mae = SpecMAE(enc_dim=64, enc_depth=2, enc_heads=4, dec_dim=64, dec_depth=1, dec_heads=4)
    clf = MAEClassifier(mae, MLPHead(64, depth=2, hidden=32), pooling="cls", freeze_encoder=True)
    out = clf(torch.randn(2, SPEC_C, SPEC_F, SPEC_T))
    out.sum().backward()
    assert all(p.grad is None for p in clf.encoder.parameters())
    assert any(p.grad is not None for p in clf.head.parameters())


def test_param_groups_discriminative_lr():
    mae = SpecMAE(enc_dim=64, enc_depth=2, enc_heads=4, dec_dim=64, dec_depth=1, dec_heads=4)
    clf = MAEClassifier(mae, MLPHead(64, depth=2, hidden=32))
    groups = clf.param_groups(encoder_lr=1e-5, head_lr=1e-3)
    assert groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 1e-3
