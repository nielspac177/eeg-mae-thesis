"""The core guarantee: train-N-straight == train-k-then-resume-(N-k), bit-for-bit (CPU).

This is what makes long MPS runs safe to interrupt. We use a tiny supervised setup
(linear model + soft-label KL) on CPU, where all RNG is serialisable.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_mae.losses import SoftLabelKLLoss
from eeg_mae.trainer import ResumableTrainer

N, D, C = 64, 8, 6


def _fixed_dataset():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(N, D, generator=g)
    logits = torch.randn(N, C, generator=g)
    y = torch.softmax(logits, dim=-1)
    return TensorDataset(x, y)


def _build_model():
    torch.manual_seed(7)  # identical init across runs
    return nn.Linear(D, C)


def _make_loader_factory(ds):
    def make(generator):
        return DataLoader(ds, batch_size=16, shuffle=True, drop_last=True, generator=generator)

    return make


def _flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_resume_equals_straight_through(tmp_path):
    ds = _fixed_dataset()
    make_loader = _make_loader_factory(ds)
    loss_fn = SoftLabelKLLoss()

    # Run A: 4 epochs straight.
    model_a = _build_model()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2)
    ResumableTrainer(model_a, opt_a, loss_fn, run_dir=tmp_path / "a", seed=0).fit(make_loader, 4)

    # Run B: 2 epochs, then a fresh trainer resumes from the checkpoint for 2 more.
    model_b1 = _build_model()
    opt_b1 = torch.optim.AdamW(model_b1.parameters(), lr=1e-2)
    ResumableTrainer(model_b1, opt_b1, loss_fn, run_dir=tmp_path / "b", seed=0).fit(make_loader, 2)

    model_b2 = _build_model()
    opt_b2 = torch.optim.AdamW(model_b2.parameters(), lr=1e-2)
    trainer_b2 = ResumableTrainer(model_b2, opt_b2, loss_fn, run_dir=tmp_path / "b", seed=0)
    assert trainer_b2.start_epoch == 2  # resumed
    trainer_b2.fit(make_loader, 4)

    assert torch.allclose(_flat_params(model_a), _flat_params(model_b2), atol=1e-6)


def test_already_complete_is_noop(tmp_path):
    ds = _fixed_dataset()
    make_loader = _make_loader_factory(ds)
    loss_fn = SoftLabelKLLoss()

    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ResumableTrainer(model, opt, loss_fn, run_dir=tmp_path / "c", seed=0).fit(make_loader, 3)
    before = _flat_params(model).clone()

    # Re-fitting to the same epoch count should not change anything.
    model2 = _build_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    t = ResumableTrainer(model2, opt2, loss_fn, run_dir=tmp_path / "c", seed=0)
    t.fit(make_loader, 3)
    assert torch.allclose(before, _flat_params(model2), atol=1e-6)
