"""Patient-grouped 5-fold out-of-fold (OOF) training and evaluation.

Ports ``train_oof`` from the ``03b2`` notebook onto the :class:`ResumableTrainer`,
so every fold is independently checkpointed and resumable, and the whole study is
**idempotent**: a completed run is cached to ``<results>/oof/<name>.npz`` and reloaded
instead of retrained. Splits use :class:`sklearn.model_selection.GroupKFold` on
``patient_id`` — no patient appears in both train and val (prevents leakage).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from . import paths
from .augment import SpecAugment, mixup
from .constants import CLASSES_6
from .data import SpecDataset
from .losses import SoftLabelKLLoss
from .metrics import kl_divergence
from .trainer import ResumableTrainer


@dataclass
class OOFConfig:
    """Hyper-parameters for one OOF study."""

    name: str
    epochs: int = 15
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.05
    n_splits: int = 5
    use_specaugment: bool = True
    mixup_alpha: float = 0.2
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    encoder_lr: float | None = None  # if set, discriminative LR (exp 5) via param_groups
    extra: dict = field(default_factory=dict)


def _make_optimizer(model: torch.nn.Module, cfg: OOFConfig) -> torch.optim.Optimizer:
    if cfg.encoder_lr is not None and hasattr(model, "param_groups"):
        return torch.optim.AdamW(
            model.param_groups(encoder_lr=cfg.encoder_lr, head_lr=cfg.lr),
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def run_oof(
    cfg: OOFConfig,
    model_factory: Callable[[], torch.nn.Module],
    label_meta: pd.DataFrame,
    soft_labels: np.ndarray,
    *,
    load_fn=None,
    device: str | torch.device = "cpu",
    cache_dir: Path | None = None,
    runs_dir: Path | None = None,
    progress: bool = False,
) -> dict:
    """Train one model per fold; return OOF predictions, per-fold KL, and overall KL.

    Caches the result to ``cache_dir/<name>.npz``; a second call returns the cache.
    Each fold's per-epoch checkpoints live under ``runs_dir/<name>/fold{k}`` so an
    interrupted study resumes fold-by-fold.
    """
    cache_dir = cache_dir or (paths.RESULTS_DIR / "oof")
    runs_dir = runs_dir or (paths.RUNS_DIR / "oof")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.name}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return {"oof": d["oof"], "fold_kls": d["fold_kls"], "kl_overall": float(d["kl_overall"])}

    soft_labels = soft_labels.astype(np.float32)
    n, n_classes = soft_labels.shape
    hard = soft_labels.argmax(axis=1)
    groups = label_meta["patient_id"].to_numpy()
    gkf = GroupKFold(n_splits=cfg.n_splits)

    aug = SpecAugment() if cfg.use_specaugment else None
    loss_fn = SoftLabelKLLoss()

    def on_batch(x, y):
        if aug is not None:
            x = aug(x)
        if cfg.mixup_alpha > 0:
            x, y = mixup(x, y, alpha=cfg.mixup_alpha)
        return x, y

    oof = np.full((n, n_classes), np.nan, dtype=np.float32)
    fold_kls = []
    for fold, (tr, va) in enumerate(gkf.split(soft_labels, hard, groups)):
        model = model_factory()
        optimizer = _make_optimizer(model, cfg)
        trainer = ResumableTrainer(
            model, optimizer, loss_fn, device=device,
            run_dir=runs_dir / cfg.name / f"fold{fold}",
            grad_clip=cfg.grad_clip, seed=cfg.seed + fold,
        )

        tr_df = label_meta.iloc[tr].reset_index(drop=True)
        va_df = label_meta.iloc[va].reset_index(drop=True)

        def make_train_loader(generator, _df=tr_df):
            ds = SpecDataset(_df, load_fn=load_fn, with_label=True, soft_classes=CLASSES_6)
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=cfg.num_workers, generator=generator)

        trainer.fit(make_train_loader, num_epochs=cfg.epochs, on_batch=on_batch, progress=progress)

        va_ds = SpecDataset(va_df, load_fn=load_fn, with_label=True, soft_classes=CLASSES_6)
        va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        preds = trainer.predict_proba(va_loader)
        oof[va] = preds
        fold_kls.append(kl_divergence(soft_labels[va], preds))

    fold_kls = np.array(fold_kls, dtype=np.float32)
    kl_overall = kl_divergence(soft_labels, oof)
    np.savez(cache_path, oof=oof, fold_kls=fold_kls, kl_overall=kl_overall)
    return {"oof": oof, "fold_kls": fold_kls, "kl_overall": float(kl_overall)}
