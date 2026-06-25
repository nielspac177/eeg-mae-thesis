"""Two-stage fine-tuning on the full data — the competition's key lever, for the MAE.

Stage 1 fine-tunes the (SSL-pretrained) MAE on a large, noisier set (windows with
``>= min_votes_s1`` votes); stage 2 sharpens on the high-quality set (``>= min_votes_eval``
votes). Evaluation is out-of-fold KL on held-out **patients** of the high-quality set —
matching how the competition's ~0.30 target is defined.

Both stages and the evaluation share one patient-grouped split, so no patient leaks
between train and val. Each stage is independently checkpointed/resumable; the stage-2
model continues from the stage-1 weights.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from . import paths
from .augment import SpecAugment, mixup
from .constants import CLASSES_6
from .data import SpecDataset, soft_label_matrix, vote_subset
from .losses import SoftLabelKLLoss
from .metrics import kl_divergence
from .trainer import ResumableTrainer


@dataclass
class TwoStageConfig:
    name: str
    min_votes_s1: int = 4
    min_votes_eval: int = 8
    epochs_s1: int = 6
    epochs_s2: int = 8
    lr_head: float = 5e-4
    encoder_lr_s1: float = 5e-5   # stage 1: more data -> slightly higher encoder LR ok
    encoder_lr_s2: float = 1e-5   # stage 2: gentle (exp5 winner)
    weight_decay: float = 0.05
    batch_size: int = 32
    n_splits: int = 5
    mixup_alpha: float = 0.2
    use_specaugment: bool = True
    grad_clip: float = 1.0
    seed: int = 42
    extra: dict = field(default_factory=dict)


def _opt(model, encoder_lr, head_lr, wd):
    if hasattr(model, "param_groups"):
        return torch.optim.AdamW(model.param_groups(encoder_lr=encoder_lr, head_lr=head_lr), weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=wd)


def run_twostage_oof(
    cfg: TwoStageConfig,
    mae_factory: Callable[[], torch.nn.Module],
    meta: pd.DataFrame,
    *,
    load_fn=None,
    device: str | torch.device = "cpu",
    cache_dir: object | None = None,
    runs_dir: object | None = None,
    progress: bool = False,
) -> dict:
    """Run the two-stage OOF and return ``{oof, fold_kls, kl_overall}`` (cached)."""
    cache_dir = cache_dir or (paths.RESULTS_DIR / "oof")
    runs_dir = runs_dir or (paths.RUNS_DIR / "twostage")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.name}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return {"oof": d["oof"], "fold_kls": d["fold_kls"], "kl_overall": float(d["kl_overall"])}

    eval_meta = vote_subset(meta, cfg.min_votes_eval)
    s1_meta = vote_subset(meta, cfg.min_votes_s1)
    eval_labels = soft_label_matrix(eval_meta)
    eval_hard = eval_labels.argmax(axis=1)
    groups = eval_meta["patient_id"].to_numpy()
    gkf = GroupKFold(n_splits=cfg.n_splits)

    aug = SpecAugment() if cfg.use_specaugment else None
    loss_fn = SoftLabelKLLoss()

    def on_batch(x, y):
        if aug is not None:
            x = aug(x)
        if cfg.mixup_alpha > 0:
            x, y = mixup(x, y, alpha=cfg.mixup_alpha)
        return x, y

    def loader_for(df, generator=None, shuffle=True):
        ds = SpecDataset(df.reset_index(drop=True), load_fn=load_fn, with_label=True, soft_classes=CLASSES_6)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=shuffle,
                          num_workers=0, generator=generator)

    oof = np.full_like(eval_labels, np.nan, dtype=np.float32)
    fold_kls = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(eval_labels, eval_hard, groups)):
        val_patients = set(eval_meta.iloc[va_idx]["patient_id"])
        s1_train = s1_meta[~s1_meta["patient_id"].isin(val_patients)]
        s2_train = eval_meta.iloc[tr_idx]
        val_df = eval_meta.iloc[va_idx]
        print(f"[{cfg.name}] fold {fold+1}: stage1 {len(s1_train):,} | stage2 {len(s2_train):,} | val {len(val_df):,}")

        model = mae_factory()

        # Stage 1 — larger, noisier set.
        t1 = ResumableTrainer(model, _opt(model, cfg.encoder_lr_s1, cfg.lr_head, cfg.weight_decay),
                              loss_fn, device=device, run_dir=runs_dir / cfg.name / f"fold{fold}" / "s1",
                              grad_clip=cfg.grad_clip, seed=cfg.seed + fold)
        t1.fit(lambda g, _df=s1_train: loader_for(_df, g), cfg.epochs_s1, on_batch=on_batch, progress=progress)

        # Stage 2 — sharpen on high-quality set, continuing from stage-1 weights.
        t2 = ResumableTrainer(model, _opt(model, cfg.encoder_lr_s2, cfg.lr_head, cfg.weight_decay),
                              loss_fn, device=device, run_dir=runs_dir / cfg.name / f"fold{fold}" / "s2",
                              grad_clip=cfg.grad_clip, seed=cfg.seed + fold)
        t2.fit(lambda g, _df=s2_train: loader_for(_df, g), cfg.epochs_s2, on_batch=on_batch, progress=progress)

        preds = t2.predict_proba(loader_for(val_df, shuffle=False))
        oof[va_idx] = preds
        kl = kl_divergence(eval_labels[va_idx], preds)
        fold_kls.append(kl)
        print(f"[{cfg.name}] fold {fold+1}: KL = {kl:.4f}")
        del model, t1, t2
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    fold_kls = np.array(fold_kls, dtype=np.float32)
    kl_overall = kl_divergence(eval_labels, oof)
    np.savez(cache_path, oof=oof, fold_kls=fold_kls, kl_overall=kl_overall)
    return {"oof": oof, "fold_kls": fold_kls, "kl_overall": float(kl_overall)}
