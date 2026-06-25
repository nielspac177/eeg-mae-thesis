"""Test-time augmentation (TTA) over the saved out-of-fold checkpoints.

For each fold, the trained classifier is reloaded and run on its validation samples
several times under a *mild* SpecAugment, averaging the softmax outputs. Averaging
slightly-perturbed views reduces variance and typically shaves a few hundredths off KL
— for free, since no model is retrained. Reuses the per-fold checkpoints that
``cv.run_oof`` already wrote under ``runs/oof/<name>/fold{k}``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from . import paths
from .augment import SpecAugment
from .data import SpecDataset
from .metrics import kl_divergence
from .trainer import CKPT_NAME


def tta_oof(
    name: str,
    model_factory: Callable[[], torch.nn.Module],
    label_meta,
    soft_labels: np.ndarray,
    *,
    load_fn=None,
    device: str | torch.device = "cpu",
    n_passes: int = 6,
    n_splits: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """Return a TTA'd OOF prediction matrix ``(n, 6)`` for the model ``name``.

    Regenerates the same patient-grouped folds used at training time, loads each fold's
    checkpoint, and averages ``n_passes`` mildly-augmented inference passes (one clean +
    the rest augmented) over that fold's validation samples.
    """
    cache_path = paths.RESULTS_DIR / "oof" / f"{name}_tta.npz"
    if cache_path.exists():
        return np.load(cache_path)["oof"]

    runs_dir = paths.RUNS_DIR / "oof" / name
    hard = soft_labels.argmax(axis=1)
    groups = label_meta["patient_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    mild = SpecAugment(n_time=1, n_freq=1, max_time=15, max_freq=10)

    oof = np.full_like(soft_labels, np.nan, dtype=np.float32)
    for fold, (_, va) in enumerate(gkf.split(soft_labels, hard, groups)):
        ckpt = runs_dir / f"fold{fold}" / CKPT_NAME
        if not ckpt.exists():
            raise FileNotFoundError(f"missing fold checkpoint for TTA: {ckpt}")
        model = model_factory().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        model.eval()

        va_df = label_meta.iloc[va].reset_index(drop=True)
        loader = DataLoader(
            SpecDataset(va_df, load_fn=load_fn, with_label=False),
            batch_size=batch_size, shuffle=False,
        )
        preds = []
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                p = F.softmax(model(x), dim=-1)
                for _ in range(n_passes - 1):
                    p = p + F.softmax(model(mild(x)), dim=-1)
                preds.append((p / n_passes).cpu().numpy())
        oof[va] = np.concatenate(preds)
        del model

    kl = kl_divergence(soft_labels, oof)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, oof=oof, kl_overall=kl)
    return oof


# Model-factory registry: maps an OOF name to a builder for its architecture so the
# saved checkpoint can be reloaded for TTA. Encoder/head shapes must match training.
def build_factories(device):
    from .cnn import make_effnet_classifier
    from .heads import MLPHead
    from .models import MAEClassifier, SpecMAE

    def mae(pooling):
        def f():
            enc = SpecMAE(enc_dim=256, enc_heads=8)
            head = MLPHead(256, n_classes=6, depth=3, hidden=512, dropout=0.2)
            return MAEClassifier(enc, head, pooling=pooling, freeze_encoder=False).to(device)
        return f

    return {
        "exp5_finetune__finetune_enc1e5": mae("cls"),
        "push_mae_mean": mae("mean"),
        "push_efficientnet_b0": lambda: make_effnet_classifier("efficientnet_b0", n_classes=6).to(device),
        "push_efficientnet_b1": lambda: make_effnet_classifier("efficientnet_b1", n_classes=6).to(device),
    }
