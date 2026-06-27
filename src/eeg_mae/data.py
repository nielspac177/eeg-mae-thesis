"""Data loading: train metadata, soft labels, spectrogram tensors, and the Dataset.

Self-contained re-implementation of the loaders in the project's ``src/hms_utils.py``
and the ``load_spec_tensor`` helper from ``03b2``.  Keeping it standalone means the
``eeg_mae`` package can be cloned and run without the rest of the original project,
as long as the raw data is reachable via :mod:`eeg_mae.paths`.

The canonical spectrogram tensor is ``(4, 100, 300)`` float32 — 4 EEG regions, 100
frequency bins, 300 time steps (~10 minutes) — matching the MAE patchifier.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import paths
from .constants import CLASSES_6, SPEC_C, SPEC_F, SPEC_T

# Expert vote columns in train.csv, aligned 1:1 with CLASSES_6.
VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
SOFT_COLS_6 = [f"soft_label_{c}" for c in CLASSES_6]

# A spectrogram parquet row spans ~2 seconds; used to convert offset seconds -> rows.
SECONDS_PER_ROW = 2.0


# --------------------------------------------------------------------------- #
# Metadata + soft labels
# --------------------------------------------------------------------------- #
def load_train_meta(data_root: Path | None = None, high_agreement_thr: float = 0.66) -> pd.DataFrame:
    """Load ``train.csv`` and attach soft labels, total votes, and an agreement flag.

    Adds columns:
      - ``total_votes``: sum of the six expert-vote columns;
      - ``soft_label_{class}``: vote fraction per class (a probability distribution);
      - ``max_vote_fraction`` and ``high_agreement`` (dominant class >= threshold).
    """
    # Prefer a local off-iCloud copy of train.csv if present: the iCloud-synced original
    # gets evicted under disk pressure and pd.read_csv then times out mid-run. Refresh it
    # with `cp <data_root>/train.csv $EEG_MAE_CACHE/train.csv` (paths.LOCAL_BASE).
    local_csv = paths.LOCAL_BASE / "train.csv"
    if data_root is None and local_csv.exists():
        df = pd.read_csv(local_csv)
    else:
        root = data_root or paths.DATA_ROOT
        df = pd.read_csv(root / "train.csv")

    df["total_votes"] = df[VOTE_COLS].sum(axis=1)
    for col, name in zip(VOTE_COLS, CLASSES_6, strict=True):
        df[f"soft_label_{name}"] = df[col] / df["total_votes"].clip(lower=1)

    df["max_vote_fraction"] = df[SOFT_COLS_6].max(axis=1)
    df["high_agreement"] = df["max_vote_fraction"] >= high_agreement_thr
    return df


def soft_label_matrix(df: pd.DataFrame, classes: Sequence[str] = CLASSES_6) -> np.ndarray:
    """Stack the ``soft_label_*`` columns into a normalised ``(n, n_classes)`` array."""
    cols = [f"soft_label_{c}" for c in classes]
    y = np.stack([df[c].to_numpy() for c in cols], axis=1).astype(np.float32)
    return y / np.clip(y.sum(axis=1, keepdims=True), 1e-8, None)


def label_subset(meta: pd.DataFrame, high_agreement_only: bool = True) -> pd.DataFrame:
    """High-agreement, one-row-per-spectrogram view used for supervised training."""
    sub = meta[meta["high_agreement"]] if high_agreement_only else meta
    return sub.drop_duplicates("spectrogram_id").reset_index(drop=True)


def vote_subset(meta: pd.DataFrame, min_votes: int) -> pd.DataFrame:
    """All labelled *windows* (not deduped) with at least ``min_votes`` expert votes.

    Unlike :func:`label_subset` this keeps every (spectrogram_id, offset) window, which
    is what the two-stage recipe needs: a large ``min_votes>=4`` set for stage 1 and a
    high-quality ``min_votes>=8`` set for stage 2 / evaluation.
    """
    return meta[meta["total_votes"] >= min_votes].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Spectrogram tensor loading
# --------------------------------------------------------------------------- #
def load_spectrogram(spec_id: int, data_root: Path | None = None) -> pd.DataFrame:
    """Read one spectrogram parquet (``time`` + 400 frequency-region columns)."""
    root = data_root or paths.DATA_ROOT
    return pd.read_parquet(root / "train_spectrograms" / f"{spec_id}.parquet")


def extract_spectrogram_segment(
    spec_df: pd.DataFrame, offset_seconds: float = 0.0, duration_seconds: float = 600.0
) -> np.ndarray:
    """Reshape a raw spectrogram into ``(4, 100, n_time)`` with log + per-region z-score.

    Mirrors ``hms_utils.extract_spectrogram_segment`` exactly (log-clip to
    ``[e^-4, e^8]`` then per-region standardisation), so tensors are identical to
    those the original notebooks trained on.
    """
    values = spec_df.drop(columns=["time"], errors="ignore").fillna(0).to_numpy(dtype=np.float32)
    offset_idx = int(offset_seconds / SECONDS_PER_ROW)
    end_idx = min(offset_idx + int(duration_seconds / SECONDS_PER_ROW), len(values))
    segment = values[offset_idx:end_idx] if offset_idx < len(values) else values
    segment = segment.T  # (400, n_time)

    n_time = segment.shape[1]
    result = np.zeros((SPEC_C, SPEC_F, n_time), dtype=np.float32)
    for r in range(SPEC_C):
        lo, hi = r * SPEC_F, (r + 1) * SPEC_F
        if hi <= segment.shape[0]:
            result[r] = segment[lo:hi]

    result = np.clip(result, np.exp(-4.0), np.exp(8.0))
    result = np.log(result)
    for r in range(SPEC_C):
        mean, std = result[r].mean(), result[r].std()
        if std > 1e-6:
            result[r] = (result[r] - mean) / std
    return result


def load_spec_tensor(
    spectrogram_id: int,
    offset_seconds: float = 0.0,
    data_root: Path | None = None,
) -> np.ndarray:
    """Load one spectrogram as a fixed ``(4, 100, 300)`` tensor (center-crop / zero-pad time).

    Returns an all-zero tensor for unreadable/corrupt parquet files (matching the
    defensive behaviour in ``03b2``), so a single bad file never kills a run.
    """
    try:
        df = load_spectrogram(int(spectrogram_id), data_root=data_root)
        arr = extract_spectrogram_segment(df, offset_seconds=offset_seconds, duration_seconds=600)
    except Exception:
        return np.zeros((SPEC_C, SPEC_F, SPEC_T), dtype=np.float32)

    _, f, t = arr.shape
    if t < SPEC_T:
        arr = np.concatenate([arr, np.zeros((SPEC_C, f, SPEC_T - t), dtype=np.float32)], axis=2)
    else:
        s = (t - SPEC_T) // 2
        arr = arr[:, :, s : s + SPEC_T]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class InMemorySpecCache:
    """Load each ``(spectrogram_id, offset)`` from parquet once, then serve from RAM.

    Parquet I/O on the near-full iCloud disk dominates wall-clock when every epoch
    re-reads ~11k files; caching makes all epochs after the first compute-bound.
    Tensors are stored as float16 (~2.7 GB for the full set) and returned as float32.

    Requires ``num_workers=0`` so the single-process cache persists across epochs
    (DataLoader workers would each keep a private, throwaway copy).
    """

    def __init__(self, data_root: Path | None = None, dtype=np.float16) -> None:
        self.cache: dict[tuple[int, float], np.ndarray] = {}
        self.data_root = data_root
        self.dtype = dtype

    def __call__(self, spectrogram_id, offset_seconds: float = 0.0) -> np.ndarray:
        key = (int(spectrogram_id), round(float(offset_seconds), 3))
        arr = self.cache.get(key)
        if arr is None:
            arr = load_spec_tensor(spectrogram_id, offset_seconds=offset_seconds, data_root=self.data_root)
            arr = arr.astype(self.dtype)
            self.cache[key] = arr
        return arr.astype(np.float32)

    def __len__(self) -> int:
        return len(self.cache)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SpecDataset(Dataset):
    """``(spectrogram, soft_label)`` loader over a metadata DataFrame.

    Parameters
    ----------
    df
        Rows with at least ``spectrogram_id`` (and ``soft_label_*`` if labelled).
    load_fn
        Callable ``(spectrogram_id, offset_seconds) -> (4, F, T) ndarray``.
        Defaults to :func:`load_spec_tensor`; inject a stub in tests.
    with_label
        If True, ``__getitem__`` returns ``(x, soft_label)``; else just ``x``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        load_fn: Callable[..., np.ndarray] | None = None,
        with_label: bool = False,
        soft_classes: Sequence[str] = CLASSES_6,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.load_fn = load_fn or load_spec_tensor
        self.with_label = with_label
        self.soft_cols = [f"soft_label_{c}" for c in soft_classes]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        x = self.load_fn(
            row["spectrogram_id"],
            offset_seconds=row.get("spectrogram_label_offset_seconds", 0) or 0,
        )
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if not self.with_label:
            return x
        soft = np.array([row[c] for c in self.soft_cols], dtype=np.float32)
        soft = soft / max(soft.sum(), 1e-8)
        return x, torch.from_numpy(soft)
