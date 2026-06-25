"""Filesystem layout — all paths are configurable via environment variables.

The raw HMS dataset and large derived artifacts (``*.parquet``, ``*.npz``,
``*.pt``) are **never** committed to git (see ``.gitignore``).  They live outside
the repo and are referenced through these resolved paths, so the package stays
small and reproducible while reading data in place.

Defaults assume this package sits at ``<project>/eeg_mae`` and the dataset at
``<project>/hms-harmful-brain-activity-classification`` with derived files under
``<project>/data/processed`` — i.e. the existing project layout.  Override any of
them with environment variables:

    EEG_MAE_DATA_ROOT    -> raw competition data (train.csv, train_spectrograms/, ...)
    EEG_MAE_PROCESSED    -> derived arrays (data/processed)
    EEG_MAE_RUNS         -> checkpoints + run state (default: <repo>/runs)
    EEG_MAE_RESULTS      -> small tracked CSV/JSON results (default: <repo>/results)
    EEG_MAE_FIGURES      -> generated figures (default: <repo>/results/figures)
"""
from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/eeg_mae/paths.py  -> parents[2] == <repo> (the eeg_mae/ git root)
REPO_ROOT = Path(__file__).resolve().parents[2]
# <repo>/..  == the surrounding project that holds the dataset + data/processed.
PROJECT_ROOT = REPO_ROOT.parent


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


# Raw competition data: train.csv, train_spectrograms/, train_eegs/, ...
DATA_ROOT = _env_path(
    "EEG_MAE_DATA_ROOT", PROJECT_ROOT / "hms-harmful-brain-activity-classification"
)

# Derived arrays produced by the original notebooks (mae_features.npz, OOF caches).
PROCESSED_DIR = _env_path("EEG_MAE_PROCESSED", PROJECT_ROOT / "data" / "processed")

# Local (non-iCloud) base for large run artifacts. Checkpoints MUST NOT live on the
# iCloud-synced project disk: macOS evicts them under disk pressure, and a read during
# eviction yields a truncated/dataless file (torch.load -> UnpicklingError). Keep them
# next to the spectrogram cache on local disk.
LOCAL_BASE = Path(
    os.environ.get("EEG_MAE_CACHE", str(Path.home() / ".cache" / "eeg_mae"))
).expanduser()

# Run outputs (checkpoints, snapshots, logs) — local disk, git-ignored.
RUNS_DIR = _env_path("EEG_MAE_RUNS", LOCAL_BASE / "runs")
# Small, tracked results stay in the repo; figures are regenerable.
RESULTS_DIR = _env_path("EEG_MAE_RESULTS", REPO_ROOT / "results")
FIGURES_DIR = _env_path("EEG_MAE_FIGURES", REPO_ROOT / "results" / "figures")

TRAIN_CSV = DATA_ROOT / "train.csv"
TRAIN_SPECTROGRAMS = DATA_ROOT / "train_spectrograms"
TRAIN_EEGS = DATA_ROOT / "train_eegs"


def ensure_dirs() -> None:
    """Create the writable output directories if they do not yet exist."""
    for d in (RUNS_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable summary of the resolved layout (used by CLIs at startup)."""
    return (
        f"REPO_ROOT     = {REPO_ROOT}\n"
        f"DATA_ROOT     = {DATA_ROOT}  (exists={DATA_ROOT.exists()})\n"
        f"PROCESSED_DIR = {PROCESSED_DIR}  (exists={PROCESSED_DIR.exists()})\n"
        f"RUNS_DIR      = {RUNS_DIR}\n"
        f"RESULTS_DIR   = {RESULTS_DIR}\n"
        f"FIGURES_DIR   = {FIGURES_DIR}"
    )
