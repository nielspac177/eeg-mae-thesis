"""Device selection — prefers Apple MPS, then CUDA, then CPU."""
from __future__ import annotations

import torch


def pick_device(requested: str | None = None) -> torch.device:
    """Return a torch device. ``requested`` (cpu|mps|cuda) overrides autodetection."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
