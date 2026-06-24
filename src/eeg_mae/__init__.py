"""eeg_mae — self-supervised spectrogram Masked Autoencoder for HMS / EEG.

A clean, reproducible re-implementation of the notebook pipeline in
``notebooks/03b_spectrogram_mae`` and ``03b2_push_to_competitive_kl``, organised as
an installable package with resumable training and config-driven experiments.

Public surface (import from the top level):

    from eeg_mae import SpecMAE, MLPHead, MAEClassifier, kl_divergence

Author: Niels Pacheco Barrios.
"""
from __future__ import annotations

from .constants import CLASSES_5, CLASSES_6, PATCH_F, PATCH_T, SPEC_C, SPEC_F, SPEC_T
from .heads import MLPHead
from .metrics import kl_divergence
from .models import MAEClassifier, SpecMAE, patchify, unpatchify

__all__ = [
    "SpecMAE",
    "MAEClassifier",
    "MLPHead",
    "patchify",
    "unpatchify",
    "kl_divergence",
    "CLASSES_5",
    "CLASSES_6",
    "SPEC_C",
    "SPEC_F",
    "SPEC_T",
    "PATCH_F",
    "PATCH_T",
]

__version__ = "0.1.0"
