"""Canonical shapes and class names shared across the package.

These mirror the data pipeline used in the original ``03b`` notebook so that a
model trained there and one trained here see byte-identical tensors.
"""
from __future__ import annotations

# Spectrogram canonical shape: 4 EEG regions x 100 frequency bins x 300 time bins (~10 min).
SPEC_C = 4
SPEC_F = 100
SPEC_T = 300

# Patch size for the ViT tokeniser -> (100/10) x (300/10) = 10 x 30 = 300 patches.
PATCH_F = 10
PATCH_T = 10

# The 6-class soft-label target used by the Kaggle KL metric, and the 5-class
# (non-"Other") view used by some ablations.
CLASSES_6 = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
CLASSES_5 = ["Seizure", "LPD", "GPD", "LRDA", "GRDA"]
