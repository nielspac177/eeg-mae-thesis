"""EfficientNet (timm) baseline for the ensemble — a different inductive bias than the MAE.

A convolutional model trained on the same 4-region spectrograms gives predictions that
are decorrelated from the ViT-MAE's, so a soft ensemble of the two typically beats either
alone (the strategy that took the original notebook to ~0.74 KL). Ported from
``src/mae_models.py``.
"""
from __future__ import annotations

from .constants import SPEC_C


def make_effnet_classifier(
    model_name: str = "efficientnet_b0",
    in_chans: int = SPEC_C,
    n_classes: int = 6,
    pretrained: bool = True,
    drop_rate: float = 0.2,
):
    """A timm CNN with a 4-channel stem and a 6-logit head.

    ``in_chans=4`` keeps the EEG regions as separate channels; timm seeds the first
    conv by averaging ImageNet weights across input channels. Output logits feed the
    same :class:`~eeg_mae.losses.SoftLabelKLLoss` as the MAE classifier.
    """
    import timm

    return timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=n_classes,
        drop_rate=drop_rate,
    )
