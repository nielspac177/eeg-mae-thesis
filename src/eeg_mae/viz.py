"""Publication figure styling — colourblind-safe palette and consistent defaults.

Used by the latent map and the ``make_figures`` CLI so every figure in the memoria
shares one visual language (Okabe–Ito palette, readable fonts, light grid).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from .constants import CLASSES_6

# Okabe-Ito colourblind-safe palette, mapped to the six HMS classes.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#999999"]
CLASS_COLORS = dict(zip(CLASSES_6, OKABE_ITO, strict=True))


def set_style() -> None:
    """Apply a clean, consistent matplotlib style for all report figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def class_color(name: str) -> str:
    return CLASS_COLORS.get(name, "#333333")


def new_axes(figsize=(6, 4)):
    set_style()
    return plt.subplots(figsize=figsize)
