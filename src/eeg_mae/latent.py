"""Experiment 7: t-SNE + UMAP map of the MAE latent space.

Encodes the labelled spectrograms with the (frozen) encoder, projects to 2-D with
both t-SNE and UMAP, and plots them side by side. Per the thesis spec, each marker's
**colour = hard label** (argmax class) and **alpha = soft-label confidence** (max
class probability) — so confident points are opaque and ambiguous ones fade out,
visualising how the latent space separates certain vs uncertain cases.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import paths
from .constants import CLASSES_6
from .data import SpecDataset
from .viz import OKABE_ITO, set_style


@torch.no_grad()
def encode_dataset(encoder, label_meta, pooling="cls", device="cpu", batch_size=64, limit=None) -> np.ndarray:
    """Return ``(n, enc_dim)`` pooled encoder features for the labelled set."""
    if limit:
        label_meta = label_meta.head(limit)
    encoder = encoder.to(device).eval()
    loader = DataLoader(SpecDataset(label_meta, with_label=False), batch_size=batch_size, shuffle=False)
    feats = []
    for x in loader:
        feats.append(encoder.encode(x.to(device), pooling=pooling).cpu().numpy())
    return np.concatenate(feats)


def _scatter(ax, xy, hard, conf, title):
    for c, _name in enumerate(CLASSES_6):
        m = hard == c
        if not m.any():
            continue
        rgba = np.zeros((m.sum(), 4))
        rgba[:, :3] = mpl_to_rgb(OKABE_ITO[c])
        rgba[:, 3] = np.clip(conf[m], 0.15, 1.0)  # alpha = soft-label confidence
        ax.scatter(xy[m, 0], xy[m, 1], s=14, c=rgba, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def mpl_to_rgb(hexcolor: str):
    import matplotlib.colors as mcolors

    return np.array(mcolors.to_rgb(hexcolor))


def latent_map(
    encoder,
    label_meta,
    labels: np.ndarray,
    *,
    pooling: str = "cls",
    device: str | torch.device = "cpu",
    name: str = "latent",
    perplexity: float = 30.0,
    limit: int | None = None,
) -> Path:
    """Compute features, run t-SNE + UMAP, and save a two-panel figure. Returns its path."""
    from sklearn.manifold import TSNE

    if limit:
        labels = labels[:limit]
    feats = encode_dataset(encoder, label_meta, pooling=pooling, device=device, limit=limit)
    hard = labels.argmax(axis=1)
    conf = labels.max(axis=1)

    tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(feats) // 4)),
                init="pca", random_state=0).fit_transform(feats)
    try:
        import umap

        umap_xy = umap.UMAP(n_components=2, random_state=0).fit_transform(feats)
    except Exception as exc:  # umap optional; t-SNE alone still produces a figure
        print(f"[latent] UMAP unavailable ({exc}); plotting t-SNE only")
        umap_xy = None

    set_style()
    ncols = 2 if umap_xy is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.5), squeeze=False)
    _scatter(axes[0, 0], tsne, hard, conf, f"t-SNE · {pooling} pooling")
    if umap_xy is not None:
        _scatter(axes[0, 1], umap_xy, hard, conf, f"UMAP · {pooling} pooling")

    handles = [mpatches.Patch(color=OKABE_ITO[c], label=CLASSES_6[c]) for c in range(len(CLASSES_6))]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("MAE latent space — colour = class, opacity = soft-label confidence", y=1.02)

    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES_DIR / f"{name}_latent.png"
    fig.savefig(out)
    plt.close(fig)

    # Save the embedding so figures can be regenerated/recoloured without re-encoding.
    np.savez(paths.RESULTS_DIR / f"{name}_embedding.npz",
             tsne=tsne, umap=umap_xy if umap_xy is not None else np.empty((0, 2)),
             hard=hard, conf=conf)
    return out
