"""CLI: TTA the saved fold models and re-ensemble (no retraining).

Loads each model's per-fold checkpoints, produces a TTA'd OOF (mild SpecAugment,
averaged passes), and ensembles all TTA'd OOFs. Compares against the no-TTA ensemble.

Example
-------
    eeg-mae-tta-push --passes 6
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from .. import paths
from ..cache import default_loader
from ..data import label_subset, load_train_meta, soft_label_matrix
from ..device import pick_device
from ..ensemble import geometric_mean, weighted_search
from ..metrics import kl_divergence
from ..tta import build_factories, tta_oof


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="TTA the fold models and re-ensemble.")
    p.add_argument("--passes", type=int, default=6)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    device = pick_device(args.device)
    paths.ensure_dirs()

    meta = load_train_meta()
    label_meta = label_subset(meta, high_agreement_only=True)
    labels = soft_label_matrix(label_meta)
    cache = default_loader()
    factories = build_factories(device)

    tta_oofs = {}
    for name, factory in factories.items():
        print(f"TTA {name} ...")
        oof = tta_oof(name, factory, label_meta, labels, load_fn=cache, device=device, n_passes=args.passes)
        tta_oofs[name] = oof
        print(f"  {name} +TTA KL = {kl_divergence(labels, oof):.4f}")

    names = list(tta_oofs)
    stack = np.stack([tta_oofs[n] for n in names])
    gm = geometric_mean(stack)
    w, _, kl_w = weighted_search(stack, labels)

    rows = [{"method": f"{n}+TTA", "kl_overall": round(kl_divergence(labels, tta_oofs[n]), 4)} for n in names]
    rows.append({"method": "geometric+TTA", "kl_overall": round(kl_divergence(labels, gm), 4)})
    rows.append({"method": "weighted+TTA", "kl_overall": round(kl_w, 4)})

    print("\n=== Ensemble (with TTA) OOF KL ===")
    for r in rows:
        print(f"  {r['method']:36s} KL = {r['kl_overall']}")
    print(f"  weights ({', '.join(names)}) = {np.round(w, 3).tolist()}")
    best = min(rows, key=lambda r: r["kl_overall"])
    print(f"\n-> best with TTA: {best['method']}  KL = {best['kl_overall']}")

    out = paths.RESULTS_DIR / "ensemble_tta.csv"
    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["method", "kl_overall"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
