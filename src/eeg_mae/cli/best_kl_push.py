"""CLI: best-KL push — train CNN backbones and ensemble several OOF models.

The ViT-MAE and CNNs make decorrelated errors, so a soft blend of their out-of-fold
predictions beats any single model. Trains each requested CNN backbone OOF (resumable/
cached), loads any pre-computed OOF caches (e.g. fine-tuned MAE variants), and reports
arithmetic / geometric / weighted ensembles over all of them.

Example
-------
    eeg-mae-push --models efficientnet_b0,efficientnet_b1 \\
                 --oof exp5_finetune__finetune_enc1e5,push_mae_mean --epochs 12
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from .. import paths
from ..cache import default_loader
from ..cnn import make_effnet_classifier
from ..cv import OOFConfig, run_oof
from ..data import label_subset, load_train_meta, soft_label_matrix
from ..device import pick_device
from ..ensemble import arithmetic_mean, geometric_mean, weighted_search
from ..metrics import kl_divergence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CNN backbones and ensemble OOF models.")
    p.add_argument("--models", default="efficientnet_b0",
                   help="comma list of timm CNN archs to train OOF (e.g. efficientnet_b0,efficientnet_b1)")
    p.add_argument("--oof", default="exp5_finetune__finetune_enc1e5",
                   help="comma list of existing OOF npz names under results/oof/ to include")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p


def _load_oof(name: str) -> np.ndarray:
    path = paths.RESULTS_DIR / "oof" / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"OOF cache not found: {path}")
    return np.load(path)["oof"]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    paths.ensure_dirs()

    meta = load_train_meta()
    label_meta = label_subset(meta, high_agreement_only=True)
    labels = soft_label_matrix(label_meta)
    cache = default_loader()

    members: dict[str, np.ndarray] = {}

    # 1. Train each requested CNN backbone OOF (same patient-grouped folds).
    for arch in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"Training {arch} OOF on {device} ...")
        cfg = OOFConfig(
            name=f"push_{arch}", epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=0.01, use_specaugment=True, mixup_alpha=0.2, num_workers=0,
        )
        res = run_oof(
            cfg, lambda a=arch: make_effnet_classifier(a, n_classes=6).to(device),
            label_meta, labels, load_fn=cache, device=device, progress=True,
        )
        members[arch] = res["oof"]
        print(f"  {arch} OOF KL = {res['kl_overall']:.4f}")

    # 2. Load pre-computed OOF caches (fine-tuned MAE variants, etc.).
    for name in [o.strip() for o in args.oof.split(",") if o.strip()]:
        members[name] = _load_oof(name)
        print(f"  loaded {name} OOF KL = {kl_divergence(labels, members[name]):.4f}")

    names = list(members)
    stack = np.stack([members[n] for n in names])

    # 3. Ensemble.
    rows = [{"method": n, "kl_overall": round(kl_divergence(labels, members[n]), 4)} for n in names]
    am = arithmetic_mean(stack)
    rows.append({"method": "arithmetic", "kl_overall": round(kl_divergence(labels, am), 4)})
    gm = geometric_mean(stack)
    rows.append({"method": "geometric", "kl_overall": round(kl_divergence(labels, gm), 4)})
    w, blend, kl_w = weighted_search(stack, labels)
    rows.append({"method": "weighted", "kl_overall": round(kl_w, 4)})

    print("\n=== Ensemble OOF KL ===")
    for r in rows:
        print(f"  {r['method']:32s} KL = {r['kl_overall']}")
    print(f"  weighted weights ({', '.join(names)}) = {np.round(w, 3).tolist()}")
    best = min(rows, key=lambda r: r["kl_overall"])
    print(f"\n-> best: {best['method']}  KL = {best['kl_overall']}")

    out = paths.RESULTS_DIR / "ensemble.csv"
    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["method", "kl_overall"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
