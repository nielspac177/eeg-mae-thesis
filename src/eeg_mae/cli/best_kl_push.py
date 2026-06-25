"""CLI: best-KL push — add an EfficientNet OOF and ensemble it with the champion MAE.

The ViT-MAE and a CNN make decorrelated errors, so a soft blend of their out-of-fold
predictions beats either alone. Trains an EfficientNet-B0 OOF (resumable/cached), loads
the champion fine-tuned MAE OOF, and reports arithmetic/geometric/weighted ensembles.

Example
-------
    eeg-mae-push --mae-oof exp5_finetune__finetune_enc1e5 --epochs 12
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
    p = argparse.ArgumentParser(description="EfficientNet OOF + ensemble with champion MAE.")
    p.add_argument("--mae-oof", default="exp5_finetune__finetune_enc1e5",
                   help="name of the champion MAE OOF npz under results/oof/")
    p.add_argument("--effnet", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    paths.ensure_dirs()

    meta = load_train_meta()
    label_meta = label_subset(meta, high_agreement_only=True)
    labels = soft_label_matrix(label_meta)
    cache = default_loader()

    # 1. EfficientNet OOF (same patient-grouped folds as the MAE experiments).
    print(f"Training {args.effnet} OOF on {device} ...")
    cfg = OOFConfig(
        name=f"push_{args.effnet}", epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=0.01, use_specaugment=True, mixup_alpha=0.2, num_workers=0,
    )
    effnet_res = run_oof(
        cfg, lambda: make_effnet_classifier(args.effnet, n_classes=6).to(device),
        label_meta, labels, load_fn=cache, device=device, progress=True,
    )
    effnet_oof = effnet_res["oof"]
    print(f"EfficientNet OOF KL = {effnet_res['kl_overall']:.4f}")

    # 2. Champion MAE OOF (rows aligned: same label_subset order + same folds).
    mae_path = paths.RESULTS_DIR / "oof" / f"{args.mae_oof}.npz"
    if not mae_path.exists():
        raise FileNotFoundError(f"champion MAE OOF not found: {mae_path} (run exp5 first)")
    mae_oof = np.load(mae_path)["oof"]
    mae_kl = kl_divergence(labels, mae_oof)
    print(f"Champion MAE OOF KL  = {mae_kl:.4f}")

    # 3. Ensemble.
    stack = np.stack([mae_oof, effnet_oof])
    ens = {
        "mae_alone": (mae_oof, mae_kl),
        "effnet_alone": (effnet_oof, effnet_res["kl_overall"]),
        "arithmetic": (am := arithmetic_mean(stack), kl_divergence(labels, am)),
        "geometric": (gm := geometric_mean(stack), kl_divergence(labels, gm)),
    }
    w, blend, kl_w = weighted_search(stack, labels)
    ens["weighted"] = (blend, kl_w)

    print("\n=== Ensemble OOF KL ===")
    rows = []
    for name, (_, kl) in ens.items():
        print(f"  {name:14s} KL = {kl:.4f}")
        rows.append({"method": name, "kl_overall": round(float(kl), 4)})
    print(f"  weighted weights (mae, effnet) = {np.round(w, 3).tolist()}")

    best = min(ens.items(), key=lambda kv: kv[1][1])
    print(f"\n-> best: {best[0]}  KL = {best[1][1]:.4f}")

    out = paths.RESULTS_DIR / "ensemble.csv"
    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["method", "kl_overall"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
