"""CLI: two-stage full-data fine-tuning of the champion MAE (the push toward ~0.30).

Stage 1 on the >=4-vote set, stage 2 on the >=8-vote high-quality set, OOF KL on
held-out >=8-vote patients. This is the MAE-centric version of the competition recipe.

Example
-------
    eeg-mae-twostage --pooling mean --epochs-s1 6 --epochs-s2 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .. import paths
from ..cache import default_loader
from ..data import load_train_meta
from ..device import pick_device
from ..heads import MLPHead
from ..metrics import kl_divergence
from ..models import MAEClassifier, SpecMAE
from ..twostage import TwoStageConfig, run_twostage_oof


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Two-stage full-data MAE fine-tune.")
    p.add_argument("--name", default=None)
    p.add_argument("--enc-dim", type=int, default=256)
    p.add_argument("--enc-heads", type=int, default=8)
    p.add_argument("--pretrained", default="runs/pretrain/mae_d256_h8/snapshots/encoder_ep90.pt")
    p.add_argument("--pooling", default="mean", choices=["cls", "mean"])
    p.add_argument("--head-depth", type=int, default=3)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--min-votes-s1", type=int, default=4)
    p.add_argument("--min-votes-eval", type=int, default=8)
    p.add_argument("--epochs-s1", type=int, default=6)
    p.add_argument("--epochs-s2", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    paths.ensure_dirs()
    name = args.name or f"twostage_mae_{args.pooling}_v{args.min_votes_s1}-{args.min_votes_eval}"

    snap = Path(args.pretrained)
    if not snap.is_absolute():
        parts = snap.parts
        snap = paths.RUNS_DIR.joinpath(*parts[1:]) if parts and parts[0] == "runs" else paths.REPO_ROOT / snap

    def mae_factory():
        enc = SpecMAE(enc_dim=args.enc_dim, enc_heads=args.enc_heads)
        enc.load_state_dict(torch.load(snap, map_location=device, weights_only=False)["state_dict"])
        head = MLPHead(args.enc_dim, n_classes=6, depth=args.head_depth, hidden=args.head_hidden, dropout=0.2)
        return MAEClassifier(enc, head, pooling=args.pooling, freeze_encoder=False).to(device)

    cfg = TwoStageConfig(
        name=name, min_votes_s1=args.min_votes_s1, min_votes_eval=args.min_votes_eval,
        epochs_s1=args.epochs_s1, epochs_s2=args.epochs_s2,
    )
    print(f"Two-stage '{name}' on {device}: stage1>={args.min_votes_s1}v, stage2/eval>={args.min_votes_eval}v")
    meta = load_train_meta()
    cache = default_loader()
    res = run_twostage_oof(cfg, mae_factory, meta, load_fn=cache, device=device, progress=not args.no_progress)
    print(f"\n=== {name} OOF KL (>= {args.min_votes_eval}-vote eval) = {res['kl_overall']:.4f} "
          f"(folds {res['fold_kls'].mean():.4f} ± {res['fold_kls'].std():.4f}) ===")

    # Append to a dedicated results CSV.
    import csv
    out = paths.RESULTS_DIR / "twostage.csv"
    write_header = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "pooling", "min_votes_s1", "min_votes_eval", "kl_overall"])
        if write_header:
            w.writeheader()
        w.writerow({"name": name, "pooling": args.pooling, "min_votes_s1": args.min_votes_s1,
                    "min_votes_eval": args.min_votes_eval, "kl_overall": round(res["kl_overall"], 4)})
    _ = kl_divergence  # (kept import explicit for reporting parity)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
