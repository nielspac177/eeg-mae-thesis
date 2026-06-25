"""CLI: self-supervised MAE pretraining (resumable), with encoder snapshots.

The MAE is trained on the **unique unlabelled spectrograms**. Training is resumable:
rerun the same command after an interruption and it continues from the last epoch.
Encoder snapshots are saved at the epochs given by ``--snapshots`` (default 30/60/90)
so experiment 3 (epoch sweep) can evaluate each without retraining.

Example
-------
    eeg-mae-pretrain --epochs 90 --enc-dim 192 --enc-heads 3 --snapshots 30 60 90
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .. import paths
from ..cache import default_loader
from ..data import SpecDataset, load_train_meta
from ..device import pick_device
from ..models import SpecMAE
from ..trainer import ResumableTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-supervised spectrogram MAE pretraining.")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--enc-dim", type=int, default=192)
    p.add_argument("--enc-heads", type=int, default=3)
    p.add_argument("--enc-depth", type=int, default=6)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--snapshots", type=int, nargs="*", default=[30, 60, 90])
    p.add_argument("--run-name", type=str, default=None, help="defaults to mae_d{enc_dim}_h{heads}")
    p.add_argument("--device", type=str, default=None, help="cpu|mps|cuda (auto if unset)")
    p.add_argument("--limit", type=int, default=None, help="cap #spectrograms (smoke tests)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    paths.ensure_dirs()
    run_name = args.run_name or f"mae_d{args.enc_dim}_h{args.enc_heads}"
    run_dir = paths.RUNS_DIR / "pretrain" / run_name
    print(paths.describe())
    print(f"\nPretraining '{run_name}' on {device} for {args.epochs} epochs.")

    # Unlabelled pretraining set: one row per unique spectrogram.
    meta = load_train_meta()
    unique = meta.drop_duplicates("spectrogram_id").reset_index(drop=True)
    if args.limit:
        unique = unique.head(args.limit)
    # Persistent memmap cache (if built) keeps every epoch compute-bound on MPS.
    cache = default_loader()
    ds = SpecDataset(unique, load_fn=cache, with_label=False)

    # Small held-out split for monitoring reconstruction loss (no labels involved).
    n = len(ds)
    n_val = max(min(200, n // 10), 1)
    perm = np.random.RandomState(0).permutation(n)
    val_ds = Subset(ds, perm[:n_val].tolist())
    train_ds = Subset(ds, perm[n_val:].tolist())
    print(f"spectrograms: {n}  (train {len(train_ds)} / val {len(val_ds)})")

    model = SpecMAE(
        enc_dim=args.enc_dim, enc_heads=args.enc_heads, enc_depth=args.enc_depth,
        mask_ratio=args.mask_ratio,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    trainer = ResumableTrainer(
        model, optimizer, loss_fn=lambda *_: None, scheduler=scheduler,
        device=device, run_dir=run_dir, grad_clip=1.0, seed=args.seed,
    )

    def make_train_loader(generator):
        return DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, generator=generator)

    def make_val_loader():
        return DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Train in segments so we can drop an encoder snapshot at each requested epoch.
    snap_dir = run_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for target in sorted(set(args.snapshots) | {args.epochs}):
        if target > args.epochs:
            continue
        trainer.fit(make_train_loader, num_epochs=target, make_val_loader=make_val_loader, progress=True)
        if target in args.snapshots:
            snap_path = snap_dir / f"encoder_ep{target}.pt"
            torch.save(
                {"state_dict": model.state_dict(),
                 "enc_dim": args.enc_dim, "enc_heads": args.enc_heads, "enc_depth": args.enc_depth,
                 "epoch": target, "val_recon_loss": trainer.history["val_loss"][-1]},
                snap_path,
            )
            print(f"  snapshot @ epoch {target} -> {snap_path}  (val recon {trainer.history['val_loss'][-1]:.4f})")

    # Persist the recon-loss curve for experiment 3 figures.
    import json

    (run_dir / "recon_history.json").write_text(json.dumps(trainer.history, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
