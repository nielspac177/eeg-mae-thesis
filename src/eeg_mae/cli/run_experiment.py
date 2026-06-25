"""CLI: run a thesis experiment from a YAML config (resumable, cached OOF).

A config describes one study. Its ``variants`` list expands into multiple runs
(e.g. head depths, ``enc_dim`` values, frozen-vs-finetune), each sharing the base
settings. Results are written as one CSV row per variant under ``results/`` and the
OOF predictions are cached, so re-running only computes what is missing.

Two kinds are supported:

* ``supervised`` — MAE encoder + MLP head, patient-grouped 5-fold OOF, soft-label KL.
* ``latent`` — encode the labelled set and write a t-SNE + UMAP map (experiment 7).

Example
-------
    eeg-mae-experiment configs/exp3_epoch_sweep.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import torch
import yaml

from .. import paths
from ..cv import OOFConfig, run_oof
from ..data import InMemorySpecCache, label_subset, load_train_meta, soft_label_matrix
from ..device import pick_device
from ..heads import MLPHead
from ..models import MAEClassifier, SpecMAE


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_variants(cfg: dict) -> list[dict]:
    """A config with N ``variants`` -> N flat run-configs; otherwise a single run."""
    variants = cfg.get("variants")
    base = {k: v for k, v in cfg.items() if k != "variants"}
    if not variants:
        return [base]
    runs = []
    for v in variants:
        merged = _merge(base, v)
        merged["name"] = f"{base['name']}__{v.get('name', 'variant')}"
        runs.append(merged)
    return runs


def load_encoder(run: dict, device: torch.device) -> SpecMAE:
    """Build a SpecMAE and load a pretrained snapshot if ``pretrained`` is set."""
    enc = SpecMAE(
        enc_dim=run.get("enc_dim", 192),
        enc_heads=run.get("enc_heads", 3),
        enc_depth=run.get("enc_depth", 6),
    )
    snap = run.get("pretrained")
    if snap:
        snap_path = Path(snap)
        if not snap_path.is_absolute():
            snap_path = paths.REPO_ROOT / snap_path
        ckpt = torch.load(snap_path, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["state_dict"])
        run.setdefault("_val_recon_loss", ckpt.get("val_recon_loss"))
    return enc.to(device)


def make_classifier_factory(run: dict, device: torch.device):
    head_cfg = run.get("head", {})

    def factory():
        enc = load_encoder(run, device)
        head = MLPHead(
            enc.enc_dim,
            n_classes=6,
            depth=head_cfg.get("depth", 2),
            hidden=head_cfg.get("hidden", 256),
            dropout=head_cfg.get("dropout", 0.1),
        )
        clf = MAEClassifier(
            enc, head, pooling=run.get("pooling", "cls"),
            freeze_encoder=run.get("freeze_encoder", True),
        )
        return clf.to(device)

    return factory


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_supervised(run: dict, meta, device, progress: bool, cache=None) -> dict:
    label_meta = label_subset(meta, high_agreement_only=run.get("high_agreement_only", True))
    if run.get("limit"):  # dev/smoke subsample
        label_meta = label_meta.head(int(run["limit"])).reset_index(drop=True)
    labels = soft_label_matrix(label_meta)

    t = run.get("train", {})
    oof_cfg = OOFConfig(
        name=run["name"],
        epochs=t.get("epochs", 15),
        batch_size=t.get("batch_size", 32),
        lr=t.get("lr", 5e-5),
        weight_decay=t.get("weight_decay", 0.05),
        use_specaugment=t.get("use_specaugment", True),
        mixup_alpha=t.get("mixup_alpha", 0.2),
        encoder_lr=t.get("encoder_lr", None),
        num_workers=0,  # in-memory cache requires a single process (see InMemorySpecCache)
        seed=run.get("seed", 42),
    )
    factory = make_classifier_factory(run, device)
    n_params = sum(p.numel() for p in factory().parameters())
    res = run_oof(oof_cfg, factory, label_meta, labels, load_fn=cache, device=device, progress=progress)

    return {
        "name": run["name"],
        "kind": "supervised",
        "enc_dim": run.get("enc_dim", 192),
        "enc_heads": run.get("enc_heads", 3),
        "pooling": run.get("pooling", "cls"),
        "freeze_encoder": run.get("freeze_encoder", True),
        "head_depth": run.get("head", {}).get("depth", 2),
        "head_hidden": run.get("head", {}).get("hidden", 256),
        "encoder_lr": run.get("train", {}).get("encoder_lr"),
        "epochs": oof_cfg.epochs,
        "params": int(n_params),
        "val_recon_loss": run.get("_val_recon_loss"),
        "kl_overall": round(res["kl_overall"], 4),
        "kl_fold_mean": round(float(res["fold_kls"].mean()), 4),
        "kl_fold_std": round(float(res["fold_kls"].std()), 4),
    }


def run_latent(run: dict, meta, device, progress: bool) -> dict:
    """Experiment 7: encode the labelled set, write a t-SNE + UMAP latent map."""
    from ..latent import latent_map

    label_meta = label_subset(meta, high_agreement_only=run.get("high_agreement_only", True))
    labels = soft_label_matrix(label_meta)
    enc = load_encoder(run, device)
    out_path = latent_map(
        enc, label_meta, labels, pooling=run.get("pooling", "cls"),
        device=device, name=run["name"], limit=run.get("limit"),
    )
    return {"name": run["name"], "kind": "latent", "figure": str(out_path)}


def write_results(study_name: str, rows: list[dict]) -> Path:
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.RESULTS_DIR / f"{study_name}.csv"
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a eeg_mae experiment config.")
    p.add_argument("config", type=str, help="path to a configs/*.yaml file")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = pick_device(args.device)
    paths.ensure_dirs()

    cfg = yaml.safe_load(Path(args.config).read_text())
    study_name = cfg["name"]
    runs = _expand_variants(cfg)
    print(f"Study '{study_name}': {len(runs)} run(s) on {device}")
    meta = load_train_meta()
    # One in-memory spectrogram cache shared across all variants of this study.
    cache = InMemorySpecCache()

    rows = []
    for run in runs:
        kind = run.get("kind", "supervised")
        print(f"\n--> {run['name']} [{kind}]")
        if kind == "supervised":
            row = run_supervised(run, meta, device, progress=not args.no_progress, cache=cache)
            print(f"    KL = {row['kl_overall']}  (folds {row['kl_fold_mean']} ± {row['kl_fold_std']})")
        elif kind == "latent":
            row = run_latent(run, meta, device, progress=not args.no_progress)
            print(f"    figure -> {row['figure']}")
        else:
            raise ValueError(f"unknown kind={kind!r}")
        rows.append(row)

    out = write_results(study_name, rows)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
