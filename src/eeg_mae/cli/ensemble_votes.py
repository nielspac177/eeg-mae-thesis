"""CLI: ensemble two-stage OOF models on the >=N-vote high-quality eval.

All listed OOF caches must have been produced with the same ``min_votes_eval`` (so
their rows align to the same vote_subset). Reports arithmetic/geometric/weighted KL.

Example
-------
    eeg-mae-ensemble-votes --oof twostage_mae_mean_v4-8,twostage_mae_cls_v4-8,twostage_cnn_efficientnet_b0_v4-8
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from .. import paths
from ..data import load_train_meta, soft_label_matrix, vote_subset
from ..ensemble import arithmetic_mean, geometric_mean, weighted_search
from ..metrics import kl_divergence


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Ensemble two-stage OOFs on the >=N-vote eval.")
    p.add_argument("--oof", required=True, help="comma list of OOF npz names under results/oof/")
    p.add_argument("--min-votes-eval", type=int, default=8)
    args = p.parse_args(argv)

    meta = load_train_meta()
    labels = soft_label_matrix(vote_subset(meta, args.min_votes_eval))

    names = [n.strip() for n in args.oof.split(",") if n.strip()]
    members = {}
    for n in names:
        oof = np.load(paths.RESULTS_DIR / "oof" / f"{n}.npz")["oof"]
        if oof.shape[0] != labels.shape[0]:
            raise ValueError(f"{n}: {oof.shape[0]} rows != {labels.shape[0]} eval rows (vote mismatch?)")
        members[n] = oof

    stack = np.stack([members[n] for n in names])
    rows = [{"method": n, "kl_overall": round(kl_divergence(labels, members[n]), 4)} for n in names]
    rows.append({"method": "arithmetic", "kl_overall": round(kl_divergence(labels, arithmetic_mean(stack)), 4)})
    rows.append({"method": "geometric", "kl_overall": round(kl_divergence(labels, geometric_mean(stack)), 4)})
    w, _, kl_w = weighted_search(stack, labels)
    rows.append({"method": "weighted", "kl_overall": round(kl_w, 4)})

    print(f"=== Ensemble on >= {args.min_votes_eval}-vote eval ({len(names)} models) ===")
    for r in rows:
        print(f"  {r['method']:44s} KL = {r['kl_overall']}")
    print(f"  weighted weights = {dict(zip(names, np.round(w, 3).tolist(), strict=True))}")
    best = min(rows, key=lambda r: r["kl_overall"])
    print(f"\n-> best: {best['method']}  KL = {best['kl_overall']}")

    out = paths.RESULTS_DIR / "ensemble_votes.csv"
    with out.open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["method", "kl_overall"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
