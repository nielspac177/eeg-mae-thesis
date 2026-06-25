"""CLI: regenerate report figures from ``results/`` — no manual steps.

Reads the per-study CSVs written by ``run_experiment`` (and the KL-progression CSV if
present) and renders the figures that go into the memoria's "Ensayos y resultados".
Latent maps are produced by the ``latent`` experiment itself; this CLI draws the
quantitative figures (KL vs head depth, KL vs epochs, KL vs enc_dim, frozen-vs-finetune,
CLS-vs-mean, and the overall KL-progression bar/line).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import paths
from ..viz import set_style


def _bar(df: pd.DataFrame, x: str, title: str, out: Path, ystd: str | None = None):
    set_style()
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(df)), 4))
    yerr = df[ystd] if ystd and ystd in df else None
    ax.bar(df[x].astype(str), df["kl_overall"], yerr=yerr, capsize=4, color="#0072B2")
    ax.set_ylabel("OOF KL divergence (lower is better)")
    ax.set_xlabel(x)
    ax.set_title(title)
    for i, v in enumerate(df["kl_overall"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def _study_figure(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if "kl_overall" not in df.columns:
        return
    name = csv_path.stem
    out = paths.FIGURES_DIR / f"{name}.png"
    # Pick the most informative x-axis that actually exists in this study's CSV.
    candidates = [
        ("head_depth", "Head depth vs OOF KL"),
        ("enc_dim", "Encoder width vs OOF KL"),
        ("epochs", "MAE pretraining epochs vs OOF KL"),
        ("pooling", "CLS vs mean pooling"),
        ("method", "Ensemble methods vs OOF KL"),
        ("name", name),
    ]
    for x, title in candidates:
        if x in df.columns:
            ystd = "kl_fold_std" if "kl_fold_std" in df.columns else None
            _bar(df, x, title, out, ystd=ystd)
            break


def _progression_figure() -> None:
    """KL progression line, continuing the project's existing CSV if available."""
    candidates = [
        paths.RESULTS_DIR / "kl_progression.csv",
        paths.PROCESSED_DIR / "03b2_kl_progression.csv",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        return
    df = pd.read_csv(src)
    ycol = "OOF KL" if "OOF KL" in df.columns else df.columns[-1]
    xcol = "stage" if "stage" in df.columns else df.columns[0]
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(df)), df[ycol], "o-", color="#0072B2")
    ax.axhline(0.30, ls="--", color="#009E73", label="Kaggle top zone (~0.30)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df[xcol], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("OOF KL divergence")
    ax.set_title("KL progression across stages")
    ax.legend()
    out = paths.FIGURES_DIR / "kl_progression.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Regenerate report figures from results/.")
    p.add_argument("--results", type=str, default=None, help="results dir (default: package results/)")
    args = p.parse_args(argv)

    results_dir = Path(args.results) if args.results else paths.RESULTS_DIR
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rendering figures from {results_dir} -> {paths.FIGURES_DIR}")

    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name == "kl_progression.csv":
            continue
        _study_figure(csv_path)
    _progression_figure()
    print("done.")


if __name__ == "__main__":
    main()
