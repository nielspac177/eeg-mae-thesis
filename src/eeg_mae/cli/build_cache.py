"""CLI: pre-build the persistent spectrogram cache (one-time, parallel I/O).

Materialises every ``(spectrogram_id, offset)`` used by pretraining (unique
spectrograms) and by the supervised experiments (the labelled high-agreement subset)
into a local-disk memmap, so all later runs read from RAM/page-cache instead of the
slow iCloud parquet. Idempotent: rerun to add any new keys.

Example
-------
    eeg-mae-build-cache --threads 8
"""
from __future__ import annotations

import argparse

from ..cache import DEFAULT_CACHE_DIR, build_cache, pairs_from_meta
from ..data import label_subset, load_train_meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pre-build the persistent spectrogram cache.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--limit", type=int, default=None, help="cap #unique spectrograms (smoke)")
    args = p.parse_args(argv)

    meta = load_train_meta()
    unique = meta.drop_duplicates("spectrogram_id").reset_index(drop=True)
    if args.limit:
        unique = unique.head(args.limit)
    labelled = label_subset(meta, high_agreement_only=True)

    pairs = set(pairs_from_meta(unique)) | set(pairs_from_meta(labelled))
    print(f"unique pretrain specs: {len(unique)} · labelled specs: {len(labelled)} "
          f"· distinct (id,offset) pairs: {len(pairs)}")
    build_cache(sorted(pairs), cache_dir=args.cache_dir, n_threads=args.threads)


if __name__ == "__main__":
    main()
