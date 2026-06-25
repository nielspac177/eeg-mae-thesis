"""Persistent, memmap-backed spectrogram cache on local (non-iCloud) disk.

The raw parquet lives on a near-full, iCloud-synced volume, so files are frequently
evicted and re-downloaded on demand — making each epoch's reads pathologically slow
(~1 s/spec). We therefore materialise every needed ``(spectrogram_id, offset)`` once
into a single float16 memmap on **local disk** (``~/.cache/eeg_mae`` by default), with
a JSON index. Training then reads from the memmap (kept warm in the OS page cache, and
shared across processes), so all four pretrain runs and every experiment pay the slow
parquet cost only once, collectively.

Build is parallelised with threads because the bottleneck is I/O / iCloud download,
not CPU — `pandas.read_parquet` releases the GIL during the read.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SPEC_C, SPEC_F, SPEC_T
from .data import load_spec_tensor

DEFAULT_CACHE_DIR = Path(
    os.environ.get("EEG_MAE_CACHE", str(Path.home() / ".cache" / "eeg_mae"))
).expanduser()
DATA_NPY = "spec_cache_f16.npy"
INDEX_JSON = "spec_cache_index.json"


def _key(spec_id, offset) -> str:
    return f"{int(spec_id)}|{round(float(offset), 3)}"


def pairs_from_meta(df: pd.DataFrame) -> list[tuple[int, float]]:
    """Extract the ``(spectrogram_id, offset_seconds)`` pairs a meta frame will request."""
    offs = df.get("spectrogram_label_offset_seconds")
    out = []
    for i, sid in enumerate(df["spectrogram_id"].to_numpy()):
        off = 0.0 if offs is None else (offs.iloc[i] or 0.0)
        out.append((int(sid), float(off)))
    return out


def build_cache(pairs, cache_dir: Path = DEFAULT_CACHE_DIR, n_threads: int = 8, verbose: bool = True) -> Path:
    """Materialise ``pairs`` into a memmap + index under ``cache_dir`` (skips existing keys).

    Returns the cache directory. Safe to call repeatedly: if the memmap exists, only
    missing keys are appended (a new memmap is written merging old + new).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / INDEX_JSON
    data_path = cache_dir / DATA_NPY

    existing: dict[str, int] = {}
    old_data = None
    if index_path.exists() and data_path.exists():
        existing = json.loads(index_path.read_text())
        old_data = np.load(data_path, mmap_mode="r")

    wanted = {_key(s, o): (s, o) for s, o in pairs}
    missing = {k: v for k, v in wanted.items() if k not in existing}
    if not missing:
        if verbose:
            print(f"[cache] up to date: {len(existing)} entries at {cache_dir}")
        return cache_dir

    n_total = len(existing) + len(missing)
    if verbose:
        print(f"[cache] building {len(missing)} new / {n_total} total entries "
              f"with {n_threads} threads -> {cache_dir}")

    new_index = dict(existing)
    tmp = cache_dir / (DATA_NPY + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(n_total, SPEC_C, SPEC_F, SPEC_T))
    # Copy already-cached rows over.
    if old_data is not None:
        out[: len(existing)] = old_data[: len(existing)]

    rows = iter(range(len(existing), n_total))
    keys = list(missing)

    def _load(k):
        s, o = missing[k]
        return k, load_spec_tensor(s, offset_seconds=o).astype(np.float16)

    done = 0
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(_load, k) for k in keys]
        for fut in as_completed(futures):
            k, arr = fut.result()
            r = next(rows)
            out[r] = arr
            new_index[k] = r
            done += 1
            if verbose and done % 500 == 0:
                print(f"[cache]   {done}/{len(missing)}")

    out.flush()
    del out
    os.replace(tmp, data_path)
    index_path.write_text(json.dumps(new_index))
    if verbose:
        gb = data_path.stat().st_size / 1e9
        print(f"[cache] done: {n_total} entries, {gb:.2f} GB at {data_path}")
    return cache_dir


def default_loader(cache_dir: Path = DEFAULT_CACHE_DIR):
    """Return a memmap cache loader if the persistent cache exists, else an in-RAM one."""
    cache_dir = Path(cache_dir)
    if (cache_dir / INDEX_JSON).exists() and (cache_dir / DATA_NPY).exists():
        print(f"[cache] using persistent memmap cache at {cache_dir}")
        return MemmapSpecCache(cache_dir)
    from .data import InMemorySpecCache

    print("[cache] persistent cache not found; falling back to in-RAM cache "
          "(run eeg-mae-build-cache to avoid re-reading parquet each process)")
    return InMemorySpecCache()


class MemmapSpecCache:
    """Read cached spectrogram tensors from the persistent memmap; fall back to parquet.

    A drop-in ``load_fn`` for :class:`~eeg_mae.data.SpecDataset` (call signature
    ``(spectrogram_id, offset_seconds=0.0) -> (4, 100, 300) float32``).
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        cache_dir = Path(cache_dir)
        self.index = json.loads((cache_dir / INDEX_JSON).read_text())
        self.data = np.load(cache_dir / DATA_NPY, mmap_mode="r")
        self.misses = 0

    def __call__(self, spectrogram_id, offset_seconds: float = 0.0) -> np.ndarray:
        row = self.index.get(_key(spectrogram_id, offset_seconds))
        if row is None:
            self.misses += 1
            return load_spec_tensor(spectrogram_id, offset_seconds=offset_seconds)
        return np.asarray(self.data[row], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.index)
