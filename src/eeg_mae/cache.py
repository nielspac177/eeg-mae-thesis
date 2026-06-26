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


def _shards(cache_dir: Path) -> list[tuple[Path, Path]]:
    """All (data_npy, index_json) shard pairs: the base plus any spec_cache_shardN.*."""
    pairs = []
    base_d, base_i = cache_dir / DATA_NPY, cache_dir / INDEX_JSON
    if base_d.exists() and base_i.exists():
        pairs.append((base_d, base_i))
    for d in sorted(cache_dir.glob("spec_cache_shard*.npy")):
        i = d.with_name(d.stem + "_index.json")
        if i.exists():
            pairs.append((d, i))
    return pairs


def _all_keys(cache_dir: Path) -> set[str]:
    keys: set[str] = set()
    for _, idx in _shards(cache_dir):
        keys |= set(json.loads(idx.read_text()))
    return keys


def pairs_from_meta(df: pd.DataFrame) -> list[tuple[int, float]]:
    """Extract the ``(spectrogram_id, offset_seconds)`` pairs a meta frame will request."""
    offs = df.get("spectrogram_label_offset_seconds")
    out = []
    for i, sid in enumerate(df["spectrogram_id"].to_numpy()):
        off = 0.0 if offs is None else (offs.iloc[i] or 0.0)
        out.append((int(sid), float(off)))
    return out


def build_cache(pairs, cache_dir: Path = DEFAULT_CACHE_DIR, n_threads: int = 8, verbose: bool = True) -> Path:
    """Materialise ``pairs`` into the cache under ``cache_dir`` (skips already-cached keys).

    Sharded + append-only: the first build writes the base memmap; later builds write
    only the *missing* keys to a new ``spec_cache_shardN.npy`` instead of rewriting the
    base. This keeps peak disk at ``base + new_shard`` rather than ``2 * total`` — essential
    on the disk-constrained Mac when growing to the full >=3-vote (100k) set.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = _all_keys(cache_dir)
    wanted = {_key(s, o): (s, o) for s, o in pairs}
    missing = {k: v for k, v in wanted.items() if k not in existing}
    if not missing:
        if verbose:
            print(f"[cache] up to date: {len(existing)} entries at {cache_dir}")
        return cache_dir

    # First build -> base files; subsequent builds -> a fresh shard (no base rewrite).
    n_existing_shards = len(_shards(cache_dir))
    if n_existing_shards == 0:
        data_path, index_path = cache_dir / DATA_NPY, cache_dir / INDEX_JSON
    else:
        n = 1
        while (cache_dir / f"spec_cache_shard{n}.npy").exists():
            n += 1
        data_path = cache_dir / f"spec_cache_shard{n}.npy"
        index_path = cache_dir / f"spec_cache_shard{n}_index.json"
    if verbose:
        print(f"[cache] building {len(missing)} new entries (total will be {len(existing)+len(missing)}) "
              f"with {n_threads} threads -> {data_path.name}")

    tmp = data_path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(len(missing), SPEC_C, SPEC_F, SPEC_T))
    keys = list(missing)
    index: dict[str, int] = {}

    def _load(k):
        s, o = missing[k]
        return k, load_spec_tensor(s, offset_seconds=o).astype(np.float16)

    rows = iter(range(len(missing)))
    done = 0
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(_load, k) for k in keys]
        for fut in as_completed(futures):
            k, arr = fut.result()
            r = next(rows)
            out[r] = arr
            index[k] = r
            done += 1
            if verbose and done % 500 == 0:
                print(f"[cache]   {done}/{len(missing)}")

    out.flush()
    del out
    os.replace(tmp, data_path)
    index_path.write_text(json.dumps(index))
    if verbose:
        gb = data_path.stat().st_size / 1e9
        print(f"[cache] done: wrote {len(missing)} entries ({gb:.2f} GB) to {data_path.name}; "
              f"{len(existing)+len(missing)} total across {len(_shards(cache_dir))} shard(s)")
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
    """Read cached spectrogram tensors from the (possibly sharded) memmaps; fall back to parquet.

    A drop-in ``load_fn`` for :class:`~eeg_mae.data.SpecDataset` (call signature
    ``(spectrogram_id, offset_seconds=0.0) -> (4, 100, 300) float32``). Loads the base
    memmap plus every ``spec_cache_shardN`` and routes each key to the shard that holds it.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        cache_dir = Path(cache_dir)
        self.shards = []          # list of (index_dict, memmap)
        self.key_to_shard = {}    # key -> shard position in self.shards
        for pos, (data_path, idx_path) in enumerate(_shards(cache_dir)):
            idx = json.loads(idx_path.read_text())
            mm = np.load(data_path, mmap_mode="r")
            self.shards.append((idx, mm))
            for k in idx:
                self.key_to_shard[k] = pos
        self.misses = 0

    def __call__(self, spectrogram_id, offset_seconds: float = 0.0) -> np.ndarray:
        k = _key(spectrogram_id, offset_seconds)
        pos = self.key_to_shard.get(k)
        if pos is None:
            self.misses += 1
            return load_spec_tensor(spectrogram_id, offset_seconds=offset_seconds)
        idx, mm = self.shards[pos]
        return np.asarray(mm[idx[k]], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.key_to_shard)
