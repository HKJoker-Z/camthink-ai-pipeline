"""Deterministic train/val/test assignment.

Algorithm (documented in docs/datasets.md):

1. Every asset gets a rank key ``sha256(f"{seed}:{asset_id}")``. The key depends only on the
   seed and the asset's content hash — not on filenames, paths, import order or filesystem
   iteration order — so it is identical on every machine.
2. Assets are sorted by that key.
3. Split sizes are computed from the configured fractions with the largest-remainder method
   so that counts always sum to N and rounding is unbiased.
4. The sorted list is cut into ``train | val | test`` in that order.

Adding or removing assets changes ranks and therefore may move existing assets between
splits; this is accepted in V1 (stability under growth would require per-asset hash
bucketing, which cannot hit exact fractions on small datasets).
"""

from __future__ import annotations

import hashlib

from ai_pipeline.core.config.schema import SplitConfig

SPLITS = ("train", "val", "test")


def rank_key(seed: int, asset_id: str) -> str:
    return hashlib.sha256(f"{seed}:{asset_id}".encode()).hexdigest()


def split_sizes(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of ``n`` items over ``fractions`` (which sum to 1)."""
    raw = {k: n * f for k, f in fractions.items()}
    sizes = {k: int(v) for k, v in raw.items()}
    remaining = n - sum(sizes.values())
    # Ties broken by fixed split order so the result is fully deterministic.
    order = sorted(fractions, key=lambda k: (-(raw[k] - sizes[k]), SPLITS.index(k)))
    for k in order[:remaining]:
        sizes[k] += 1
    return sizes


def assign_splits(asset_ids: list[str], cfg: SplitConfig) -> dict[str, str]:
    ids = sorted(set(asset_ids))
    ranked = sorted(ids, key=lambda a: rank_key(cfg.seed, a))
    fractions = {"train": cfg.train, "val": cfg.val, "test": cfg.test}
    sizes = split_sizes(len(ranked), fractions)
    out: dict[str, str] = {}
    pos = 0
    for name in SPLITS:
        for aid in ranked[pos : pos + sizes[name]]:
            out[aid] = name
        pos += sizes[name]
    return out
