"""Deterministic train/val/test assignment.

Image datasets (documented in docs/datasets.md):

1. Every asset gets a rank key ``sha256(f"{seed}:{asset_id}")``. The key depends only on the
   seed and the asset's content hash — not on filenames, paths, import order or filesystem
   iteration order — so it is identical on every machine.
2. Assets are sorted by that key.
3. Split sizes are computed from the configured fractions with the largest-remainder method
   so that counts always sum to N and rounding is unbiased.
4. The sorted list is cut into ``train | val | test`` in that order.

Video-derived frames use :func:`split_assets` instead. Frames of one clip, and consecutive
clips of one recording session, are one group and are not shuffled across partitions.
A single session that has to fill more than one split is cut only on clip boundaries, into
contiguous time ranges (earlier clips train, then val, then test).

Adding or removing assets changes ranks and therefore may move existing assets between
splits; this is accepted in V1 (stability under growth would require per-asset hash
bucketing, which cannot hit exact fractions on small datasets).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from ai_pipeline.core.config.schema import SplitConfig
from ai_pipeline.core.dataset.grouping import Clip, Grouping, LeakageGroup

SPLITS = ("train", "val", "test")
# When several splits are similarly full, prefer the one that improves class balance.
FILL_SLACK = 0.05
# A temporal cut may move by this fraction of the dataset when that balances classes.
SIZE_SLACK_FRAC = 0.02


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
    """Per-image split. Order of ``asset_ids`` does not matter; duplicates are ignored."""
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


def split_assets(
    asset_ids: list[str],
    cfg: SplitConfig,
    grouping: Grouping | None,
    class_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[dict[str, str], dict]:
    """Assign every asset id.

    Without video groups this is :func:`assign_splits` (``mode="image"``). With video groups,
    each recording session is kept whole (``mode="group"``), unless the dataset is a single
    multi-clip session and more than one split is required — then the session is divided on
    clip boundaries into contiguous ranges (``mode="temporal"``).

    ``class_counts`` maps asset id → {class name: instance count} and is used only to break
    near-ties. Fractions stay the primary target.
    """
    ids = sorted(set(asset_ids))
    counts = class_counts or {}
    if grouping is None or not grouping.uses_video_groups:
        return assign_splits(ids, cfg), {
            "mode": "image",
            "n_groups": len(ids),
            "n_clips": 0,
            "group_ids": [],
            "clips": [],
            "cuts": [],
        }

    fractions = {"train": cfg.train, "val": cfg.val, "test": cfg.test}
    targets = split_sizes(len(ids), fractions)
    id_set = set(ids)
    sessions = [_limit_group(group, id_set) for group in grouping.groups]
    sessions = [group for group in sessions if group.clips]
    singletons = [aid for aid in ids if aid not in grouping.by_asset]
    positive = [name for name in SPLITS if targets[name] > 0]
    one_session = len(sessions) == 1 and not singletons and len(sessions[0].clips) > 1
    if one_session and len(positive) > 1:
        assignment = _assign_temporal(sessions[0].clips, targets, counts, cfg.seed)
        mode = "temporal"
    else:
        units = _units(sessions, singletons, cfg.seed)
        assignment = _assign_units(units, targets, counts)
        mode = "group"
    return assignment, _summary(mode, sessions, assignment)


def _limit_group(group: LeakageGroup, ids: set[str]) -> LeakageGroup:
    clips = []
    for clip in group.clips:
        kept = tuple(aid for aid in clip.asset_ids if aid in ids)
        if kept:
            clips.append(
                Clip(
                    clip_id=clip.clip_id,
                    asset_ids=kept,
                    start_s=clip.start_s,
                    end_s=clip.end_s,
                    camera=clip.camera,
                )
            )
    return LeakageGroup(group_id=group.group_id, clips=tuple(clips))


@dataclass(frozen=True)
class _Unit:
    unit_id: str
    asset_ids: tuple[str, ...]
    rank: str


def _units(sessions: list[LeakageGroup], singletons: list[str], seed: int) -> list[_Unit]:
    units = [
        _Unit(group.group_id, group.asset_ids, rank_key(seed, group.group_id)) for group in sessions
    ]
    units.extend(_Unit(aid, (aid,), rank_key(seed, aid)) for aid in singletons)
    return units


def _assign_units(
    units: list[_Unit],
    targets: dict[str, int],
    class_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, str]:
    """Place whole groups. Fill ratio first; class balance only among similarly full splits."""
    positive = [name for name in SPLITS if targets[name] > 0]
    if not positive:
        positive = ["train"]
    placed = dict.fromkeys(SPLITS, 0)
    placed_classes = {name: Counter() for name in SPLITS}
    per_unit: dict[str, Counter[str]] = {}
    overall: Counter[str] = Counter()
    for unit in units:
        counter: Counter[str] = Counter()
        for aid in unit.asset_ids:
            counter.update(class_counts.get(aid, {}))
        per_unit[unit.unit_id] = counter
        overall.update(counter)
    n = sum(targets.values()) or 1
    out: dict[str, str] = {}
    for unit in sorted(units, key=lambda u: (-len(u.asset_ids), u.rank, u.unit_id)):
        fill = {name: (placed[name] + len(unit.asset_ids)) / targets[name] for name in positive}
        best_fill = min(fill.values())
        candidates = [name for name in positive if fill[name] <= best_fill + FILL_SLACK]

        def cost(name: str, unit: _Unit = unit, fill: dict[str, float] = fill) -> tuple:
            got = placed_classes[name] + per_unit[unit.unit_id]
            share = targets[name] / n
            err = sum(abs(got[cls] - overall[cls] * share) for cls in overall)
            return (err, fill[name], SPLITS.index(name))

        choice = min(candidates, key=cost)
        for aid in unit.asset_ids:
            out[aid] = choice
        placed[choice] += len(unit.asset_ids)
        placed_classes[choice].update(per_unit[unit.unit_id])
    return out


def _assign_temporal(
    clips: tuple[Clip, ...],
    targets: dict[str, int],
    class_counts: Mapping[str, Mapping[str, int]],
    seed: int,
) -> dict[str, str]:
    """Cut an ordered clip list into contiguous ``train | val | test`` ranges."""
    positive = [name for name in SPLITS if targets[name] > 0]
    n_clips = len(clips)
    if len(positive) <= 1 or n_clips == 1:
        only = positive[0] if positive else "train"
        return {aid: only for clip in clips for aid in clip.asset_ids}

    per_clip: list[Counter[str]] = []
    overall: Counter[str] = Counter()
    for clip in clips:
        counter: Counter[str] = Counter()
        for aid in clip.asset_ids:
            counter.update(class_counts.get(aid, {}))
        per_clip.append(counter)
        overall.update(counter)
    n_images = sum(targets.values()) or 1
    slack = max(1, int(round(SIZE_SLACK_FRAC * n_images)))

    def evaluate(ends: tuple[int, ...]) -> tuple[int, float]:
        size_err = 0
        class_err = 0.0
        start = 0
        for name, end in zip(positive, ends, strict=True):
            count = sum(len(clip.asset_ids) for clip in clips[start:end])
            size_err += abs(count - targets[name])
            got: Counter[str] = Counter()
            for counter in per_clip[start:end]:
                got.update(counter)
            share = targets[name] / n_images
            class_err += sum(abs(got[cls] - overall[cls] * share) for cls in overall)
            start = end
        return size_err, class_err

    candidates: list[tuple[int, float, tuple[int, ...]]] = []
    if len(positive) == 2:
        boundaries = ((i, n_clips) for i in range(1, n_clips))
    else:
        # Three positive splits. Fewer clips than splits cannot fill every range; the search
        # still requires each range to be non-empty, so this branch is only used when it can.
        boundaries = ((i, j, n_clips) for i in range(1, n_clips - 1) for j in range(i + 1, n_clips))
    for ends in boundaries:
        size_err, class_err = evaluate(ends)
        candidates.append((size_err, class_err, ends))
    if not candidates:
        only = positive[0]
        return {aid: only for clip in clips for aid in clip.asset_ids}
    min_size = min(item[0] for item in candidates)
    pool = [item for item in candidates if item[0] <= min_size + slack]
    _size, _cls, ends = min(pool, key=lambda item: (item[1], item[0], rank_key(seed, str(item[2]))))
    out: dict[str, str] = {}
    start = 0
    for name, end in zip(positive, ends, strict=True):
        for clip in clips[start:end]:
            for aid in clip.asset_ids:
                out[aid] = name
        start = end
    return out


def _summary(mode: str, sessions: list[LeakageGroup], assignment: dict[str, str]) -> dict:
    clips = []
    cuts = []
    for group in sessions:
        previous_split = None
        previous_clip = None
        for clip in group.clips:
            split = assignment[clip.asset_ids[0]]
            clips.append(
                {
                    "clip_id": clip.clip_id,
                    "group_id": group.group_id,
                    "frames": len(clip.asset_ids),
                    "split": split,
                }
            )
            if previous_split is not None and split != previous_split:
                cuts.append(
                    {
                        "group_id": group.group_id,
                        "left_clip": previous_clip,
                        "right_clip": clip.clip_id,
                        "left_split": previous_split,
                        "right_split": split,
                    }
                )
            previous_split = split
            previous_clip = clip.clip_id
    return {
        "mode": mode,
        "n_groups": len(sessions),
        "n_clips": len(clips),
        "group_ids": [group.group_id for group in sessions],
        "clips": clips,
        "cuts": cuts,
    }
