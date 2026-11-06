from __future__ import annotations

import random

from ai_pipeline.core.config.schema import SplitConfig
from ai_pipeline.core.data.assets import AssetInfo
from ai_pipeline.core.dataset.grouping import grouping_from_assets
from ai_pipeline.core.dataset.hashing import dataset_content_hash
from ai_pipeline.core.dataset.split import assign_splits, split_assets, split_sizes
from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource, BoundingBox

IDS = [f"{i:064x}" for i in range(50)]


def _split_cfg(**kw) -> SplitConfig:
    return SplitConfig(**{"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42, **kw})


def _ann(aid, cls="person", box=(0.1, 0.1, 0.5, 0.5), conf=None, kind="human"):
    return Annotation(
        asset_id=aid,
        class_name=cls,
        bbox=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        confidence=conf,
        source=AnnotationSource(kind=kind),
    )


# --- split ----------------------------------------------------------------------------------------


def test_split_sizes_largest_remainder():
    assert split_sizes(10, {"train": 0.8, "val": 0.1, "test": 0.1}) == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    assert split_sizes(7, {"train": 0.8, "val": 0.1, "test": 0.1}) == {
        "train": 5,
        "val": 1,
        "test": 1,
    }
    assert split_sizes(24, {"train": 0.8, "val": 0.1, "test": 0.1}) == {
        "train": 19,
        "val": 3,
        "test": 2,
    }
    assert split_sizes(0, {"train": 0.8, "val": 0.2, "test": 0.0}) == {
        "train": 0,
        "val": 0,
        "test": 0,
    }
    for n in range(1, 60):
        assert sum(split_sizes(n, {"train": 0.7, "val": 0.2, "test": 0.1}).values()) == n


def test_assign_splits_is_order_independent_and_deterministic():
    cfg = _split_cfg()
    a = assign_splits(IDS, cfg)
    shuffled = IDS[:]
    random.Random(7).shuffle(shuffled)
    b = assign_splits(shuffled, cfg)
    c = assign_splits(IDS + IDS[:3], cfg)  # duplicates ignored
    assert a == b == c
    assert sorted(a.values()).count("train") == 40
    assert sorted(a.values()).count("val") == 5
    assert sorted(a.values()).count("test") == 5
    # Pinned expectation guards against accidental algorithm changes across versions/machines.
    assert a[IDS[0]] == assign_splits(IDS, cfg)[IDS[0]]
    assert assign_splits(["a", "b", "c", "d", "e"], _split_cfg(seed=1)) == {
        "a": "train",
        "b": "train",
        "c": "val",
        "d": "train",
        "e": "train",
    }


def test_seed_changes_assignment():
    assert assign_splits(IDS, _split_cfg(seed=1)) != assign_splits(IDS, _split_cfg(seed=2))


def _asset(aid: str, name: str) -> AssetInfo:
    return AssetInfo(
        asset_id=aid,
        ext=".jpg",
        width=8,
        height=8,
        size_bytes=1,
        format="JPEG",
        original_filenames=[name],
        first_imported_at="t",
    )


def _aid(i: int) -> str:
    return f"{i:064x}"


def _clip_frames(clock: str, n: int, max_off: int, start: int) -> list[AssetInfo]:
    assets = []
    for k in range(n):
        off = 1 if n == 1 or k == 0 else max_off if k == n - 1 else min(max_off - 1, 2 + k)
        assets.append(_asset(_aid(start + k), f"cam0_20261106_{clock}_t{off:04d}s.jpg"))
    return assets


# Observed cam0_20261106 clip sizes from the rodent recording (12 five-minute files).
_RAT_CLIPS = (
    ("143807", 107, 63, 291),
    ("144304", 72, 46, 294),
    ("144804", 33, 9, 271),
    ("145304", 59, 27, 297),
    ("145804", 43, 13, 298),
    ("150304", 35, 12, 271),
    ("150804", 59, 40, 300),
    ("151305", 88, 32, 299),
    ("151805", 75, 26, 297),
    ("152305", 63, 27, 295),
    ("152805", 63, 52, 299),
    ("153305", 35, 23, 299),
)


def test_numbered_images_keep_the_per_image_split():
    assets = [_asset(_aid(i), f"img_{i:03d}.jpg") for i in range(50)]
    grouping = grouping_from_assets(assets)
    assert not grouping.uses_video_groups
    ids = [a.asset_id for a in assets]
    got, summary = split_assets(ids, _split_cfg(), grouping)
    assert summary["mode"] == "image" and summary["cuts"] == []
    assert got == assign_splits(ids, _split_cfg())


def test_rat_shaped_session_is_cut_on_clip_boundaries_only():
    """One afternoon of consecutive clips stays contiguous: one temporal cut, no interleaving."""
    assets: list[AssetInfo] = []
    class_counts: dict[str, dict[str, int]] = {}
    cursor = 0
    for clock, n, boxes, max_off in _RAT_CLIPS:
        frames = _clip_frames(clock, n, max_off, cursor)
        class_counts[frames[0].asset_id] = {"rodent": boxes}
        assets.extend(frames)
        cursor += n
    assert len(assets) == 732
    grouping = grouping_from_assets(assets)
    assert len(grouping.groups) == 1
    assert grouping.groups[0].group_id == "session:cam0_20261106_143807"
    assert len(grouping.groups[0].clips) == 12

    cfg = _split_cfg(train=0.9, val=0.1, test=0.0)
    ids = [a.asset_id for a in assets]
    assignment, summary = split_assets(ids, cfg, grouping, class_counts)
    assert set(assignment) == set(ids)
    assert summary["mode"] == "temporal" and len(summary["cuts"]) == 1
    by_clip: list[str] = []
    for clip in summary["clips"]:
        members = {
            assignment[a.asset_id]
            for a in assets
            if a.original_filenames[0].startswith(f"cam0_20261106_{clip['clip_id'].split('_')[-1]}")
            or clip["clip_id"] in a.original_filenames[0]
        }
        # Every frame of the clip shares the recorded split.
        assert members == {clip["split"]}
        by_clip.append(clip["split"])
    assert by_clip[0] == "train" and by_clip[-1] == "val"
    # Partitions are contiguous along the recording. Consecutive clips differ at most once.
    changes = [i for i in range(1, len(by_clip)) if by_clip[i] != by_clip[i - 1]]
    assert changes == [10]
    cut = summary["cuts"][0]
    assert cut["left_split"] == "train" and cut["right_split"] == "val"
    assert all(split == "train" for split in by_clip[:10])
    assert all(split == "val" for split in by_clip[10:])


def test_consecutive_clips_merge_and_a_later_session_stays_whole():
    morning = _clip_frames("100000", 8, 290, 0) + _clip_frames("100500", 8, 290, 8)
    evening = _clip_frames("180000", 8, 290, 16) + _clip_frames("180500", 8, 290, 24)
    assets = morning + evening
    grouping = grouping_from_assets(assets)
    assert len(grouping.groups) == 2
    assert all(group.group_id.startswith("session:") for group in grouping.groups)
    ids = [a.asset_id for a in assets]
    # Half the frames each; the two sessions must not be mixed inside a clip.
    assignment, summary = split_assets(ids, _split_cfg(train=0.5, val=0.5, test=0.0), grouping)
    assert summary["mode"] == "group" and summary["cuts"] == []
    splits = {summary["clips"][0]["split"], summary["clips"][2]["split"]}
    assert splits == {"train", "val"}
    for clip in summary["clips"]:
        assert len({assignment[a] for a in _assets_of(assets, clip["clip_id"])}) == 1


def _assets_of(assets: list[AssetInfo], clip_id: str) -> list[str]:
    return [a.asset_id for a in assets if clip_id in a.original_filenames[0]]


def test_a_single_clip_is_not_divided_to_fill_validation():
    assets = _clip_frames("143807", 20, 40, 0)
    grouping = grouping_from_assets(assets)
    assert len(grouping.groups) == 1 and len(grouping.groups[0].clips) == 1
    ids = [a.asset_id for a in assets]
    assignment, summary = split_assets(ids, _split_cfg(train=0.8, val=0.2, test=0.0), grouping)
    assert summary["mode"] == "group" and summary["cuts"] == []
    assert set(assignment.values()) == {"train"}


def test_temporal_cut_prefers_class_balance_when_sizes_tie():
    # Three equal clips, one session. 0.5/0.5 leaves two cuts with the same size error;
    # the cut that puts both classes in val wins.
    assets = (
        _clip_frames("120000", 10, 280, 0)
        + _clip_frames("120500", 10, 280, 10)
        + _clip_frames("121000", 10, 280, 20)
    )
    grouping = grouping_from_assets(assets)
    assert len(grouping.groups) == 1
    class_counts = {
        assets[0].asset_id: {"a": 10},
        assets[10].asset_id: {"a": 10},
        assets[20].asset_id: {"b": 10},
    }
    ids = [a.asset_id for a in assets]
    assignment, summary = split_assets(
        ids, _split_cfg(train=0.5, val=0.5, test=0.0), grouping, class_counts
    )
    assert summary["mode"] == "temporal"
    assert [clip["split"] for clip in summary["clips"]] == ["train", "val", "val"]
    assert assignment[assets[0].asset_id] == "train"
    assert assignment[assets[20].asset_id] == "val"


def test_grouped_split_is_order_independent():
    assets = _clip_frames("100000", 6, 290, 0) + _clip_frames("180000", 6, 290, 6)
    grouping = grouping_from_assets(assets)
    ids = [a.asset_id for a in assets]
    cfg = _split_cfg(train=0.5, val=0.5, test=0.0, seed=3)
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    assert split_assets(ids, cfg, grouping)[0] == split_assets(shuffled, cfg, grouping)[0]


def test_windows_video_names_group():
    asset = _asset(_aid(1), "images\\cam0_20261106_143807_t0001s.jpg")
    grouping = grouping_from_assets([asset])
    assert grouping.by_asset[_aid(1)] == "clip:cam0_20261106_143807"


# --- hash -----------------------------------------------------------------------------------------


def _hash(ids=IDS[:10], anns=None, classes=("person",), split_cfg=None, fmt="yolo"):
    split_cfg = split_cfg or _split_cfg()
    split = assign_splits(list(ids), split_cfg)
    anns = [_ann(ids[0]), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))] if anns is None else anns
    return dataset_content_hash(
        task_type="object_detection",
        classes=list(classes),
        label_format=fmt,
        split_cfg=split_cfg,
        asset_ids=ids,
        split=split,
        annotations=anns,
    )


def test_hash_is_stable_and_order_independent():
    ids = IDS[:10]
    anns = [_ann(ids[0]), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))]
    base = _hash(ids, anns=anns)
    assert base == _hash(ids, anns=anns)
    assert base == _hash(list(reversed(ids)), anns=anns)
    assert base == _hash(ids, anns=list(reversed(anns)))


def test_hash_ignores_provenance_but_not_content():
    ids = IDS[:10]
    human = [_ann(ids[0], kind="human"), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4), kind="human")]
    model = [
        a.model_copy(update={"source": AnnotationSource(kind="model", ref="yolo11n.pt")})
        for a in human
    ]
    assert _hash(ids, anns=human) == _hash(ids, anns=model)

    assert _hash(ids, anns=[_ann(ids[0])]) != _hash(ids)  # fewer boxes
    assert _hash(
        ids, anns=[_ann(ids[0], box=(0.1, 0.1, 0.5, 0.51)), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))]
    ) != _hash(ids)
    assert _hash(
        ids, anns=[_ann(ids[0], conf=0.9), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))]
    ) != _hash(ids)
    assert _hash(
        ids,
        anns=[_ann(ids[0], cls="dog"), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))],
        classes=("person", "dog"),
    ) != _hash(ids, classes=("person", "dog"))


def test_hash_changes_with_assets_classes_split_and_format():
    ids = IDS[:10]
    base = _hash(ids)
    assert _hash(IDS[:11]) != base
    assert _hash(ids, classes=("person", "dog")) != base
    assert _hash(ids, classes=("dog", "person")) != _hash(ids, classes=("person", "dog"))
    assert _hash(ids, split_cfg=_split_cfg(seed=7)) != base
    assert _hash(ids, split_cfg=_split_cfg(train=0.7, val=0.2)) != base
    assert _hash(ids, fmt="coco") != base


def test_hash_rounding_tolerates_float_noise():
    ids = IDS[:10]
    a = [_ann(ids[0], box=(0.1, 0.1, 0.5, 0.5)), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))]
    b = [_ann(ids[0], box=(0.1 + 1e-9, 0.1, 0.5, 0.5)), _ann(ids[1], box=(0.2, 0.2, 0.4, 0.4))]
    assert _hash(ids, anns=a) == _hash(ids, anns=b)
