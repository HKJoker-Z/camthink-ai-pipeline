from __future__ import annotations

import random

from ai_pipeline.core.config.schema import SplitConfig
from ai_pipeline.core.dataset.hashing import dataset_content_hash
from ai_pipeline.core.dataset.split import assign_splits, split_sizes
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
