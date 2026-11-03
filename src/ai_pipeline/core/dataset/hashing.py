"""Dataset content hash.

``content_hash`` identifies *what a dataset is*: which images (by content), which boxes with
which class names, how classes are ordered, and how samples are split. Two builds with the same
logical content produce the same hash on any machine, so ``dataset build`` is idempotent.

Contributes to the hash (``HASH_SPEC``):
- task type
- ordered class list
- label format
- split configuration (train/val/test fractions and seed)
- for every sample, sorted by asset_id:
    asset_id (= SHA-256 of the image bytes), split name,
    annotations sorted by content key: class_name, bbox rounded to 6 decimals, confidence

Deliberately excluded:
- absolute/relative paths, original filenames, import ids, label set ids, timestamps,
  hostnames, dataset version number (provenance, recorded in dataset.json instead)
- annotation provenance (``source``) and ``metadata`` — identical boxes from a human and a
  model are the same dataset content
- validation thresholds — they decide whether a build is accepted, not what it contains
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from ai_pipeline.core.config.schema import SplitConfig
from ai_pipeline.core.labeling.annotations import Annotation, group_by_asset

HASH_SPEC = "dataset-hash-v1"


def dataset_content_hash(
    *,
    task_type: str,
    classes: list[str],
    label_format: str,
    split_cfg: SplitConfig,
    asset_ids: Iterable[str],
    split: dict[str, str],
    annotations: Iterable[Annotation],
) -> str:
    by_asset = group_by_asset(annotations)
    samples = []
    for aid in sorted(set(asset_ids)):
        anns = sorted((a.content_key() for a in by_asset.get(aid, [])), key=_sortable)
        samples.append([aid, split[aid], [list(a) for a in anns]])
    payload = {
        "spec": HASH_SPEC,
        "task_type": task_type,
        "classes": list(classes),
        "label_format": label_format,
        "split": {
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
            "seed": split_cfg.seed,
        },
        "samples": samples,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sortable(key: tuple) -> tuple:
    # confidence may be None; make the tuple orderable
    return tuple((v is None, v if v is not None else 0) for v in key)
