"""Object-detection dataset validation rules.

ERRORs make a dataset invalid. WARNINGs are reported and only fail the dataset when
``dataset.validation.fail_on_warnings`` is set in the project configuration.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ai_pipeline.core.data.assets import AssetInfo
from ai_pipeline.core.dataset.validation import ValidationReport
from ai_pipeline.core.labeling.annotations import Annotation, group_by_asset
from ai_pipeline.tasks.base import ParseIssue

if TYPE_CHECKING:
    from ai_pipeline.core.config.schema import ProjectConfig

BOUNDS_TOLERANCE = 1e-6  # float noise from other tools (e.g. 1.0000001) is not an error
TINY_BOX_AREA = 1e-5  # normalised area below which a box is almost certainly a mistake


def validate_detection_dataset(
    cfg: ProjectConfig,
    assets: list[AssetInfo],
    annotations: list[Annotation],
    *,
    parse_issues: list[ParseIssue],
    split: dict[str, str],
    asset_files_ok: dict[str, bool] | None = None,
) -> ValidationReport:
    rep = ValidationReport()
    classes = cfg.dataset.classes
    vcfg = cfg.dataset.validation
    asset_ids = {a.asset_id for a in assets}
    by_asset = group_by_asset(annotations)

    # -- structural: label files that could not be parsed at all
    for pi in parse_issues:
        rep.error(
            "LABEL_PARSE_ERROR",
            f"{pi.file}:{pi.line}: {pi.message}" if pi.line else f"{pi.file}: {pi.message}",
            file=pi.file,
            line=pi.line,
        )

    # -- assets
    if len(assets) != len(asset_ids):
        rep.error("DUPLICATE_ASSET", "asset list contains duplicate asset ids")
    if asset_files_ok:
        for aid, ok in asset_files_ok.items():
            if not ok:
                rep.error("ASSET_FILE_MISSING", "stored image is missing or corrupt", asset_id=aid)
    for a in assets:
        if a.width <= 0 or a.height <= 0:
            rep.error("ASSET_BAD_DIMENSIONS", f"{a.width}x{a.height}", asset_id=a.asset_id)

    # -- annotations
    class_counts: Counter[str] = Counter()
    for aid, anns in by_asset.items():
        if aid not in asset_ids:
            rep.error(
                "ANNOTATION_UNKNOWN_ASSET",
                f"{len(anns)} annotation(s) reference an asset outside this dataset",
                asset_id=aid,
            )
            continue
        seen_keys: set[tuple] = set()
        for ann in anns:
            key = ann.content_key()
            if key in seen_keys:
                rep.warning(
                    "DUPLICATE_ANNOTATION", "identical box repeated", asset_id=aid, box=key[1:5]
                )
            seen_keys.add(key)

            if ann.class_name not in classes:
                rep.error(
                    "UNKNOWN_CLASS",
                    f"class {ann.class_name!r} not in project classes {classes}",
                    asset_id=aid,
                    class_name=ann.class_name,
                )
            b = ann.bbox
            coords = (b.x_min, b.y_min, b.x_max, b.y_max)
            if any(v != v for v in coords):  # NaN
                rep.error("BBOX_INVALID", "non-finite coordinate", asset_id=aid)
                continue
            if min(coords) < -BOUNDS_TOLERANCE or max(coords) > 1 + BOUNDS_TOLERANCE:
                rep.error(
                    "BBOX_OUT_OF_BOUNDS",
                    f"box {tuple(round(v, 4) for v in coords)} exceeds image bounds",
                    asset_id=aid,
                    box=coords,
                )
            if b.width <= 0 or b.height <= 0:
                rep.error(
                    "BBOX_EMPTY",
                    f"zero/negative size {b.width:.6f}x{b.height:.6f}",
                    asset_id=aid,
                    box=coords,
                )
            elif b.area < TINY_BOX_AREA:
                rep.warning("BBOX_TINY", f"area {b.area:.2e} below {TINY_BOX_AREA}", asset_id=aid)
            if ann.confidence is not None and not 0 <= ann.confidence <= 1:
                rep.error("CONFIDENCE_OUT_OF_RANGE", f"{ann.confidence}", asset_id=aid)
            class_counts[ann.class_name] += 1

    # -- dataset-level policy
    n_images = len(asset_ids)
    empty = [aid for aid in sorted(asset_ids) if not by_asset.get(aid)]
    if n_images < vcfg.min_images:
        rep.error("TOO_FEW_IMAGES", f"{n_images} images < min_images {vcfg.min_images}")
    if n_images and len(empty) / n_images > vcfg.max_empty_label_ratio:
        rep.error(
            "TOO_MANY_EMPTY_IMAGES",
            f"{len(empty)}/{n_images} images have no annotations "
            f"(> max_empty_label_ratio {vcfg.max_empty_label_ratio})",
        )
    elif empty:
        rep.warning("EMPTY_IMAGES", f"{len(empty)} image(s) without annotations (background)")
    for cls in classes:
        n = class_counts.get(cls, 0)
        if n < vcfg.min_instances_per_class:
            rep.error(
                "TOO_FEW_INSTANCES",
                f"class {cls!r} has {n} instance(s) < min_instances_per_class "
                f"{vcfg.min_instances_per_class}",
                class_name=cls,
            )

    # -- splits
    split_counts = Counter(split.get(aid, "unassigned") for aid in asset_ids)
    fractions = {
        "train": cfg.dataset.split.train,
        "val": cfg.dataset.split.val,
        "test": cfg.dataset.split.test,
    }
    for name, frac in fractions.items():
        if frac > 0 and n_images and split_counts.get(name, 0) == 0:
            rep.error("EMPTY_SPLIT", f"split {name!r} has 0 images although fraction is {frac}")
    split_class_counts = {
        s: dict(
            sorted(
                Counter(
                    a.class_name
                    for aid in asset_ids
                    if split.get(aid) == s
                    for a in by_asset.get(aid, [])
                ).items()
            )
        )
        for s in ("train", "val", "test")
    }

    rep.stats = {
        "images": n_images,
        "annotations": sum(class_counts.values()),
        "empty_images": len(empty),
        "class_distribution": dict(sorted(class_counts.items())),
        "split_counts": {s: split_counts.get(s, 0) for s in ("train", "val", "test")},
        "split_class_distribution": split_class_counts,
    }
    return rep
