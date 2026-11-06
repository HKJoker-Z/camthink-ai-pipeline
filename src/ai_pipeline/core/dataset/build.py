"""Dataset validation and immutable version build.

A dataset version directory ``artifacts/<p>/datasets/<ds_version>/``::

    dataset.json        provenance, config snapshot, counts, validation report, content_hash
    manifest.jsonl      one line per sample: asset_id, split, files, dimensions, n_annotations
    data.yaml           provider-facing description (written by the task module)
    images/{train,val,test}/<asset_id>.<ext>   hardlink to the asset store (copy if impossible)
    labels/{train,val,test}/<asset_id>.txt     labels in the project's label_format

Once READY a version is never modified: files are read-only and no code path writes into an
existing version directory. Changing anything produces a new version (or returns the existing
one with the same content hash).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_pipeline.core.data.assets import READ_ONLY, AssetInfo
from ai_pipeline.core.data.importer import asset_store, scoped_asset_infos
from ai_pipeline.core.dataset.grouping import grouping_from_assets
from ai_pipeline.core.dataset.hashing import HASH_SPEC, dataset_content_hash
from ai_pipeline.core.dataset.split import split_assets
from ai_pipeline.core.dataset.validation import ValidationReport
from ai_pipeline.core.errors import OperationError, StateError
from ai_pipeline.core.ids import next_sequential_id, utc_now_iso
from ai_pipeline.core.labeling.annotations import Annotation, group_by_asset
from ai_pipeline.core.labeling.labelsets import latest_ready_label_set_id, load_label_set
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import DatasetState
from ai_pipeline.core.store import write_json_atomic
from ai_pipeline.tasks import get_task
from ai_pipeline.tasks.base import ParseIssue

log = get_logger("dataset")
DATASET_ID_PREFIX = "ds-v"


@dataclass
class DatasetPlan:
    """Everything needed to build a version, computed before any file is written."""

    label_set_id: str | None
    label_set_record: dict[str, Any] | None
    import_ids: list[str]
    assets: list[AssetInfo]
    annotations: list[Annotation]
    parse_issues: list[ParseIssue]
    split: dict[str, str]
    report: ValidationReport
    content_hash: str
    classes: list[str]
    fail_on_warnings: bool = False
    existing_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.report.is_valid(fail_on_warnings=self.fail_on_warnings)


def _warn_if_groups_block_a_split(report: ValidationReport, summary: dict, split_cfg) -> None:
    """A recording kept whole can leave val or test empty. Say so; do not break the group."""
    if summary.get("mode") == "image":
        return
    fractions = {"train": split_cfg.train, "val": split_cfg.val, "test": split_cfg.test}
    counts = report.stats.get("split_counts") or {}
    missing = [name for name, frac in fractions.items() if frac > 0 and not counts.get(name)]
    if not missing:
        return
    report.warning(
        "INDIVISIBLE_GROUPS",
        f"kept {summary.get('n_groups')} video group(s) whole, so split(s) {missing} received "
        "no images. Frames from one recording stay together.",
        splits=missing,
        group_ids=summary.get("group_ids") or [],
    )


def dataset_dir(project: Project, version: str) -> Path:
    return project.store.root / "datasets" / version


def plan_dataset(
    project: Project,
    *,
    label_set_id: str | None = None,
    import_ids: list[str] | None = None,
    verify_assets: bool = False,
) -> DatasetPlan:
    cfg = project.config
    task = get_task(cfg.project.task_type)
    assets = scoped_asset_infos(project, import_ids)

    label_set_id = label_set_id or latest_ready_label_set_id(project)
    record: dict[str, Any] | None = None
    annotations: list[Annotation] = []
    parse_issues: list[ParseIssue] = []
    if label_set_id is not None:
        record, annotations = load_label_set(project, label_set_id)
        parse_issues = [
            ParseIssue(p["file"], p.get("line"), p["message"])
            for p in record.get("parse_issues", [])
        ]
        # When the caller narrows the scope to specific imports, annotations for assets outside
        # that scope are intentionally excluded. Without a scope they stay, so the validator can
        # report ANNOTATION_UNKNOWN_ASSET for genuinely dangling references.
        if import_ids:
            in_scope = {a.asset_id for a in assets}
            annotations = [a for a in annotations if a.asset_id in in_scope]

    asset_ids = [a.asset_id for a in assets]
    grouping = grouping_from_assets(assets)
    per_asset_classes: dict[str, dict[str, int]] = {}
    for ann in annotations:
        per_asset_classes.setdefault(ann.asset_id, {})
        bucket = per_asset_classes[ann.asset_id]
        bucket[ann.class_name] = bucket.get(ann.class_name, 0) + 1
    split, grouping_summary = split_assets(
        asset_ids, cfg.dataset.split, grouping, per_asset_classes
    )

    store = asset_store(project)
    files_ok = {
        a.asset_id: (store.verify(a) if verify_assets else store.file_path(a).is_file())
        for a in assets
    }

    source = ((record or {}).get("source") or {}) if record else {}
    source_kind = source.get("kind")
    report = task.validate(
        cfg,
        assets,
        annotations,
        parse_issues=parse_issues,
        split=split,
        asset_files_ok=files_ok,
        label_source_kind=source_kind,
    )
    report.stats["grouping"] = grouping_summary
    _warn_if_groups_block_a_split(report, grouping_summary, cfg.dataset.split)
    if label_set_id is None:
        report.warning("NO_LABEL_SET", "no label set available; all images are unlabeled")

    content_hash = dataset_content_hash(
        task_type=cfg.project.task_type,
        classes=list(cfg.dataset.classes),
        label_format=cfg.dataset.label_format,
        split_cfg=cfg.dataset.split,
        asset_ids=asset_ids,
        split=split,
        annotations=annotations,
    )
    existing = find_version_by_hash(project, content_hash)
    return DatasetPlan(
        label_set_id=label_set_id,
        label_set_record=record,
        import_ids=import_ids or [],
        assets=assets,
        annotations=annotations,
        parse_issues=parse_issues,
        split=split,
        report=report,
        content_hash=content_hash,
        classes=list(cfg.dataset.classes),
        fail_on_warnings=cfg.dataset.validation.fail_on_warnings,
        existing_version=existing,
        extra={"group_of": dict(grouping.by_asset) if grouping.uses_video_groups else {}},
    )


def find_version_by_hash(project: Project, content_hash: str) -> str | None:
    for rec in project.store.list("datasets"):
        if rec.get("content_hash") == content_hash and rec.get("state") == DatasetState.READY.value:
            return rec["dataset_version"]
    return None


def build_dataset(project: Project, plan: DatasetPlan) -> tuple[dict[str, Any], bool]:
    """Materialise ``plan``. Returns ``(record, created)``.

    Idempotent: a READY version with the same content hash is returned unchanged.
    Never builds from an invalid plan.
    """
    if not plan.valid:
        raise OperationError(
            f"dataset is invalid: {len(plan.report.errors)} error(s), "
            f"{len(plan.report.warnings)} warning(s)",
            code="DATASET_INVALID",
            details={
                "report": plan.report.to_dict(fail_on_warnings=plan.fail_on_warnings, max_issues=50)
            },
            hint="Run `ai-pipeline dataset validate <project>` for the full report and fix the "
            "labels/images or adjust dataset.validation in project.yaml.",
        )
    if plan.existing_version:
        # Content identity is reused; the *attempt* is still provenance worth keeping. The
        # store record grows a build_history entry; the immutable directory is untouched.
        record = project.store.read("datasets", plan.existing_version)
        record.setdefault("build_history", []).append(_build_event(plan, created=False))
        project.store.write("datasets", plan.existing_version, record)
        return record, False

    cfg = project.config
    task = get_task(cfg.project.task_type)
    store = asset_store(project)
    version = next_sequential_id(DATASET_ID_PREFIX, project.store.list_ids("datasets"))
    ddir = dataset_dir(project, version)
    if ddir.exists():
        raise StateError(
            f"dataset directory {ddir} already exists without a record",
            code="DATASET_DIR_ORPHANED",
            details={"path": str(ddir)},
            hint="Remove or rename the orphaned directory; versions are never rebuilt in place.",
        )
    blog = log.bind(project=project.name, operation="dataset.build", dataset_version=version)
    now = utc_now_iso()
    record: dict[str, Any] = {
        "dataset_version": version,
        "state": DatasetState.BUILDING.value,
        "created_at": now,
        "project": project.name,
        "task_type": cfg.project.task_type,
        "content_hash": plan.content_hash,
        "hash_spec": HASH_SPEC,
        "classes": plan.classes,
        "label_format": cfg.dataset.label_format,
        "provenance": {
            "import_ids": plan.import_ids or sorted({i for a in plan.assets for i in a.import_ids}),
            "label_set_id": plan.label_set_id,
            "label_source": (plan.label_set_record or {}).get("source"),
            "annotations_sha256": (plan.label_set_record or {}).get("annotations_sha256"),
            "config_path": str(project.config_path),
        },
        "config_snapshot": {
            "classes": plan.classes,
            "split": cfg.dataset.split.model_dump(),
            "validation": cfg.dataset.validation.model_dump(),
        },
        "validation": plan.report.to_dict(fail_on_warnings=plan.fail_on_warnings),
        "counts": {
            "images": len(plan.assets),
            "annotations": len(plan.annotations),
            "splits": plan.report.stats.get("split_counts", {}),
        },
        "class_distribution": plan.report.stats.get("class_distribution", {}),
        "files": {"manifest": "manifest.jsonl", "data_config": None},
        "size_bytes": 0,
        "error": None,
        # Every build request that resolved to this version, first entry = the one that
        # materialised it. Lives only in the store record, never in the hash.
        "build_history": [_build_event(plan, created=True)],
    }
    project.store.create("datasets", version, record)

    try:
        ddir.mkdir(parents=True, exist_ok=False)
        size = _materialise(project, task, store, plan, ddir, record)
        record["files"]["data_config"] = str(
            task.write_data_config(ddir, classes=plan.classes).relative_to(ddir)
        )
        record["size_bytes"] = size
        record["state"] = DatasetState.READY.value
        record["completed_at"] = utc_now_iso()
        write_json_atomic(ddir / "dataset.json", record)
        _make_tree_read_only(ddir)
        project.store.write("datasets", version, record)
        project.set_pointer("current_dataset_version", version)
    except Exception as exc:
        record["state"] = DatasetState.FAILED.value
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        project.store.write("datasets", version, record)
        blog.exception("dataset build failed")
        raise OperationError(
            f"building {version} failed: {exc}",
            code="DATASET_BUILD_FAILED",
            details={"dataset_version": version, "path": str(ddir)},
            hint="The failed version is recorded as FAILED and will not be reused; fix the "
            "cause and run `dataset build` again.",
        ) from exc

    blog.info(
        "built %s (%d images, %d annotations)", version, len(plan.assets), len(plan.annotations)
    )
    return record, True


def _build_event(plan: DatasetPlan, *, created: bool) -> dict[str, Any]:
    return {
        "at": utc_now_iso(),
        "created": created,
        "label_set_id": plan.label_set_id,
        "import_ids": plan.import_ids,
        "annotations_sha256": (plan.label_set_record or {}).get("annotations_sha256"),
    }


def _materialise(project: Project, task, store, plan: DatasetPlan, ddir: Path, record: dict) -> int:
    by_asset = group_by_asset(plan.annotations)
    class_index = {c: i for i, c in enumerate(plan.classes)}
    fmt = project.config.dataset.label_format
    size = 0
    with (ddir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for a in sorted(plan.assets, key=lambda x: x.asset_id):
            split = plan.split[a.asset_id]
            img_rel = Path("images") / split / a.filename
            lbl_rel = Path("labels") / split / f"{a.asset_id}.txt"
            _link_or_copy(store.file_path(a), ddir / img_rel)
            (ddir / lbl_rel).parent.mkdir(parents=True, exist_ok=True)
            (ddir / lbl_rel).write_text(
                task.format_labels(fmt, by_asset.get(a.asset_id, []), class_index=class_index),
                encoding="utf-8",
            )
            size += a.size_bytes
            row = {
                "asset_id": a.asset_id,
                "split": split,
                "image": str(img_rel),
                "label": str(lbl_rel),
                "width": a.width,
                "height": a.height,
                "n_annotations": len(by_asset.get(a.asset_id, [])),
                "original_filename": a.original_filenames[0] if a.original_filenames else None,
            }
            group_id = (plan.extra.get("group_of") or {}).get(a.asset_id)
            if group_id:
                row["group_id"] = group_id
            manifest.write(json.dumps(row, sort_keys=True) + "\n")
    for split in ("train", "val", "test"):  # providers expect the directories to exist
        (ddir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ddir / "labels" / split).mkdir(parents=True, exist_ok=True)
    return size


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:  # cross-device or unsupported: fall back to a copy
        shutil.copyfile(src, dest)


def _make_tree_read_only(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                os.chmod(p, READ_ONLY)


def list_datasets(project: Project) -> list[dict[str, Any]]:
    return [
        {
            "dataset_version": r["dataset_version"],
            "state": r["state"],
            "created_at": r["created_at"],
            "content_hash": r["content_hash"][:12],
            "images": r["counts"]["images"],
            "annotations": r["counts"]["annotations"],
            "splits": r["counts"].get("splits", {}),
            "label_set_id": r["provenance"]["label_set_id"],
        }
        for r in project.store.list("datasets")
    ]
