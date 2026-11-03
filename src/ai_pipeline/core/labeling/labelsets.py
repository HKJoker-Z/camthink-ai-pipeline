"""Label sets: a named collection of canonical annotations with provenance.

Record: ``artifacts/<p>/labels/<label_set_id>.json``
Annotations: ``artifacts/<p>/labels/<label_set_id>/annotations.jsonl``

A label set covers a *scope* of assets (by import ids; default all). Assets in scope with no
annotations are negatives/background, which object detection allows and validation policy
bounds (``max_empty_label_ratio``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ai_pipeline.core.data.assets import AssetInfo
from ai_pipeline.core.data.importer import scoped_asset_infos
from ai_pipeline.core.errors import NotFoundError, OperationError
from ai_pipeline.core.ids import new_event_id, utc_now_iso
from ai_pipeline.core.labeling.annotations import (
    Annotation,
    AnnotationSource,
    read_annotations_jsonl,
    write_annotations_jsonl,
)
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import LabelSetState
from ai_pipeline.tasks import get_task

log = get_logger("label")


def annotations_path(project: Project, label_set_id: str) -> Path:
    return project.store.root / "labels" / label_set_id / "annotations.jsonl"


def stem_index(assets: list[AssetInfo]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map original filename stems to asset ids. Returns ``(unique, ambiguous)``."""
    seen: dict[str, set[str]] = {}
    for a in assets:
        for stem in a.stems:
            seen.setdefault(stem, set()).add(a.asset_id)
    unique = {s: next(iter(ids)) for s, ids in seen.items() if len(ids) == 1}
    ambiguous = {s: sorted(ids) for s, ids in seen.items() if len(ids) > 1}
    return unique, ambiguous


def read_class_names(path: Path) -> list[str]:
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n and not n.startswith("#")]
    if not names:
        raise OperationError(
            f"class names file {path} is empty",
            code="CLASS_FILE_EMPTY",
            details={"path": str(path)},
        )
    return names


def import_labels(
    project: Project,
    root: Path,
    *,
    fmt: str,
    class_names: list[str] | None = None,
    import_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Convert label files under ``root`` to a canonical label set (``source.kind='imported'``)."""
    if not root.exists():
        raise OperationError(
            f"Label source does not exist: {root}",
            code="LABEL_SOURCE_MISSING",
            details={"path": str(root)},
        )
    task = get_task(project.config.project.task_type)
    class_names = class_names or list(project.config.dataset.classes)
    assets = scoped_asset_infos(project, import_ids)
    if not assets:
        raise OperationError(
            "No imported assets to attach labels to",
            code="NO_ASSETS",
            hint="Run `ai-pipeline data import <project> <source>` first.",
        )
    unique, ambiguous = stem_index(assets)
    if ambiguous:
        raise OperationError(
            f"{len(ambiguous)} filename stem(s) map to several assets; cannot match label files",
            code="AMBIGUOUS_STEMS",
            details={"ambiguous": dict(list(ambiguous.items())[:20])},
            hint="Restrict the scope with --import <import_id> so each stem is unique.",
        )

    label_set_id = new_event_id("lbl")
    source = AnnotationSource(kind="imported", format=fmt, label_set_id=label_set_id)
    parsed = task.parse_labels(
        fmt, root.resolve(), class_names=class_names, stem_to_asset=unique, source=source
    )

    path = annotations_path(project, label_set_id)
    n = write_annotations_jsonl(path, parsed.annotations)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    scope_ids = {a.asset_id for a in assets}
    record = {
        "label_set_id": label_set_id,
        "state": LabelSetState.READY.value,
        "created_at": utc_now_iso(),
        "source": {
            "kind": "imported",
            "format": fmt,
            "path": str(root.resolve()),
            "class_names": class_names,
        },
        "scope": {"import_ids": import_ids or [], "asset_count": len(assets)},
        "annotations_file": str(path.relative_to(project.store.root)),
        "annotations_sha256": digest,
        "counts": {
            "annotations": n,
            "assets_with_annotations": len(parsed.labeled_asset_ids & scope_ids),
            "assets_without_annotations": len(scope_ids - parsed.labeled_asset_ids),
            "unmatched_label_files": len(parsed.unmatched_files),
            "parse_issues": len(parsed.parse_issues),
        },
        "unmatched_label_files": parsed.unmatched_files[:200],
        "parse_issues": [
            {"file": p.file, "line": p.line, "message": p.message} for p in parsed.parse_issues
        ],
    }
    project.store.create("labels", label_set_id, record)
    log.bind(project=project.name, operation="label.import").info(
        "label set %s: %d annotations, %d parse issues, %d unmatched files",
        label_set_id,
        n,
        len(parsed.parse_issues),
        len(parsed.unmatched_files),
    )
    return record


def load_label_set(project: Project, label_set_id: str) -> tuple[dict[str, Any], list[Annotation]]:
    record = project.store.read("labels", label_set_id)
    path = project.store.root / record["annotations_file"]
    if not path.is_file():
        raise NotFoundError(
            f"annotations file for label set {label_set_id!r} is missing",
            code="LABEL_FILE_MISSING",
            details={"path": str(path)},
        )
    return record, list(read_annotations_jsonl(path))


def latest_ready_label_set_id(project: Project) -> str | None:
    ready = [r for r in project.store.list("labels") if r.get("state") == LabelSetState.READY.value]
    if not ready:
        return None
    return max(ready, key=lambda r: r["created_at"])["label_set_id"]
