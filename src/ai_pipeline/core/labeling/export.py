"""Export a label set as a review package (YOLO txt + manifest) for an external annotation tool.

Record:  ``artifacts/<p>/label_exports/<export_id>.json``
Archive: ``artifacts/<p>/label_exports/<export_id>/<project>_review_<label_set_id>.zip``

Archive layout (one top-level folder, as X-AnyLabeling and ``upload`` expect)::

    <project>_review/
        labels/<stem>.txt   one per frame in scope; empty file = no proposed objects
        images/<name>       only with ``include_images``
        classes.txt         class names in project index order (line N = class id N)
        manifest.json       frames, asset ids, original names, confidences, timestamps, sequences
        README.txt

Label files are named after each image's original file name, so they sit next to the images the
user already has. Exports are read-only snapshots: nothing in the label store changes, and every
export gets its own id and directory. Confidences and timestamps are copied only when recorded
(model confidence; EXIF capture time); otherwise they are ``null``. ``sequence`` is parsed from
the file name (``<group>_t<seconds>s`` or ``<group>_<index>``) and marked as such.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from ai_pipeline.core.data.assets import AssetInfo, original_basename
from ai_pipeline.core.data.importer import asset_store, scoped_asset_infos
from ai_pipeline.core.errors import ConfigError, OperationError
from ai_pipeline.core.ids import new_event_id, utc_now_iso
from ai_pipeline.core.labeling.annotations import Annotation, BoundingBox, group_by_asset
from ai_pipeline.core.labeling.labelsets import load_label_set
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.tasks import get_task

log = get_logger("label")

KIND = "label_exports"
MANIFEST_SCHEMA_VERSION = 1
LABEL_SET_ID_RE = re.compile(r"^lbl-\d{8}-\d{6}-[0-9a-f]{4}$")
REVIEW_STATUS = {"model": "unreviewed", "imported": "imported", "human": "human_reviewed"}
_SEQ_OFFSET = re.compile(r"^(?P<group>.+?)[_-]t(?P<offset>\d+(?:\.\d+)?)s$")
_SEQ_INDEX = re.compile(r"^(?P<group>.*?[A-Za-z].*?)[_-]?(?P<index>\d+)$")
_EXIF_DATETIME_ORIGINAL = 0x9003
_EXIF_DATETIME = 0x0132
_EXIF_IFD = 0x8769


def sequence_hint(stem: str) -> dict[str, Any] | None:
    m = _SEQ_OFFSET.match(stem)
    if m:
        off = float(m["offset"])
        return {
            "group": m["group"],
            "offset_s": int(off) if off.is_integer() else off,
            "source": "filename",
        }
    m = _SEQ_INDEX.match(stem)
    if m:
        return {"group": m["group"].rstrip("_-"), "index": int(m["index"]), "source": "filename"}
    return None


def exif_timestamp(path: Path) -> str | None:
    """EXIF capture time as written by the camera (``YYYY:MM:DD HH:MM:SS``), or None."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            value = exif.get_ifd(_EXIF_IFD).get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
    except Exception:  # noqa: BLE001 - unreadable/absent EXIF means "not available"
        return None
    return (str(value).strip() or None) if value else None


def clamp_bbox(b: BoundingBox) -> tuple[BoundingBox, bool]:
    c = BoundingBox(
        x_min=min(1.0, max(0.0, b.x_min)),
        y_min=min(1.0, max(0.0, b.y_min)),
        x_max=min(1.0, max(0.0, b.x_max)),
        y_max=min(1.0, max(0.0, b.y_max)),
    )
    return c, c != b


def _import_filenames(project: Project, import_ids: list[str]) -> dict[str, str]:
    """asset id -> file name it had in the given imports (first one wins)."""
    names: dict[str, str] = {}
    for iid in import_ids:
        for entry in project.store.read("imports", iid).get("assets", []):
            names.setdefault(entry["asset_id"], entry["filename"])
    return names


def _model_info(project: Project, source: dict[str, Any]) -> dict[str, Any] | None:
    if source.get("kind") != "model":
        return None
    weights = source.get("weights")
    model_version = source.get("model_version")
    if not model_version and weights:
        parts = Path(weights).parts
        for rec in project.store.list("models"):
            run_id = rec.get("training_run_id")
            if run_id and "runs" in parts and run_id in parts:
                model_version = rec["model_version"]
                break
    return {
        "model_version": model_version,
        "weights": weights,
        "provider": source.get("provider"),
        "confidence_threshold": source.get("confidence"),
        "run_id": source.get("run_id"),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _readme(project: Project, label_set_id: str, model: dict[str, Any] | None) -> str:
    by = f" by model {model['model_version'] or model['weights']}" if model else ""
    return (
        f"{project.name} review package for label set {label_set_id}{by}.\n\n"
        "These boxes are PROPOSALS, not verified ground truth. Review every frame:\n"
        "remove false positives, add missed objects, fix inaccurate boxes. An empty\n"
        "labels/<frame>.txt means no object was proposed; leave it empty only if the frame\n"
        "really contains none.\n\n"
        "labels/<frame>.txt  YOLO: <class_id> <cx> <cy> <w> <h>, normalised to [0, 1]\n"
        "classes.txt         line N is class id N\n"
        "manifest.json       asset ids, original names, confidences, timestamps, sequences\n\n"
        "Return the corrected labels/ folder and classes.txt (zip) through `upload`, then\n"
        f"`label import {project.name} <labels dir> --format yolo --parent {label_set_id}`.\n"
    )


def export_label_set(
    project: Project,
    label_set_id: str,
    *,
    fmt: str = "yolo",
    include_images: bool = False,
) -> dict[str, Any]:
    if not LABEL_SET_ID_RE.match(label_set_id):
        raise ConfigError(
            f"invalid label set id {label_set_id!r}",
            code="LABEL_SET_ID_INVALID",
            details={"label_set_id": label_set_id, "pattern": LABEL_SET_ID_RE.pattern},
            hint=f"List label sets with `label list {project.name}`.",
        )
    task = get_task(project.config.project.task_type)
    if fmt not in task.label_formats:
        raise ConfigError(
            f"label export format {fmt!r} not supported by {task.name}",
            code="LABEL_FORMAT_UNSUPPORTED",
            details={"format": fmt, "supported": list(task.label_formats)},
        )
    record, annotations = load_label_set(project, label_set_id)
    classes = list(project.config.dataset.classes)
    class_index = {name: i for i, name in enumerate(classes)}
    unknown = sorted({a.class_name for a in annotations} - set(class_index))
    if unknown:
        raise OperationError(
            f"label set {label_set_id} uses classes not in the project: {unknown}",
            code="LABEL_CLASS_UNKNOWN",
            details={"unknown_classes": unknown, "project_classes": classes},
            hint="Add the classes with `project configure` or export a matching label set.",
        )

    import_ids = list(record.get("scope", {}).get("import_ids") or [])
    store = asset_store(project)
    assets = {a.asset_id: a for a in scoped_asset_infos(project, import_ids or None)}
    by_asset = group_by_asset(annotations)
    for aid in by_asset.keys() - assets.keys():
        assets[aid] = store.info(aid)
    names = _import_filenames(project, import_ids)

    def image_name(a: AssetInfo) -> str:
        raw = names.get(a.asset_id) or (a.original_filenames[0] if a.original_filenames else "")
        return original_basename(raw) or a.filename

    # Windows file names are case-insensitive: "Frame.txt" and "frame.txt" are one file there.
    frames_by_stem: dict[str, tuple[str, AssetInfo]] = {}
    seen: dict[str, str] = {}
    clashes: dict[str, list[str]] = {}
    for a in assets.values():
        name = image_name(a)
        stem = Path(name).stem
        key = stem.lower()
        if key in seen:
            clashes.setdefault(stem, [frames_by_stem[seen[key]][1].asset_id]).append(a.asset_id)
        seen[key] = stem
        frames_by_stem[stem] = (name, a)
    if clashes:
        raise OperationError(
            f"{len(clashes)} file name(s) map to several images; label files would collide",
            code="AMBIGUOUS_STEMS",
            details={"ambiguous": dict(list(clashes.items())[:20])},
            hint="Export a label set scoped to a single import.",
        )

    export_id = new_event_id("lex")
    top = f"{project.name}_review"
    out_dir = project.store.root / KIND / export_id
    out_dir.mkdir(parents=True, exist_ok=False)
    archive = out_dir / f"{project.name}_review_{label_set_id}.zip"
    model = _model_info(project, record["source"])

    frames: list[dict[str, Any]] = []
    sequences: dict[str, dict[str, int]] = {}
    counts = dict.fromkeys(
        (
            "boxes",
            "boxes_clamped",
            "boxes_skipped",
            "boxes_with_confidence",
            "frames_with_boxes",
            "frames_with_timestamp",
        ),
        0,
    )
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, prefix=".zip-")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for stem in sorted(frames_by_stem):
                name, a = frames_by_stem[stem]
                kept: list[Annotation] = []
                skipped: list[dict[str, Any]] = []
                for ann in sorted(by_asset.get(a.asset_id, []), key=lambda x: x.content_key()):
                    box, clamped = clamp_bbox(ann.bbox)
                    if box.width <= 0 or box.height <= 0:
                        skipped.append({"bbox": ann.bbox.model_dump(), "reason": "zero area"})
                        continue
                    counts["boxes_clamped"] += clamped
                    kept.append(ann.model_copy(update={"bbox": box}))
                # format_labels orders lines by content_key; the manifest's "line" relies on it.
                kept.sort(key=lambda x: x.content_key())
                text = task.format_labels(fmt, kept, class_index=class_index)
                z.writestr(f"{top}/labels/{stem}.txt", text)
                if include_images:
                    z.write(store.file_path(a), f"{top}/images/{name}")
                boxes = []
                for line, ann in enumerate(kept, start=1):
                    cx, cy, w, h = ann.bbox.to_xywh()
                    boxes.append(
                        {
                            "line": line,
                            "class_id": class_index[ann.class_name],
                            "class_name": ann.class_name,
                            "confidence": ann.confidence,
                            "bbox_xywh": [round(v, 6) for v in (cx, cy, w, h)],
                        }
                    )
                    counts["boxes_with_confidence"] += ann.confidence is not None
                ts = exif_timestamp(store.file_path(a))
                seq = sequence_hint(stem)
                frame = {
                    "image": name,
                    "label_file": f"labels/{stem}.txt",
                    "asset_id": a.asset_id,
                    "original_filenames": a.original_filenames,
                    "import_ids": a.import_ids,
                    "width": a.width,
                    "height": a.height,
                    "boxes": boxes,
                    "timestamp": ts,
                    "timestamp_source": "exif" if ts else None,
                    "sequence": seq,
                }
                if skipped:
                    frame["skipped_boxes"] = skipped
                frames.append(frame)
                counts["boxes"] += len(boxes)
                counts["boxes_skipped"] += len(skipped)
                counts["frames_with_boxes"] += bool(boxes)
                counts["frames_with_timestamp"] += ts is not None
                if seq:
                    g = sequences.setdefault(seq["group"], {"frames": 0, "frames_with_boxes": 0})
                    g["frames"] += 1
                    g["frames_with_boxes"] += bool(boxes)

            counts = {
                "frames": len(frames),
                "frames_without_boxes": len(frames) - counts["frames_with_boxes"],
                **counts,
            }
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": "label_review_export",
                "export_id": export_id,
                "created_at": utc_now_iso(),
                "project": project.name,
                "task_type": task.name,
                "format": fmt,
                "label_set": {
                    "label_set_id": label_set_id,
                    "source_kind": record["source"]["kind"],
                    "created_at": record["created_at"],
                    "annotations_sha256": record.get("annotations_sha256"),
                    "parent_label_set_id": record.get("parent_label_set_id"),
                    "scope_import_ids": import_ids,
                },
                "review": {
                    "status": REVIEW_STATUS.get(record["source"]["kind"], "unknown"),
                    "ground_truth": False,
                    "note": "Proposed labels; a human must review them before training."
                    if record["source"]["kind"] == "model"
                    else "Labels as stored in the label set; not re-verified by this export.",
                },
                "auto_label": model,
                "classes": [{"id": i, "name": n} for i, n in enumerate(classes)],
                "images_included": include_images,
                "counts": counts,
                "sequences": [{"group": g, **v} for g, v in sorted(sequences.items())],
                "frames": frames,
            }
            z.writestr(f"{top}/classes.txt", "\n".join(classes) + "\n")
            z.writestr(f"{top}/manifest.json", json.dumps(manifest, indent=1) + "\n")
            z.writestr(f"{top}/README.txt", _readme(project, label_set_id, model))
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
            n_files = len(z.infolist())
        if bad is not None:
            raise OperationError(
                f"export archive failed its CRC check at {bad}", code="EXPORT_CORRUPTED"
            )
        os.replace(tmp, archive)
    except BaseException:
        tmp.unlink(missing_ok=True)
        if not any(out_dir.iterdir()):
            out_dir.rmdir()
        raise

    rec = {
        "export_id": export_id,
        "label_set_id": label_set_id,
        "format": fmt,
        "created_at": manifest["created_at"],
        "include_images": include_images,
        "archive": str(archive.relative_to(project.store.root)),
        "archive_name": archive.name,
        "top_folder": top,
        "files": n_files,
        "bytes": archive.stat().st_size,
        "sha256": _sha256(archive),
        "counts": counts,
        "auto_label": model,
    }
    project.store.create(KIND, export_id, rec)
    log.bind(project=project.name, operation="label.export").info(
        "label export %s of %s: %d frames, %d boxes, %d bytes",
        export_id,
        label_set_id,
        counts["frames"],
        counts["boxes"],
        rec["bytes"],
    )
    return rec
