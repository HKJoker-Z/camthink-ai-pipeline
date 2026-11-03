"""YOLO txt label format <-> canonical annotations.

One ``<stem>.txt`` per image, one object per line::

    <class_index> <cx> <cy> <w> <h> [<confidence>]

All values normalised to [0, 1]. Nothing here is a source of truth; it is a conversion.
"""

from __future__ import annotations

from pathlib import Path

from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource, BoundingBox
from ai_pipeline.tasks.base import ParsedLabels, ParseIssue


def parse_yolo_tree(
    root: Path,
    *,
    class_names: list[str],
    stem_to_asset: dict[str, str],
    source: AnnotationSource,
) -> ParsedLabels:
    out = ParsedLabels()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*.txt") if p.is_file())
    for path in files:
        asset_id = stem_to_asset.get(path.stem)
        if asset_id is None:
            out.unmatched_files.append(str(path))
            continue
        out.labeled_asset_ids.add(asset_id)
        src = source.model_copy(update={"ref": str(path)})
        anns, issues = parse_yolo_file(path, asset_id=asset_id, class_names=class_names, source=src)
        out.annotations.extend(anns)
        out.parse_issues.extend(issues)
    return out


def parse_yolo_file(
    path: Path, *, asset_id: str, class_names: list[str], source: AnnotationSource
) -> tuple[list[Annotation], list[ParseIssue]]:
    anns: list[Annotation] = []
    issues: list[ParseIssue] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) not in (5, 6):
            issues.append(
                ParseIssue(str(path), lineno, f"expected 5 or 6 fields, got {len(parts)}: {raw!r}")
            )
            continue
        try:
            cls_idx = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            conf = float(parts[5]) if len(parts) == 6 else None
        except ValueError:
            issues.append(ParseIssue(str(path), lineno, f"non-numeric field: {raw!r}"))
            continue
        if not 0 <= cls_idx < len(class_names):
            issues.append(
                ParseIssue(
                    str(path),
                    lineno,
                    f"class index {cls_idx} out of range for {len(class_names)} classes",
                )
            )
            continue
        anns.append(
            Annotation(
                asset_id=asset_id,
                class_name=class_names[cls_idx],
                bbox=BoundingBox.from_xywh(cx, cy, w, h),
                confidence=conf,
                source=source,
                metadata={"line": lineno},
            )
        )
    return anns, issues


def format_yolo(annotations: list[Annotation], *, class_index: dict[str, int]) -> str:
    """Render annotations for one asset. Boxes are clamped to [0, 1] here because the dataset
    has already been validated (out-of-range beyond tolerance is an error upstream)."""
    lines = []
    for ann in sorted(annotations, key=lambda a: a.content_key()):
        cx, cy, w, h = ann.bbox.to_xywh()
        cx, cy, w, h = (min(1.0, max(0.0, v)) for v in (cx, cy, w, h))
        lines.append(f"{class_index[ann.class_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")
