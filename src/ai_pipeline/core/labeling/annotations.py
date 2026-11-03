"""Canonical object-detection annotation.

This is the internal source of truth for labels; YOLO txt (and any future format) is only an
import/export representation handled by the task module.

Design choices:
- Coordinates are **normalised** ``[0, 1]`` ``x_min, y_min, x_max, y_max`` relative to image
  width/height. Resolution-independent; converting to/from pixel formats uses the asset's
  recorded dimensions. Values are stored as parsed — out-of-range boxes are kept so validation
  can report them rather than silently clamping.
- Classes are referenced by **name**, not index. Indices depend on the project's class order
  and are assigned when a dataset version is built.
- ``source`` records provenance so human, imported and model-generated labels can be told
  apart later (auto-labeling arrives in M2; this schema already accommodates it).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ANNOTATION_SCHEMA_VERSION = 1

SourceKind = Literal["human", "imported", "model"]


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Centre x, centre y, width, height (YOLO convention, still normalised)."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            self.width,
            self.height,
        )

    @classmethod
    def from_xywh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        return cls(x_min=cx - w / 2, y_min=cy - h / 2, x_max=cx + w / 2, y_max=cy + h / 2)


class AnnotationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: SourceKind
    format: str | None = None  # e.g. "yolo" for imported labels
    ref: str | None = None  # file the label came from, or model/weights identifier
    label_set_id: str | None = None


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    class_name: str
    bbox: BoundingBox
    confidence: float | None = None
    source: AnnotationSource
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_key(self, ndigits: int = 6) -> tuple:
        """What this annotation *is*, independent of provenance; used for hashing and
        duplicate detection."""
        b = self.bbox
        return (
            self.class_name,
            round(b.x_min, ndigits),
            round(b.y_min, ndigits),
            round(b.x_max, ndigits),
            round(b.y_max, ndigits),
            None if self.confidence is None else round(self.confidence, ndigits),
        )


def write_annotations_jsonl(path: Path, annotations: Iterable[Annotation]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ann in annotations:
            fh.write(ann.model_dump_json(exclude_none=True) + "\n")
            n += 1
    return n


def read_annotations_jsonl(path: Path) -> Iterator[Annotation]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Annotation.model_validate_json(line)


def group_by_asset(annotations: Iterable[Annotation]) -> dict[str, list[Annotation]]:
    out: dict[str, list[Annotation]] = {}
    for ann in annotations:
        out.setdefault(ann.asset_id, []).append(ann)
    return out
