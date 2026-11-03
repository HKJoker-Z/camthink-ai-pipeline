"""Object detection task module (the only fully implemented task in V1)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ai_pipeline.core.data.assets import AssetInfo
from ai_pipeline.core.dataset.validation import ValidationReport
from ai_pipeline.core.errors import ConfigError
from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource
from ai_pipeline.tasks.base import ParsedLabels, ParseIssue
from ai_pipeline.tasks.object_detection.validation import validate_detection_dataset
from ai_pipeline.tasks.object_detection.yolo import format_yolo, parse_yolo_tree

if TYPE_CHECKING:  # the config loader imports the task registry; avoid the import cycle
    from ai_pipeline.core.config.schema import ProjectConfig


class ObjectDetectionTask:
    name = "object_detection"
    label_formats: tuple[str, ...] = ("yolo",)
    # Scalar metrics an evaluation may report; per-class metrics are nested under these names.
    metric_names: tuple[str, ...] = ("precision", "recall", "f1", "mAP50", "mAP50-95")

    def _check_format(self, fmt: str) -> None:
        if fmt not in self.label_formats:
            raise ConfigError(
                f"label format {fmt!r} not supported by {self.name}",
                code="LABEL_FORMAT_UNSUPPORTED",
                details={"format": fmt, "supported": list(self.label_formats)},
            )

    def parse_labels(
        self,
        fmt: str,
        root: Path,
        *,
        class_names: list[str],
        stem_to_asset: dict[str, str],
        source: AnnotationSource,
    ) -> ParsedLabels:
        self._check_format(fmt)
        return parse_yolo_tree(
            root, class_names=class_names, stem_to_asset=stem_to_asset, source=source
        )

    def format_labels(
        self, fmt: str, annotations: list[Annotation], *, class_index: dict[str, int]
    ) -> str:
        self._check_format(fmt)
        return format_yolo(annotations, class_index=class_index)

    def validate(
        self,
        cfg: ProjectConfig,
        assets: list[AssetInfo],
        annotations: list[Annotation],
        *,
        parse_issues: list[ParseIssue],
        split: dict[str, str],
        asset_files_ok: dict[str, bool] | None = None,
    ) -> ValidationReport:
        return validate_detection_dataset(
            cfg,
            assets,
            annotations,
            parse_issues=parse_issues,
            split=split,
            asset_files_ok=asset_files_ok,
        )

    def write_data_config(self, dataset_dir: Path, *, classes: list[str]) -> Path:
        """YOLO-style ``data.yaml`` with paths relative to the dataset directory."""
        data = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": dict(enumerate(classes)),
        }
        path = dataset_dir / "data.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path
