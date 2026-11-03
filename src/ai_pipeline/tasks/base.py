"""TaskModule protocol.

A task module owns what is specific to a computer-vision task type and nothing else:
label formats and their conversion to/from the canonical annotation model, dataset validation
rules, and the provider-facing data description. It never references a model library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ai_pipeline.core.config.schema import ProjectConfig
    from ai_pipeline.core.data.assets import AssetInfo
    from ai_pipeline.core.dataset.validation import ValidationReport
    from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource


@dataclass
class ParseIssue:
    """A label file (or line) that could not be converted to canonical annotations."""

    file: str
    line: int | None
    message: str


@dataclass
class ParsedLabels:
    annotations: list[Annotation] = field(default_factory=list)
    labeled_asset_ids: set[str] = field(default_factory=set)
    unmatched_files: list[str] = field(default_factory=list)  # label files with no asset
    parse_issues: list[ParseIssue] = field(default_factory=list)


@runtime_checkable
class TaskModule(Protocol):
    name: str
    label_formats: tuple[str, ...]
    metric_names: tuple[str, ...]

    def parse_labels(
        self,
        fmt: str,
        root: Path,
        *,
        class_names: list[str],
        stem_to_asset: dict[str, str],
        source: AnnotationSource,
    ) -> ParsedLabels:
        """Convert label files under ``root`` to canonical annotations."""
        ...

    def format_labels(
        self, fmt: str, annotations: list[Annotation], *, class_index: dict[str, int]
    ) -> str:
        """Render one asset's annotations in ``fmt``."""
        ...

    def validate(
        self,
        cfg: ProjectConfig,
        assets: list[AssetInfo],
        annotations: list[Annotation],
        *,
        parse_issues: list[ParseIssue],
        split: dict[str, str],
    ) -> ValidationReport: ...

    def write_data_config(self, dataset_dir: Path, *, classes: list[str]) -> Path:
        """Write the provider-facing dataset description (e.g. ``data.yaml``) and return it."""
        ...
