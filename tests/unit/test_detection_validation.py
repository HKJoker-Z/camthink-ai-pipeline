from __future__ import annotations

import pytest
import yaml

from ai_pipeline.core.config import ProjectConfig
from ai_pipeline.core.data.assets import AssetInfo
from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource, BoundingBox
from ai_pipeline.core.project import render_template
from ai_pipeline.tasks.base import ParseIssue
from ai_pipeline.tasks.object_detection.validation import validate_detection_dataset

SRC = AnnotationSource(kind="human")


def _cfg(**validation) -> ProjectConfig:
    data = yaml.safe_load(
        render_template("default", project_name="demo", task_type="object_detection")[
            "project.yaml"
        ]
    )
    data["dataset"]["classes"] = ["person", "dog"]
    data["dataset"]["validation"] = {
        "min_images": 1,
        "min_instances_per_class": 0,
        "max_empty_label_ratio": 1.0,
        **validation,
    }
    return ProjectConfig.model_validate(data)


def _asset(i: int, w: int = 100, h: int = 80) -> AssetInfo:
    return AssetInfo(
        asset_id=f"{i:064x}",
        ext=".jpg",
        width=w,
        height=h,
        size_bytes=1,
        format="JPEG",
        first_imported_at="t",
    )


def _ann(i: int, cls="person", box=(0.1, 0.1, 0.5, 0.5), conf=None) -> Annotation:
    return Annotation(
        asset_id=f"{i:064x}",
        class_name=cls,
        bbox=BoundingBox(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]),
        confidence=conf,
        source=SRC,
    )


def _split(n: int) -> dict[str, str]:
    names = {0: "val", 1: "test"}
    return {f"{i:064x}": names.get(i % 4, "train") for i in range(n)}


def _codes(rep, severity=None):
    return sorted(i.code for i in rep.issues if severity is None or i.severity.value == severity)


def test_valid_dataset_has_no_errors():
    assets = [_asset(i) for i in range(8)]
    anns = [_ann(i) for i in range(8)] + [_ann(0, "dog", (0.6, 0.6, 0.9, 0.9))]
    rep = validate_detection_dataset(_cfg(), assets, anns, parse_issues=[], split=_split(8))
    assert rep.is_valid()
    assert rep.stats["images"] == 8 and rep.stats["annotations"] == 9
    assert rep.stats["class_distribution"] == {"dog": 1, "person": 8}
    assert rep.stats["split_counts"] == {"train": 4, "val": 2, "test": 2}


@pytest.mark.parametrize(
    "ann, code",
    [
        (_ann(0, cls="cat"), "UNKNOWN_CLASS"),
        (_ann(0, box=(0.5, 0.5, 1.2, 0.9)), "BBOX_OUT_OF_BOUNDS"),
        (_ann(0, box=(-0.1, 0.1, 0.5, 0.5)), "BBOX_OUT_OF_BOUNDS"),
        (_ann(0, box=(0.5, 0.1, 0.5, 0.5)), "BBOX_EMPTY"),
        (_ann(0, box=(0.6, 0.1, 0.5, 0.5)), "BBOX_EMPTY"),
        (_ann(0, box=(0.1, 0.1, 0.5, float("nan"))), "BBOX_INVALID"),
        (_ann(0, conf=1.5), "CONFIDENCE_OUT_OF_RANGE"),
        (_ann(7), "ANNOTATION_UNKNOWN_ASSET"),
    ],
)
def test_annotation_errors(ann, code):
    assets = [_asset(i) for i in range(4)]
    rep = validate_detection_dataset(
        _cfg(), assets, [_ann(1), ann], parse_issues=[], split=_split(4)
    )
    assert code in _codes(rep, "ERROR"), _codes(rep)
    assert not rep.is_valid()


def test_bounds_tolerance_accepts_float_noise():
    assets = [_asset(0)]
    rep = validate_detection_dataset(
        _cfg(),
        assets,
        [_ann(0, box=(-1e-9, 0.0, 1.0, 1.0 + 1e-9))],
        parse_issues=[],
        split=_split(1),
    )
    assert "BBOX_OUT_OF_BOUNDS" not in _codes(rep)


def test_warnings_do_not_invalidate_unless_policy_says_so():
    assets = [_asset(i) for i in range(4)]
    anns = [_ann(0), _ann(0), _ann(1, box=(0.5, 0.5, 0.501, 0.501))]  # duplicate + tiny; 2 empty
    rep = validate_detection_dataset(_cfg(), assets, anns, parse_issues=[], split=_split(4))
    assert _codes(rep, "WARNING") == ["BBOX_TINY", "DUPLICATE_ANNOTATION", "EMPTY_IMAGES"]
    assert _codes(rep, "ERROR") == []
    assert rep.is_valid() and not rep.is_valid(fail_on_warnings=True)
    assert rep.to_dict(fail_on_warnings=True)["valid"] is False


def test_dataset_level_policies():
    assets = [_asset(i) for i in range(4)]
    anns = [_ann(0)]
    rep = validate_detection_dataset(
        _cfg(min_images=10, min_instances_per_class=2, max_empty_label_ratio=0.5),
        assets,
        anns,
        parse_issues=[],
        split=_split(4),
    )
    assert _codes(rep, "ERROR") == [
        "TOO_FEW_IMAGES",
        "TOO_FEW_INSTANCES",  # person: 1 < 2
        "TOO_FEW_INSTANCES",  # dog: 0 < 2
        "TOO_MANY_EMPTY_IMAGES",
    ]


def test_parse_issues_missing_files_and_empty_split():
    assets = [_asset(0), _asset(1)]
    split = {assets[0].asset_id: "train", assets[1].asset_id: "train"}  # val/test empty
    rep = validate_detection_dataset(
        _cfg(),
        assets,
        [_ann(0), _ann(1)],
        parse_issues=[ParseIssue("a.txt", 3, "bad")],
        split=split,
        asset_files_ok={assets[1].asset_id: False},
    )
    codes = _codes(rep, "ERROR")
    assert "LABEL_PARSE_ERROR" in codes
    assert "ASSET_FILE_MISSING" in codes
    assert codes.count("EMPTY_SPLIT") == 2  # val and test have fraction > 0 in the template


def test_report_serialisation_truncates():
    assets = [_asset(i) for i in range(3)]
    anns = [_ann(i, cls="cat") for i in range(3)]
    rep = validate_detection_dataset(_cfg(), assets, anns, parse_issues=[], split=_split(3))
    d = rep.to_dict(max_issues=1)
    assert d["error_count"] == 3 and len(d["issues"]) == 1 and d["issues_truncated"]
    assert d["issues_by_code"] == {"UNKNOWN_CLASS": 3}
