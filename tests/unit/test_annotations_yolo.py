from __future__ import annotations

from pathlib import Path

import pytest

from ai_pipeline.core.errors import ConfigError
from ai_pipeline.core.labeling.annotations import (
    Annotation,
    AnnotationSource,
    BoundingBox,
    read_annotations_jsonl,
    write_annotations_jsonl,
)
from ai_pipeline.tasks import get_task
from ai_pipeline.tasks.object_detection.yolo import format_yolo, parse_yolo_file, parse_yolo_tree

SRC = AnnotationSource(kind="imported", format="yolo")


def test_bbox_conversions_roundtrip():
    b = BoundingBox.from_xywh(0.5, 0.5, 0.4, 0.2)
    assert (b.x_min, b.y_min, b.x_max, b.y_max) == pytest.approx((0.3, 0.4, 0.7, 0.6))
    assert b.to_xywh() == pytest.approx((0.5, 0.5, 0.4, 0.2))
    assert b.area == pytest.approx(0.08)


def test_parse_yolo_file_good_and_bad_lines(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text(
        "0 0.5 0.5 0.4 0.4\n"
        "# comment\n"
        "\n"
        "1 0.2 0.2 0.1 0.1 0.87\n"
        "2 0.5 0.5 0.1 0.1\n"  # class out of range
        "0 0.5 0.5 0.1\n"  # too few fields
        "0 a b c d\n"  # non-numeric
    )
    anns, issues = parse_yolo_file(f, asset_id="A", class_names=["person", "dog"], source=SRC)
    assert [a.class_name for a in anns] == ["person", "dog"]
    assert anns[1].confidence == pytest.approx(0.87)
    assert anns[0].source.kind == "imported" and anns[0].metadata == {"line": 1}
    assert [(i.line, i.message.split(":")[0]) for i in issues] == [
        (5, "class index 2 out of range for 2 classes"),
        (6, "expected 5 or 6 fields, got 4"),
        (7, "non-numeric field"),
    ]


def test_parse_yolo_tree_matches_by_stem(tmp_path: Path):
    (tmp_path / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.txt").write_text("")  # labeled as empty/background
    (tmp_path / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    parsed = parse_yolo_tree(
        tmp_path, class_names=["person"], stem_to_asset={"x": "AX", "y": "AY"}, source=SRC
    )
    assert len(parsed.annotations) == 1 and parsed.annotations[0].asset_id == "AX"
    assert parsed.labeled_asset_ids == {"AX", "AY"}
    assert parsed.unmatched_files == [str(tmp_path / "orphan.txt")]
    assert parsed.annotations[0].source.ref == str(tmp_path / "x.txt")


def test_format_yolo_is_sorted_and_clamped():
    anns = [
        Annotation(
            asset_id="A",
            class_name="dog",
            bbox=BoundingBox.from_xywh(0.5, 0.5, 0.2, 0.2),
            source=SRC,
        ),
        Annotation(
            asset_id="A",
            class_name="person",
            bbox=BoundingBox(x_min=-1e-7, y_min=0.1, x_max=0.5, y_max=1.0000001),
            source=SRC,
        ),
    ]
    text = format_yolo(anns, class_index={"person": 0, "dog": 1})
    assert text == "1 0.500000 0.500000 0.200000 0.200000\n0 0.250000 0.550000 0.500000 0.900000\n"
    assert format_yolo([], class_index={}) == ""


def test_canonical_roundtrip_through_jsonl_and_yolo(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.4 0.4\n1 0.25 0.25 0.1 0.1\n")
    anns, _ = parse_yolo_file(f, asset_id="A", class_names=["person", "dog"], source=SRC)
    jl = tmp_path / "ann.jsonl"
    assert write_annotations_jsonl(jl, anns) == 2
    back = list(read_annotations_jsonl(jl))
    assert back == anns
    out = format_yolo(back, class_index={"person": 0, "dog": 1})
    assert out.splitlines() == [
        "1 0.250000 0.250000 0.100000 0.100000",
        "0 0.500000 0.500000 0.400000 0.400000",
    ]


def test_task_module_rejects_unknown_format(tmp_path: Path):
    task = get_task("object_detection")
    with pytest.raises(ConfigError) as ei:
        task.parse_labels("coco", tmp_path, class_names=[], stem_to_asset={}, source=SRC)
    assert ei.value.code == "LABEL_FORMAT_UNSUPPORTED"
    with pytest.raises(ConfigError):
        task.format_labels("coco", [], class_index={})


def test_content_key_ignores_provenance():
    a = Annotation(
        asset_id="A", class_name="p", bbox=BoundingBox.from_xywh(0.5, 0.5, 0.2, 0.2), source=SRC
    )
    b = a.model_copy(
        update={"source": AnnotationSource(kind="model", ref="yolo11n.pt"), "metadata": {"x": 1}}
    )
    assert a.content_key() == b.content_key()
