from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import make_image, make_image_set, write_yolo_labels
from typer.testing import CliRunner

from ai_pipeline.cli.main import app

runner = CliRunner()


def invoke_json(*args: str):
    result = runner.invoke(app, ["-o", "json", *args])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:  # pragma: no cover - surfaces stray stdout noise
        raise AssertionError(f"stdout is not a single JSON document:\n{result.stdout}") from None
    return result, payload


def _relax(workspace: Path, name: str = "demo") -> None:
    cfg = workspace / "projects" / name / "project.yaml"
    cfg.write_text(
        cfg.read_text()
        .replace("min_images: 20", "min_images: 4")
        .replace("min_instances_per_class: 10", "min_instances_per_class: 1")
    )


def test_full_m1_flow_json(workspace: Path, tmp_path: Path):
    src = tmp_path / "src"
    imgs = make_image_set(src, 10)
    labels = write_yolo_labels(tmp_path / "labels", [p.stem for p in imgs])
    (src / "junk.bin").write_bytes(b"x")

    r, p = invoke_json("project", "create", "demo")
    assert r.exit_code == 0
    _relax(workspace)

    r, p = invoke_json("data", "import", "demo", str(src))
    assert r.exit_code == 0, p
    assert p["result"]["counts"]["images_new"] == 10 and p["result"]["total_assets"] == 10
    assert p["next_operations"] == ["label import demo <path>"]
    import_id = p["result"]["import_id"]

    r, p = invoke_json("data", "import", "demo", str(src))
    assert p["result"]["counts"] == {
        "files_seen": 11,
        "images_new": 0,
        "images_existing": 10,
        "rejected": 0,
        "skipped_non_image": 1,
    }

    r, p = invoke_json("data", "list", "demo")
    assert [i["import_id"] for i in p["result"]["imports"]][0] == import_id
    assert len(p["result"]["imports"]) == 2

    r, p = invoke_json("label", "import", "demo", str(labels))
    assert r.exit_code == 0, p
    assert p["result"]["counts"]["annotations"] == 15
    assert p["next_operations"] == ["dataset validate demo", "dataset build demo"]

    r, p = invoke_json("label", "list", "demo")
    assert len(p["result"]["label_sets"]) == 1

    r, p = invoke_json("dataset", "validate", "demo")
    assert r.exit_code == 0 and p["result"]["report"]["valid"] is True
    assert p["result"]["existing_version"] is None
    content_hash = p["result"]["content_hash"]

    r, p = invoke_json("dataset", "build", "demo")
    assert r.exit_code == 0, p
    assert p["result"]["created"] is True and p["result"]["dataset_version"] == "ds-v001"
    assert p["result"]["content_hash"] == content_hash
    assert p["next_operations"] == ["train start demo"]

    r, p = invoke_json("dataset", "build", "demo")
    assert r.exit_code == 0 and p["result"]["created"] is False
    assert p["result"]["dataset_version"] == "ds-v001"

    r, p = invoke_json("dataset", "list", "demo")
    assert [d["dataset_version"] for d in p["result"]["datasets"]] == ["ds-v001"]
    assert p["result"]["current_dataset_version"] == "ds-v001"

    r, p = invoke_json("dataset", "show", "demo", "ds-v001", "--manifest")
    assert r.exit_code == 0
    rec = p["result"]
    assert rec["state"] == "READY" and rec["content_hash"] == content_hash
    assert rec["provenance"]["import_ids"] and rec["provenance"]["label_set_id"]
    assert rec["provenance"]["label_source"]["kind"] == "imported"
    assert rec["config_snapshot"]["split"]["seed"] == 42
    assert rec["validation"]["valid"] is True
    assert len(rec["manifest"]) == 10 and {m["split"] for m in rec["manifest"]} <= {
        "train",
        "val",
        "test",
    }
    assert rec["counts"]["splits"] == {"train": 8, "val": 1, "test": 1}

    r, p = invoke_json("status", "demo")
    assert p["result"]["counts"]["assets"] == 10 and p["result"]["counts"]["datasets"] == 1
    assert p["result"]["pointers"]["current_dataset_version"] == "ds-v001"

    r, p = invoke_json("dataset", "show", "demo", "ds-v999")
    assert r.exit_code == 3 and p["error"]["code"] == "DATASET_NOT_FOUND"


def test_invalid_labels_are_reported_and_block_build(workspace: Path, tmp_path: Path):
    src = tmp_path / "src"
    imgs = make_image_set(src, 10)
    labels = write_yolo_labels(
        tmp_path / "labels",
        [p.stem for p in imgs],
        boxes_for=lambda i, s: (
            ["1 0.5 0.5 0.2 0.2", "0 0.9 0.9 0.4 0.4", "garbage"]
            if i == 0
            else ["0 0.5 0.5 0.2 0.2"]
        ),
    )
    invoke_json("project", "create", "demo")
    _relax(workspace)
    invoke_json("data", "import", "demo", str(src))
    r, p = invoke_json("label", "import", "demo", str(labels))
    assert r.exit_code == 0 and p["result"]["counts"]["parse_issues"] == 2

    r, p = invoke_json("dataset", "validate", "demo")
    assert r.exit_code == 1
    assert p["status"] == "failed" and p["error"]["code"] == "DATASET_INVALID"
    assert p["error"]["details"]["issues_by_code"] == {
        "BBOX_OUT_OF_BOUNDS": 1,
        "LABEL_PARSE_ERROR": 2,
    }
    assert p["result"]["report"]["valid"] is False
    messages = [i["message"] for i in p["result"]["report"]["issues"]]
    assert any("class index 1 out of range" in m for m in messages)
    assert any("expected 5 or 6 fields" in m for m in messages)

    r, p = invoke_json("dataset", "build", "demo")
    assert r.exit_code == 1 and p["error"]["code"] == "DATASET_INVALID"
    r, p = invoke_json("dataset", "list", "demo")
    assert p["result"]["datasets"] == []


def test_human_output_and_default_source(workspace: Path):
    runner.invoke(app, ["project", "create", "demo"])
    _relax(workspace)
    raw = workspace / "projects" / "demo" / "data" / "raw"
    imgs = make_image_set(raw, 10)
    write_yolo_labels(workspace / "labels", [p.stem for p in imgs])

    r = runner.invoke(app, ["data", "import", "demo"])  # no source -> project config
    assert r.exit_code == 0 and "10 new" in r.stdout
    r = runner.invoke(app, ["label", "import", "demo", str(workspace / "labels")])
    assert r.exit_code == 0 and "annotations on 10 images" in r.stdout
    r = runner.invoke(app, ["dataset", "validate", "demo"])
    assert r.exit_code == 0 and "VALID" in r.stdout and "INVALID" not in r.stdout
    r = runner.invoke(app, ["dataset", "build", "demo"])
    assert r.exit_code == 0 and "built ds-v001" in r.stdout
    r = runner.invoke(app, ["dataset", "list", "demo"])
    assert "* ds-v001  READY" in r.stdout
    r = runner.invoke(app, ["data", "list", "demo"])
    assert "assets in store: 10" in r.stdout


def test_label_import_with_classes_file_and_scope(workspace: Path, tmp_path: Path):
    src = tmp_path / "src"
    imgs = make_image_set(src, 10)
    labels = write_yolo_labels(
        tmp_path / "labels", [p.stem for p in imgs], boxes_for=lambda i, s: ["1 0.5 0.5 0.2 0.2"]
    )
    (tmp_path / "names.txt").write_text("background\nperson\n")
    invoke_json("project", "create", "demo")
    _relax(workspace)
    r, p = invoke_json("data", "import", "demo", str(src))
    imp = p["result"]["import_id"]
    r, p = invoke_json(
        "label",
        "import",
        "demo",
        str(labels),
        "--classes-file",
        str(tmp_path / "names.txt"),
        "--import",
        imp,
    )
    assert r.exit_code == 0, p
    assert p["result"]["class_names"] == ["background", "person"]
    r, p = invoke_json("dataset", "validate", "demo")
    assert r.exit_code == 0 and p["result"]["report"]["stats"]["class_distribution"] == {
        "person": 10
    }


def test_corrupt_image_is_rejected_not_fatal(workspace: Path, tmp_path: Path):
    src = tmp_path / "src"
    make_image(src / "ok.jpg", 1)
    (src / "bad.jpg").write_bytes(b"\xff\xd8\xff")
    invoke_json("project", "create", "demo")
    r, p = invoke_json("data", "import", "demo", str(src))
    assert r.exit_code == 0
    assert p["result"]["counts"]["rejected"] == 1 and p["result"]["rejected"][0]["path"].endswith(
        "bad.jpg"
    )
