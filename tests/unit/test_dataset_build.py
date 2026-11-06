from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from tests.conftest import make_image, make_image_set, write_yolo_labels

from ai_pipeline.core.data.importer import import_images
from ai_pipeline.core.dataset.build import build_dataset, dataset_dir, list_datasets, plan_dataset
from ai_pipeline.core.errors import NotFoundError, OperationError
from ai_pipeline.core.labeling.labelsets import (
    import_labels,
    latest_ready_label_set_id,
    load_label_set,
    stem_index,
)
from ai_pipeline.core.project import Project


def _setup(project: Project, tmp_path: Path, n: int = 12):
    src = tmp_path / "src"
    imgs = make_image_set(src, n)
    labels = write_yolo_labels(tmp_path / "labels", [p.stem for p in imgs])
    imp = import_images(project, [src])
    lbl = import_labels(project, labels, fmt="yolo")
    return src, labels, imp, lbl


# --- label sets -----------------------------------------------------------------------------------


def test_import_labels_creates_canonical_set(demo_project, tmp_path):
    _, labels, imp, lbl = _setup(demo_project, tmp_path, 6)
    assert lbl["state"] == "READY"
    assert lbl["counts"] == {
        "annotations": 9,  # 6 + 3 even indices
        "assets_with_annotations": 6,
        "assets_without_annotations": 0,
        "unmatched_label_files": 0,
        "parse_issues": 0,
    }
    rec, anns = load_label_set(demo_project, lbl["label_set_id"])
    assert len(anns) == 9
    assert {a.source.kind for a in anns} == {"imported"}
    assert all(a.source.label_set_id == lbl["label_set_id"] for a in anns)
    assert latest_ready_label_set_id(demo_project) == lbl["label_set_id"]
    assert rec["annotations_sha256"]


def test_import_labels_requires_assets_and_detects_ambiguity(demo_project, tmp_path):
    labels = write_yolo_labels(tmp_path / "labels", ["a"])
    with pytest.raises(OperationError) as ei:
        import_labels(demo_project, labels, fmt="yolo")
    assert ei.value.code == "NO_ASSETS"
    with pytest.raises(OperationError) as ei:
        import_labels(demo_project, tmp_path / "nope", fmt="yolo")
    assert ei.value.code == "LABEL_SOURCE_MISSING"

    # Same stem, different content, two directories -> ambiguous
    make_image(tmp_path / "s1" / "a.jpg", 1)
    make_image(tmp_path / "s2" / "a.jpg", 2)
    imp1 = import_images(demo_project, [tmp_path / "s1"])
    import_images(demo_project, [tmp_path / "s2"])
    with pytest.raises(OperationError) as ei:
        import_labels(demo_project, labels, fmt="yolo")
    assert ei.value.code == "AMBIGUOUS_STEMS"
    # Scoping to one import resolves it
    rec = import_labels(demo_project, labels, fmt="yolo", import_ids=[imp1["import_id"]])
    assert rec["counts"]["annotations"] == 2  # default fixture writes two boxes for index 0


def test_stem_index():
    from ai_pipeline.core.data.assets import AssetInfo

    def a(aid, names):
        return AssetInfo(
            asset_id=aid,
            ext=".jpg",
            width=1,
            height=1,
            size_bytes=1,
            format="J",
            original_filenames=names,
            first_imported_at="t",
        )

    unique, ambiguous = stem_index(
        [a("A", ["x/1.jpg", "y/1.jpg"]), a("B", ["2.jpg", "3.png"]), a("C", ["3.jpg"])]
    )
    assert unique == {"1": "A", "2": "B"}
    assert ambiguous == {"3": ["B", "C"]}


# --- build ----------------------------------------------------------------------------------------


def test_build_is_deterministic_idempotent_and_immutable(demo_project, tmp_path):
    _setup(demo_project, tmp_path)
    plan = plan_dataset(demo_project)
    assert plan.valid, plan.report.to_dict()
    rec, created = build_dataset(demo_project, plan)
    assert created and rec["state"] == "READY" and rec["dataset_version"] == "ds-v001"
    assert demo_project.pointers()["current_dataset_version"] == "ds-v001"

    ddir = dataset_dir(demo_project, "ds-v001")
    manifest = [json.loads(line) for line in (ddir / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 12
    assert manifest == sorted(manifest, key=lambda m: m["asset_id"])
    for m in manifest:
        assert (ddir / m["image"]).is_file() and (ddir / m["label"]).is_file()
        assert m["split"] in ("train", "val", "test")
        assert "group_id" not in m
    assert rec["validation"]["stats"]["grouping"]["mode"] == "image"
    assert {m["split"] for m in manifest} == {"train", "val", "test"}
    assert (ddir / "data.yaml").read_text().startswith("path: .")
    assert json.loads((ddir / "dataset.json").read_text())["content_hash"] == rec["content_hash"]
    assert rec["provenance"]["label_set_id"] == plan.label_set_id
    assert rec["provenance"]["import_ids"] and rec["config_snapshot"]["classes"] == ["person"]
    assert rec["validation"]["valid"] is True
    for f in ddir.rglob("*"):
        if f.is_file():
            assert not stat.S_IMODE(os.stat(f).st_mode) & stat.S_IWUSR, f

    # Second build with unchanged content: same version, nothing new.
    plan2 = plan_dataset(demo_project)
    assert plan2.content_hash == plan.content_hash and plan2.existing_version == "ds-v001"
    rec2, created2 = build_dataset(demo_project, plan2)
    assert not created2 and rec2["dataset_version"] == "ds-v001"
    assert demo_project.store.count("datasets") == 1

    # Re-importing the same labels (new label set id) still yields the same identity.
    import_labels(demo_project, tmp_path / "labels", fmt="yolo")
    rec3, created3 = build_dataset(demo_project, plan_dataset(demo_project))
    assert not created3 and rec3["dataset_version"] == "ds-v001"


def test_hash_matches_across_two_independent_workspaces(
    settings, tmp_path, tmp_path_factory, monkeypatch
):
    """Same logical content in two different artifact roots/paths -> identical hash."""
    src = tmp_path / "src"
    imgs = make_image_set(src, 8)
    labels = write_yolo_labels(tmp_path / "labels", [p.stem for p in imgs])
    hashes = []
    for i in range(2):
        other = tmp_path_factory.mktemp(f"ws{i}")
        s = settings.__class__(
            infra_path=settings.infra_path,
            infra=settings.infra,
            projects_root=other / "projects",
            artifact_root=other / "artifacts",
        )
        p = Project.create("demo", s)
        p.config_path.write_text(
            p.config_path.read_text()
            .replace("min_images: 20", "min_images: 4")
            .replace("min_instances_per_class: 10", "min_instances_per_class: 1")
        )
        p = Project.load("demo", s)
        # import in a different order via a renamed copy of the source tree
        import_images(p, [src])
        import_labels(p, labels, fmt="yolo")
        rec, _ = build_dataset(p, plan_dataset(p))
        hashes.append(rec["content_hash"])
    assert hashes[0] == hashes[1]


def test_changes_produce_new_versions(demo_project, tmp_path):
    src, labels, _, _ = _setup(demo_project, tmp_path)
    v1, _ = build_dataset(demo_project, plan_dataset(demo_project))

    # 1. label change
    (labels / "img_000.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    import_labels(demo_project, labels, fmt="yolo")
    v2, created = build_dataset(demo_project, plan_dataset(demo_project))
    assert (
        created and v2["dataset_version"] == "ds-v002" and v2["content_hash"] != v1["content_hash"]
    )

    # 2. new image
    make_image(src / "extra.jpg", 1234)
    (labels / "extra.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    import_images(demo_project, [src])
    import_labels(demo_project, labels, fmt="yolo")
    v3, created = build_dataset(demo_project, plan_dataset(demo_project))
    assert created and v3["counts"]["images"] == 13 and v3["content_hash"] != v2["content_hash"]

    # 3. config change (split seed)
    demo_project.config_path.write_text(
        demo_project.config_path.read_text().replace("seed: 42", "seed: 43")
    )
    p = Project.load("demo", demo_project.settings)
    v4, created = build_dataset(p, plan_dataset(p))
    assert created and v4["content_hash"] != v3["content_hash"]
    assert v4["config_snapshot"]["split"]["seed"] == 43

    # 4. validation-only config change does NOT change identity
    p.config_path.write_text(p.config_path.read_text().replace("min_images: 4", "min_images: 5"))
    p = Project.load("demo", p.settings)
    v5, created = build_dataset(p, plan_dataset(p))
    assert not created and v5["dataset_version"] == v4["dataset_version"]

    # Earlier versions untouched
    assert demo_project.store.read("datasets", "ds-v001") == v1
    assert [d["dataset_version"] for d in list_datasets(p)] == [
        "ds-v001",
        "ds-v002",
        "ds-v003",
        "ds-v004",
    ]


def test_invalid_plan_is_not_built(demo_project, tmp_path):
    src = tmp_path / "src"
    imgs = make_image_set(src, 10)
    labels = write_yolo_labels(
        tmp_path / "labels",
        [p.stem for p in imgs],
        boxes_for=lambda i, s: ["0 0.9 0.9 0.5 0.5"] if i == 0 else ["0 0.5 0.5 0.2 0.2"],
    )
    import_images(demo_project, [src])
    import_labels(demo_project, labels, fmt="yolo")
    plan = plan_dataset(demo_project)
    assert not plan.valid
    assert [i.code for i in plan.report.errors] == ["BBOX_OUT_OF_BOUNDS"]
    with pytest.raises(OperationError) as ei:
        build_dataset(demo_project, plan)
    assert ei.value.code == "DATASET_INVALID"
    assert ei.value.details["report"]["error_count"] == 1
    assert demo_project.store.count("datasets") == 0
    assert demo_project.pointers()["current_dataset_version"] is None


def test_fail_on_warnings_policy(demo_project, tmp_path):
    src = tmp_path / "src"
    imgs = make_image_set(src, 10)
    labels = write_yolo_labels(
        tmp_path / "labels",
        [p.stem for p in imgs],
        boxes_for=lambda i, s: [] if i == 0 else ["0 0.5 0.5 0.2 0.2"],
    )
    import_images(demo_project, [src])
    import_labels(demo_project, labels, fmt="yolo")
    assert plan_dataset(demo_project).valid  # EMPTY_IMAGES is only a warning
    demo_project.config_path.write_text(
        demo_project.config_path.read_text().replace(
            "fail_on_warnings: false", "fail_on_warnings: true"
        )
    )
    p = Project.load("demo", demo_project.settings)
    plan = plan_dataset(p)
    assert not plan.valid and plan.report.errors == [] and plan.report.warnings


def test_no_label_set_is_a_warning_and_scoped_imports(demo_project, tmp_path):
    src = tmp_path / "src"
    make_image_set(src, 10)
    imp = import_images(demo_project, [src])
    plan = plan_dataset(demo_project, import_ids=[imp["import_id"]])
    assert "NO_LABEL_SET" in [w.code for w in plan.report.warnings]
    assert "TOO_FEW_INSTANCES" in [e.code for e in plan.report.errors]  # min 1 instance, none
    with pytest.raises(NotFoundError):
        plan_dataset(demo_project, label_set_id="lbl-missing")


def test_video_session_split_is_contiguous_and_reuses_the_version(demo_project, tmp_path):
    src = tmp_path / "src"
    paths = []
    for n, clock in enumerate(("120000", "120500", "121000")):
        for k in range(10):
            off = 1 if k == 0 else 280 if k == 9 else 2 + k
            paths.append(
                make_image(src / f"cam0_20261106_{clock}_t{off:04d}s.jpg", seed=n * 10 + k + 1)
            )
    write_yolo_labels(tmp_path / "labels", [p.stem for p in paths])
    import_images(demo_project, [src])
    import_labels(demo_project, tmp_path / "labels", fmt="yolo")
    plan = plan_dataset(demo_project)
    assert plan.valid, plan.report.to_dict()
    grouping = plan.report.stats["grouping"]
    assert grouping["mode"] == "temporal" and grouping["n_groups"] == 1 and grouping["n_clips"] == 3
    assert [c["split"] for c in grouping["clips"]] == ["train", "val", "test"]
    assert len(grouping["cuts"]) == 2
    rec, created = build_dataset(demo_project, plan)
    assert created
    manifest = [
        json.loads(line)
        for line in (dataset_dir(demo_project, rec["dataset_version"]) / "manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {m["group_id"] for m in manifest} == {grouping["group_ids"][0]}
    again, created_again = build_dataset(demo_project, plan_dataset(demo_project))
    assert not created_again and again["content_hash"] == rec["content_hash"]


def test_model_label_set_is_not_treated_as_ground_truth(demo_project, tmp_path):
    from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource, BoundingBox
    from ai_pipeline.core.labeling.labelsets import create_label_set

    src = tmp_path / "src"
    make_image_set(src, 4)
    imp = import_images(demo_project, [src])
    from ai_pipeline.core.data.importer import scoped_asset_infos

    assets = scoped_asset_infos(demo_project, [imp["import_id"]])
    lid = "lbl-20260101-000000-abcd"
    src_ann = AnnotationSource(kind="model", label_set_id=lid, provider="fake")
    anns = [
        Annotation(
            asset_id=a.asset_id,
            class_name="person",
            bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4),
            confidence=0.5,
            source=src_ann,
        )
        for a in assets
    ]
    create_label_set(
        demo_project,
        label_set_id=lid,
        annotations=anns,
        source={"kind": "model", "provider": "fake"},
        scope_assets=assets,
        import_ids=[imp["import_id"]],
        labeled_asset_ids={a.asset_id for a in assets},
    )
    plan = plan_dataset(demo_project, label_set_id=lid, import_ids=[imp["import_id"]])
    assert plan.report.stats["labels_ground_truth"] is False
    assert plan.report.stats["verified_negative_images"] == 0
    assert "UNREVIEWED_LABELS" in {i.code for i in plan.report.warnings}
    assert "UNREVIEWED_LABELS" not in {i.code for i in plan.report.errors}
    # The warning does not drop frames; every imported image is still in the plan.
    assert len(plan.assets) == 4
    assert set(plan.split) == {a.asset_id for a in plan.assets}


def test_failed_build_is_recorded_and_not_reused(demo_project, tmp_path, monkeypatch):
    _setup(demo_project, tmp_path)
    import ai_pipeline.core.dataset.build as build_mod

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(build_mod, "_materialise", boom)
    with pytest.raises(OperationError) as ei:
        build_dataset(demo_project, plan_dataset(demo_project))
    assert ei.value.code == "DATASET_BUILD_FAILED"
    failed = demo_project.store.read("datasets", "ds-v001")
    assert failed["state"] == "FAILED" and "disk on fire" in failed["error"]["message"]
    assert demo_project.pointers()["current_dataset_version"] is None

    monkeypatch.undo()
    rec, created = build_dataset(demo_project, plan_dataset(demo_project))
    assert created and rec["dataset_version"] == "ds-v002"
