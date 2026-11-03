from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.conftest import make_image_set, write_yolo_labels

from ai_pipeline.core.config.schema import Rule
from ai_pipeline.core.data.importer import import_images
from ai_pipeline.core.dataset.build import build_dataset, plan_dataset
from ai_pipeline.core.errors import NotFoundError, OperationError, StateError
from ai_pipeline.core.evaluation import apply_rules
from ai_pipeline.core.jobs import read_spec
from ai_pipeline.core.labeling.labelsets import import_labels, load_label_set
from ai_pipeline.core.project import Project
from ai_pipeline.core.registry import list_models, register_from_run, select_model
from ai_pipeline.core.runs import (
    cancel_run,
    job_dir,
    list_runs,
    run_logs,
    start_autolabel,
    start_evaluation,
    start_training,
    sync_run,
    wait_run,
)


@pytest.fixture
def fake_project(demo_project: Project, tmp_path: Path) -> Project:
    """demo project switched to the fake provider with a READY dataset."""
    cfg = demo_project.config_path
    cfg.write_text(cfg.read_text().replace("provider: ultralytics", "provider: fake"))
    p = Project.load("demo", demo_project.settings)
    src = tmp_path / "src"
    imgs = make_image_set(src, 12)
    write_yolo_labels(tmp_path / "labels", [i.stem for i in imgs])
    import_images(p, [src])
    import_labels(p, tmp_path / "labels", fmt="yolo")
    build_dataset(p, plan_dataset(p))
    return p


def _set_provider_args(project: Project, **args) -> Project:
    text = project.config_path.read_text().replace(
        "provider_args: {}", f"provider_args: {json.dumps(args)}"
    )
    project.config_path.write_text(text)
    return Project.load("demo", project.settings)


# --- policy ---------------------------------------------------------------------------------------


def test_apply_rules():
    rules = {"mAP50": Rule(min=0.3), "latency": Rule(max=10)}
    v = apply_rules({"mAP50": 0.5, "latency": 5}, rules)
    assert v["passed"] and [c["metric"] for c in v["checks"]] == ["latency", "mAP50"]
    v = apply_rules({"mAP50": 0.2, "latency": 5}, rules)
    assert not v["passed"] and [c["passed"] for c in v["checks"]] == [True, False]
    v = apply_rules({"latency": 5}, rules)
    assert not v["passed"] and v["checks"][1]["reason"] == "metric missing or non-numeric"
    assert apply_rules({"x": 1}, {})["passed"] is True


# --- training run lifecycle -----------------------------------------------------------------------


def test_training_run_record_and_registration(fake_project: Project):
    p = fake_project
    run = start_training(p)
    assert run["state"] == "SUBMITTED" and run["kind"] == "train"
    assert run["inputs"]["dataset_version"] == "ds-v001" and run["inputs"]["dataset_content_hash"]
    assert run["candidate"]["name"] == "yolo11n" and run["params"]["epochs"] == 5
    assert run["backend"] == "local" and run["provider"] == "fake"
    assert run["handle"]["pid"]
    spec = read_spec(job_dir(p, run["run_id"]))
    assert spec.kind == "train" and spec.params["classes"] == ["person"]
    assert Path(spec.inputs["data_config"]).is_file()

    run = wait_run(p, run["run_id"], timeout_s=30)
    assert run["state"] == "COMPLETED", run
    ts = run["timestamps"]
    assert ts["created"] <= ts["submitted"] <= ts["started"] <= ts["finished"]
    assert run["metrics"]["mAP50"] == 0.75 and "per_class" in run["metrics"]
    assert run["environment"]["provider"] == "fake"
    assert run["outputs"]["weights"].startswith("demo/runs/") and run["outputs"]["result"]
    assert (p.settings.artifact_root / run["outputs"]["weights"]).is_file()

    # post hook registered a model
    assert run["post_processed"] and run["model_version"] == "mv-001"
    model = p.store.read("models", "mv-001")
    assert model["state"] == "TRAINED"
    assert model["training_run_id"] == run["run_id"] and model["dataset_version"] == "ds-v001"
    assert (
        model["artifacts"]["weights"]["sha256"] and model["artifacts"]["weights"]["size_bytes"] > 0
    )
    assert model["metrics"]["train"]["mAP50"] == 0.75 and model["metrics"]["eval"] is None
    assert model["history"][0]["state"] == "TRAINED"

    # idempotent sync / register
    assert sync_run(p, run["run_id"])["model_version"] == "mv-001"
    rec, created = register_from_run(p, run)
    assert not created and rec["model_version"] == "mv-001"
    assert p.store.count("models") == 1

    # a second run gets a new id and a new model version
    run2 = wait_run(p, start_training(p)["run_id"], timeout_s=30)
    assert run2["run_id"] != run["run_id"] and run2["model_version"] == "mv-002"
    assert [r["run_id"] for r in list_runs(p, "train")] == [run["run_id"], run2["run_id"]]
    assert "fake training" in run_logs(p, run["run_id"])


def test_training_preconditions(demo_project: Project):
    with pytest.raises(StateError) as ei:
        start_training(demo_project)
    assert ei.value.code == "NO_DATASET"
    with pytest.raises(NotFoundError):
        start_training(demo_project, dataset_version="ds-v009")


def test_candidate_and_backend_selection(fake_project: Project):
    with pytest.raises(StateError) as ei:
        start_training(fake_project, candidate_name="nope")
    assert ei.value.code == "CANDIDATE_UNKNOWN"
    with pytest.raises(OperationError) as ei:
        start_training(fake_project, backend="remote")
    assert ei.value.code == "BACKEND_NOT_IMPLEMENTED"
    assert fake_project.store.count("runs") == 0  # nothing half-created


def test_failed_job_marks_run_failed_without_model(fake_project: Project):
    p = _set_provider_args(fake_project, fake_fail="train")
    run = wait_run(p, start_training(p)["run_id"], timeout_s=30)
    assert run["state"] == "FAILED"
    assert run["error"]["type"] == "RuntimeError" and "asked to fail" in run["error"]["message"]
    assert run["model_version"] is None and p.store.count("models") == 0
    with pytest.raises(StateError) as ei:
        register_from_run(p, run)
    assert ei.value.code == "RUN_NOT_COMPLETED"


def test_cancel_running_job(fake_project: Project):
    p = _set_provider_args(fake_project, fake_sleep_s=30)
    run = start_training(p)
    import time

    deadline = time.monotonic() + 20
    while sync_run(p, run["run_id"])["state"] != "RUNNING":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    run = cancel_run(p, run["run_id"])
    assert run["state"] == "CANCELLED" and run["timestamps"]["finished"]
    assert cancel_run(p, run["run_id"])["state"] == "CANCELLED"  # idempotent


def test_post_hook_failure_is_retryable(fake_project: Project, monkeypatch):
    p = fake_project
    run = start_training(p)
    import ai_pipeline.core.runs as runs_mod

    calls = {"n": 0}
    real = runs_mod._post_process

    def flaky(project, r):
        calls["n"] += 1
        if calls["n"] == 1:
            r["post_processed"] = False
            r["post_error"] = {"type": "Boom", "message": "x"}
            project.store.write("runs", r["run_id"], r)
            raise OperationError("boom", code="RUN_POST_PROCESS_FAILED")
        return real(project, r)

    monkeypatch.setattr(runs_mod, "_post_process", flaky)
    with pytest.raises(OperationError):
        wait_run(p, run["run_id"], timeout_s=30)
    r = p.store.read("runs", run["run_id"])
    assert (
        r["state"] == "COMPLETED" and not r["post_processed"] and r["post_error"]["type"] == "Boom"
    )
    r = sync_run(p, run["run_id"])  # retried
    assert r["post_processed"] and r["model_version"] == "mv-001"


# --- evaluation + selection -----------------------------------------------------------------------


def test_evaluation_applies_policy_and_selection(fake_project: Project):
    p = fake_project
    run = wait_run(p, start_training(p)["run_id"], timeout_s=30)
    mv = run["model_version"]

    ev = start_evaluation(p, mv)
    assert ev["kind"] == "evaluate" and ev["inputs"]["model_version"] == mv
    assert ev["inputs"]["split"] == "test" and Path(ev["inputs"]["weights"]).is_file()
    ev = wait_run(p, ev["run_id"], timeout_s=30)
    assert ev["state"] == "COMPLETED" and ev["metrics"]["split"] == "test"

    model = p.store.read("models", mv)
    assert model["state"] == "ACCEPTED"
    assert model["metrics"]["eval"]["mAP50"] == 0.75
    assert (
        model["evaluation"]["run_id"] == ev["run_id"]
        and model["evaluation"]["acceptance"]["passed"]
    )
    assert [h["state"] for h in model["history"]] == ["TRAINED", "EVALUATED", "ACCEPTED"]

    selected = select_model(p, None)
    assert selected["model_version"] == mv
    assert p.pointers()["selected_model_version"] == mv
    assert list_models(p)[0]["accepted"] is True

    # rejection path via fake metrics below the mAP50 >= 0.3 rule
    p2 = _set_provider_args(
        p, fake_metrics={"precision": 0.1, "recall": 0.1, "mAP50": 0.1, "mAP50-95": 0.05}
    )
    run2 = wait_run(p2, start_training(p2)["run_id"], timeout_s=30)
    ev2 = wait_run(p2, start_evaluation(p2, run2["model_version"])["run_id"], timeout_s=30)
    assert ev2["state"] == "COMPLETED"
    m2 = p2.store.read("models", run2["model_version"])
    assert m2["state"] == "REJECTED"
    assert [c["passed"] for c in m2["evaluation"]["acceptance"]["checks"]] == [False]
    with pytest.raises(StateError) as ei:
        select_model(p2, m2["model_version"])
    assert ei.value.code == "MODEL_NOT_ACCEPTED"
    # implicit selection still picks the accepted one (and it stays selected)
    assert select_model(p2, None)["model_version"] == mv

    # re-evaluation of an accepted model is allowed (ACCEPTED -> EVALUATED -> ...)
    ev3 = wait_run(p2, start_evaluation(p2, mv)["run_id"], timeout_s=30)
    assert ev3["state"] == "COMPLETED"
    assert p2.store.read("models", mv)["state"] == "REJECTED"  # fake_metrics now apply to it too


def test_select_requires_accepted_model(fake_project: Project):
    with pytest.raises(StateError) as ei:
        select_model(fake_project, None)
    assert ei.value.code == "NO_ACCEPTED_MODEL"
    run = wait_run(fake_project, start_training(fake_project)["run_id"], timeout_s=30)
    with pytest.raises(StateError) as ei:
        select_model(fake_project, run["model_version"])
    assert ei.value.code == "MODEL_NOT_ACCEPTED"


def test_evaluate_unknown_model(fake_project: Project):
    with pytest.raises(NotFoundError):
        start_evaluation(fake_project, "mv-999")


# --- autolabel ------------------------------------------------------------------------------------


def test_autolabel_creates_model_label_set_without_building(fake_project: Project):
    p = fake_project
    assert p.config.labeling.auto.provider == "fake"  # fixture swapped every provider
    datasets_before = p.store.count("datasets")

    run = start_autolabel(p, confidence=0.5)
    assert run["kind"] == "autolabel" and run["inputs"]["asset_count"] == 12
    assert run["params"]["class_map"] == {"person": "person"} and run["params"]["confidence"] == 0.5
    lst = job_dir(p, run["run_id"]) / "inputs" / "images.txt"
    assert len(lst.read_text().splitlines()) == 12

    run = wait_run(p, run["run_id"], timeout_s=30)
    assert run["state"] == "COMPLETED" and run["label_set_id"]
    rec, anns = load_label_set(p, run["label_set_id"])
    assert rec["source"]["kind"] == "model" and rec["source"]["run_id"] == run["run_id"]
    assert rec["source"]["provider"] == "fake" and rec["source"]["weights"] == "yolo11n.pt"
    assert rec["run_id"] == run["run_id"] and rec["parent_label_set_id"] is None
    assert rec["counts"]["annotations"] == len(anns) == run["metrics"]["detections"]
    assert rec["counts"]["assets_with_annotations"] == 12
    assert {a.source.kind for a in anns} == {"model"}
    assert all(a.source.run_id == run["run_id"] and a.source.ref == "yolo11n.pt" for a in anns)
    assert all(a.confidence is not None for a in anns)
    assert all(a.metadata["source_class"] == "person" for a in anns)

    # Nothing was built or merged automatically.
    assert p.store.count("datasets") == datasets_before
    assert p.store.count("labels") == 2

    # The new label set is "latest" and can feed a new, distinct dataset version on request.
    plan = plan_dataset(p)
    assert plan.label_set_id == run["label_set_id"]
    rec2, created = build_dataset(p, plan)
    assert created and rec2["dataset_version"] == "ds-v002"
    assert rec2["provenance"]["label_source"]["kind"] == "model"


def test_autolabel_keeps_unknown_mapped_classes_for_validation(fake_project: Project):
    """class_map maps *source* classes to project classes; the hook applies the map and drops
    unmapped source classes. Mapped-to names that are not project classes are kept so that
    dataset validation (UNKNOWN_CLASS) reports them instead of silently losing boxes."""
    p = fake_project
    # The fake provider emits the first and last class_map keys as source classes.
    text = p.config_path.read_text().replace(
        "class_map: { person: person }", "class_map: { car: person, bike: zebra }"
    )
    p.config_path.write_text(text)
    p = Project.load("demo", p.settings)
    assert p.config.labeling.auto.class_map == {"car": "person", "bike": "zebra"}
    run = wait_run(p, start_autolabel(p, confidence=0.0)["run_id"], timeout_s=30)
    rec, anns = load_label_set(p, run["label_set_id"])
    assert rec["source"]["dropped_unmapped"] == 0
    assert {a.class_name for a in anns} == {"person", "zebra"}
    assert {a.metadata["source_class"] for a in anns} == {"car", "bike"}
    plan = plan_dataset(p)
    assert "UNKNOWN_CLASS" in [e.code for e in plan.report.errors]


def test_autolabel_drops_unmapped_source_classes(fake_project: Project):
    p = fake_project
    # Only one of the two emitted source classes is mapped -> the other is dropped.
    import ai_pipeline.providers.fake as fake_mod

    text = p.config_path.read_text().replace(
        "class_map: { person: person }", "class_map: { car: person }"
    )
    p.config_path.write_text(text)
    p = Project.load("demo", p.settings)
    run = wait_run(p, start_autolabel(p, confidence=0.0)["run_id"], timeout_s=30)
    rec, anns = load_label_set(p, run["label_set_id"])
    assert fake_mod.FakeProvider.name == "fake"
    assert rec["source"]["dropped_unmapped"] == 0  # single key: both emitted boxes are 'car'
    assert {a.class_name for a in anns} == {"person"}

    # Force an unmapped source class by editing the detections the hook consumes.
    import json as _json

    det = p.settings.artifact_root / run["outputs"]["detections"]
    rows = [_json.loads(line) for line in det.read_text().splitlines()]
    rows[0]["class_name"] = "giraffe"
    det.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    run["label_set_id"] = None
    run["post_processed"] = False
    p.store.write("runs", run["run_id"], run)
    run = sync_run(p, run["run_id"])
    rec2 = p.store.read("labels", run["label_set_id"])
    assert rec2["source"]["dropped_unmapped"] == 1
    assert rec2["counts"]["annotations"] == len(rows) - 1


def test_autolabel_requires_config_and_assets(demo_project: Project):
    with pytest.raises(StateError) as ei:
        start_autolabel(demo_project)
    assert ei.value.code == "NO_ASSETS"


# --- dataset build provenance (item 4) ------------------------------------------------------------


def test_rebuild_records_build_history_without_touching_version(
    fake_project: Project, tmp_path: Path
):
    p = fake_project
    rec = p.store.read("datasets", "ds-v001")
    assert len(rec["build_history"]) == 1 and rec["build_history"][0]["created"] is True
    first_lbl = rec["build_history"][0]["label_set_id"]

    import_labels(p, tmp_path / "labels", fmt="yolo")  # identical content, new label set id
    rec2, created = build_dataset(p, plan_dataset(p))
    assert not created and rec2["dataset_version"] == "ds-v001"
    hist = rec2["build_history"]
    assert len(hist) == 2 and hist[1]["created"] is False
    assert hist[1]["label_set_id"] != first_lbl
    # annotations.jsonl embeds provenance (label_set_id, file refs) so its file hash differs;
    # identity equality is carried by content_hash below.
    assert hist[0]["annotations_sha256"] != hist[1]["annotations_sha256"]
    assert rec2["content_hash"] == rec["content_hash"] and rec2["provenance"] == rec["provenance"]
    # the immutable directory copy is unchanged
    on_disk = json.loads((p.store.root / "datasets" / "ds-v001" / "dataset.json").read_text())
    assert len(on_disk["build_history"]) == 1
