from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.conftest import export_model, make_image_set, write_yolo_labels  # noqa: F401

from ai_pipeline.core.deployment import (
    health,
    list_deployments,
    raise_for_state,
    rollback_deployment,
    start_deployment,
)
from ai_pipeline.core.errors import ConfigError, OperationError, PolicyRejected, StateError
from ai_pipeline.core.exports import start_export
from ai_pipeline.core.project import Project
from ai_pipeline.core.runs import start_evaluation, start_training, wait_run
from ai_pipeline.core.store import write_json_atomic
from ai_pipeline.targets import get_target
from ai_pipeline.targets.fake import DEFAULT_BENCHMARK, FakeTarget


def _export(p: Project, mv: str | None = None, **kw):
    return export_model(p, mv, **kw)


def _target(p: Project) -> FakeTarget:
    return get_target("fake", p.settings)  # type: ignore[return-value]


def _inject(p: Project, **data) -> None:
    t = _target(p)
    t.root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(t.root / "inject.json", data)


def _clear_inject(p: Project) -> None:
    (_target(p).root / "inject.json").unlink(missing_ok=True)


# --- exports --------------------------------------------------------------------------------------


def test_export_creates_immutable_artifact_and_links_model(accepted_project: Project):
    p = accepted_project
    rec = _export(p)
    assert rec["export_id"] == "exp-001" and rec["state"] == "READY"
    assert rec["model_version"] == "mv-001" and rec["format"] == "onnx"
    assert rec["export_run_id"].startswith("run-") and rec["provider"] == "fake"
    model = p.store.read("models", "mv-001")
    assert rec["source_weights"]["sha256"] == model["artifacts"]["weights"]["sha256"]
    assert rec["source_weights"]["path"] == model["artifacts"]["weights"]["path"]
    assert rec["export_params"] == {"opset": 17, "image_size": 640, "dynamic": False}
    art = p.settings.artifact_root / rec["artifact"]["path"]
    assert art.is_file() and art.name == "model.onnx"
    assert rec["artifact"]["sha256"] and rec["artifact"]["size_bytes"] == art.stat().st_size
    assert rec["details"]["exporter"] == "fake"
    assert rec["environment"]["provider"] == "fake"
    # model state unchanged; traceability list updated
    assert model["state"] == "ACCEPTED" and model["exports"] == ["exp-001"]
    run = p.store.read("runs", rec["export_run_id"])
    assert run["kind"] == "export" and run["export_id"] == "exp-001"


def test_export_reuse_and_rerun(accepted_project: Project):
    p = accepted_project
    first = _export(p)
    run, existing = start_export(p)
    assert run is None and existing["export_id"] == first["export_id"]
    assert p.store.count("exports") == 1
    second = _export(p, rerun=True)
    assert second["export_id"] == "exp-002" and second["export_run_id"] != first["export_run_id"]
    assert p.store.read("exports", "exp-001") == first  # never overwritten
    assert p.store.read("models", "mv-001")["exports"] == ["exp-001", "exp-002"]

    # a parameter change is a different request -> new export without --rerun
    p.config_path.write_text(p.config_path.read_text().replace("opset: 17", "opset: 13"))
    p2 = Project.load("demo", p.settings)
    third = _export(p2)
    assert third["export_id"] == "exp-003" and third["export_params"]["opset"] == 13


def test_export_explicit_model_and_errors(accepted_project: Project):
    p = accepted_project
    # a second, unselected model can be exported explicitly
    run2 = wait_run(p, start_training(p)["run_id"], timeout_s=30)
    rec = _export(p, run2["model_version"])
    assert rec["model_version"] == "mv-002"
    p.set_pointer("selected_model_version", None)
    with pytest.raises(StateError) as ei:
        start_export(p)
    assert ei.value.code == "NO_SELECTED_MODEL"


def test_export_job_failure(accepted_project: Project):
    p = accepted_project
    p.config_path.write_text(
        p.config_path.read_text().replace(
            "dynamic: false\n  provider_args: {}",
            'dynamic: false\n  provider_args: {"fake_fail": "export"}',
        )
    )
    p = Project.load("demo", p.settings)
    assert p.config.export.provider_args == {"fake_fail": "export"}
    run, _ = start_export(p)
    run = wait_run(p, run["run_id"], timeout_s=30)
    assert run["state"] == "FAILED" and run["export_id"] is None
    assert p.store.count("exports") == 0


def test_export_does_not_block_reevaluation(exported_project: Project):
    p = exported_project
    ev = wait_run(p, start_evaluation(p, "mv-001")["run_id"], timeout_s=30)
    assert ev["state"] == "COMPLETED"
    assert p.store.read("models", "mv-001")["state"] == "ACCEPTED"


# --- fake target ----------------------------------------------------------------------------------


def test_fake_target_phases_and_atomic_current(tmp_path: Path):
    t = FakeTarget(name="fake", root=tmp_path / "dev")
    art = tmp_path / "m.onnx"
    art.write_text("onnx")
    info = t.verify()
    assert info["reachable"] and info["current"] is None and info["staged"] == []
    staged = t.stage("dep-001", art, {"model_version": "mv-001"})
    assert Path(staged["path"]).is_file() and staged["size_bytes"] == 4
    with pytest.raises(OperationError) as ei:
        t.stage("dep-001", art, {})
    assert ei.value.code == "TARGET_DEPLOYMENT_EXISTS"
    with pytest.raises(OperationError) as ei:
        t.activate("dep-001")  # not prepared yet
    assert ei.value.code == "TARGET_NOT_PREPARED"
    prep = t.prepare("dep-001", precision="fp16", params={"runtime": "tensorrt"})
    assert Path(prep["engine_path"]).is_file() and prep["build_params"]["precision"] == "fp16"
    assert t.smoke_test("dep-001")["passed"]
    bm = t.benchmark("dep-001")
    assert bm["metrics"] == DEFAULT_BENCHMARK and bm["iterations"] == 50
    assert t.health_check()["healthy"] is False  # nothing active yet
    act = t.activate("dep-001")
    assert act == {"previous": None, "current": "dep-001", "at": act["at"]}
    assert t.current() == "dep-001" and os.readlink(t.current_link) == "deployments/dep-001"
    assert t.health_check()["healthy"]
    t.stage("dep-002", art, {})
    t.prepare("dep-002", precision="fp16", params={})
    assert t.activate("dep-002")["previous"] == "dep-001"
    rb = t.rollback("dep-001")
    assert rb["previous"] == "dep-002" and rb["current"] == "dep-001"
    assert t.current() == "dep-001"
    assert t.rollback(None)["current"] is None and t.current() is None
    with pytest.raises(OperationError) as ei:
        t.rollback("dep-999")
    assert ei.value.code == "TARGET_ROLLBACK_MISSING"


def test_fake_target_injection(tmp_path: Path):
    t = FakeTarget(name="fake", root=tmp_path / "dev")
    art = tmp_path / "m.onnx"
    art.write_text("onnx")
    write_json_atomic(
        t.root / "inject.json", {"fail": "prepare", "benchmark_metrics": {"fps": 3.0}}
    )
    t.stage("dep-001", art, {})
    with pytest.raises(OperationError) as ei:
        t.prepare("dep-001", precision="fp16", params={})
    assert ei.value.code == "TARGET_PREPARE_FAILED" and ei.value.details["injected"]
    write_json_atomic(
        t.root / "inject.json", {"benchmark_metrics": {"fps": 3.0}, "health_fail_for": ["dep-001"]}
    )
    t.prepare("dep-001", precision="fp16", params={})
    assert t.benchmark("dep-001")["metrics"]["fps"] == 3.0
    t.activate("dep-001")
    assert t.health_check()["healthy"] is False
    write_json_atomic(t.root / "inject.json", {"fail": "smoke_test"})
    with pytest.raises(OperationError):
        t.smoke_test("dep-001")


def test_target_factory(settings):
    t = get_target("fake", settings)
    assert isinstance(t, FakeTarget) and t.root == settings.infra_path.parent / ".fake-target"
    from ai_pipeline.targets.camthink_ng4511 import CamThinkNG4511Target

    ng = get_target("ng4511-lab", settings)
    assert isinstance(ng, CamThinkNG4511Target) and ng.cfg.ssh_host == "ng4511"  # no SSH yet
    with pytest.raises(ConfigError):
        get_target("nope", settings)


# --- deployment: success --------------------------------------------------------------------------


def test_deploy_success_path(exported_project: Project):
    p = exported_project
    rec = start_deployment(p)
    assert rec["deployment_id"] == "dep-001" and rec["state"] == "ACTIVE"
    assert [ph["state"] for ph in rec["phases"]] == [
        "CREATED",
        "VERIFYING",
        "STAGING",
        "PREPARING",
        "SMOKE_TESTING",
        "BENCHMARKING",
        "READY_TO_ACTIVATE",
        "ACTIVATING",
        "HEALTH_CHECKING",
        "ACTIVE",
    ]
    assert rec["model_version"] == "mv-001" and rec["export_id"] == "exp-001"
    assert (
        rec["export_artifact"]["sha256"] == p.store.read("exports", "exp-001")["artifact"]["sha256"]
    )
    assert rec["target"] == "fake" and rec["target_type"] == "fake" and rec["precision"] == "fp16"
    assert (
        rec["target_info"]["reachable"]
        and rec["staged"]["sha256"] == rec["export_artifact"]["sha256"]
    )
    assert (
        rec["runtime_artifact"]["engine_sha256"]
        and rec["runtime_artifact"]["build_params"]["precision"] == "fp16"
    )
    assert rec["smoke"]["passed"] and rec["benchmark"]["metrics"]["fps"] == 32.0
    assert rec["acceptance"]["passed"] and {c["metric"] for c in rec["acceptance"]["checks"]} == {
        "fps",
        "latency_ms_p50",
    }
    assert rec["activation"]["previous"] is None and rec["health"]["healthy"]
    assert rec["target_ref"] == "demo-dep-001"
    assert rec["previous_deployment"] is None and rec["rollback"] is None and rec["error"] is None
    assert rec["timestamps"]["finished"]
    assert p.pointers()["active_deployment"] == "dep-001"
    assert p.pointers()["last_known_good_deployment"] == "dep-001"
    assert p.store.read("models", "mv-001")["deployments"] == ["dep-001"]
    assert p.store.read("models", "mv-001")["state"] == "ACCEPTED"  # untouched
    assert _target(p).current() == "demo-dep-001"
    h = health(p)
    assert h["healthy"] and h["consistent"] and h["target_current"] == "demo-dep-001"
    raise_for_state(rec)  # no exception


def test_second_deployment_supersedes_and_manual_rollback(exported_project: Project):
    p = exported_project
    d1 = start_deployment(p)
    d2 = start_deployment(p)
    assert d2["deployment_id"] == "dep-002" and d2["state"] == "ACTIVE"
    assert d2["previous_deployment"] == "dep-001" and d2["activation"]["previous"] == "demo-dep-001"
    assert p.store.read("deployments", "dep-001")["state"] == "SUPERSEDED"
    assert p.pointers()["active_deployment"] == "dep-002"
    assert _target(p).current() == "demo-dep-002"

    rb = rollback_deployment(p)
    assert rb["deployment_id"] == "dep-002" and rb["state"] == "ROLLED_BACK"
    assert rb["rollback"]["trigger"] == "manual" and rb["rollback"]["to"] == "dep-001"
    assert rb["rollback"]["health"]["healthy"]
    assert p.store.read("deployments", "dep-001")["state"] == "ACTIVE"
    assert p.pointers()["active_deployment"] == "dep-001"
    assert p.pointers()["last_known_good_deployment"] == "dep-001"
    assert _target(p).current() == "demo-dep-001"
    assert d1["deployment_id"] == "dep-001"
    with pytest.raises(StateError) as ei:
        rollback_deployment(p)  # dep-001 has no previous
    assert ei.value.code == "NO_PREVIOUS_DEPLOYMENT"


def test_deploy_without_acceptance_rules_still_persists_benchmark(exported_project: Project):
    p = exported_project
    text = p.config_path.read_text()
    start = text.index("  acceptance:\n    latency_ms_p50")
    end = text.index("  health:", start)
    p.config_path.write_text(text[:start] + "  acceptance: {}\n" + text[end:])
    p = Project.load("demo", p.settings)
    rec = start_deployment(p)
    assert rec["state"] == "ACTIVE"
    assert rec["benchmark"]["metrics"] == DEFAULT_BENCHMARK
    assert rec["acceptance"] == {"passed": True, "checks": []}


# --- deployment: preconditions --------------------------------------------------------------------


def test_deploy_preconditions(accepted_project: Project):
    p = accepted_project
    with pytest.raises(StateError) as ei:
        start_deployment(p)
    assert ei.value.code == "NO_EXPORT"
    _export(p)
    # a REJECTED model may not be deployed even explicitly
    p.config_path.write_text(
        p.config_path.read_text().replace(
            "provider_args: {}",
            'provider_args: {"fake_metrics": {"precision": 0.1, "recall": 0.1, '
            '"mAP50": 0.1, "mAP50-95": 0.05}}',
        )
    )
    p2 = Project.load("demo", p.settings)
    run = wait_run(p2, start_training(p2)["run_id"], timeout_s=30)
    wait_run(p2, start_evaluation(p2, run["model_version"])["run_id"], timeout_s=30)
    assert p2.store.read("models", "mv-002")["state"] == "REJECTED"
    with pytest.raises(StateError) as ei:
        start_deployment(p2, model_version="mv-002")
    assert ei.value.code == "MODEL_REJECTED"
    # a merely TRAINED model may not be deployed either
    run3 = wait_run(p2, start_training(p2)["run_id"], timeout_s=30)
    with pytest.raises(StateError) as ei:
        start_deployment(p2, model_version=run3["model_version"])
    assert ei.value.code == "MODEL_NOT_ACCEPTED"
    # export/model mismatch: exp-001 belongs to mv-001
    rec3 = _export(p2, run3["model_version"])
    with pytest.raises(StateError) as ei:
        start_deployment(p2, model_version="mv-001", export_id=rec3["export_id"])
    assert ei.value.code == "EXPORT_MODEL_MISMATCH"
    assert p2.store.count("deployments") == 0  # nothing half-created by failed preconditions


# --- deployment: failures before activation -------------------------------------------------------


@pytest.mark.parametrize("phase", ["verify", "stage", "prepare", "benchmark"])
def test_failure_before_activation_leaves_active_untouched(exported_project: Project, phase: str):
    p = exported_project
    good = start_deployment(p)
    assert good["state"] == "ACTIVE"
    _inject(p, fail=phase)
    bad = start_deployment(p)
    assert bad["deployment_id"] == "dep-002" and bad["state"] == "FAILED"
    assert (
        bad["error"]["phase"] == phase and bad["error"]["code"] == f"TARGET_{phase.upper()}_FAILED"
    )
    assert bad["rollback"] is None and bad["activation"] is None
    assert bad["phases"][-1]["state"] == "FAILED"
    # active deployment and target pointer untouched
    assert p.pointers()["active_deployment"] == "dep-001"
    assert p.store.read("deployments", "dep-001")["state"] == "ACTIVE"
    assert _target(p).current() == "demo-dep-001"
    with pytest.raises(OperationError) as ei:
        raise_for_state(bad)
    assert ei.value.code == "DEPLOYMENT_FAILED" and ei.value.exit_code == 1


def test_smoke_test_failure(exported_project: Project):
    p = exported_project
    start_deployment(p)
    _inject(p, fail="smoke_test")
    bad = start_deployment(p)
    assert bad["state"] == "FAILED" and bad["error"]["phase"] == "smoke_test"
    assert bad["benchmark"] is None  # never reached
    assert p.pointers()["active_deployment"] == "dep-001"


def test_policy_rejection_does_not_activate(exported_project: Project):
    p = exported_project
    start_deployment(p)
    _inject(p, benchmark_metrics={"fps": 9.0})
    rej = start_deployment(p)
    assert rej["state"] == "REJECTED"
    assert rej["benchmark"]["metrics"]["fps"] == 9.0  # execution result persisted
    assert rej["acceptance"]["passed"] is False
    assert [c["metric"] for c in rej["acceptance"]["checks"] if not c["passed"]] == ["fps"]
    assert rej["error"]["code"] == "POLICY_REJECTED" and rej["activation"] is None
    assert rej["rollback"] is None
    assert p.pointers()["active_deployment"] == "dep-001"
    assert _target(p).current() == "demo-dep-001"
    with pytest.raises(PolicyRejected) as ei:
        raise_for_state(rej)
    assert ei.value.exit_code == 5
    # model acceptance is a separate concept and unaffected
    assert p.store.read("models", "mv-001")["state"] == "ACCEPTED"


# --- deployment: failures after activation --------------------------------------------------------


def test_health_failure_rolls_back_to_previous(exported_project: Project):
    p = exported_project
    start_deployment(p)
    _inject(p, health_fail_for=["demo-dep-002"])
    bad = start_deployment(p)
    assert bad["deployment_id"] == "dep-002" and bad["state"] == "ROLLED_BACK"
    phases = [ph["state"] for ph in bad["phases"]]
    assert phases[-4:] == ["ACTIVATING", "HEALTH_CHECKING", "ROLLING_BACK", "ROLLED_BACK"]
    assert bad["activation"]["previous"] == "demo-dep-001" and bad["health"]["healthy"] is False
    assert bad["error"]["code"] == "HEALTH_CHECK_FAILED"
    rb = bad["rollback"]
    assert rb["trigger"] == "automatic" and rb["to"] == "dep-001" and rb["health"]["healthy"]
    assert p.store.read("deployments", "dep-001")["state"] == "ACTIVE"
    assert p.pointers()["active_deployment"] == "dep-001"
    assert p.pointers()["last_known_good_deployment"] == "dep-001"
    assert _target(p).current() == "demo-dep-001"
    with pytest.raises(OperationError) as ei:
        raise_for_state(bad)
    assert ei.value.code == "DEPLOYMENT_ROLLED_BACK"


def test_activation_failure_rolls_back(exported_project: Project):
    p = exported_project
    start_deployment(p)
    _inject(p, fail="activate")
    bad = start_deployment(p)
    assert bad["state"] == "ROLLED_BACK" and bad["error"]["phase"] == "activate"
    assert bad["rollback"]["to"] == "dep-001"
    assert p.pointers()["active_deployment"] == "dep-001" and _target(p).current() == "demo-dep-001"


def test_first_deployment_health_failure_rolls_back_to_nothing(exported_project: Project):
    p = exported_project
    _inject(p, health_fail_for=["demo-dep-001"])
    bad = start_deployment(p)
    assert bad["state"] == "ROLLED_BACK" and bad["rollback"]["to"] is None
    assert p.pointers()["active_deployment"] is None
    assert p.pointers()["last_known_good_deployment"] is None
    assert _target(p).current() is None


def test_rollback_failure_is_explicit(exported_project: Project):
    p = exported_project
    start_deployment(p)
    _inject(p, fail="rollback", health_fail_for=["demo-dep-002"])
    bad = start_deployment(p)
    assert bad["state"] == "ROLLBACK_FAILED"
    assert bad["rollback"]["error"]["code"] == "TARGET_ROLLBACK_FAILED"
    assert bad["rollback"]["target_current"] == "demo-dep-002"  # what the target actually shows
    # we refuse to claim anything is active; last known good is preserved for recovery
    assert p.pointers()["active_deployment"] is None
    assert p.pointers()["last_known_good_deployment"] == "dep-001"
    assert p.store.read("deployments", "dep-001")["state"] == "SUPERSEDED"
    with pytest.raises(OperationError) as ei:
        raise_for_state(bad)
    assert ei.value.code == "DEPLOYMENT_ROLLBACK_FAILED"
    h = health(p)
    assert not h["consistent"]

    # recovery: explicit rollback to the last known good
    _clear_inject(p)
    with pytest.raises(StateError) as ei:
        rollback_deployment(p)
    assert ei.value.code == "NO_ACTIVE_DEPLOYMENT"
    restored = rollback_deployment(p, to="dep-001")
    assert restored["deployment_id"] == "dep-001" and restored["state"] == "ACTIVE"
    assert p.pointers()["active_deployment"] == "dep-001" and _target(p).current() == "demo-dep-001"
    assert p.store.read("deployments", "dep-002")["state"] == "ROLLBACK_FAILED"  # history kept


def test_rollback_health_failure_is_rollback_failed(exported_project: Project):
    p = exported_project
    start_deployment(p)
    _inject(p, fail="health_check")  # every health check fails, including after rollback
    bad = start_deployment(p)
    assert bad["state"] == "ROLLBACK_FAILED"
    assert bad["rollback"]["error"]["code"] == "TARGET_HEALTH_CHECK_FAILED"


def test_records_are_never_overwritten(exported_project: Project):
    p = exported_project
    a = start_deployment(p)
    _inject(p, benchmark_metrics={"fps": 1.0})
    b = start_deployment(p)
    _clear_inject(p)
    c = start_deployment(p)
    ids = [d["deployment_id"] for d in list_deployments(p)]
    assert ids == ["dep-001", "dep-002", "dep-003"]
    assert [d["state"] for d in list_deployments(p)] == ["SUPERSEDED", "REJECTED", "ACTIVE"]
    assert a["export_id"] == b["export_id"] == c["export_id"] == "exp-001"
    assert p.store.read("deployments", "dep-002")["benchmark"]["metrics"]["fps"] == 1.0
    assert p.store.read("models", "mv-001")["deployments"] == ["dep-001", "dep-002", "dep-003"]
    assert (
        json.loads(
            (_target(p).root / "deployments" / "demo-dep-002" / "deployment.json").read_text()
        )["deployment_id"]
        == "dep-002"
    )
