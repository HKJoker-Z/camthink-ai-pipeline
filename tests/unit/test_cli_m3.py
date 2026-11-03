from __future__ import annotations

import json
from pathlib import Path

from tests.conftest import make_image_set, write_yolo_labels
from typer.testing import CliRunner

from ai_pipeline.cli.main import app

runner = CliRunner()


def invoke_json(*args: str):
    result = runner.invoke(app, ["-o", "json", *args])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:  # pragma: no cover
        raise AssertionError(f"stdout is not a single JSON document:\n{result.stdout}") from None
    return result, payload


def _prepare_accepted(workspace: Path, tmp_path: Path) -> None:
    src = tmp_path / "src"
    imgs = make_image_set(src, 12)
    write_yolo_labels(tmp_path / "labels", [p.stem for p in imgs])
    invoke_json("project", "create", "demo")
    cfg = workspace / "projects" / "demo" / "project.yaml"
    cfg.write_text(
        cfg.read_text()
        .replace("min_images: 20", "min_images: 4")
        .replace("min_instances_per_class: 10", "min_instances_per_class: 1")
        .replace("provider: ultralytics", "provider: fake")
    )
    invoke_json("data", "import", "demo", str(src))
    invoke_json("label", "import", "demo", str(tmp_path / "labels"))
    invoke_json("dataset", "build", "demo")
    r, p = invoke_json("train", "start", "demo", "--wait")
    assert r.exit_code == 0, p
    r, p = invoke_json("evaluate", "run", "demo", "mv-001", "--wait")
    assert r.exit_code == 0 and p["result"]["model_state"] == "ACCEPTED"
    r, p = invoke_json("model", "select", "demo")
    assert r.exit_code == 0


def _inject(workspace: Path, **data) -> None:
    root = workspace / ".fake-target"
    root.mkdir(exist_ok=True)
    (root / "inject.json").write_text(json.dumps(data))


def test_export_and_deploy_success_json(workspace: Path, tmp_path: Path):
    _prepare_accepted(workspace, tmp_path)
    r, p = invoke_json("status", "demo")
    assert p["next_operations"] == ["export start demo"]

    r, p = invoke_json("export", "start", "demo", "--wait")
    assert r.exit_code == 0, p
    assert p["result"]["kind"] == "export" and p["result"]["state"] == "COMPLETED"
    assert p["result"]["created"] is True and p["result"]["export_id"] == "exp-001"
    exp = p["result"]["export"]
    assert (
        exp["format"] == "onnx" and exp["artifact"]["sha256"] and exp["model_version"] == "mv-001"
    )
    assert p["next_operations"] == ["deploy start demo"]

    r, p = invoke_json("export", "start", "demo", "--wait")
    assert (
        r.exit_code == 0
        and p["result"]["created"] is False
        and p["result"]["export_id"] == "exp-001"
    )
    r, p = invoke_json("export", "list", "demo")
    assert [e["export_id"] for e in p["result"]["exports"]] == ["exp-001"]
    r, p = invoke_json("export", "show", "demo", "exp-001")
    assert (
        p["result"]["export_run_id"].startswith("run-") and p["result"]["source_weights"]["sha256"]
    )
    r, p = invoke_json("model", "show", "demo", "mv-001")
    assert p["result"]["exports"] == ["exp-001"] and p["result"]["state"] == "ACCEPTED"

    r, p = invoke_json("target", "verify", "demo")
    assert r.exit_code == 0 and p["result"]["reachable"] and p["result"]["current"] is None

    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["deployment_id"] == "dep-001" and res["state"] == "ACTIVE"
    assert res["phases"][-1] == "ACTIVE" and res["benchmark"]["fps"] == 32.0
    assert res["acceptance"]["passed"] is True and res["health"] is True
    assert res["pointers"]["active_deployment"] == "dep-001"
    assert p["next_operations"][0] == "health demo"

    r, p = invoke_json("health", "demo")
    assert r.exit_code == 0 and p["result"]["healthy"] and p["result"]["consistent"]
    r, p = invoke_json("deploy", "status", "demo")
    assert p["result"]["deployment_id"] == "dep-001" and p["result"]["state"] == "ACTIVE"
    r, p = invoke_json("deploy", "status", "demo", "dep-001", "--full")
    assert p["result"]["record"]["runtime_artifact"]["engine_sha256"]
    r, p = invoke_json("deploy", "list", "demo")
    assert [d["deployment_id"] for d in p["result"]["deployments"]] == ["dep-001"]
    r, p = invoke_json("status", "demo")
    assert p["result"]["deployment_states"] == {"dep-001": "ACTIVE"}


def test_pre_activation_rejection_and_failure_json(workspace: Path, tmp_path: Path):
    _prepare_accepted(workspace, tmp_path)
    invoke_json("export", "start", "demo", "--wait")
    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 0

    _inject(workspace, benchmark_metrics={"fps": 9.0})
    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 5
    assert p["status"] == "rejected" and p["error"]["code"] == "POLICY_REJECTED"
    assert p["result"]["deployment_id"] == "dep-002" and p["result"]["state"] == "REJECTED"
    assert p["result"]["benchmark"]["fps"] == 9.0
    assert [c["metric"] for c in p["result"]["acceptance"]["checks"] if not c["passed"]] == ["fps"]
    assert p["result"]["pointers"]["active_deployment"] == "dep-001"

    _inject(workspace, fail="prepare")
    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 1
    assert p["error"]["code"] == "DEPLOYMENT_FAILED" and p["error"]["details"]["phase"] == "prepare"
    assert p["result"]["state"] == "FAILED" and p["result"]["rollback"] is None
    assert p["result"]["pointers"]["active_deployment"] == "dep-001"

    r, p = invoke_json("deploy", "status", "demo")
    assert p["result"]["deployment_id"] == "dep-001" and p["result"]["state"] == "ACTIVE"
    r, p = invoke_json("health", "demo")
    assert r.exit_code == 0 and p["result"]["target_current"] == "dep-001"


def test_post_activation_rollback_json(workspace: Path, tmp_path: Path):
    _prepare_accepted(workspace, tmp_path)
    invoke_json("export", "start", "demo", "--wait")
    invoke_json("deploy", "start", "demo")

    _inject(workspace, health_fail_for=["dep-002"])
    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 1
    assert p["error"]["code"] == "DEPLOYMENT_ROLLED_BACK"
    res = p["result"]
    assert res["state"] == "ROLLED_BACK" and res["health"] is False
    assert res["rollback"]["trigger"] == "automatic" and res["rollback"]["to"] == "dep-001"
    assert res["phases"][-3:] == ["HEALTH_CHECKING", "ROLLING_BACK", "ROLLED_BACK"]
    assert res["pointers"]["active_deployment"] == "dep-001"
    r, p = invoke_json("deploy", "list", "demo")
    assert [d["state"] for d in p["result"]["deployments"]] == ["ACTIVE", "ROLLED_BACK"]
    r, p = invoke_json("health", "demo")
    assert r.exit_code == 0 and p["result"]["target_current"] == "dep-001"

    # rollback failure -> explicit state, recovery via --to
    _inject(workspace, fail="rollback", health_fail_for=["dep-003"])
    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 1 and p["error"]["code"] == "DEPLOYMENT_ROLLBACK_FAILED"
    assert (
        p["result"]["state"] == "ROLLBACK_FAILED"
        and p["result"]["pointers"]["active_deployment"] is None
    )
    assert p["next_operations"][0] == "health demo"
    r, p = invoke_json("health", "demo")
    assert r.exit_code == 1 and p["error"]["code"] == "UNHEALTHY"
    assert p["result"]["target_current"] == "dep-003" and p["result"]["consistent"] is False
    (workspace / ".fake-target" / "inject.json").unlink()
    r, p = invoke_json("health", "demo")  # target now healthy but nothing is marked active
    assert r.exit_code == 1 and p["error"]["code"] == "POINTER_MISMATCH"
    r, p = invoke_json("rollback", "demo")
    assert r.exit_code == 3 and p["error"]["code"] == "NO_ACTIVE_DEPLOYMENT"
    r, p = invoke_json("rollback", "demo", "--to", "dep-001")
    assert (
        r.exit_code == 0
        and p["result"]["deployment_id"] == "dep-001"
        and p["result"]["state"] == "ACTIVE"
    )
    r, p = invoke_json("health", "demo")
    assert r.exit_code == 0 and p["result"]["consistent"]


def test_manual_rollback_json(workspace: Path, tmp_path: Path):
    _prepare_accepted(workspace, tmp_path)
    invoke_json("export", "start", "demo", "--wait")
    invoke_json("deploy", "start", "demo")
    r, p = invoke_json("deploy", "start", "demo")
    assert (
        p["result"]["deployment_id"] == "dep-002"
        and p["result"]["previous_deployment"] == "dep-001"
    )
    r, p = invoke_json("rollback", "demo")
    assert r.exit_code == 0, p
    assert p["result"]["deployment_id"] == "dep-002" and p["result"]["state"] == "ROLLED_BACK"
    assert (
        p["result"]["rollback"]["trigger"] == "manual"
        and p["result"]["pointers"]["active_deployment"] == "dep-001"
    )
    r, p = invoke_json("deploy", "list", "demo")
    assert [d["state"] for d in p["result"]["deployments"]] == ["ACTIVE", "ROLLED_BACK"]


def test_deploy_preconditions_json(workspace: Path, tmp_path: Path):
    _prepare_accepted(workspace, tmp_path)
    r, p = invoke_json("deploy", "start", "demo")
    assert r.exit_code == 3 and p["error"]["code"] == "NO_EXPORT"
    r, p = invoke_json("deploy", "status", "demo")
    assert r.exit_code == 0 and p["result"]["active_deployment"] is None
    r, p = invoke_json("rollback", "demo")
    assert r.exit_code == 3 and p["error"]["code"] == "NO_ACTIVE_DEPLOYMENT"
    r, p = invoke_json("export", "show", "demo", "exp-404")
    assert r.exit_code == 3 and p["error"]["code"] == "EXPORT_NOT_FOUND"
    r, p = invoke_json("deploy", "start", "demo", "--target", "ng4511-lab")
    assert r.exit_code == 3  # no export yet is checked first
    invoke_json("export", "start", "demo", "--wait")
    r, p = invoke_json("deploy", "start", "demo", "--target", "ng4511-lab")
    assert r.exit_code == 1 and p["error"]["code"] == "TARGET_NOT_IMPLEMENTED"


def test_human_output_m3(workspace: Path, tmp_path: Path):
    _prepare_accepted(workspace, tmp_path)
    r = runner.invoke(app, ["export", "start", "demo", "--wait"])
    assert r.exit_code == 0 and "export exp-001: onnx" in r.stdout
    r = runner.invoke(app, ["deploy", "start", "demo"])
    assert r.exit_code == 0 and "state=ACTIVE" in r.stdout and "PASS fps=32.0" in r.stdout
    _inject(workspace, benchmark_metrics={"fps": 9.0})
    r = runner.invoke(app, ["deploy", "start", "demo"])
    assert r.exit_code == 5 and "REJECTED" in r.stdout and "FAIL fps=9.0" in r.stdout
    r = runner.invoke(app, ["deploy", "list", "demo"])
    assert "* dep-001  ACTIVE" in r.stdout and "dep-002  REJECTED" in r.stdout
    r = runner.invoke(app, ["health", "demo"])
    assert r.exit_code == 0 and "healthy; current=dep-001" in r.stdout
