"""M7: project creation/configuration from the CLI and edge service control."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from tests.conftest import make_image_set, write_yolo_labels
from tests.unit.test_cli_m3 import invoke_json

# -- project create / configure / show ------------------------------------------------------------


def _load(workspace: Path, name: str = "demo") -> dict:
    return yaml.safe_load((workspace / "projects" / name / "project.yaml").read_text())


def test_project_create_with_options(workspace: Path):
    r, p = invoke_json(
        "project",
        "create",
        "cat_dog",
        "--classes",
        "cat,dog",
        "--epochs",
        "3",
        "--weights",
        "yolo11s.pt",
        "--target",
        "ng4511-lab",
        "--min-images",
        "5",
        "--min-map50",
        "0.2",
        "--set",
        "training.device=1",
        "--set",
        "deployment.keep_versions=2",
        "--description",
        "Cats and dogs",
    )
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["created"] is True and "dataset.classes" in res["settings_applied"]
    assert res["config"]["classes"] == ["cat", "dog"]
    assert res["config"]["candidates"] == [{"name": "yolo11s", "weights": "yolo11s.pt"}]
    assert res["config"]["training"]["epochs"] == 3 and res["config"]["deployment"]["target"] == (
        "ng4511-lab"
    )
    data = _load(workspace, "cat_dog")
    assert data["training"]["device"] == "1"  # kept a string even though YAML parses 1 as int
    assert data["labeling"]["auto"]["class_map"] == {"cat": "cat", "dog": "dog"}
    assert data["evaluation"]["acceptance"]["mAP50"] == {"min": 0.2}
    assert data["deployment"]["keep_versions"] == 2
    assert data["project"]["description"] == "Cats and dogs"
    r, p = invoke_json("project", "validate", "cat_dog")
    assert r.exit_code == 0 and p["result"]["valid"] is True


def test_project_create_invalid_option_creates_nothing(workspace: Path):
    r, p = invoke_json("project", "create", "bad", "--target", "no-such-target")
    assert r.exit_code == 2 and p["status"] == "failed"
    assert p["error"]["code"] == "CONFIG_INFRA_MISMATCH"
    assert not (workspace / "projects" / "bad").exists()
    r, p = invoke_json("project", "create", "bad", "--set", "training.epochs=0")
    assert r.exit_code == 2 and p["error"]["code"] == "CONFIG_INVALID"
    assert not (workspace / "projects" / "bad").exists()
    r, p = invoke_json("project", "create", "bad", "--set", "project.name=other")
    assert r.exit_code == 2 and p["error"]["code"] == "SET_PATH_PROTECTED"
    assert not (workspace / "projects" / "bad").exists()


def test_project_configure_and_show(workspace: Path):
    invoke_json("project", "create", "demo")
    before = (workspace / "projects" / "demo" / "project.yaml").read_text()

    r, p = invoke_json("project", "configure", "demo")
    assert r.exit_code == 2 and p["error"]["code"] == "NO_CHANGES"

    r, p = invoke_json(
        "project", "configure", "demo", "--classes", "cat,dog", "--epochs", "7", "--batch-size", "8"
    )
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["changed"] is True
    paths = {c["path"]: c for c in res["changes"]}
    assert paths["dataset.classes"]["old"] == ["person"]
    assert paths["dataset.classes"]["new"] == ["cat", "dog"]
    assert paths["training.epochs"] == {"path": "training.epochs", "old": 5, "new": 7}
    assert "labeling.auto.class_map" in paths  # identity map regenerated with the classes
    data = _load(workspace)
    assert data["dataset"]["classes"] == ["cat", "dog"] and data["training"]["batch_size"] == 8
    assert data["labeling"]["auto"]["class_map"] == {"cat": "cat", "dog": "dog"}

    # explicit mapping from pretrained classes to project classes
    r, p = invoke_json("project", "configure", "demo", "--auto-label-map", "cat:cat,dog:dog")
    assert r.exit_code == 0 and p["result"]["changed"] is False  # same as the identity map
    r, p = invoke_json(
        "project", "configure", "demo", "--classes", "pet", "--auto-label-map", "cat:pet,dog:pet"
    )
    assert r.exit_code == 0, p
    assert _load(workspace)["labeling"]["auto"]["class_map"] == {"cat": "pet", "dog": "pet"}

    # invalid change leaves the file untouched
    snapshot = (workspace / "projects" / "demo" / "project.yaml").read_text()
    r, p = invoke_json("project", "configure", "demo", "--set", "training.epochs=0")
    assert r.exit_code == 2 and p["error"]["code"] == "CONFIG_INVALID", p
    assert (workspace / "projects" / "demo" / "project.yaml").read_text() == snapshot
    r, p = invoke_json("project", "configure", "demo", "--set", "training.nope=1")
    assert r.exit_code == 2 and p["error"]["code"] == "CONFIG_INVALID"
    r, p = invoke_json("project", "configure", "demo", "--set", "project.task_type=classification")
    assert r.exit_code == 2 and p["error"]["code"] == "SET_PATH_PROTECTED"
    r, p = invoke_json("project", "configure", "demo", "--set", "garbage")
    assert r.exit_code == 2 and p["error"]["code"] == "SET_SYNTAX"
    assert (workspace / "projects" / "demo" / "project.yaml").read_text() == snapshot
    assert snapshot != before

    r, p = invoke_json("project", "show", "demo")
    assert r.exit_code == 0
    assert p["result"]["config"]["dataset"]["classes"] == ["pet"]
    assert p["result"]["config"]["training"]["epochs"] == 7


def test_project_configure_source_and_image_size(workspace: Path):
    invoke_json("project", "create", "demo")
    r, p = invoke_json(
        "project", "configure", "demo", "--source", "./data/uploads/ds", "--image-size", "320"
    )
    assert r.exit_code == 0, p
    data = _load(workspace)
    assert data["dataset"]["sources"] == [{"type": "local_images", "path": "./data/uploads/ds"}]
    assert data["training"]["image_size"] == 320 and data["export"]["image_size"] == 320


# -- edge service control ----------------------------------------------------------------------


def _deployed(workspace: Path, tmp_path: Path, name: str = "demo") -> None:
    src = tmp_path / f"src-{name}"
    imgs = make_image_set(src, 12)
    write_yolo_labels(tmp_path / f"labels-{name}", [p.stem for p in imgs])
    r, p = invoke_json("project", "create", name, "--min-images", "4", "--min-instances", "1")
    assert r.exit_code == 0, p
    cfg = workspace / "projects" / name / "project.yaml"
    cfg.write_text(
        cfg.read_text()
        .replace("provider: ultralytics", "provider: fake")
        .replace("timeout_s: 120", "timeout_s: 0.5")
        .replace("interval_s: 5", "interval_s: 0.1")
    )
    invoke_json("data", "import", name, str(src))
    invoke_json("label", "import", name, str(tmp_path / f"labels-{name}"))
    invoke_json("dataset", "build", name)
    r, p = invoke_json("train", "start", name, "--wait")
    assert r.exit_code == 0, p
    r, p = invoke_json("evaluate", "run", name, "mv-001", "--wait")
    assert r.exit_code == 0 and p["result"]["model_state"] == "ACCEPTED"
    invoke_json("model", "select", name)
    r, p = invoke_json("export", "start", name, "--wait")
    assert r.exit_code == 0, p
    r, p = invoke_json("deploy", "start", name)
    assert r.exit_code == 0, p


def _counts(name: str) -> tuple[int, int]:
    _, runs = invoke_json("run", "list", name)
    _, exps = invoke_json("export", "list", name)
    return len(runs["result"]["runs"]), len(exps["result"]["exports"])


def test_service_stop_start_restart_cycle(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path)
    counts = _counts("demo")

    r, p = invoke_json("target", "status", "demo")
    assert r.exit_code == 0 and p["operation"] == "target.status"
    dev = p["result"]
    assert dev["target"] == "fake" and dev["reachable"] is True
    assert dev["current"] == "demo-dep-001" and dev["project"] == "demo"
    assert dev["deployment_id"] == "dep-001" and dev["model_version"] == "mv-001"
    assert dev["service"]["state"] == "running" and dev["consistent"] is True
    assert p["result"]["this_project_is_current"] is True

    # stop: intentional, keeps the deployment installed and the pointer untouched
    r, p = invoke_json("service", "stop", "demo")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["action"] == "stop" and res["changed"] is True
    assert res["service"]["state"] == "stopped" and res["deployment_id"] == "dep-001"
    assert res["stopped"]["deployment"] == "demo-dep-001"
    assert "service start demo" in p["next_operations"]

    r, p = invoke_json("health", "demo")
    assert r.exit_code == 1 and p["error"]["code"] == "SERVICE_STOPPED"
    assert "service start demo" in p["error"]["hint"]
    r, p = invoke_json("target", "status", "demo")
    dev = p["result"]
    assert dev["service"]["state"] == "stopped" and dev["current"] == "demo-dep-001"
    assert dev["service"]["stopped"]["deployment"] == "demo-dep-001"
    assert dev["healthy"] is False
    r, p = invoke_json("deploy", "status", "demo")
    assert p["result"]["state"] == "ACTIVE"  # not undeployed

    # idempotent stop
    r, p = invoke_json("service", "stop", "demo")
    assert r.exit_code == 0 and p["result"]["changed"] is False

    # start resumes the same deployment
    r, p = invoke_json("service", "start", "demo")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["changed"] is True and res["service"]["state"] == "running"
    assert res["health"]["healthy"] is True and res["current_after"] == "demo-dep-001"
    assert res["switched_from"] is None
    r, p = invoke_json("health", "demo")
    assert r.exit_code == 0 and p["result"]["healthy"] and p["result"]["consistent"]
    r, p = invoke_json("service", "start", "demo")
    assert r.exit_code == 0 and p["result"]["changed"] is False

    # restart keeps deployment id and model version
    r, p = invoke_json("service", "restart", "demo")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["deployment_id"] == "dep-001" and res["model_version"] == "mv-001"
    assert res["current_after"] == "demo-dep-001" and res["service"]["state"] == "running"

    # no training/export runs were created by service control
    assert _counts("demo") == counts
    r, p = invoke_json("deploy", "status", "demo", "dep-001", "--full")
    events = p["result"]["record"]["service_events"]
    assert [e["action"] for e in events] == ["stop", "stop", "start", "start", "restart"]


def test_service_start_refuses_to_replace_other_project(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path, "alpha")
    _deployed(workspace, tmp_path, "beta")  # device now runs beta-dep-001

    r, p = invoke_json("target", "status")
    assert r.exit_code == 0
    fake = next(t for t in p["result"]["targets"] if t["target"] == "fake")
    assert fake["current"] == "beta-dep-001" and fake["project"] == "beta"

    # alpha's own pointer still says ACTIVE, but the device state wins
    r, p = invoke_json("target", "status", "alpha")
    assert p["result"]["this_project_is_current"] is False
    assert p["result"]["project"] == "beta"
    assert p["result"]["project_pointers"]["active_deployment"] == "dep-001"

    r, p = invoke_json("service", "stop", "alpha")
    assert r.exit_code == 3 and p["error"]["code"] == "DEVICE_OCCUPIED"
    r, p = invoke_json("service", "start", "alpha")
    assert r.exit_code == 3 and p["error"]["code"] == "DEVICE_OCCUPIED"
    assert p["error"]["details"]["current_project"] == "beta"
    assert "--switch" in p["error"]["hint"]
    r, p = invoke_json("service", "restart", "alpha")
    assert r.exit_code == 3 and p["error"]["code"] == "DEVICE_OCCUPIED"
    # beta is untouched
    r, p = invoke_json("health", "beta")
    assert r.exit_code == 0 and p["result"]["healthy"]

    # explicit switch re-points the device without rebuilding anything
    counts = _counts("alpha")
    r, p = invoke_json("service", "start", "alpha", "--switch")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["switched_from"] == "beta-dep-001" and res["current_after"] == "alpha-dep-001"
    assert res["health"]["healthy"] is True
    assert _counts("alpha") == counts
    r, p = invoke_json("health", "alpha")
    assert r.exit_code == 0 and p["result"]["consistent"] is True
    r, p = invoke_json("health", "beta")
    assert r.exit_code == 1 and p["error"]["code"] == "POINTER_MISMATCH"


def test_service_requires_deployment_and_handles_unreachable(workspace: Path, tmp_path: Path):
    invoke_json("project", "create", "fresh")
    r, p = invoke_json("service", "stop", "fresh")
    assert r.exit_code == 3 and p["error"]["code"] == "NO_ACTIVE_DEPLOYMENT"

    r, p = invoke_json("target", "status", "fresh", "--target", "ng4511-lab")
    assert r.exit_code == 0  # status is informational even for unreachable devices
    dev = p["result"]
    assert dev["reachable"] is False and dev["service"]["state"] == "unreachable"
    assert dev["error"]

    _deployed(workspace, tmp_path)
    (workspace / ".fake-target" / "inject.json").write_text(json.dumps({"fail": "service_stop"}))
    r, p = invoke_json("service", "stop", "demo")
    assert r.exit_code == 1 and p["error"]["code"] == "TARGET_SERVICE_STOP_FAILED"
    (workspace / ".fake-target" / "inject.json").unlink()
    r, p = invoke_json("health", "demo")
    assert r.exit_code == 0 and p["result"]["healthy"]
