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


def _prepare(workspace: Path, tmp_path: Path) -> None:
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
    r, p = invoke_json("dataset", "build", "demo")
    assert r.exit_code == 0, p


def test_full_m2_flow_json(workspace: Path, tmp_path: Path):
    _prepare(workspace, tmp_path)

    r, p = invoke_json("status", "demo")
    assert p["next_operations"] == ["train start demo"]

    r, p = invoke_json("train", "start", "demo", "--wait", "--timeout", "60")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["kind"] == "train" and res["state"] == "COMPLETED"
    assert res["model_version"] == "mv-001" and res["metrics"]["mAP50"] == 0.75
    assert res["inputs"]["dataset_version"] == "ds-v001"
    assert "/" not in json.dumps(res["inputs"])  # no host paths leak into the envelope inputs
    run_id = res["run_id"]
    assert p["next_operations"] == ["evaluate run demo mv-001"]

    r, p = invoke_json("run", "status", "demo", run_id, "--tail", "2")
    assert r.exit_code == 0 and "completed" in p["result"]["log_tail"]

    r, p = invoke_json("train", "list", "demo")
    assert [x["run_id"] for x in p["result"]["runs"]] == [run_id]

    r, p = invoke_json("model", "list", "demo")
    assert p["result"]["models"][0]["state"] == "TRAINED"

    r, p = invoke_json("model", "register", "demo", run_id)
    assert r.exit_code == 0 and p["result"]["created"] is False

    r, p = invoke_json("evaluate", "run", "demo", "mv-001", "--wait", "--timeout", "60")
    assert r.exit_code == 0, p
    assert p["result"]["model_state"] == "ACCEPTED"
    assert p["result"]["acceptance"]["passed"] is True
    assert p["next_operations"] == ["model select demo"]

    r, p = invoke_json("model", "select", "demo")
    assert r.exit_code == 0 and p["result"]["selected_model_version"] == "mv-001"
    assert p["result"]["primary_metric"] == "mAP50-95" and p["result"]["value"] == 0.525

    r, p = invoke_json("model", "show", "demo", "mv-001")
    m = p["result"]
    assert m["training_run_id"] == run_id and m["dataset_version"] == "ds-v001"
    assert m["artifacts"]["weights"]["sha256"] and m["evaluation"]["run_id"]
    assert [h["state"] for h in m["history"]] == ["TRAINED", "EVALUATED", "ACCEPTED"]

    r, p = invoke_json("status", "demo")
    assert p["result"]["pointers"]["selected_model_version"] == "mv-001"
    assert p["result"]["counts"]["runs"] == 2 and p["result"]["counts"]["models"] == 1
    assert p["result"]["model_states"] == {"mv-001": "ACCEPTED"}

    r, p = invoke_json("run", "list", "demo", "--kind", "evaluate")
    assert len(p["result"]["runs"]) == 1 and p["result"]["runs"][0]["kind"] == "evaluate"


def test_label_auto_json(workspace: Path, tmp_path: Path):
    _prepare(workspace, tmp_path)
    cfg = workspace / "projects" / "demo" / "project.yaml"
    cfg.write_text(
        cfg.read_text().replace(
            "    provider: ultralytics\n    weights:", "    provider: fake\n    weights:"
        )
    )
    r, p = invoke_json("label", "auto", "demo", "--wait", "--confidence", "0.5")
    assert r.exit_code == 0, p
    assert p["result"]["kind"] == "autolabel" and p["result"]["state"] == "COMPLETED"
    ls = p["result"]["label_set"]
    assert (
        ls["source"]["kind"] == "model"
        and ls["counts"]["annotations"] == p["result"]["metrics"]["detections"]
    )
    r, p = invoke_json("label", "list", "demo")
    assert [x["source_kind"] for x in p["result"]["label_sets"]] == ["imported", "model"]
    assert p["result"]["label_sets"][1]["run_id"]
    r, p = invoke_json("dataset", "list", "demo")
    assert len(p["result"]["datasets"]) == 1  # autolabel did not build anything


def test_failed_run_exit_code_and_cancel(workspace: Path, tmp_path: Path):
    _prepare(workspace, tmp_path)
    cfg = workspace / "projects" / "demo" / "project.yaml"
    cfg.write_text(
        cfg.read_text().replace("provider_args: {}", 'provider_args: {"fake_fail": "train"}')
    )
    r, p = invoke_json("train", "start", "demo", "--wait", "--timeout", "60")
    assert r.exit_code == 1
    assert p["status"] == "failed" and p["error"]["code"] == "RUN_FAILED"
    assert "asked to fail" in p["error"]["message"]
    assert p["result"]["state"] == "FAILED" and p["result"]["model_version"] is None
    r, p = invoke_json("run", "logs", "demo", p["result"]["run_id"], "--tail", "30")
    assert r.exit_code == 0 and "asked to fail" in p["result"]["logs"]

    cfg.write_text(cfg.read_text().replace('{"fake_fail": "train"}', '{"fake_sleep_s": 30}'))
    r, p = invoke_json("train", "start", "demo")
    assert r.exit_code == 0 and p["result"]["state"] == "SUBMITTED"
    run_id = p["result"]["run_id"]
    r, p = invoke_json("status", "demo")
    assert p["next_operations"] == [f"run status demo {run_id}"]
    r, p = invoke_json("run", "cancel", "demo", run_id)
    assert r.exit_code == 0 and p["result"]["state"] == "CANCELLED"
    r, p = invoke_json("run", "status", "demo", run_id)
    assert r.exit_code == 1 and p["error"]["code"] == "RUN_CANCELLED"


def test_state_errors(workspace: Path, tmp_path: Path):
    invoke_json("project", "create", "demo")
    r, p = invoke_json("train", "start", "demo")
    assert r.exit_code == 3 and p["error"]["code"] == "NO_DATASET"
    r, p = invoke_json("model", "select", "demo")
    assert r.exit_code == 3 and p["error"]["code"] == "NO_ACCEPTED_MODEL"
    r, p = invoke_json("evaluate", "run", "demo", "mv-001")
    assert r.exit_code == 3 and p["error"]["code"] == "MODEL_NOT_FOUND"
    r, p = invoke_json("run", "status", "demo", "run-nope")
    assert r.exit_code == 3 and p["error"]["code"] == "RUN_NOT_FOUND"


def test_human_output(workspace: Path, tmp_path: Path):
    _prepare(workspace, tmp_path)
    r = runner.invoke(app, ["train", "start", "demo", "--wait"])
    assert (
        r.exit_code == 0 and "state=COMPLETED" in r.stdout and "model_version: mv-001" in r.stdout
    )
    r = runner.invoke(app, ["evaluate", "run", "demo", "mv-001", "--wait"])
    assert "model mv-001: ACCEPTED" in r.stdout and "PASS mAP50=0.75" in r.stdout
    r = runner.invoke(app, ["model", "list", "demo"])
    assert "mv-001  ACCEPTED" in r.stdout
    r = runner.invoke(app, ["model", "select", "demo", "mv-001"])
    assert "selected mv-001" in r.stdout
    r = runner.invoke(app, ["run", "list", "demo"])
    assert "train     COMPLETED" in r.stdout and "evaluate  COMPLETED" in r.stdout
