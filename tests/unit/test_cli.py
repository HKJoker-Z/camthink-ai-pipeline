from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_pipeline import __version__
from ai_pipeline.cli.main import app
from ai_pipeline.cli.output import Envelope

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def invoke_json(*args: str):
    result = invoke("-o", "json", *args)
    payload = json.loads(result.stdout)
    return result, payload


def test_version():
    r = invoke("--version")
    assert r.exit_code == 0
    assert r.stdout.strip() == __version__


def test_project_create_validate_status_human(workspace: Path):
    r = invoke("project", "create", "demo")
    assert r.exit_code == 0, r.output
    assert "Created project 'demo'" in r.stdout
    assert (workspace / "projects" / "demo" / "project.yaml").is_file()
    assert (workspace / "artifacts" / "demo" / "project.json").is_file()

    r = invoke("project", "validate", "demo")
    assert r.exit_code == 0 and "valid" in r.stdout

    r = invoke("status", "demo")
    assert r.exit_code == 0
    assert "counts: imports=0" in r.stdout
    assert "next: ai-pipeline data import demo <path>" in r.stdout

    r = invoke("project", "list")
    assert r.exit_code == 0 and "demo" in r.stdout and "valid" in r.stdout


def test_json_envelope_shape_and_stdout_purity(workspace: Path):
    r, payload = invoke_json("project", "create", "demo", "--task", "object_detection")
    assert r.exit_code == 0
    assert set(payload) == {
        "schema_version",
        "project",
        "operation",
        "status",
        "result",
        "error",
        "next_operations",
        "timestamp",
    }
    assert payload["schema_version"] == 1
    assert payload["operation"] == "project.create"
    assert payload["status"] == "ok"
    assert payload["result"]["created"] is True
    assert payload["next_operations"][0] == "project validate demo"

    r, payload = invoke_json("status", "demo")
    assert payload["result"]["counts"]["runs"] == 0
    assert payload["result"]["pointers"]["active_deployment"] is None


def test_exit_codes_and_error_envelope(workspace: Path):
    # 3: state error (not found)
    r, payload = invoke_json("status", "nothing_here")
    assert r.exit_code == 3
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "PROJECT_NOT_FOUND"
    assert payload["error"]["hint"]

    # 3: already exists, nothing destroyed
    invoke("project", "create", "demo")
    before = (workspace / "projects" / "demo" / "project.yaml").read_text()
    r, payload = invoke_json("project", "create", "demo")
    assert r.exit_code == 3 and payload["error"]["code"] == "PROJECT_EXISTS"
    assert (workspace / "projects" / "demo" / "project.yaml").read_text() == before

    # 2: config error (bad name / invalid yaml content)
    r, payload = invoke_json("project", "create", "Bad Name")
    assert r.exit_code == 2 and payload["error"]["code"] == "PROJECT_NAME_INVALID"

    cfg = workspace / "projects" / "demo" / "project.yaml"
    cfg.write_text(cfg.read_text().replace("epochs: 5", "epochs: 0"))
    r, payload = invoke_json("project", "validate", "demo")
    assert r.exit_code == 2
    assert payload["error"]["code"] == "CONFIG_INVALID"
    assert payload["error"]["details"]["errors"][0]["path"] == "training.epochs"

    r, payload = invoke_json("project", "list")
    assert r.exit_code == 0
    assert payload["result"]["projects"][0]["valid"] is False


def test_missing_infrastructure_is_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AIP_INFRA_FILE", raising=False)
    r, payload = invoke_json("project", "list")
    assert r.exit_code == 2
    assert payload["error"]["code"] == "INFRA_FILE_MISSING"


def test_explicit_infra_and_roots(workspace: Path, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    r, payload = invoke_json(
        "--infra",
        str(workspace / "infrastructure.yaml"),
        "--projects-root",
        str(elsewhere / "p"),
        "--artifact-root",
        str(elsewhere / "a"),
        "project",
        "create",
        "demo",
    )
    assert r.exit_code == 0, r.output
    assert payload["result"]["config_path"].startswith(str(elsewhere / "p"))
    assert (elsewhere / "a" / "demo" / "project.json").is_file()


def test_bad_output_mode():
    r = invoke("-o", "xml", "project", "list")
    assert r.exit_code == 2


def test_envelope_human_rendering():
    env = Envelope(
        operation="x",
        project="p",
        status="failed",
        error={"code": "E", "message": "boom", "details": {"path": "/x"}, "hint": "try y"},
        next_operations=["status p"],
    )
    text = env.to_human()
    assert "x: FAILED" in text and "error [E]: boom" in text
    assert "path: /x" in text and "hint: try y" in text
    assert "next: ai-pipeline status p" in text
    assert json.loads(env.to_json())["error"]["code"] == "E"
