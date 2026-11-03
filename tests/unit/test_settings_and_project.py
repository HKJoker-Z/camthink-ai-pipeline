from __future__ import annotations

from pathlib import Path

import pytest

from ai_pipeline.core.errors import AlreadyExistsError, ConfigError, NotFoundError
from ai_pipeline.core.project import Project
from ai_pipeline.core.settings import Settings, find_infrastructure_file
from ai_pipeline.core.status import next_operations, project_status

# --- settings -----------------------------------------------------------------------------------


def test_settings_resolve_from_infra_file_dir(workspace: Path):
    sub = workspace / "a" / "b"
    sub.mkdir(parents=True)
    s = Settings.resolve(cwd=sub)
    assert s.infra_path == workspace / "infrastructure.yaml"
    assert s.projects_root == workspace / "projects"
    assert s.artifact_root == workspace / "artifacts"


def test_settings_precedence(workspace: Path, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    env = {"AIP_ARTIFACT_ROOT": str(other / "env-art"), "AIP_PROJECTS_ROOT": "env-proj"}
    s = Settings.resolve(cwd=workspace, env=env, artifact_root=Path("cli-art"))
    assert s.artifact_root == workspace / "cli-art"  # CLI beats env
    assert s.projects_root == workspace / "env-proj"  # env beats infra


def test_settings_missing_infra(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        find_infrastructure_file(tmp_path) is None
        or tmp_path not in find_infrastructure_file(tmp_path).parents
    )
    with pytest.raises(ConfigError) as ei:
        Settings.resolve(cwd=tmp_path, env={})
    assert ei.value.code in ("INFRA_FILE_MISSING", "CONFIG_FILE_MISSING")


# --- project ------------------------------------------------------------------------------------


def test_create_load_list(settings: Settings):
    assert Project.list_names(settings) == []
    p = Project.create("demo_a", settings)
    assert p.config_path.is_file()
    assert (p.dir / "README.md").is_file()
    assert (p.data_dir / "raw").is_dir()
    assert p.store.project_exists()
    assert "{{" not in p.config_path.read_text()

    loaded = Project.load("demo_a", settings)
    assert loaded.config.name == "demo_a"
    assert loaded.config.project.task_type == "object_detection"
    assert Project.list_names(settings) == ["demo_a"]
    assert loaded.pointers() == {
        "current_dataset_version": None,
        "selected_model_version": None,
        "active_deployment": None,
        "last_known_good_deployment": None,
    }


def test_create_is_not_destructive(settings: Settings):
    Project.create("demo_a", settings)
    with pytest.raises(AlreadyExistsError) as ei:
        Project.create("demo_a", settings)
    assert ei.value.code == "PROJECT_EXISTS"


def test_create_refuses_stale_artifact_record(settings: Settings):
    p = Project("ghost", settings)
    p.store.write_project({"name": "ghost"})
    with pytest.raises(AlreadyExistsError) as ei:
        Project.create("ghost", settings)
    assert ei.value.code == "PROJECT_RECORD_EXISTS"
    assert not p.config_path.exists()


def test_create_validates_name_and_task(settings: Settings):
    with pytest.raises(ConfigError) as ei:
        Project.create("Demo", settings)
    assert ei.value.code == "PROJECT_NAME_INVALID"
    with pytest.raises(ConfigError) as ei:
        Project.create("demo", settings, task_type="pose")
    assert ei.value.code == "TASK_TYPE_UNKNOWN"
    with pytest.raises(ConfigError) as ei:
        Project.create("demo", settings, template="nope")
    assert ei.value.code == "TEMPLATE_UNKNOWN"
    assert Project.list_names(settings) == []


def test_load_missing_and_name_mismatch(settings: Settings):
    with pytest.raises(NotFoundError) as ei:
        Project.load("missing", settings)
    assert ei.value.code == "PROJECT_NOT_FOUND"

    p = Project.create("demo_b", settings)
    text = p.config_path.read_text().replace("name: demo_b", "name: other")
    p.config_path.write_text(text)
    with pytest.raises(ConfigError) as ei:
        Project.load("demo_b", settings)
    assert ei.value.code == "PROJECT_NAME_MISMATCH"


def test_load_checks_infra_references(settings: Settings):
    p = Project.create("demo_c", settings)
    text = p.config_path.read_text().replace("target: fake", "target: moon-base")
    p.config_path.write_text(text)
    with pytest.raises(ConfigError) as ei:
        Project.load("demo_c", settings)
    assert ei.value.code == "CONFIG_INFRA_MISMATCH"


def test_load_initialises_missing_record_for_committed_config(settings: Settings):
    """A git checkout has project.yaml but no artifacts/: load must self-heal, once."""
    p = Project.create("demo_g", settings)
    p.store.project_record_path.unlink()
    loaded = Project.load("demo_g", settings)
    assert loaded.store.project_exists()
    rec = loaded.record()
    assert rec["task_type"] == "object_detection" and rec["template"] is None
    loaded.set_pointer("selected_model_version", "mv-001")
    Project.load("demo_g", settings)  # second load must not reset the record
    assert loaded.pointers()["selected_model_version"] == "mv-001"


def test_pointers_roundtrip(settings: Settings):
    p = Project.create("demo_d", settings)
    p.set_pointer("selected_model_version", "mv-001")
    assert p.pointers()["selected_model_version"] == "mv-001"
    with pytest.raises(ValueError):
        p.set_pointer("favourite_colour", "blue")


# --- status --------------------------------------------------------------------------------------


def test_project_status_fresh(settings: Settings):
    p = Project.create("demo_e", settings)
    s = project_status(p)
    assert s["project"] == "demo_e"
    assert s["counts"] == {
        k: 0 for k in ("imports", "labels", "datasets", "runs", "models", "deployments", "assets")
    }
    assert s["latest_run"] is None
    assert s["next_operations"] == ["data import demo_e <path>"]


def test_project_status_reflects_runs(settings: Settings):
    p = Project.create("demo_f", settings)
    p.store.create("imports", "imp-1", {})
    p.store.create("labels", "lbl-1", {})
    p.store.create("datasets", "ds-v001", {})
    p.store.create(
        "runs",
        "run-1",
        {"run_id": "run-1", "kind": "train", "state": "RUNNING", "created_at": "2026"},
    )
    s = project_status(p)
    assert s["active_runs"] == ["run-1"]
    assert s["latest_run"]["run_id"] == "run-1"
    assert s["next_operations"] == ["run status demo_f run-1"]


def test_next_operations_progression():
    zero = {k: 0 for k in ("imports", "labels", "datasets", "runs", "models", "deployments")}
    ptr = {
        "current_dataset_version": "ds-v001",
        "selected_model_version": None,
        "active_deployment": None,
    }
    assert next_operations("p", zero, ptr, []) == ["data import p <path>"]
    assert next_operations("p", {**zero, "imports": 1}, ptr, []) == [
        "label import p <path>",
        "label auto p",
    ]
    no_ds = {**ptr, "current_dataset_version": None}
    assert (
        next_operations("p", {**zero, "imports": 1, "labels": 1}, no_ds, [])[0]
        == "dataset validate p"
    )
    with_ds = {**zero, "imports": 1, "labels": 1}
    assert next_operations("p", with_ds, ptr, []) == ["train start p"]
    assert next_operations("p", with_ds, ptr, [], unregistered_train_runs=["run-1"]) == [
        "model register p run-1"
    ]
    full = {**with_ds, "runs": 1, "models": 1}
    assert next_operations("p", full, ptr, [], model_states={"mv-001": "TRAINED"}) == [
        "evaluate run p mv-001"
    ]
    assert next_operations("p", full, ptr, [], model_states={"mv-001": "REJECTED"}) == [
        "train start p",
        "model list p",
    ]
    assert next_operations("p", full, ptr, [], model_states={"mv-001": "ACCEPTED"}) == [
        "model select p"
    ]
    selected = {**ptr, "selected_model_version": "mv-001"}
    assert next_operations("p", full, selected, [], model_states={"mv-001": "ACCEPTED"}) == [
        "model show p mv-001",
        "train start p",
    ]
    assert next_operations("p", full, selected, ["run-9"]) == ["run status p run-9"]
