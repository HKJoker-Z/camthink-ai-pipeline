from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ai_pipeline.core.config import (
    InfrastructureConfig,
    ProjectConfig,
    load_infrastructure,
    load_project_config,
    validate_project_against_infra,
)
from ai_pipeline.core.config.env import substitute_env
from ai_pipeline.core.config.schema import Rule
from ai_pipeline.core.errors import ConfigError, ExitCode
from ai_pipeline.core.project import render_template

VALID_INFRA = yaml.safe_load(
    """
schema_version: 1
compute_backends:
  local: {type: local}
deployment_targets:
  fake: {type: fake}
"""
)


@pytest.fixture
def valid_project() -> dict:
    text = render_template("default", project_name="demo", task_type="object_detection")[
        "project.yaml"
    ]
    return yaml.safe_load(text)


def _write(tmp_path: Path, name: str, data) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data, encoding="utf-8")
    return p


def _errors(exc_info) -> list[str]:
    return [e["path"] for e in exc_info.value.details["errors"]]


# --- env substitution ---------------------------------------------------------------------


def test_env_substitution_variants():
    env = {"HOST": "ng4511", "EMPTY": ""}
    data = {
        "a": "${HOST}",
        "b": "x-${HOST}-y",
        "c": "${MISSING:-dflt}",
        "d": "${EMPTY:-z}",
        "n": [1, "${HOST}"],
    }
    out = substitute_env(data, env=env)
    assert out == {"a": "ng4511", "b": "x-ng4511-y", "c": "dflt", "d": "", "n": [1, "ng4511"]}


def test_env_substitution_missing_var_fails_with_path():
    with pytest.raises(ConfigError) as ei:
        substitute_env({"deployment_targets": {"x": {"url": "${NOPE}"}}}, env={})
    assert ei.value.code == "ENV_VAR_MISSING"
    assert ei.value.details == {"variable": "NOPE", "path": "deployment_targets.x.url"}
    assert "NOPE" in ei.value.hint


# --- Rule ---------------------------------------------------------------------------------------


def test_rule_bounds():
    assert Rule(min=0.3).check(0.3) and not Rule(min=0.3).check(0.29)
    assert Rule(max=60).check(60) and not Rule(max=60).check(61)
    with pytest.raises(ValueError):
        Rule()
    with pytest.raises(ValueError):
        Rule(min=2, max=1)


# --- project schema ---------------------------------------------------------------------------


def test_template_project_is_valid(valid_project):
    cfg = ProjectConfig.model_validate(valid_project)
    assert cfg.name == "demo"
    assert cfg.dataset.classes == ["person"]


@pytest.mark.parametrize(
    "mutate, expected_path",
    [
        (lambda d: d["training"].__setitem__("epoch", 3), "training.epoch"),
        (lambda d: d["project"].__setitem__("name", "Bad-Name"), "project.name"),
        (lambda d: d["dataset"].__setitem__("classes", ["a", "a"]), "dataset.classes"),
        (lambda d: d["dataset"].__setitem__("classes", []), "dataset.classes"),
        (lambda d: d["dataset"]["split"].__setitem__("test", 0.5), "dataset.split"),
        (lambda d: d["training"].__setitem__("epochs", 0), "training.epochs"),
        (
            lambda d: d["evaluation"]["selection"].__setitem__("primary_metric", "recall_x"),
            "evaluation",
        ),
        (lambda d: d["evaluation"].__setitem__("acceptance", {"f1": {"min": 0.1}}), "evaluation"),
        (
            lambda d: d["deployment"].__setitem__("acceptance", {"tops": {"min": 1}}),
            "deployment.acceptance",
        ),
        (lambda d: d["deployment"].__setitem__("precision", "int8"), "deployment.precision"),
        (
            lambda d: d["dataset"].__setitem__("sources", [{"type": "s3", "path": "x"}]),
            "dataset.sources",
        ),
        (lambda d: d.__setitem__("schema_version", 2), "schema_version"),
    ],
)
def test_project_schema_rejects_bad_values(tmp_path, valid_project, mutate, expected_path):
    data = copy.deepcopy(valid_project)
    mutate(data)
    path = _write(tmp_path, "project.yaml", data)
    with pytest.raises(ConfigError) as ei:
        load_project_config(path)
    assert ei.value.exit_code == ExitCode.CONFIG
    assert ei.value.code == "CONFIG_INVALID"
    assert any(p.startswith(expected_path) for p in _errors(ei)), _errors(ei)


def test_project_semantic_checks(tmp_path, valid_project):
    data = copy.deepcopy(valid_project)
    data["project"]["task_type"] = "segmentation"
    data["model"]["provider"] = "detectron"
    with pytest.raises(ConfigError) as ei:
        load_project_config(_write(tmp_path, "p.yaml", data))
    assert _errors(ei) == ["project.task_type", "model.provider"]

    data = copy.deepcopy(valid_project)
    data["dataset"]["label_format"] = "coco"
    data["evaluation"]["metrics"] = ["mAP50", "iou"]
    data["evaluation"]["selection"]["primary_metric"] = "mAP50"
    data["evaluation"]["acceptance"] = {}
    with pytest.raises(ConfigError) as ei:
        load_project_config(_write(tmp_path, "p.yaml", data))
    assert _errors(ei) == ["dataset.label_format", "evaluation.metrics"]


def test_project_infra_cross_validation(valid_project):
    cfg = ProjectConfig.model_validate(valid_project)
    infra = InfrastructureConfig.model_validate(VALID_INFRA)
    validate_project_against_infra(cfg, infra)  # template references local/fake

    data = copy.deepcopy(valid_project)
    data["training"]["backend"] = "gpu-farm"
    data["deployment"]["target"] = "ng-9000"
    with pytest.raises(ConfigError) as ei:
        validate_project_against_infra(ProjectConfig.model_validate(data), infra)
    assert ei.value.code == "CONFIG_INFRA_MISMATCH"
    assert _errors(ei) == ["training.backend", "deployment.target"]


# --- infrastructure ---------------------------------------------------------------------------


def test_infrastructure_discriminated_unions(tmp_path):
    infra = load_infrastructure(_write(tmp_path, "i.yaml", VALID_INFRA))
    assert infra.compute_backends["local"].type == "local"
    assert infra.projects_root == "./projects"

    bad = copy.deepcopy(VALID_INFRA)
    bad["compute_backends"]["x"] = {"type": "ssh"}  # missing ssh_host/workdir
    with pytest.raises(ConfigError) as ei:
        load_infrastructure(_write(tmp_path, "i.yaml", bad))
    assert all(p.startswith("compute_backends.x") for p in _errors(ei))

    bad = copy.deepcopy(VALID_INFRA)
    bad["deployment_targets"]["d"] = {"type": "jetson_xavier"}
    with pytest.raises(ConfigError):
        load_infrastructure(_write(tmp_path, "i.yaml", bad))


def test_yaml_errors_and_missing_file(tmp_path):
    with pytest.raises(ConfigError) as ei:
        load_infrastructure(tmp_path / "nothing.yaml")
    assert ei.value.code == "CONFIG_FILE_MISSING"
    with pytest.raises(ConfigError) as ei:
        load_infrastructure(_write(tmp_path, "bad.yaml", "a: [1, 2"))
    assert ei.value.code == "CONFIG_YAML_INVALID"
    with pytest.raises(ConfigError) as ei:
        load_infrastructure(_write(tmp_path, "list.yaml", "- a\n- b\n"))
    assert ei.value.code == "CONFIG_YAML_INVALID"
