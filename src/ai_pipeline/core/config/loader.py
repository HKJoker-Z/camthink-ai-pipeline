"""Load and validate configuration files, turning pydantic errors into ConfigError."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ai_pipeline.core.config.env import substitute_env
from ai_pipeline.core.config.schema import InfrastructureConfig, ProjectConfig
from ai_pipeline.core.errors import ConfigError
from ai_pipeline.providers import provider_names
from ai_pipeline.tasks import get_task, task_types


def load_yaml_with_env(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(
            f"Configuration file not found: {path}",
            code="CONFIG_FILE_MISSING",
            details={"path": str(path)},
            hint="Check the path, --infra / AIP_INFRA_FILE, or create the project first.",
        )
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML in {path}: {exc}",
            code="CONFIG_YAML_INVALID",
            details={"path": str(path)},
        ) from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top level of {path} must be a mapping",
            code="CONFIG_YAML_INVALID",
            details={"path": str(path)},
        )
    return substitute_env(raw)


def format_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """Flatten a pydantic error into ``[{path, message, type}]``."""
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append({"path": loc, "message": err["msg"], "type": err["type"]})
    return out


def _raise_config_error(path: Path, exc: ValidationError, what: str) -> None:
    errors = format_validation_error(exc)
    first = errors[0]
    summary = f"{what} {path} is invalid: {first['path']}: {first['message']}"
    if len(errors) > 1:
        summary += f" (+{len(errors) - 1} more)"
    raise ConfigError(
        summary,
        code="CONFIG_INVALID",
        details={"path": str(path), "errors": errors},
        hint="See docs/configuration.md for the schema. Unknown keys are rejected on purpose.",
    ) from exc


def load_infrastructure(path: Path) -> InfrastructureConfig:
    data = load_yaml_with_env(path)
    try:
        return InfrastructureConfig.model_validate(data)
    except ValidationError as exc:
        _raise_config_error(path, exc, "Infrastructure file")
        raise  # unreachable; keeps type checkers happy


def load_project_config(path: Path) -> ProjectConfig:
    return validate_project_data(load_yaml_with_env(path), path)


def validate_project_data(data: dict[str, Any], path: Path) -> ProjectConfig:
    """Validate an in-memory project mapping (already env-substituted); ``path`` is for
    error messages only. Used before writing a configuration back to disk."""
    try:
        cfg = ProjectConfig.model_validate(data)
    except ValidationError as exc:
        _raise_config_error(path, exc, "Project file")
        raise
    _validate_project_semantics(cfg, path)
    return cfg


def _validate_project_semantics(cfg: ProjectConfig, path: Path) -> None:
    """Cross-field checks that need registries (task modules, providers)."""
    problems: list[dict[str, Any]] = []

    try:
        task = get_task(cfg.project.task_type)
    except KeyError:
        problems.append(
            {
                "path": "project.task_type",
                "message": f"unknown task type {cfg.project.task_type!r}; known: {task_types()}",
            }
        )
        task = None

    if task is not None:
        if cfg.dataset.label_format not in task.label_formats:
            problems.append(
                {
                    "path": "dataset.label_format",
                    "message": f"{cfg.dataset.label_format!r} not supported by "
                    f"{task.name}; allowed: {list(task.label_formats)}",
                }
            )
        unknown_metrics = sorted(set(cfg.evaluation.metrics) - set(task.metric_names))
        if unknown_metrics:
            problems.append(
                {
                    "path": "evaluation.metrics",
                    "message": f"unknown metrics for {task.name}: {unknown_metrics}; "
                    f"known: {list(task.metric_names)}",
                }
            )

    known_providers = provider_names()
    if cfg.model.provider not in known_providers:
        problems.append(
            {
                "path": "model.provider",
                "message": f"unknown provider {cfg.model.provider!r}; known: {known_providers}",
            }
        )
    if cfg.labeling.auto and cfg.labeling.auto.provider not in known_providers:
        problems.append(
            {
                "path": "labeling.auto.provider",
                "message": f"unknown provider {cfg.labeling.auto.provider!r}; "
                f"known: {known_providers}",
            }
        )

    if problems:
        first = problems[0]
        raise ConfigError(
            f"Project file {path} is invalid: {first['path']}: {first['message']}"
            + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
            code="CONFIG_INVALID",
            details={"path": str(path), "errors": problems},
            hint="See docs/configuration.md.",
        )


def validate_project_against_infra(cfg: ProjectConfig, infra: InfrastructureConfig) -> None:
    """Every backend/target name referenced by the project must exist in the infrastructure."""
    problems: list[dict[str, Any]] = []
    backends = sorted(infra.compute_backends)
    targets = sorted(infra.deployment_targets)

    if cfg.training.backend not in infra.compute_backends:
        problems.append(
            {
                "path": "training.backend",
                "message": f"compute backend {cfg.training.backend!r} not defined in "
                f"infrastructure; defined: {backends}",
            }
        )
    if cfg.labeling.auto and cfg.labeling.auto.backend not in infra.compute_backends:
        problems.append(
            {
                "path": "labeling.auto.backend",
                "message": f"compute backend {cfg.labeling.auto.backend!r} not defined in "
                f"infrastructure; defined: {backends}",
            }
        )
    if cfg.deployment.target not in infra.deployment_targets:
        problems.append(
            {
                "path": "deployment.target",
                "message": f"deployment target {cfg.deployment.target!r} not defined in "
                f"infrastructure; defined: {targets}",
            }
        )
    if problems:
        first = problems[0]
        raise ConfigError(
            f"Project {cfg.name!r} references unknown infrastructure: "
            f"{first['path']}: {first['message']}"
            + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
            code="CONFIG_INFRA_MISMATCH",
            details={"errors": problems},
            hint="Add the missing entry to infrastructure.yaml or change the project reference.",
        )
