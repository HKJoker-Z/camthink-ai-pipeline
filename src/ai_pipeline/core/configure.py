"""Validated edits of ``project.yaml`` without a text editor.

An operator (or an agent over a restricted SSH command) names what to change; the change is
applied to the YAML mapping, validated with the same schema and cross-checks as loading, and
only then written atomically. A rejected change leaves the file untouched.

Two layers:

- ``Change`` = (dotted path, value), the generic primitive (``--set training.epochs=30``);
- :func:`named_changes` maps the common, supported fields (classes, source, weights, epochs,
  target, …) onto those paths, including the derived edits that keep the file coherent
  (``labeling.auto.class_map`` follows ``dataset.classes`` unless given explicitly).
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ai_pipeline.core.config import (
    ProjectConfig,
    validate_project_against_infra,
    validate_project_data,
)
from ai_pipeline.core.config.env import substitute_env
from ai_pipeline.core.config.schema import InfrastructureConfig
from ai_pipeline.core.errors import ConfigError

Change = tuple[str, Any]

# Identity of a project is not a setting.
PROTECTED_PATHS = ("schema_version", "project.name", "project.task_type")

HEADER = (
    "# Project configuration for {name}. Schema: docs/configuration.md. Unknown keys are\n"
    "# rejected. Edited by `ai-pipeline project configure`; comments are not preserved.\n"
)


def parse_set_option(item: str) -> Change:
    """``key.path=value`` -> (path, value); the value is parsed as YAML (``30``, ``0.5``,
    ``true``, ``[a, b]``, ``{min: 0.3}``). A value that YAML cannot parse stays a string."""
    if "=" not in item:
        raise ConfigError(
            f"--set expects key.path=value, got {item!r}",
            code="SET_SYNTAX",
            details={"item": item},
            hint="Example: --set training.epochs=30 --set dataset.classes='[cat, dog]'",
        )
    path, raw = item.split("=", 1)
    path = path.strip()
    if not path:
        raise ConfigError(f"--set has an empty key in {item!r}", code="SET_SYNTAX")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    if value is None and raw.strip() not in ("", "null", "~"):
        value = raw
    return path, value


def parse_csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_map(text: str) -> dict[str, str]:
    """``a:b,c:d`` -> {a: b, c: d}; a bare ``a`` means ``a:a``."""
    out: dict[str, str] = {}
    for item in parse_csv(text):
        src, _, dst = item.partition(":")
        out[src.strip()] = (dst or src).strip()
    return out


def named_changes(
    *,
    description: str | None = None,
    classes: str | None = None,
    source: str | None = None,
    weights: str | None = None,
    auto_label_weights: str | None = None,
    auto_label_map: str | None = None,
    auto_label_confidence: float | None = None,
    epochs: int | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    min_images: int | None = None,
    min_instances: int | None = None,
    max_empty_ratio: float | None = None,
    min_map50: float | None = None,
    target: str | None = None,
    precision: str | None = None,
    confidence: float | None = None,
    sets: list[str] | None = None,
    has_auto_labeling: bool = True,
) -> list[Change]:
    """Translate the supported CLI options into dotted-path changes (in application order)."""
    out: list[Change] = []
    if description is not None:
        out.append(("project.description", description))
    if classes is not None:
        names = parse_csv(classes)
        out.append(("dataset.classes", names))
        if auto_label_map is None and has_auto_labeling:
            # A stale class_map would silently label the wrong classes; default to identity
            # (pretrained COCO names == project names), overridable with --auto-label-map.
            out.append(("labeling.auto.class_map", {c: c for c in names}))
    if source is not None:
        out.append(("dataset.sources", [{"type": "local_images", "path": source}]))
    if weights is not None:
        name = Path(weights).stem.replace(".", "_")
        out.append(("model.candidates", [{"name": name, "weights": weights}]))
    if auto_label_weights is not None:
        out.append(("labeling.auto.weights", auto_label_weights))
    if auto_label_map is not None:
        out.append(("labeling.auto.class_map", parse_map(auto_label_map)))
    if auto_label_confidence is not None:
        out.append(("labeling.auto.confidence", auto_label_confidence))
    if epochs is not None:
        out.append(("training.epochs", epochs))
    if image_size is not None:
        out.append(("training.image_size", image_size))
        out.append(("export.image_size", image_size))
    if batch_size is not None:
        out.append(("training.batch_size", batch_size))
    if device is not None:
        out.append(("training.device", device))
    if min_images is not None:
        out.append(("dataset.validation.min_images", min_images))
    if min_instances is not None:
        out.append(("dataset.validation.min_instances_per_class", min_instances))
    if max_empty_ratio is not None:
        out.append(("dataset.validation.max_empty_label_ratio", max_empty_ratio))
    if min_map50 is not None:
        out.append(("evaluation.acceptance.mAP50", {"min": min_map50}))
    if target is not None:
        out.append(("deployment.target", target))
    if precision is not None:
        out.append(("deployment.precision", precision))
    if confidence is not None:
        out.append(("deployment.inference.confidence", confidence))
    for item in sets or []:
        out.append(parse_set_option(item))
    return out


def _get(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return copy.deepcopy(node)


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node: Any = data
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            if key in node and node[key] is not None:
                raise ConfigError(
                    f"cannot set {path}: {key!r} is not a mapping",
                    code="SET_PATH_INVALID",
                    details={"path": path},
                )
            node[key] = {}
        node = node[key]
    old = node.get(keys[-1])
    # YAML parses `0` as int; keep the field's existing scalar type for string settings
    # (training.device: "0") so --set behaves like editing the file.
    if isinstance(old, str) and value is not None and not isinstance(value, (str, dict, list)):
        value = str(value)
    node[keys[-1]] = value


def apply_changes(
    data: dict[str, Any],
    changes: list[Change],
    *,
    path: Path,
    infra: InfrastructureConfig,
) -> tuple[dict[str, Any], ProjectConfig, list[dict[str, Any]]]:
    """Return ``(new_data, validated_config, diff)``; raises ``ConfigError`` on any problem."""
    for p, _ in changes:
        if p in PROTECTED_PATHS or any(p.startswith(x + ".") for x in PROTECTED_PATHS):
            raise ConfigError(
                f"{p} identifies the project and cannot be changed",
                code="SET_PATH_PROTECTED",
                details={"path": p, "protected": list(PROTECTED_PATHS)},
                hint="Create a new project instead.",
            )
    new = copy.deepcopy(data)
    diff: list[dict[str, Any]] = []
    for p, value in changes:
        old = _get(new, p)
        _set(new, p, value)
        if old != _get(new, p):
            diff.append({"path": p, "old": old, "new": _get(new, p)})
    cfg = validate_project_data(substitute_env(copy.deepcopy(new)), path)
    validate_project_against_infra(cfg, infra)
    return new, cfg, diff


def dump_yaml(name: str, data: dict[str, Any]) -> str:
    return HEADER.format(name=name) + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def write_yaml_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".project.", suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def configure_file(
    path: Path, changes: list[Change], *, infra: InfrastructureConfig
) -> tuple[ProjectConfig, list[dict[str, Any]]]:
    """Apply ``changes`` to the project file at ``path`` (validate first, then write)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping", code="CONFIG_YAML_INVALID")
    new, cfg, diff = apply_changes(raw, changes, path=path, infra=infra)
    if diff:
        write_yaml_atomic(path, dump_yaml(cfg.project.name, new))
    return cfg, diff
