"""Project abstraction (Layer C handle).

A project is ``projects/<name>/project.yaml`` plus gitignored ``data/``; its generated state
lives in ``artifacts/<name>/`` behind a :class:`~ai_pipeline.core.store.Store`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any

from ai_pipeline.core.config import (
    ProjectConfig,
    load_project_config,
    validate_project_against_infra,
)
from ai_pipeline.core.config.schema import PROJECT_NAME_RE
from ai_pipeline.core.errors import AlreadyExistsError, ConfigError, NotFoundError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.settings import Settings
from ai_pipeline.core.store import Store
from ai_pipeline.tasks import task_types

PROJECT_FILE = "project.yaml"
RECORD_SCHEMA_VERSION = 1

POINTER_KEYS = (
    "current_dataset_version",
    "selected_model_version",
    "active_deployment",
    "last_known_good_deployment",
)


def validate_project_name(name: str) -> str:
    if not PROJECT_NAME_RE.match(name):
        raise ConfigError(
            f"Invalid project name {name!r}",
            code="PROJECT_NAME_INVALID",
            details={"name": name, "pattern": PROJECT_NAME_RE.pattern},
            hint="Use lowercase letters, digits and underscores, starting with a letter.",
        )
    return name


@dataclass
class Project:
    name: str
    settings: Settings
    _config: ProjectConfig | None = field(default=None, repr=False)

    # -- paths ------------------------------------------------------------------

    @property
    def dir(self) -> Path:
        return self.settings.projects_root / self.name

    @property
    def config_path(self) -> Path:
        return self.dir / PROJECT_FILE

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"

    @property
    def artifact_dir(self) -> Path:
        return self.settings.artifact_root / self.name

    @property
    def logs_dir(self) -> Path:
        return self.artifact_dir / "logs"

    @cached_property
    def store(self) -> Store:
        return Store(self.artifact_dir)

    # -- config -------------------------------------------------------------------

    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self) -> ProjectConfig:
        if not self.config_path.is_file():
            raise NotFoundError(
                f"Project {self.name!r} not found: {self.config_path} does not exist",
                code="PROJECT_NOT_FOUND",
                details={"project": self.name, "path": str(self.config_path)},
                hint="Run `ai-pipeline project list` to see projects, or "
                "`ai-pipeline project create <name>`.",
            )
        cfg = load_project_config(self.config_path)
        if cfg.project.name != self.name:
            raise ConfigError(
                f"project.name {cfg.project.name!r} in {self.config_path} does not match "
                f"directory name {self.name!r}",
                code="PROJECT_NAME_MISMATCH",
                details={"path": str(self.config_path), "expected": self.name},
            )
        validate_project_against_infra(cfg, self.settings.infra)
        return cfg

    # -- record / pointers ----------------------------------------------------------

    def record(self) -> dict[str, Any]:
        return self.store.read_project()

    def ensure_record(self, *, task_type: str, template: str | None = None) -> bool:
        """Create ``project.json`` if missing. Returns True when it was created.

        ``project.yaml`` is committed to git while the artifact root is not, so a fresh
        checkout has configs without records; the record is derivable and holds only
        pointers, so initialising it lazily is safe and idempotent.
        """
        if self.store.project_exists():
            return False
        now = utc_now_iso()
        self.store.write_project(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "name": self.name,
                "task_type": task_type,
                "template": template,
                "created_at": now,
                "updated_at": now,
                "pointers": dict.fromkeys(POINTER_KEYS),
            }
        )
        return True

    def pointers(self) -> dict[str, str | None]:
        rec = self.record()
        return {k: rec.get("pointers", {}).get(k) for k in POINTER_KEYS}

    def set_pointer(self, key: str, value: str | None) -> None:
        if key not in POINTER_KEYS:
            raise ValueError(f"unknown pointer {key!r}")
        rec = self.record()
        rec.setdefault("pointers", {})[key] = value
        rec["updated_at"] = utc_now_iso()
        self.store.write_project(rec)

    # -- construction -------------------------------------------------------------------

    @classmethod
    def load(cls, name: str, settings: Settings) -> Project:
        project = cls(validate_project_name(name), settings)
        cfg = project.load_config()  # fail early
        project.ensure_record(task_type=cfg.project.task_type)
        return project

    @classmethod
    def exists(cls, name: str, settings: Settings) -> bool:
        return (settings.projects_root / name / PROJECT_FILE).is_file()

    @classmethod
    def list_names(cls, settings: Settings) -> list[str]:
        root = settings.projects_root
        if not root.is_dir():
            return []
        return sorted(p.parent.name for p in root.glob(f"*/{PROJECT_FILE}"))

    @classmethod
    def create(
        cls,
        name: str,
        settings: Settings,
        *,
        task_type: str = "object_detection",
        template: str = "default",
    ) -> Project:
        validate_project_name(name)
        if task_type not in task_types():
            raise ConfigError(
                f"Unknown task type {task_type!r}",
                code="TASK_TYPE_UNKNOWN",
                details={"task_type": task_type, "known": task_types()},
            )
        project = cls(name, settings)
        if project.config_path.exists():
            raise AlreadyExistsError(
                f"Project {name!r} already exists at {project.dir}",
                code="PROJECT_EXISTS",
                details={"project": name, "path": str(project.dir)},
                hint="Choose another name, or edit the existing project.yaml.",
            )
        if project.store.project_exists():
            raise AlreadyExistsError(
                f"Artifact directory for {name!r} already has a project record "
                f"at {project.store.project_record_path}",
                code="PROJECT_RECORD_EXISTS",
                details={"project": name, "path": str(project.store.project_record_path)},
                hint="Remove or rename the stale artifact directory before recreating "
                "the project, or pick a different --artifact-root.",
            )

        files = render_template(template, project_name=name, task_type=task_type)
        project.dir.mkdir(parents=True, exist_ok=False)
        for rel, content in files.items():
            dest = project.dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        (project.data_dir / "raw").mkdir(parents=True, exist_ok=True)

        project.ensure_record(task_type=task_type, template=template)
        project.load_config()  # the rendered template must itself be valid
        return project


def render_template(template: str, **values: str) -> dict[str, str]:
    """Return ``{relative_path: content}`` for every file in ``templates/<template>/``.

    Placeholders are ``{{name}}``; deliberately not Jinja to keep templates trivially readable.
    """
    base = resources.files("ai_pipeline.templates")
    tdir = base / template
    if not tdir.is_dir():
        raise ConfigError(
            f"Unknown project template {template!r}",
            code="TEMPLATE_UNKNOWN",
            details={
                "template": template,
                "known": sorted(p.name for p in base.iterdir() if p.is_dir()),
            },
        )
    values = {**values, "created_at": utc_now_iso()}
    out: dict[str, str] = {}
    for rel, entry in _walk(tdir, ""):
        content = entry.read_text(encoding="utf-8")
        for key, val in values.items():
            content = content.replace("{{" + key + "}}", val)
        out[rel] = content
    return out


def _walk(node: Any, prefix: str):
    for child in node.iterdir():
        rel = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            yield from _walk(child, rel)
        else:
            yield rel, child
