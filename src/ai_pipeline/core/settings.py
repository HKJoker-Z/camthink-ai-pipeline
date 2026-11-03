"""Resolution of where things live: infrastructure file, projects root, artifact root.

Precedence (highest first): explicit CLI option > environment variable > infrastructure.yaml
value > built-in default. Relative paths inside infrastructure.yaml are resolved against the
file's own directory so the CLI works from any working directory; relative CLI/env paths are
resolved against the current working directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ai_pipeline.core.config import InfrastructureConfig, load_infrastructure
from ai_pipeline.core.errors import ConfigError

ENV_INFRA_FILE = "AIP_INFRA_FILE"
ENV_PROJECTS_ROOT = "AIP_PROJECTS_ROOT"
ENV_ARTIFACT_ROOT = "AIP_ARTIFACT_ROOT"
INFRA_FILENAME = "infrastructure.yaml"


def find_infrastructure_file(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default CWD) looking for ``infrastructure.yaml``, like git."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / INFRA_FILENAME
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class Settings:
    infra_path: Path
    infra: InfrastructureConfig
    projects_root: Path
    artifact_root: Path

    @classmethod
    def resolve(
        cls,
        *,
        infra_path: Path | None = None,
        projects_root: Path | None = None,
        artifact_root: Path | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> Settings:
        env = os.environ if env is None else env
        cwd = (cwd or Path.cwd()).resolve()

        if infra_path is None and env.get(ENV_INFRA_FILE):
            infra_path = Path(env[ENV_INFRA_FILE])
        if infra_path is None:
            infra_path = find_infrastructure_file(cwd)
        if infra_path is None:
            raise ConfigError(
                f"No {INFRA_FILENAME} found in {cwd} or its parents",
                code="INFRA_FILE_MISSING",
                details={"cwd": str(cwd)},
                hint=f"Run from the repository, pass --infra <file>, or set {ENV_INFRA_FILE}.",
            )
        infra_path = (cwd / infra_path).resolve() if not infra_path.is_absolute() else infra_path
        infra = load_infrastructure(infra_path)
        base = infra_path.parent

        def pick(explicit: Path | None, env_key: str, from_infra: str) -> Path:
            if explicit is not None:
                return (cwd / explicit).resolve()
            if env.get(env_key):
                return (cwd / Path(env[env_key])).resolve()
            return (base / Path(from_infra)).resolve()

        return cls(
            infra_path=infra_path,
            infra=infra,
            projects_root=pick(projects_root, ENV_PROJECTS_ROOT, infra.projects_root),
            artifact_root=pick(artifact_root, ENV_ARTIFACT_ROOT, infra.artifact_root),
        )
