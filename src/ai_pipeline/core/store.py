"""Filesystem JSON metadata store: one file per entity, atomic writes.

Layout under the project's artifact directory::

    artifacts/<project>/
        project.json
        imports/<id>.json
        labels/<id>.json
        datasets/<id>.json     (plus the dataset version directory <id>/, owned by core.dataset)
        runs/<id>.json
        models/<id>.json
        deployments/<id>.json

Chosen over SQLite/MLflow for V1 because records are few, must be human-readable, and
must survive plain ``rsync``. All callers go through this class so the backend can change.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ai_pipeline.core.errors import AlreadyExistsError, NotFoundError

KINDS = ("imports", "labels", "datasets", "runs", "models", "deployments")
PROJECT_RECORD = "project.json"


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via temp file + ``os.replace`` so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        finally:
            raise


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class Store:
    """Per-project metadata store rooted at ``artifacts/<project>/``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- project record -------------------------------------------------------

    @property
    def project_record_path(self) -> Path:
        return self.root / PROJECT_RECORD

    def project_exists(self) -> bool:
        return self.project_record_path.is_file()

    def read_project(self) -> dict[str, Any]:
        if not self.project_exists():
            raise NotFoundError(
                f"No project record at {self.project_record_path}",
                code="PROJECT_RECORD_MISSING",
                details={"path": str(self.project_record_path)},
                hint="Run `ai-pipeline project create <name>` first, or check --artifact-root.",
            )
        return read_json(self.project_record_path)

    def write_project(self, data: dict[str, Any]) -> None:
        write_json_atomic(self.project_record_path, data)

    # -- entities ---------------------------------------------------------------

    def _dir(self, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"unknown entity kind {kind!r}; expected one of {KINDS}")
        return self.root / kind

    def path(self, kind: str, entity_id: str) -> Path:
        return self._dir(kind) / f"{entity_id}.json"

    def exists(self, kind: str, entity_id: str) -> bool:
        return self.path(kind, entity_id).is_file()

    def read(self, kind: str, entity_id: str) -> dict[str, Any]:
        p = self.path(kind, entity_id)
        if not p.is_file():
            raise NotFoundError(
                f"{kind[:-1]} {entity_id!r} not found",
                code=f"{kind[:-1].upper()}_NOT_FOUND",
                details={"kind": kind, "id": entity_id, "path": str(p)},
                hint=f"List existing ids with the corresponding `list` command for {kind}.",
            )
        return read_json(p)

    def write(self, kind: str, entity_id: str, data: dict[str, Any]) -> None:
        write_json_atomic(self.path(kind, entity_id), data)

    def create(self, kind: str, entity_id: str, data: dict[str, Any]) -> None:
        """Write a new record; refuse to overwrite an existing one."""
        if self.exists(kind, entity_id):
            raise AlreadyExistsError(
                f"{kind[:-1]} {entity_id!r} already exists",
                code=f"{kind[:-1].upper()}_EXISTS",
                details={"kind": kind, "id": entity_id, "path": str(self.path(kind, entity_id))},
            )
        self.write(kind, entity_id, data)

    def update(self, kind: str, entity_id: str, **fields: Any) -> dict[str, Any]:
        record = self.read(kind, entity_id)
        record.update(fields)
        self.write(kind, entity_id, record)
        return record

    def list_ids(self, kind: str) -> list[str]:
        d = self._dir(kind)
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json") if not p.name.startswith("."))

    def list(self, kind: str) -> list[dict[str, Any]]:
        return [self.read(kind, i) for i in self.list_ids(kind)]

    def count(self, kind: str) -> int:
        return len(self.list_ids(kind))
