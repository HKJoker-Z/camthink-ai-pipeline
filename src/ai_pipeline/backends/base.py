"""ComputeBackend protocol.

A backend knows how to launch the job runner somewhere, how to read the job's ``status.json``
from there, and how to move the job directory to/from there. It never interprets results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ai_pipeline.core.jobs.spec import JobStatus


@runtime_checkable
class ComputeBackend(Protocol):
    name: str

    def submit(self, job_dir: Path) -> JobStatus:
        """Stage the job directory if needed and launch the runner detached.
        Returns the PENDING status (with an opaque ``handle``)."""
        ...

    def status(self, job_dir: Path) -> JobStatus:
        """Current status, reconciled with reality (e.g. a runner that died without
        writing a terminal state is reported as FAILED)."""
        ...

    def logs(self, job_dir: Path, *, tail: int | None = None) -> str: ...

    def fetch(self, job_dir: Path) -> None:
        """Bring ``out/`` and ``status.json`` back to the control host (no-op when shared)."""
        ...

    def cancel(self, job_dir: Path) -> JobStatus: ...
