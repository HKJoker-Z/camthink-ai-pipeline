"""What a provider sees while executing a job."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.jobs.spec import JobSpec, JobState, JobStatus, read_status, write_status


@dataclass
class JobContext:
    job_dir: Path
    spec: JobSpec
    log: logging.Logger
    _status: JobStatus = field(repr=False)

    @property
    def out_dir(self) -> Path:
        d = self.job_dir / "out"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve(self, path: str) -> Path:
        """Resolve a spec path (relative to the job dir, or absolute)."""
        p = Path(path)
        return p if p.is_absolute() else (self.job_dir / p)

    def relpath(self, path: Path) -> str:
        """Make an output path relative to the job dir for result.json."""
        return str(Path(path).resolve().relative_to(self.job_dir.resolve()))

    def report_progress(self, **fields: Any) -> None:
        """Merge progress fields into status.json (e.g. epoch=3, total_epochs=5)."""
        current = read_status(self.job_dir) or self._status
        if current.state is not JobState.RUNNING:
            return
        current.progress.update(fields)
        current.progress["updated_at"] = utc_now_iso()
        write_status(self.job_dir, current)
        self._status = current
