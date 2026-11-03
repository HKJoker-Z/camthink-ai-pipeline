"""Local compute backend: runs the job runner as a detached subprocess on this machine.

The control host and compute host share the artifact root, so ``fetch`` is a no-op. This is
the production backend for the V1 topology (ControlHost == ComputeHost == AI server) and the
backend used by the test suite.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ai_pipeline.core.errors import OperationError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.jobs.spec import (
    LOG_FILE,
    TERMINAL_JOB_STATES,
    JobState,
    JobStatus,
    read_spec,
    read_status,
    write_status,
)
from ai_pipeline.core.logging import get_logger

log = get_logger("backend.local")
RUNNER_MODULE = "ai_pipeline.core.jobs.runner"


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (exited, not yet reaped) still answers kill(0); check /proc state when available.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


class LocalBackend:
    def __init__(self, *, name: str = "local", python: str | None = None) -> None:
        self.name = name
        self.python = python or sys.executable

    # -- protocol ------------------------------------------------------------------

    def submit(self, job_dir: Path) -> JobStatus:
        spec = read_spec(job_dir)
        status = JobStatus(run_id=spec.run_id, state=JobState.PENDING, submitted_at=utc_now_iso())
        write_status(job_dir, status)
        log_path = job_dir / LOG_FILE
        try:
            with log_path.open("ab") as log_fh:
                proc = subprocess.Popen(
                    [self.python, "-m", RUNNER_MODULE, str(job_dir)],
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=str(job_dir),
                    start_new_session=True,  # survives the CLI process exiting
                )
        except OSError as exc:
            status.state = JobState.FAILED
            status.finished_at = utc_now_iso()
            status.error = {"type": type(exc).__name__, "message": f"could not start runner: {exc}"}
            write_status(job_dir, status)
            raise OperationError(
                f"could not start job runner with {self.python!r}: {exc}",
                code="RUNNER_START_FAILED",
                details={"python": self.python, "job_dir": str(job_dir)},
                hint="Check compute_backends.<name>.python in infrastructure.yaml.",
            ) from exc
        status.handle = {"pid": proc.pid, "python": self.python, "backend": self.name}
        write_status(job_dir, status)
        log.bind(run_id=spec.run_id).info("submitted job pid=%s", proc.pid)
        return status

    def status(self, job_dir: Path) -> JobStatus:
        status = read_status(job_dir)
        if status is None:
            spec = read_spec(job_dir)
            return JobStatus(run_id=spec.run_id, state=JobState.PENDING)
        if status.state in TERMINAL_JOB_STATES:
            return status
        pid = status.handle.get("pid")
        if pid and not _pid_alive(pid):
            # Re-read: the runner may have written its terminal state between our read and
            # the liveness check.
            fresh = read_status(job_dir)
            if fresh is not None and fresh.state in TERMINAL_JOB_STATES:
                return fresh
            status = fresh or status
            status.state = JobState.FAILED
            status.finished_at = utc_now_iso()
            status.error = {
                "type": "RunnerDied",
                "message": f"runner process {pid} exited without reporting a terminal state",
                "log_tail": self.logs(job_dir, tail=20),
            }
            write_status(job_dir, status)
        return status

    def logs(self, job_dir: Path, *, tail: int | None = None) -> str:
        p = job_dir / LOG_FILE
        if not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if tail is None:
            return text
        return "\n".join(text.splitlines()[-tail:])

    def fetch(self, job_dir: Path) -> None:
        return None  # shared filesystem

    def cancel(self, job_dir: Path) -> JobStatus:
        status = self.status(job_dir)
        if status.state in TERMINAL_JOB_STATES:
            return status
        pid = status.handle.get("pid")
        if pid and _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while _pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _pid_alive(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    os.kill(pid, signal.SIGKILL)
        status = read_status(job_dir) or status
        status.state = JobState.CANCELLED
        status.finished_at = utc_now_iso()
        status.error = {"type": "Cancelled", "message": "cancelled by operator"}
        write_status(job_dir, status)
        return status
