"""Job spec / status / result file schemas.

Files inside a job directory::

    job.json      JobSpec   written once by the control plane
    status.json   JobStatus written by the backend (PENDING) and the runner (RUNNING → terminal)
    runner.log    stdout/stderr of the runner process plus its own logging
    out/          everything the provider produced; out/result.json is JobResult

All paths inside these files are relative to the job directory so the directory can be moved
or synced to another host without rewriting.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_pipeline.core.store import read_json, write_json_atomic

JOB_SCHEMA_VERSION = 1
JOB_KINDS = ("autolabel", "train", "evaluate", "export")
JobKind = Literal["autolabel", "train", "evaluate", "export"]

SPEC_FILE = "job.json"
STATUS_FILE = "status.json"
LOG_FILE = "runner.log"
OUT_DIR = "out"
RESULT_FILE = "out/result.json"


class JobState(str, Enum):
    PENDING = "PENDING"  # written by the backend at submit, before the runner starts
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_JOB_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})


class JobSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = JOB_SCHEMA_VERSION
    run_id: str
    kind: JobKind
    project: str
    task_type: str
    provider: str
    # Kind-specific inputs. Paths are relative to the job dir ("inputs/...") or absolute on the
    # compute host; the backend is responsible for making them resolvable there.
    inputs: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = JOB_SCHEMA_VERSION
    run_id: str
    state: JobState
    # Opaque backend/runner handle, e.g. {"pid": 123, "hostname": "ai-server"}. Generic by
    # design: the run model never interprets it, only the backend that wrote it does.
    handle: dict[str, Any] = Field(default_factory=dict)
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
    result_file: str | None = None  # relative to the job dir when COMPLETED


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = JOB_SCHEMA_VERSION
    run_id: str
    kind: JobKind
    provider: str
    # Named artifacts, paths relative to the job dir (e.g. {"weights": "out/weights/best.pt"})
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


def write_spec(job_dir: Path, spec: JobSpec) -> None:
    write_json_atomic(job_dir / SPEC_FILE, spec.model_dump())


def read_spec(job_dir: Path) -> JobSpec:
    return JobSpec.model_validate(read_json(job_dir / SPEC_FILE))


def write_status(job_dir: Path, status: JobStatus) -> None:
    write_json_atomic(job_dir / STATUS_FILE, status.model_dump(mode="json"))


def read_status(job_dir: Path) -> JobStatus | None:
    p = job_dir / STATUS_FILE
    return JobStatus.model_validate(read_json(p)) if p.is_file() else None


def read_result(job_dir: Path) -> JobResult | None:
    p = job_dir / RESULT_FILE
    return JobResult.model_validate(read_json(p)) if p.is_file() else None
