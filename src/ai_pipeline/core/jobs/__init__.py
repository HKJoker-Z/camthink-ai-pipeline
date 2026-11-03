"""Jobs: the unit of work a ComputeBackend executes.

A job is a directory. The control plane writes ``job.json`` (the spec); the runner, executing
on the compute host, writes ``status.json`` on every state change, ``runner.log``, and
``out/`` (``result.json`` + artifacts). Nothing else crosses the boundary, which is what lets
the same job run locally today and over SSH later.
"""

from ai_pipeline.core.jobs.spec import (
    JOB_KINDS,
    JobResult,
    JobSpec,
    JobState,
    JobStatus,
    read_result,
    read_spec,
    read_status,
    write_spec,
    write_status,
)

__all__ = [
    "JOB_KINDS",
    "JobResult",
    "JobSpec",
    "JobState",
    "JobStatus",
    "read_result",
    "read_spec",
    "read_status",
    "write_spec",
    "write_status",
]
