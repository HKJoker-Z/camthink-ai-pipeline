"""Job runner: executes one job directory on the compute host.

Invoked by a ComputeBackend as ``python -m ai_pipeline.core.jobs.runner <job_dir>``. This is the
only process that imports a ModelProvider implementation (and therefore ML libraries). It
reports through ``status.json`` and ``out/result.json``; the exit code mirrors the terminal
state (0 = COMPLETED, 1 = FAILED).
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
import traceback
from pathlib import Path

from ai_pipeline import __version__
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.jobs.context import JobContext
from ai_pipeline.core.jobs.spec import (
    LOG_FILE,
    RESULT_FILE,
    JobResult,
    JobSpec,
    JobState,
    JobStatus,
    read_spec,
    read_status,
    write_status,
)
from ai_pipeline.core.store import write_json_atomic
from ai_pipeline.providers import load_provider


def _logger(job_dir: Path) -> logging.Logger:
    log = logging.getLogger("ai_pipeline.job")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh = logging.FileHandler(job_dir / LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    # Backends redirect stderr into runner.log; only echo to the terminal for interactive runs.
    if sys.stderr.isatty():
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


def _environment(provider) -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "ai_pipeline": __version__,
        "provider": provider.name,
    }
    describe = getattr(provider, "describe_environment", None)
    if describe is not None:
        env.update(describe())
    return env


def execute(spec: JobSpec, ctx: JobContext, provider) -> JobResult:
    kind = spec.kind
    if kind == "train":
        out = provider.train(ctx, inputs=spec.inputs, params=spec.params)
    elif kind == "evaluate":
        out = provider.evaluate(ctx, inputs=spec.inputs, params=spec.params)
    elif kind == "autolabel":
        out = provider.autolabel(ctx, inputs=spec.inputs, params=spec.params)
    elif kind == "export":
        out = provider.export(ctx, inputs=spec.inputs, params=spec.params)
    else:  # pragma: no cover - JobSpec validation prevents this
        raise ValueError(f"unknown job kind {kind!r}")
    return JobResult(
        run_id=spec.run_id,
        kind=kind,
        provider=provider.name,
        artifacts=out.get("artifacts", {}),
        metrics=out.get("metrics", {}),
        environment=_environment(provider),
        extra=out.get("extra", {}),
    )


def run_job(job_dir: Path) -> int:
    job_dir = Path(job_dir).resolve()
    spec = read_spec(job_dir)
    log = _logger(job_dir)
    status = read_status(job_dir) or JobStatus(run_id=spec.run_id, state=JobState.PENDING)
    status.state = JobState.RUNNING
    status.started_at = utc_now_iso()
    status.handle.update({"pid": os.getpid(), "hostname": socket.gethostname()})
    write_status(job_dir, status)
    log.info("job %s kind=%s provider=%s starting", spec.run_id, spec.kind, spec.provider)

    try:
        provider = load_provider(spec.provider)
        ctx = JobContext(job_dir=job_dir, spec=spec, log=log, _status=status)
        result = execute(spec, ctx, provider)
        write_json_atomic(job_dir / RESULT_FILE, result.model_dump())
        status = read_status(job_dir) or status  # pick up progress written meanwhile
        status.state = JobState.COMPLETED
        status.finished_at = utc_now_iso()
        status.result_file = RESULT_FILE
        status.error = None
        write_status(job_dir, status)
        log.info("job %s completed", spec.run_id)
        return 0
    except BaseException as exc:  # noqa: BLE001 - must record *any* failure before exiting
        status = read_status(job_dir) or status
        status.state = JobState.FAILED
        status.finished_at = utc_now_iso()
        status.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_status(job_dir, status)
        log.error("job %s failed: %s: %s", spec.run_id, type(exc).__name__, exc)
        log.debug("%s", status.error["traceback"])
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.stderr.write("usage: python -m ai_pipeline.core.jobs.runner <job_dir>\n")
        return 2
    return run_job(Path(argv[0]))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
