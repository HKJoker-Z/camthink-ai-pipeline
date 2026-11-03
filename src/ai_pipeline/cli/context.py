"""Shared CLI plumbing: global options, lazy settings, and the command runner that turns
exceptions into envelopes and exit codes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import typer

from ai_pipeline.cli.output import Envelope, OutputMode, emit
from ai_pipeline.core.errors import ExitCode, PipelineError
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.settings import Settings

log = get_logger("cli")


@dataclass
class CliContext:
    output: OutputMode = "human"
    infra_path: Path | None = None
    projects_root: Path | None = None
    artifact_root: Path | None = None
    verbosity: int = 0
    _settings: Settings | None = field(default=None, repr=False)

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = Settings.resolve(
                infra_path=self.infra_path,
                projects_root=self.projects_root,
                artifact_root=self.artifact_root,
            )
        return self._settings


def get_ctx(ctx: typer.Context) -> CliContext:
    obj = ctx.find_root().obj
    if obj is None:  # pragma: no cover - typer always runs the root callback first
        obj = CliContext()
        ctx.find_root().obj = obj
    return obj


def run_command(
    ctx: typer.Context,
    operation: str,
    fn: Callable[[CliContext], Envelope],
    *,
    project: str | None = None,
) -> None:
    """Execute ``fn``, emit its envelope, and exit with the right code.

    - ``PipelineError`` -> status ``failed`` (or ``rejected`` for policy), its exit code.
    - any other exception -> ``INTERNAL_ERROR``, exit 1, traceback in the log.
    """
    cli = get_ctx(ctx)
    bound = log.bind(operation=operation, project=project)
    try:
        envelope = fn(cli)
    except PipelineError as exc:
        bound.error("%s failed: %s", operation, exc.message)
        envelope = Envelope(
            operation=operation,
            project=project,
            status="rejected" if exc.exit_code == ExitCode.POLICY_REJECTED else "failed",
            error=exc.to_dict(),
        )
        emit(envelope, cli.output)
        raise typer.Exit(code=exc.exit_code) from exc
    except Exception as exc:  # noqa: BLE001 - last-resort handler, reported not swallowed
        bound.exception("%s crashed", operation)
        envelope = Envelope(
            operation=operation,
            project=project,
            status="failed",
            error={
                "code": "INTERNAL_ERROR",
                "message": f"{type(exc).__name__}: {exc}",
                "details": {},
                "hint": "This is a bug in the pipeline. Re-run with -vv and report the log.",
            },
        )
        emit(envelope, cli.output)
        raise typer.Exit(code=ExitCode.INTERNAL) from exc

    envelope.operation = envelope.operation or operation
    envelope.project = envelope.project or project
    emit(envelope, cli.output)
    if envelope.status != "ok":
        code = (
            ExitCode.POLICY_REJECTED if envelope.status == "rejected" else ExitCode.OPERATION_FAILED
        )
        raise typer.Exit(code=code)
    logging.getLogger("ai_pipeline").debug("%s ok", operation)
