"""``ai-pipeline train ...`` commands."""

from __future__ import annotations

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.cli.run_cmd import (
    _TAIL,
    _TIMEOUT,
    _WAIT,
    run_envelope,
    status_impl,
)
from ai_pipeline.cli.run_cmd import (
    list_cmd as _list_runs,
)
from ai_pipeline.core.project import Project
from ai_pipeline.core.runs import start_training, wait_run

app = typer.Typer(help="Start and follow training runs.", no_args_is_help=True)


@app.command("start")
def start_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    dataset: str | None = typer.Option(
        None, "--dataset", help="Dataset version (default: current)."
    ),
    candidate: str | None = typer.Option(
        None, "--candidate", help="Candidate name (default: first)."
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Compute backend (default: training.backend)."
    ),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
) -> None:
    """Create and submit a training run (a new run id every time)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run = start_training(
            project, dataset_version=dataset, candidate_name=candidate, backend=backend
        )
        if wait:
            run = wait_run(project, run["run_id"], timeout_s=timeout)
        return run_envelope(project, run, "train.start")

    run_command(ctx, "train.start", op, project=name)


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="Run id."),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
    tail: int | None = _TAIL,
) -> None:
    """Alias of `run status` for training runs."""
    status_impl(ctx, "train.status", name, run_id, wait, timeout, tail)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List training runs."""
    _list_runs(ctx, name, kind="train", sync=True)
