"""``ai-pipeline run ...`` — generic observation/control of any run kind."""

from __future__ import annotations

import contextlib
from typing import Any

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.project import Project
from ai_pipeline.core.runs import cancel_run, list_runs, run_logs, sync_run, wait_run
from ai_pipeline.core.state import RunState
from ai_pipeline.core.status import project_status

app = typer.Typer(
    help="Inspect and control runs (train / evaluate / autolabel).", no_args_is_help=True
)


def run_result(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "kind": run["kind"],
        "state": run["state"],
        "backend": run["backend"],
        "provider": run["provider"],
        "candidate": run.get("candidate"),
        "inputs": {k: v for k, v in run["inputs"].items() if not str(v).startswith("/")},
        "timestamps": run["timestamps"],
        "progress": run.get("progress") or {},
        "metrics": {k: v for k, v in (run.get("metrics") or {}).items() if k != "per_class"},
        "environment": run.get("environment") or {},
        "outputs": run.get("outputs") or {},
        "model_version": run.get("model_version"),
        "label_set_id": run.get("label_set_id"),
        "export_id": run.get("export_id"),
        "error": run.get("error"),
        "post_error": run.get("post_error"),
    }


def run_text(run: dict[str, Any]) -> str:
    lines = [f"{run['run_id']}  kind={run['kind']}  state={run['state']}  backend={run['backend']}"]
    ts = run["timestamps"]
    lines.append(f"  created={ts['created']} started={ts['started']} finished={ts['finished']}")
    if run.get("progress"):
        lines.append(f"  progress: {run['progress']}")
    metrics = {k: v for k, v in (run.get("metrics") or {}).items() if k != "per_class"}
    if metrics:
        lines.append(f"  metrics: {metrics}")
    dev = (run.get("environment") or {}).get("device")
    if isinstance(dev, dict):
        lines.append(
            f"  device: requested={dev.get('requested')} used={dev.get('used')} "
            f"({dev.get('policy')}; cuda_available={dev.get('cuda_available')})"
        )
    if run.get("model_version"):
        lines.append(f"  model_version: {run['model_version']}")
    if run.get("label_set_id"):
        lines.append(f"  label_set_id: {run['label_set_id']}")
    if run.get("export_id"):
        lines.append(f"  export_id: {run['export_id']}")
    if run.get("error"):
        lines.append(f"  error: {run['error'].get('type')}: {run['error'].get('message')}")
    if run.get("post_error"):
        lines.append(f"  post-processing error: {run['post_error']}")
    return "\n".join(lines)


def run_envelope(project: Project, run: dict[str, Any], operation: str) -> Envelope:
    state = RunState(run["state"])
    status = "ok"
    error = None
    if state in (RunState.FAILED, RunState.CANCELLED):
        status = "failed"
        err = run.get("error") or {}
        status_code = "RUN_FAILED" if state is RunState.FAILED else "RUN_CANCELLED"
        error = {
            "code": status_code,
            "message": f"{err.get('type', state.value)}: {err.get('message', '')}".strip(": "),
            "details": {
                "run_id": run["run_id"],
                "cause": err.get("code"),  # structured code raised inside the job, if any
                **(err.get("details") or {}),
                "log_tail": err.get("log_tail"),
            },
            "hint": err.get("hint") or f"See `run logs {project.name} {run['run_id']}`.",
        }
    return Envelope(
        operation=operation,
        project=project.name,
        status=status,  # type: ignore[arg-type]
        result=run_result(run),
        error=error,
        next_operations=project_status(project)["next_operations"],
        human_text=run_text(run),
    )


def status_impl(
    ctx: typer.Context,
    operation: str,
    name: str,
    run_id: str,
    wait: bool,
    timeout: float,
    tail: int | None,
) -> None:
    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run = wait_run(project, run_id, timeout_s=timeout) if wait else sync_run(project, run_id)
        env = run_envelope(project, run, operation)
        if tail:
            logs = run_logs(project, run_id, tail=tail)
            env.result["log_tail"] = logs
            env.human_text = (env.human_text or "") + "\n--- log tail ---\n" + logs
        return env

    run_command(ctx, operation, op, project=name)


_WAIT = typer.Option(False, "--wait", help="Block until the run reaches a terminal state.")
_TIMEOUT = typer.Option(3600.0, "--timeout", help="Seconds to wait with --wait.")
_TAIL = typer.Option(None, "--tail", help="Append the last N log lines.")


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="Run id."),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
    tail: int | None = _TAIL,
) -> None:
    """Sync a run with its job status (and run post-processing on completion)."""
    status_impl(ctx, "run.status", name, run_id, wait, timeout, tail)


@app.command("list")
def list_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    kind: str | None = typer.Option(None, "--kind", help="train | evaluate | autolabel"),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Refresh non-terminal runs first."),
) -> None:
    """List runs (refreshing active ones)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        if sync:
            for r in project.store.list("runs"):
                if r["state"] in ("SUBMITTED", "RUNNING"):
                    # Listing must not fail because one run's post hook does.
                    with contextlib.suppress(Exception):
                        sync_run(project, r["run_id"])
        rows = list_runs(project, kind)
        text = (
            "\n".join(
                f"{r['run_id']}  {r['kind']:<9} {r['state']:<10} {r['candidate'] or '-':<10} "
                f"ds={r['dataset_version'] or '-'} mv={r['model_version'] or '-'} "
                f"lbl={r['label_set_id'] or '-'}"
                for r in rows
            )
            or "(no runs)"
        )
        return Envelope(operation="run.list", project=name, result={"runs": rows}, human_text=text)

    run_command(ctx, "run.list", op, project=name)


@app.command("logs")
def logs_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="Run id."),
    tail: int | None = typer.Option(None, "--tail", help="Only the last N lines."),
) -> None:
    """Print the runner log of a run."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        logs = run_logs(project, run_id, tail=tail)
        return Envelope(
            operation="run.logs",
            project=name,
            result={"run_id": run_id, "logs": logs},
            human_text=logs or "(empty)",
        )

    run_command(ctx, "run.logs", op, project=name)


@app.command("cancel")
def cancel_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="Run id."),
) -> None:
    """Cancel a run that is not yet finished."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run = cancel_run(project, run_id)
        env = run_envelope(project, run, "run.cancel")
        if run["state"] == "CANCELLED":  # cancellation that worked is a success for this command
            env.status, env.error = "ok", None
        return env

    run_command(ctx, "run.cancel", op, project=name)
