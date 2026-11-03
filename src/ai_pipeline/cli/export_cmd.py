"""``ai-pipeline export ...`` commands."""

from __future__ import annotations

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.cli.run_cmd import _TAIL, _TIMEOUT, _WAIT, run_envelope, status_impl
from ai_pipeline.core.exports import list_exports, start_export
from ai_pipeline.core.project import Project
from ai_pipeline.core.runs import wait_run
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Export model versions to portable artifacts (ONNX).", no_args_is_help=True)


def _export_result(rec: dict) -> dict:
    return {
        "export_id": rec["export_id"],
        "state": rec["state"],
        "model_version": rec["model_version"],
        "format": rec["format"],
        "export_run_id": rec["export_run_id"],
        "artifact": rec["artifact"],
        "export_params": rec["export_params"],
    }


@app.command("start")
def start_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    model_version: str | None = typer.Argument(None, help="Model version (default: selected)."),
    backend: str | None = typer.Option(None, "--backend", help="Compute backend."),
    rerun: bool = typer.Option(
        False, "--rerun", help="Export again even if an identical READY export exists."
    ),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
) -> None:
    """Submit an export job; on completion an immutable ExportArtifact (exp-NNN) is recorded.
    Reuses an existing export with the identical request unless --rerun."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run, existing = start_export(
            project, model_version=model_version, backend=backend, rerun=rerun
        )
        if existing is not None:
            return Envelope(
                operation="export.start",
                project=name,
                result={**_export_result(existing), "created": False, "run_id": None},
                next_operations=project_status(project)["next_operations"],
                human_text=(
                    f"{existing['export_id']} already exists for {existing['model_version']} "
                    f"({existing['format']}, identical parameters); nothing exported. "
                    "Use --rerun to export again."
                ),
            )
        if wait:
            run = wait_run(project, run["run_id"], timeout_s=timeout)
        env = run_envelope(project, run, "export.start")
        env.result["created"] = True
        if run.get("export_id"):
            rec = project.store.read("exports", run["export_id"])
            env.result["export"] = _export_result(rec)
            env.human_text = (env.human_text or "") + (
                f"\n  export {rec['export_id']}: {rec['format']} "
                f"{rec['artifact']['size_bytes']} bytes sha256={rec['artifact']['sha256'][:12]}"
            )
        return env

    run_command(ctx, "export.start", op, project=name)


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="Export run id."),
    wait: bool = _WAIT,
    timeout: float = _TIMEOUT,
    tail: int | None = _TAIL,
) -> None:
    """Alias of `run status` for export runs."""
    status_impl(ctx, "export.status", name, run_id, wait, timeout, tail)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List export artifacts."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rows = list_exports(project)
        text = (
            "\n".join(
                f"{r['export_id']}  {r['state']:<6} {r['model_version']}  {r['format']}  "
                f"{r['size_bytes']} bytes  sha={r['sha256']}  run={r['export_run_id']}"
                for r in rows
            )
            or "(no exports)"
        )
        return Envelope(
            operation="export.list", project=name, result={"exports": rows}, human_text=text
        )

    run_command(ctx, "export.list", op, project=name)


@app.command("show")
def show_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    export_id: str = typer.Argument(..., help="Export id, e.g. exp-001."),
) -> None:
    """Show an export artifact's full record."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        return Envelope(
            operation="export.show", project=name, result=project.store.read("exports", export_id)
        )

    run_command(ctx, "export.show", op, project=name)
