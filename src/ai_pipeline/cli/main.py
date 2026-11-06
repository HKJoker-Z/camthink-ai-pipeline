"""``ai-pipeline`` entry point: global options and command groups."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from ai_pipeline import __version__
from ai_pipeline.cli import (
    app_cmd,
    data_cmd,
    dataset_cmd,
    deploy_cmd,
    evaluate_cmd,
    export_cmd,
    label_cmd,
    model_cmd,
    project_cmd,
    run_cmd,
    service_cmd,
    train_cmd,
)
from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.logging import configure_logging
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(
    name="ai-pipeline",
    help="Reusable AI model lifecycle pipeline for CamThink edge devices.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)
app.add_typer(project_cmd.app, name="project")
app.add_typer(data_cmd.app, name="data")
app.add_typer(label_cmd.app, name="label")
app.add_typer(dataset_cmd.app, name="dataset")
app.add_typer(train_cmd.app, name="train")
app.add_typer(run_cmd.app, name="run")
app.add_typer(evaluate_cmd.app, name="evaluate")
app.add_typer(model_cmd.app, name="model")
app.add_typer(export_cmd.app, name="export")
app.add_typer(deploy_cmd.app, name="deploy")
app.add_typer(deploy_cmd.target_app, name="target")
app.add_typer(service_cmd.app, name="service")
app.add_typer(app_cmd.app, name="app")
deploy_cmd.target_app.command("status")(service_cmd.target_status_cmd)
app.command("rollback")(deploy_cmd.rollback_cmd)
app.command("health")(deploy_cmd.health_cmd)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    output: str = typer.Option(
        "human", "--output", "-o", help="Output format: human | json.", show_default=True
    ),
    infra: Path | None = typer.Option(
        None, "--infra", help="infrastructure.yaml path (default: search upward from CWD)."
    ),
    projects_root: Path | None = typer.Option(
        None, "--projects-root", help="Override projects root."
    ),
    artifact_root: Path | None = typer.Option(
        None, "--artifact-root", help="Override artifact root."
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="-v info, -vv debug."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only log errors."),
    log_json: bool = typer.Option(False, "--log-json", help="JSON lines on stderr."),
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Print version and exit."
    ),
) -> None:
    if output not in ("human", "json"):
        raise typer.BadParameter("must be 'human' or 'json'", param_hint="--output")
    load_dotenv(override=False)
    configure_logging(verbosity=verbose, quiet=quiet, json_logs=log_json)
    ctx.obj = CliContext(
        output=output,  # type: ignore[arg-type]
        infra_path=infra,
        projects_root=projects_root,
        artifact_root=artifact_root,
        verbosity=verbose,
    )


@app.command("status")
def status(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """Summarise a project's state: pointers, entity counts, active runs, next operations."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        summary = project_status(project)
        counts = " ".join(f"{k}={v}" for k, v in summary["counts"].items())
        pointers = "\n".join(f"  {k}: {v or '-'}" for k, v in summary["pointers"].items())
        return Envelope(
            operation="status",
            project=name,
            result=summary,
            next_operations=summary["next_operations"],
            human_text=f"task: {summary['task_type']}\ncounts: {counts}\npointers:\n{pointers}",
        )

    run_command(ctx, "status", op, project=name)


if __name__ == "__main__":  # pragma: no cover
    app()
