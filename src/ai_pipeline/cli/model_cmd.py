"""``ai-pipeline model ...`` commands (model registry)."""

from __future__ import annotations

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.project import Project
from ai_pipeline.core.registry import list_models, register_from_run, select_model
from ai_pipeline.core.runs import sync_run
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Model registry: list, inspect, register, select.", no_args_is_help=True)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List model versions with state and evaluation metrics."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rows = list_models(project)
        selected = project.pointers()["selected_model_version"]
        text = (
            "\n".join(
                f"{'*' if r['model_version'] == selected else ' '} {r['model_version']}  "
                f"{r['state']:<10} {r['candidate'] or '-':<10} ds={r['dataset_version'] or '-'} "
                f"run={r['training_run_id']}  eval={r['eval_metrics'] or '-'}"
                for r in rows
            )
            or "(no models)"
        )
        return Envelope(
            operation="model.list",
            project=name,
            result={"models": rows, "selected_model_version": selected},
            human_text=text,
        )

    run_command(ctx, "model.list", op, project=name)


@app.command("show")
def show_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    model_version: str = typer.Argument(..., help="Model version id."),
) -> None:
    """Show a model version's full record."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        return Envelope(
            operation="model.show", project=name, result=project.store.read("models", model_version)
        )

    run_command(ctx, "model.show", op, project=name)


@app.command("register")
def register_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    run_id: str = typer.Argument(..., help="COMPLETED train run id."),
) -> None:
    """Register a model version from a completed train run (normally automatic on completion)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run = sync_run(project, run_id)
        record, created = register_from_run(project, run)
        return Envelope(
            operation="model.register",
            project=name,
            result={
                "model_version": record["model_version"],
                "created": created,
                "state": record["state"],
                "run_id": run_id,
            },
            next_operations=project_status(project)["next_operations"],
            human_text=(
                f"{'registered' if created else 'already registered'} "
                f"{record['model_version']} ({record['state']}) from {run_id}"
            ),
        )

    run_command(ctx, "model.register", op, project=name)


@app.command("select")
def select_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    model_version: str | None = typer.Argument(
        None, help="Explicit model version (default: best accepted)."
    ),
) -> None:
    """Select the model version that later stages (export/deploy) will use."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        record = select_model(project, model_version)
        metric = project.config.evaluation.selection.primary_metric
        return Envelope(
            operation="model.select",
            project=name,
            result={
                "selected_model_version": record["model_version"],
                "state": record["state"],
                "primary_metric": metric,
                "value": (record["metrics"].get("eval") or {}).get(metric),
                "explicit": model_version is not None,
            },
            next_operations=project_status(project)["next_operations"],
            human_text=f"selected {record['model_version']} ({record['state']}, {metric}="
            f"{(record['metrics'].get('eval') or {}).get(metric)})",
        )

    run_command(ctx, "model.select", op, project=name)
