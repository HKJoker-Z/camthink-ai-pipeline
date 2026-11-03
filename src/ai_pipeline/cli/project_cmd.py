"""``ai-pipeline project ...`` commands."""

from __future__ import annotations

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.errors import ConfigError
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Create, validate and list projects.", no_args_is_help=True)


@app.command("create")
def create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name: ^[a-z][a-z0-9_]*$"),
    task: str = typer.Option("object_detection", "--task", help="Task type."),
    template: str = typer.Option("default", "--template", help="Scaffold template."),
) -> None:
    """Scaffold projects/<name>/project.yaml from a template and initialise its state."""

    def op(cli: CliContext) -> Envelope:
        project = Project.create(name, cli.settings, task_type=task, template=template)
        status = project_status(project)
        return Envelope(
            operation="project.create",
            project=name,
            result={
                "project": name,
                "task_type": task,
                "template": template,
                "config_path": str(project.config_path),
                "data_dir": str(project.data_dir),
                "artifact_dir": str(project.artifact_dir),
                "created": True,
            },
            next_operations=[f"project validate {name}", *status["next_operations"]],
            human_text=(
                f"Created project {name!r}\n"
                f"  config:    {project.config_path}\n"
                f"  data:      {project.data_dir}\n"
                f"  artifacts: {project.artifact_dir}\n"
                f"Edit project.yaml (classes, candidates, thresholds), then validate."
            ),
        )

    run_command(ctx, "project.create", op, project=name)


@app.command("validate")
def validate(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
) -> None:
    """Validate project.yaml (schema, task vocabulary, infrastructure references)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        cfg = project.config
        return Envelope(
            operation="project.validate",
            project=name,
            result={
                "project": name,
                "valid": True,
                "config_path": str(project.config_path),
                "task_type": cfg.project.task_type,
                "classes": cfg.dataset.classes,
                "provider": cfg.model.provider,
                "candidates": [c.name for c in cfg.model.candidates],
                "training_backend": cfg.training.backend,
                "deployment_target": cfg.deployment.target,
            },
            human_text=f"{project.config_path}: valid",
        )

    run_command(ctx, "project.validate", op, project=name)


@app.command("list")
def list_projects(ctx: typer.Context) -> None:
    """List projects under the projects root, with their validity."""

    def op(cli: CliContext) -> Envelope:
        rows = []
        for name in Project.list_names(cli.settings):
            entry: dict = {"project": name, "valid": True, "error": None}
            try:
                Project.load(name, cli.settings)
            except ConfigError as exc:
                entry.update(valid=False, error=exc.message)
            rows.append(entry)
        text = (
            "\n".join(
                f"{r['project']:<32} {'valid' if r['valid'] else 'INVALID: ' + str(r['error'])}"
                for r in rows
            )
            or "(no projects)"
        )
        return Envelope(
            operation="project.list",
            result={"projects_root": str(cli.settings.projects_root), "projects": rows},
            human_text=text,
        )

    run_command(ctx, "project.list", op)
