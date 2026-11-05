"""``ai-pipeline project ...`` commands."""

from __future__ import annotations

from typing import Any

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.configure import configure_file, named_changes
from ai_pipeline.core.errors import ConfigError
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Create, validate, configure and list projects.", no_args_is_help=True)

# Supported configuration fields, shared by `project create` and `project configure`.
# Anything else goes through --set key.path=value (validated against the schema).
_DESCRIPTION = typer.Option(None, "--description", help="project.description")
_CLASSES = typer.Option(
    None,
    "--classes",
    help="Comma-separated class names in index order (dataset.classes); "
    "labeling.auto.class_map becomes the identity map unless --auto-label-map is given.",
)
_SOURCE = typer.Option(
    None,
    "--source",
    help="Image folder for `data import` (dataset.sources; relative to the project dir).",
)
_WEIGHTS = typer.Option(
    None, "--weights", help="Pretrained weights to fine-tune, e.g. yolo11n.pt (model.candidates)."
)
_AUTO_WEIGHTS = typer.Option(
    None, "--auto-label-weights", help="Weights used by `label auto` (labeling.auto.weights)."
)
_AUTO_MAP = typer.Option(
    None,
    "--auto-label-map",
    help="pretrained:project class pairs for `label auto`, e.g. 'cat:cat,dog:dog' "
    "(labeling.auto.class_map).",
)
_AUTO_CONF = typer.Option(None, "--auto-label-confidence", help="labeling.auto.confidence")
_EPOCHS = typer.Option(None, "--epochs", min=1, help="training.epochs")
_IMAGE_SIZE = typer.Option(
    None, "--image-size", min=32, help="training.image_size and export.image_size"
)
_BATCH = typer.Option(None, "--batch-size", min=1, help="training.batch_size")
_DEVICE = typer.Option(None, "--device", help='training.device: "0", "1", "cpu", …')
_MIN_IMAGES = typer.Option(None, "--min-images", min=1, help="dataset.validation.min_images")
_MIN_INSTANCES = typer.Option(
    None, "--min-instances", min=0, help="dataset.validation.min_instances_per_class"
)
_MIN_MAP50 = typer.Option(None, "--min-map50", help="evaluation.acceptance.mAP50.min")
_TARGET = typer.Option(None, "--target", help="deployment.target (name from infrastructure.yaml)")
_PRECISION = typer.Option(None, "--precision", help="deployment.precision: fp16 | fp32")
_CONFIDENCE = typer.Option(None, "--confidence", help="deployment.inference.confidence")
_SET = typer.Option(
    [],
    "--set",
    help="Generic key.path=value (YAML value), repeatable, e.g. --set training.epochs=30.",
)


def _changes(has_auto: bool = True, **named: Any) -> list[tuple[str, Any]]:
    return named_changes(has_auto_labeling=has_auto, **named)


@app.command("create")
def create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name: ^[a-z][a-z0-9_]*$"),
    task: str = typer.Option("object_detection", "--task", help="Task type."),
    template: str = typer.Option("default", "--template", help="Scaffold template."),
    description: str | None = _DESCRIPTION,
    classes: str | None = _CLASSES,
    source: str | None = _SOURCE,
    weights: str | None = _WEIGHTS,
    auto_label_weights: str | None = _AUTO_WEIGHTS,
    auto_label_map: str | None = _AUTO_MAP,
    auto_label_confidence: float | None = _AUTO_CONF,
    epochs: int | None = _EPOCHS,
    image_size: int | None = _IMAGE_SIZE,
    batch_size: int | None = _BATCH,
    device: str | None = _DEVICE,
    min_images: int | None = _MIN_IMAGES,
    min_instances: int | None = _MIN_INSTANCES,
    min_map50: float | None = _MIN_MAP50,
    target: str | None = _TARGET,
    precision: str | None = _PRECISION,
    confidence: float | None = _CONFIDENCE,
    sets: list[str] = _SET,
) -> None:
    """Scaffold projects/<name>/project.yaml from a template, apply the given settings
    (validated before anything is written) and initialise the project state."""

    def op(cli: CliContext) -> Envelope:
        changes = _changes(
            description=description,
            classes=classes,
            source=source,
            weights=weights,
            auto_label_weights=auto_label_weights,
            auto_label_map=auto_label_map,
            auto_label_confidence=auto_label_confidence,
            epochs=epochs,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            min_images=min_images,
            min_instances=min_instances,
            min_map50=min_map50,
            target=target,
            precision=precision,
            confidence=confidence,
            sets=sets,
        )
        project = Project.create(
            name, cli.settings, task_type=task, template=template, changes=changes
        )
        status = project_status(project)
        cfg = project.config
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
                "settings_applied": [p for p, _ in changes],
                "config": _config_summary(project),
            },
            next_operations=[f"project validate {name}", *status["next_operations"]],
            human_text=(
                f"Created project {name!r}\n"
                f"  config:    {project.config_path}\n"
                f"  data:      {project.data_dir}\n"
                f"  artifacts: {project.artifact_dir}\n"
                f"  classes={cfg.dataset.classes} target={cfg.deployment.target} "
                f"epochs={cfg.training.epochs}\n"
                + (
                    "Adjust with `project configure`, then validate."
                    if changes
                    else "Edit project.yaml or use `project configure`, then validate."
                )
            ),
        )

    run_command(ctx, "project.create", op, project=name)


def _config_summary(project: Project) -> dict[str, Any]:
    cfg = project.config
    return {
        "classes": cfg.dataset.classes,
        "sources": [s.model_dump() for s in cfg.dataset.sources],
        "candidates": [c.model_dump() for c in cfg.model.candidates],
        "auto_label": cfg.labeling.auto.model_dump() if cfg.labeling.auto else None,
        "training": {
            "epochs": cfg.training.epochs,
            "image_size": cfg.training.image_size,
            "batch_size": cfg.training.batch_size,
            "device": cfg.training.device,
            "backend": cfg.training.backend,
        },
        "evaluation_acceptance": {
            k: v.model_dump(exclude_none=True) for k, v in cfg.evaluation.acceptance.items()
        },
        "deployment": {
            "target": cfg.deployment.target,
            "precision": cfg.deployment.precision,
            "inference": cfg.deployment.inference.model_dump(),
            "acceptance": {
                k: v.model_dump(exclude_none=True) for k, v in cfg.deployment.acceptance.items()
            },
        },
    }


@app.command("configure")
def configure(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    description: str | None = _DESCRIPTION,
    classes: str | None = _CLASSES,
    source: str | None = _SOURCE,
    weights: str | None = _WEIGHTS,
    auto_label_weights: str | None = _AUTO_WEIGHTS,
    auto_label_map: str | None = _AUTO_MAP,
    auto_label_confidence: float | None = _AUTO_CONF,
    epochs: int | None = _EPOCHS,
    image_size: int | None = _IMAGE_SIZE,
    batch_size: int | None = _BATCH,
    device: str | None = _DEVICE,
    min_images: int | None = _MIN_IMAGES,
    min_instances: int | None = _MIN_INSTANCES,
    min_map50: float | None = _MIN_MAP50,
    target: str | None = _TARGET,
    precision: str | None = _PRECISION,
    confidence: float | None = _CONFIDENCE,
    sets: list[str] = _SET,
) -> None:
    """Change supported settings of project.yaml. The new configuration is validated (schema,
    task vocabulary, infrastructure references) before the file is rewritten; a rejected
    change leaves the file untouched (exit 2, CONFIG_INVALID)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        has_auto = project.config.labeling.auto is not None
        changes = _changes(
            has_auto,
            description=description,
            classes=classes,
            source=source,
            weights=weights,
            auto_label_weights=auto_label_weights,
            auto_label_map=auto_label_map,
            auto_label_confidence=auto_label_confidence,
            epochs=epochs,
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            min_images=min_images,
            min_instances=min_instances,
            min_map50=min_map50,
            target=target,
            precision=precision,
            confidence=confidence,
            sets=sets,
        )
        if not changes:
            raise ConfigError(
                "nothing to change: pass at least one option",
                code="NO_CHANGES",
                hint="See `project configure --help`; e.g. --classes cat,dog --epochs 30.",
            )
        _cfg, diff = configure_file(project.config_path, changes, infra=cli.settings.infra)
        project = Project.load(name, cli.settings)  # re-read what is now on disk
        lines = [f"{d['path']}: {d['old']!r} -> {d['new']!r}" for d in diff] or ["(no change)"]
        return Envelope(
            operation="project.configure",
            project=name,
            result={
                "config_path": str(project.config_path),
                "changed": bool(diff),
                "changes": diff,
                "config": _config_summary(project),
            },
            next_operations=[
                f"project validate {name}",
                *project_status(project)["next_operations"],
            ],
            human_text="\n".join(lines),
        )

    run_command(ctx, "project.configure", op, project=name)


@app.command("show")
def show(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """Print the effective (validated) project configuration."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        cfg = project.config.model_dump(exclude_none=True)
        return Envelope(
            operation="project.show",
            project=name,
            result={"config_path": str(project.config_path), "config": cfg},
            human_text=project.config_path.read_text(encoding="utf-8"),
        )

    run_command(ctx, "project.show", op, project=name)


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
