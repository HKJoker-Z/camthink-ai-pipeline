"""``ai-pipeline label ...`` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.labeling.labelsets import import_labels, read_class_names
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Import and list label sets.", no_args_is_help=True)


@app.command("import")
def import_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    path: Path = typer.Argument(
        ..., help="Label file or directory (matched to images by filename stem)."
    ),
    fmt: str = typer.Option("yolo", "--format", help="Label format."),
    classes_file: Path | None = typer.Option(
        None,
        "--classes-file",
        help="Class names, one per line, in the label files' index order. "
        "Default: project dataset.classes order.",
    ),
    import_ids: list[str] = typer.Option(
        [], "--import", help="Restrict matching to these import ids (repeatable)."
    ),
) -> None:
    """Convert external label files to the canonical annotation store as an 'imported' label set."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        names = read_class_names(classes_file) if classes_file else None
        rec = import_labels(
            project, path.expanduser(), fmt=fmt, class_names=names, import_ids=import_ids or None
        )
        c = rec["counts"]
        warn = ""
        if c["parse_issues"]:
            warn += (
                f"\n  {c['parse_issues']} label line(s) could not be parsed "
                "(see `dataset validate`)"
            )
        if c["unmatched_label_files"]:
            warn += f"\n  {c['unmatched_label_files']} label file(s) matched no imported image"
        return Envelope(
            operation="label.import",
            project=name,
            result={
                "label_set_id": rec["label_set_id"],
                "format": fmt,
                "counts": c,
                "class_names": rec["source"]["class_names"],
                "parse_issues": rec["parse_issues"][:20],
                "unmatched_label_files": rec["unmatched_label_files"][:20],
            },
            next_operations=project_status(project)["next_operations"],
            human_text=(
                f"label set {rec['label_set_id']}: {c['annotations']} annotations on "
                f"{c['assets_with_annotations']} images; "
                f"{c['assets_without_annotations']} images unlabeled" + warn
            ),
        )

    run_command(ctx, "label.import", op, project=name)


@app.command("auto")
def auto_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    import_ids: list[str] = typer.Option(
        [], "--import", help="Restrict to these import ids (repeatable)."
    ),
    weights: str | None = typer.Option(None, "--weights", help="Override labeling.auto.weights."),
    confidence: float | None = typer.Option(
        None, "--confidence", help="Override labeling.auto.confidence."
    ),
    backend: str | None = typer.Option(None, "--backend", help="Override labeling.auto.backend."),
    wait: bool = typer.Option(False, "--wait", help="Block until the run finishes."),
    timeout: float = typer.Option(3600.0, "--timeout", help="Seconds to wait with --wait."),
) -> None:
    """Auto-label assets with a pretrained model via a job; on completion a new label set
    (source.kind = model) is created. Nothing is built or merged automatically."""
    from ai_pipeline.cli.run_cmd import run_envelope
    from ai_pipeline.core.runs import start_autolabel, wait_run

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        run = start_autolabel(
            project,
            import_ids=import_ids or None,
            weights=weights,
            confidence=confidence,
            backend=backend,
        )
        if wait:
            run = wait_run(project, run["run_id"], timeout_s=timeout)
        env = run_envelope(project, run, "label.auto")
        if run.get("label_set_id"):
            rec = project.store.read("labels", run["label_set_id"])
            env.result["label_set"] = {
                "label_set_id": rec["label_set_id"],
                "counts": rec["counts"],
                "source": rec["source"],
            }
            env.human_text = (env.human_text or "") + (
                f"\n  label set {rec['label_set_id']}: {rec['counts']['annotations']} "
                f"annotations on {rec['counts']['assets_with_annotations']} images "
                f"({rec['source'].get('dropped_unmapped', 0)} unmapped detections dropped)"
            )
        return env

    run_command(ctx, "label.auto", op, project=name)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List label sets."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rows = sorted(
            (
                {
                    "label_set_id": r["label_set_id"],
                    "state": r["state"],
                    "created_at": r["created_at"],
                    "source_kind": r["source"]["kind"],
                    "format": r["source"].get("format"),
                    "run_id": r.get("run_id"),
                    "parent_label_set_id": r.get("parent_label_set_id"),
                    "counts": r["counts"],
                }
                for r in project.store.list("labels")
            ),
            key=lambda r: r["created_at"],
        )
        text = (
            "\n".join(
                f"{r['label_set_id']}  {r['state']:<6} {r['source_kind']}/"
                f"{r['format'] or r['run_id'] or '-'}  "
                f"annotations={r['counts']['annotations']} issues={r['counts']['parse_issues']}"
                for r in rows
            )
            or "(no label sets)"
        )
        return Envelope(
            operation="label.list", project=name, result={"label_sets": rows}, human_text=text
        )

    run_command(ctx, "label.list", op, project=name)
