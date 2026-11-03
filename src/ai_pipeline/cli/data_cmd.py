"""``ai-pipeline data ...`` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.data.importer import asset_store, import_images, resolve_sources
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Import and list source images.", no_args_is_help=True)


@app.command("import")
def import_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    source: Path | None = typer.Argument(
        None, help="Image file or directory. Default: dataset.sources of type local_images."
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
) -> None:
    """Import images into the immutable, content-addressed asset store."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        sources = resolve_sources(project, source)
        rec = import_images(project, sources, recursive=recursive)
        c = rec["counts"]
        return Envelope(
            operation="data.import",
            project=name,
            result={
                "import_id": rec["import_id"],
                "sources": rec["sources"],
                "counts": c,
                "total_assets": asset_store(project).count(),
                "rejected": rec["rejected"],
            },
            next_operations=project_status(project)["next_operations"],
            human_text=(
                f"import {rec['import_id']}: {c['images_new']} new, "
                f"{c['images_existing']} already present, {c['rejected']} rejected, "
                f"{c['skipped_non_image']} non-image files skipped"
                + (
                    "\n"
                    + "\n".join(f"  rejected {r['path']}: {r['reason']}" for r in rec["rejected"])
                    if rec["rejected"]
                    else ""
                )
            ),
        )

    run_command(ctx, "data.import", op, project=name)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List import events and the asset count."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        imports = sorted(
            (
                {k: r[k] for k in ("import_id", "created_at", "sources", "counts")}
                for r in project.store.list("imports")
            ),
            key=lambda r: r["created_at"],
        )
        total = asset_store(project).count()
        text = (
            "\n".join(
                f"{i['import_id']}  {i['created_at']}  new={i['counts']['images_new']} "
                f"existing={i['counts']['images_existing']} rejected={i['counts']['rejected']}"
                for i in imports
            )
            or "(no imports)"
        )
        return Envelope(
            operation="data.list",
            project=name,
            result={"imports": imports, "total_assets": total},
            human_text=f"{text}\nassets in store: {total}",
        )

    run_command(ctx, "data.list", op, project=name)
