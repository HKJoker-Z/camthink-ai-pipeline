"""``ai-pipeline dataset ...`` commands."""

from __future__ import annotations

import json

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.dataset.build import (
    DatasetPlan,
    build_dataset,
    dataset_dir,
    list_datasets,
    plan_dataset,
)
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(help="Validate, build and inspect dataset versions.", no_args_is_help=True)

_LABELS_OPT = typer.Option(None, "--labels", help="Label set id (default: latest READY).")
_IMPORTS_OPT = typer.Option([], "--import", help="Restrict to these import ids (repeatable).")


def _report_text(plan: DatasetPlan, max_issues: int = 30) -> str:
    rep = plan.report
    s = rep.stats
    lines = [
        f"images={s.get('images')} annotations={s.get('annotations')} "
        f"empty={s.get('empty_images')} splits={s.get('split_counts')}",
        f"classes={s.get('class_distribution')}",
    ]
    grouping = s.get("grouping") or {}
    if grouping.get("mode") not in (None, "image"):
        lines.append(
            f"grouping={grouping.get('mode')} groups={grouping.get('n_groups')} "
            f"clips={grouping.get('n_clips')} cuts={len(grouping.get('cuts') or [])}"
        )
    lines.append(
        f"errors={len(rep.errors)} warnings={len(rep.warnings)} "
        f"(fail_on_warnings={plan.fail_on_warnings}) -> {'VALID' if plan.valid else 'INVALID'}"
    )
    for issue in rep.issues[:max_issues]:
        where = f" [{issue.asset_id[:12]}]" if issue.asset_id else ""
        lines.append(f"  {issue.severity.value:<7} {issue.code}{where}: {issue.message}")
    if len(rep.issues) > max_issues:
        lines.append(f"  ... {len(rep.issues) - max_issues} more (use --output json)")
    return "\n".join(lines)


@app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    labels: str | None = _LABELS_OPT,
    import_ids: list[str] = _IMPORTS_OPT,
    verify_assets: bool = typer.Option(False, "--verify-assets", help="Re-hash stored images."),
) -> None:
    """Run task-owned validation on the current assets + label set without building anything."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        plan = plan_dataset(
            project, label_set_id=labels, import_ids=import_ids or None, verify_assets=verify_assets
        )
        result = {
            "label_set_id": plan.label_set_id,
            "content_hash": plan.content_hash,
            "existing_version": plan.existing_version,
            "report": plan.report.to_dict(fail_on_warnings=plan.fail_on_warnings),
        }
        if plan.valid:
            return Envelope(
                operation="dataset.validate",
                project=name,
                result=result,
                next_operations=[f"dataset build {name}"],
                human_text=_report_text(plan),
            )
        return Envelope(
            operation="dataset.validate",
            project=name,
            status="failed",
            result=result,
            error={
                "code": "DATASET_INVALID",
                "message": (
                    f"{len(plan.report.errors)} error(s), {len(plan.report.warnings)} warning(s)"
                ),
                "details": {"issues_by_code": plan.report.summary_by_code()},
                "hint": "Fix the reported labels/images, or adjust dataset.validation thresholds.",
            },
            human_text=_report_text(plan),
        )

    run_command(ctx, "dataset.validate", op, project=name)


@app.command("build")
def build_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    labels: str | None = _LABELS_OPT,
    import_ids: list[str] = _IMPORTS_OPT,
) -> None:
    """Validate, then build an immutable dataset version (idempotent on content)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        plan = plan_dataset(project, label_set_id=labels, import_ids=import_ids or None)
        record, created = build_dataset(project, plan)
        v = record["dataset_version"]
        return Envelope(
            operation="dataset.build",
            project=name,
            result={
                "dataset_version": v,
                "created": created,
                "content_hash": record["content_hash"],
                "state": record["state"],
                "path": str(dataset_dir(project, v)),
                "counts": record["counts"],
                "class_distribution": record["class_distribution"],
                "label_set_id": record["provenance"]["label_set_id"],
                "warnings": record["validation"]["warning_count"],
            },
            next_operations=project_status(project)["next_operations"],
            human_text=(
                (
                    f"built {v}"
                    if created
                    else f"{v} already exists with identical content; nothing built"
                )
                + f"\n  hash:   {record['content_hash'][:16]}\n  counts: {record['counts']}"
                + f"\n  path:   {dataset_dir(project, v)}"
            ),
        )

    run_command(ctx, "dataset.build", op, project=name)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List dataset versions."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rows = list_datasets(project)
        current = project.pointers()["current_dataset_version"]
        text = (
            "\n".join(
                f"{'*' if r['dataset_version'] == current else ' '} {r['dataset_version']}  "
                f"{r['state']:<8} {r['created_at']}  images={r['images']} "
                f"ann={r['annotations']} hash={r['content_hash']}"
                for r in rows
            )
            or "(no dataset versions)"
        )
        return Envelope(
            operation="dataset.list",
            project=name,
            result={"datasets": rows, "current_dataset_version": current},
            human_text=text,
        )

    run_command(ctx, "dataset.list", op, project=name)


@app.command("show")
def show_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    version: str = typer.Argument(..., help="Dataset version id, e.g. ds-v001."),
    manifest: bool = typer.Option(False, "--manifest", help="Include manifest entries."),
) -> None:
    """Show a dataset version's full record (provenance, config snapshot, validation)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        record = project.store.read("datasets", version)
        result = dict(record)
        result["path"] = str(dataset_dir(project, version))
        if manifest:
            mpath = dataset_dir(project, version) / record["files"]["manifest"]
            result["manifest"] = (
                [
                    json.loads(line)
                    for line in mpath.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if mpath.is_file()
                else []
            )
        return Envelope(operation="dataset.show", project=name, result=result)

    run_command(ctx, "dataset.show", op, project=name)
