"""``ai-pipeline deploy ...``, ``rollback``, ``health`` and ``target verify`` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.deployment import (
    deployment_summary,
    health,
    infer_on_target,
    list_deployments,
    raise_for_state,
    rollback_deployment,
    setup_target,
    start_deployment,
)
from ai_pipeline.core.errors import PipelineError
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import DeploymentState
from ai_pipeline.core.status import project_status
from ai_pipeline.targets import get_target

app = typer.Typer(
    help="Deploy exported models to a target; inspect deployments.", no_args_is_help=True
)
target_app = typer.Typer(help="Deployment target utilities.", no_args_is_help=True)


def _deployment_text(rec: dict[str, Any]) -> str:
    s = deployment_summary(rec)
    lines = [
        f"{s['deployment_id']}  state={s['state']}  model={s['model_version']}  "
        f"export={s['export_id']}  target={s['target']}  precision={s['precision']}",
        f"  phases: {' -> '.join(p['state'] for p in rec['phases'])}",
    ]
    if s["benchmark"]:
        lines.append(f"  benchmark: {s['benchmark']}")
    if s["acceptance"]:
        for c in s["acceptance"]["checks"]:
            verdict = "PASS" if c["passed"] else "FAIL"
            lines.append(f"    {verdict} {c['metric']}={c['value']} rule={c['rule']}")
        if not s["acceptance"]["checks"]:
            lines.append("    (no deployment.acceptance rules configured)")
    if s["health"] is not None:
        lines.append(f"  health: {'healthy' if s['health'] else 'UNHEALTHY'}")
    if s["rollback"]:
        rb = s["rollback"]
        lines.append(
            f"  rollback: {rb.get('trigger')} -> {rb.get('to')} "
            f"({'FAILED: ' + rb['error']['message'] if rb.get('error') else 'restored'})"
        )
    if s["error"]:
        e = s["error"]
        lines.append(f"  error [{e.get('code')}] in {e.get('phase')}: {e.get('message')}")
    return "\n".join(lines)


def deployment_envelope(project: Project, rec: dict[str, Any], operation: str) -> Envelope:
    status, error = "ok", None
    try:
        raise_for_state(rec)
    except PipelineError as exc:
        status = "rejected" if rec["state"] == DeploymentState.REJECTED.value else "failed"
        error = exc.to_dict()
    result = {**deployment_summary(rec), "phases": [p["state"] for p in rec["phases"]]}
    result["pointers"] = project.pointers()
    return Envelope(
        operation=operation,
        project=project.name,
        status=status,  # type: ignore[arg-type]
        result=result,
        error=error,
        next_operations=project_status(project)["next_operations"],
        human_text=_deployment_text(rec),
    )


@app.command("start")
def start_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    model_version: str | None = typer.Argument(
        None, help="Model version (default: selected; must be ACCEPTED)."
    ),
    export_id: str | None = typer.Option(
        None, "--export", help="Export id (default: latest READY for the model)."
    ),
    target: str | None = typer.Option(
        None, "--target", help="Target name (default: deployment.target)."
    ),
) -> None:
    """Deploy: verify -> stage -> prepare -> smoke -> benchmark -> policy -> activate -> health.
    Exit 0 ACTIVE, 5 policy rejected (not activated), 1 failed / rolled back."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rec = start_deployment(
            project, model_version=model_version, export_id=export_id, target_name=target
        )
        return deployment_envelope(project, rec, "deploy.start")

    run_command(ctx, "deploy.start", op, project=name)


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    deployment_id: str | None = typer.Argument(None, help="Deployment id (default: active)."),
    full: bool = typer.Option(False, "--full", help="Include the complete record."),
) -> None:
    """Show a deployment (default: the active one) with its phases, benchmark and verdicts."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        dep = deployment_id or project.pointers()["active_deployment"]
        if not dep:
            return Envelope(
                operation="deploy.status",
                project=name,
                result={"active_deployment": None, "pointers": project.pointers()},
                next_operations=project_status(project)["next_operations"],
                human_text="no active deployment",
            )
        rec = project.store.read("deployments", dep)
        result = {
            **deployment_summary(rec),
            "phases": [p["state"] for p in rec["phases"]],
            "pointers": project.pointers(),
        }
        if full:
            result["record"] = rec
        return Envelope(
            operation="deploy.status", project=name, result=result, human_text=_deployment_text(rec)
        )

    run_command(ctx, "deploy.status", op, project=name)


@app.command("list")
def list_cmd(ctx: typer.Context, name: str = typer.Argument(..., help="Project name.")) -> None:
    """List deployments."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rows = list_deployments(project)
        active = project.pointers()["active_deployment"]
        text = (
            "\n".join(
                f"{'*' if r['deployment_id'] == active else ' '} {r['deployment_id']}  "
                f"{r['state']:<16} {r['model_version']}  {r['export_id']}  {r['target']}"
                for r in rows
            )
            or "(no deployments)"
        )
        return Envelope(
            operation="deploy.list",
            project=name,
            result={"deployments": rows, "pointers": project.pointers()},
            human_text=text,
        )

    run_command(ctx, "deploy.list", op, project=name)


def rollback_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    to: str | None = typer.Option(None, "--to", help="Deployment to restore (default: previous)."),
) -> None:
    """Roll the active deployment back to the previous known-good one."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        rec = rollback_deployment(project, to=to)
        env = deployment_envelope(project, rec, "rollback")
        if rec["state"] == DeploymentState.ROLLED_BACK.value:
            env.status, env.error = "ok", None  # a rollback that worked is this command's success
        return env

    run_command(ctx, "rollback", op, project=name)


def health_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = typer.Option(
        None, "--target", help="Target name (default: active deployment's)."
    ),
) -> None:
    """Health-check the target; exit 1 when unhealthy or inconsistent with the active pointer."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        result = health(project, target_name=target)
        ok = result["healthy"] and result["consistent"]
        if result["healthy"]:
            message = (
                f"target current={result['target_current']} but "
                f"active_deployment={result['active_deployment']}"
            )
        else:
            message = result["details"].get("reason", "target unhealthy")
        stopped = result["details"].get("stopped_intentionally")
        if not result["healthy"]:
            code = "SERVICE_STOPPED" if stopped else "UNHEALTHY"
            hint = (
                f"The service was stopped deliberately (`service stop`); `service start {name}` "
                "resumes it."
                if stopped
                else f"See `deploy status {name}` and consider `rollback {name}`."
            )
        else:
            code = "POINTER_MISMATCH"
            hint = f"See `target status --target {result['target']}` for the owning project."
        return Envelope(
            operation="health",
            project=name,
            status="ok" if ok else "failed",
            result=result,
            error=None
            if ok
            else {"code": code, "message": message, "details": result["details"], "hint": hint},
            next_operations=project_status(project)["next_operations"],
            human_text=(
                f"target {result['target']}: {'healthy' if result['healthy'] else 'UNHEALTHY'}; "
                f"current={result['target_current']} "
                f"active_deployment={result['active_deployment']}"
            ),
        )

    run_command(ctx, "health", op, project=name)


@app.command("infer")
def infer_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    images: list[Path] = typer.Argument(
        ..., help="Local image files to run through the deployed model."
    ),
    deployment_id: str | None = typer.Option(
        None, "--deployment", help="Deployment id (default: active)."
    ),
    target: str | None = typer.Option(
        None, "--target", help="Target name (default: the deployment's)."
    ),
) -> None:
    """Run real inference on the target with the active (or given) deployment."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        result = infer_on_target(
            project,
            [p.expanduser() for p in images],
            target_name=target,
            deployment_id=deployment_id,
        )
        lines = []
        for img in result.get("images", []):
            lines.append(f"{Path(img['image']).name}: {len(img['detections'])} detection(s)")
            for d in img["detections"][:10]:
                lines.append(f"    {d['class_name']} {d['confidence']:.2f} {d['bbox_xyxy']}")
        return Envelope(
            operation="deploy.infer",
            project=name,
            result=result,
            human_text="\n".join(lines) + f"\ntotal detections: {result.get('total_detections')}",
        )

    run_command(ctx, "deploy.infer", op, project=name)


@target_app.command("setup")
def setup_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = typer.Option(
        None, "--target", help="Target name (default: deployment.target)."
    ),
) -> None:
    """Install/update the edge runtime bundle and inference service on the target (idempotent)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        info = setup_target(project, target)
        text = f"target {info['target']}: runtime {info['bundle'].get('runtime_version')} " + (
            "installed" if info["bundle"].get("updated") else "up to date"
        )
        for w in info.get("warnings", []):
            text += f"\n  warning: {w}"
        return Envelope(operation="target.setup", project=name, result=info, human_text=text)

    run_command(ctx, "target.setup", op, project=name)


@target_app.command("verify")
def verify_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = typer.Option(
        None, "--target", help="Target name (default: deployment.target)."
    ),
) -> None:
    """Verify the deployment target is reachable and report its environment."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        tname = target or project.config.deployment.target
        info = get_target(tname, cli.settings).verify()
        return Envelope(operation="target.verify", project=name, result={"target": tname, **info})

    run_command(ctx, "target.verify", op, project=name)
