"""``ai-pipeline service ...`` — start/stop/restart the inference service of an installed
deployment (no redeploy, no engine rebuild) and ``target status`` (device-level truth)."""

from __future__ import annotations

from typing import Any

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core.edge_service import (
    device_status,
    fleet_status,
    resolve_target_name,
    service_action,
)
from ai_pipeline.core.project import Project
from ai_pipeline.core.status import project_status

app = typer.Typer(
    help="Control the inference service of an installed deployment (no redeploy).",
    no_args_is_help=True,
)

_TARGET = typer.Option(None, "--target", help="Target name (default: active deployment's).")


def device_text(d: dict[str, Any]) -> str:
    svc = d["service"]
    if d.get("reachable") is False:
        return f"{d['target']}: UNREACHABLE ({(d.get('error') or {}).get('message')})"
    hb = d.get("heartbeat") or {}
    return (
        f"{d['target']}: current={d.get('current')} project={d.get('project')} "
        f"deployment={d.get('deployment_id')} model={d.get('model_version')} "
        f"service={svc['state']} ({svc.get('systemd_state')}) healthy={d.get('healthy')} "
        f"consistent={d.get('consistent')}"
        + (f" fps={hb.get('fps')} frames={hb.get('frames')}" if hb else "")
        + (f"\n    reason: {d['health_reason']}" if d.get("health_reason") else "")
    )


def _next_for_device(d: dict[str, Any]) -> list[str]:
    p = d.get("project")
    if not p:
        return []
    if d["service"]["state"] in ("stopped", "inactive", "failed"):
        return [f"service start {p}", f"deploy status {p} {d.get('deployment_id')} --full"]
    return [f"health {p}", f"deploy status {p}"]


def target_status_cmd(
    ctx: typer.Context,
    name: str | None = typer.Argument(
        None, help="Project name (default: every target in infrastructure.yaml)."
    ),
    target: str | None = _TARGET,
) -> None:
    """What each device actually runs: current deployment, owning project, model, service
    state (running | stopped | failed | inactive | unreachable), heartbeat, health,
    consistency with the project's pointers. Read-only; exit 0 even when stopped/unreachable."""

    def op(cli: CliContext) -> Envelope:
        if name is None and target is None:
            devices = fleet_status(cli.settings)
            return Envelope(
                operation="target.status",
                result={"targets": devices},
                next_operations=[op for d in devices for op in _next_for_device(d)],
                human_text="\n".join(device_text(d) for d in devices) or "(no targets)",
            )
        tname = target
        project = None
        if name is not None:
            project = Project.load(name, cli.settings)
            tname = resolve_target_name(project, target)
        d = device_status(cli.settings, tname)  # type: ignore[arg-type]
        result: dict[str, Any] = dict(d)
        if project is not None:
            result["project_pointers"] = project.pointers()
            result["this_project_is_current"] = d.get("project") == project.name
        return Envelope(
            operation="target.status",
            project=name,
            result=result,
            next_operations=_next_for_device(d)
            or (project_status(project)["next_operations"] if project else []),
            human_text=device_text(d),
        )

    run_command(ctx, "target.status", op, project=name)


def _action_text(r: dict[str, Any]) -> str:
    svc = r["service"]
    line = (
        f"{r['action']} {r['target_ref']} (model {r['model_version']}) on {r['target']}: "
        f"service {svc.get('state_before')} -> {svc.get('state')}"
        + ("" if r["changed"] else " (no change)")
    )
    if r.get("switched_from"):
        line += f"\n  switched device from {r['switched_from']}"
    if r.get("health") is not None:
        line += f"\n  healthy={r['health'].get('healthy')} current={r.get('current_after')}"
    return line


def _service_envelope(project: Project, result: dict[str, Any], operation: str) -> Envelope:
    return Envelope(
        operation=operation,
        project=project.name,
        result=result,
        next_operations=[f"target status {project.name}", f"health {project.name}"]
        if result["action"] != "stop"
        else [f"target status {project.name}", f"service start {project.name}"],
        human_text=_action_text(result),
    )


@app.command("status")
def status_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = _TARGET,
) -> None:
    """Alias of `target status <project>`."""
    target_status_cmd(ctx, name, target)


@app.command("stop")
def stop_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = _TARGET,
) -> None:
    """Stop the inference service running this project's active deployment. Nothing is
    removed: engine, artifacts, pointers and rollback history stay; `service start` resumes.
    Refused (exit 3, DEVICE_OCCUPIED) when the device runs another project's deployment."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        result = service_action(project, "stop", target_name=target)
        return _service_envelope(project, result, "service.stop")

    run_command(ctx, "service.stop", op, project=name)


@app.command("start")
def start_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = _TARGET,
    switch: bool = typer.Option(
        False,
        "--switch",
        help="If another project's deployment is current, re-point the device to this project's "
        "installed deployment (no rebuild). Requires operator approval.",
    ),
) -> None:
    """Start the already installed deployment's service (no staging, no TensorRT build) and
    wait until it is healthy. Idempotent when it is already running."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        result = service_action(project, "start", target_name=target, switch=switch)
        return _service_envelope(project, result, "service.start")

    run_command(ctx, "service.start", op, project=name)


@app.command("restart")
def restart_cmd(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Project name."),
    target: str | None = _TARGET,
) -> None:
    """Restart the service of this project's current deployment and wait until healthy;
    verifies the same deployment is still current afterwards."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        result = service_action(project, "restart", target_name=target)
        return _service_envelope(project, result, "service.restart")

    run_command(ctx, "service.restart", op, project=name)
