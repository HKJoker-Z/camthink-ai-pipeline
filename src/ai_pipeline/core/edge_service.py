"""Edge application lifecycle: what actually runs on each device, and start/stop/restart of
an *installed* deployment's inference service without redeploying.

The device is the source of truth: its ``current`` pointer names a ``<project>-dep-NNN``
reference, which is mapped back to the owning project's records (model version, export,
pointers). Service actions never stage, build or re-point anything except the explicit,
approval-gated ``switch`` (re-point ``current`` to this project's already installed deployment
through the target's ``rollback`` primitive — no engine rebuild).
"""

from __future__ import annotations

import re
import time
from typing import Any

from ai_pipeline.core.deployment import _ref_of, wait_healthy
from ai_pipeline.core.errors import (
    ConfigError,
    OperationError,
    PipelineError,
    RemoteError,
    StateError,
)
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.settings import Settings
from ai_pipeline.targets import get_target
from ai_pipeline.targets.base import DeploymentTarget

log = get_logger("edge_service")

REF_RE = re.compile(r"^(?P<project>[a-z][a-z0-9_]*)-(?P<dep>dep-\d+)$")
STOP_POLL_S = 30.0


def parse_ref(ref: str | None) -> tuple[str | None, str | None]:
    """``v1_person-dep-003`` -> (``v1_person``, ``dep-003``); legacy ``dep-002`` -> (None, dep)."""
    if not ref:
        return None, None
    m = REF_RE.match(ref)
    if m:
        return m.group("project"), m.group("dep")
    return None, ref if ref.startswith("dep-") else None


def normalize_service_state(systemd_state: str | None, stopped: Any, *, service: str | None) -> str:
    """running | stopped (deliberate) | failed | inactive (unexpected) | unknown | none."""
    if service is None:
        return "none"
    if systemd_state == "active":
        return "running"
    if systemd_state in ("inactive", "failed", "deactivating", "activating", "unknown", None):
        if stopped:
            return "stopped"
        if systemd_state == "failed":
            return "failed"
        if systemd_state in ("inactive", "deactivating"):
            return "inactive"
        return "unknown"
    return "unknown"


def _supports_service(target: DeploymentTarget) -> bool:
    return callable(getattr(target, "service_status", None))


def _project_view(
    settings: Settings, project_name: str | None, dep_id: str | None
) -> dict[str, Any]:
    """What the owning project knows about the deployment the device runs (best effort)."""
    out: dict[str, Any] = {
        "project": project_name,
        "deployment_id": dep_id,
        "model_version": None,
        "export_id": None,
        "deployment_state": None,
        "project_active_deployment": None,
        "project_known": False,
    }
    if not project_name or not Project.exists(project_name, settings):
        return out
    try:
        project = Project.load(project_name, settings)
    except PipelineError as exc:
        out["project_error"] = exc.to_dict()
        return out
    out["project_known"] = True
    out["project_active_deployment"] = project.pointers()["active_deployment"]
    if dep_id and project.store.exists("deployments", dep_id):
        rec = project.store.read("deployments", dep_id)
        out.update(
            model_version=rec["model_version"],
            export_id=rec["export_id"],
            deployment_state=rec["state"],
            precision=rec.get("precision"),
        )
    return out


def device_status(settings: Settings, target_name: str) -> dict[str, Any]:
    """Live state of one device: current deployment, owning project, service, heartbeat, health."""
    cfg = settings.infra.deployment_targets[target_name]
    out: dict[str, Any] = {
        "target": target_name,
        "type": cfg.type,
        "reachable": None,
        "current": None,
        "project": None,
        "deployment_id": None,
        "model_version": None,
        "service": {"name": None, "state": "unknown", "systemd_state": None, "stopped": None},
        "heartbeat": None,
        "healthy": None,
        "health_reason": None,
        "consistent": None,
        "error": None,
        "at": utc_now_iso(),
    }
    try:
        target = get_target(target_name, settings)
    except (ConfigError, OperationError) as exc:
        out.update(reachable=False, error=exc.to_dict())
        out["service"]["state"] = "unknown"
        return out
    if not _supports_service(target):
        out.update(reachable=None, error={"code": "TARGET_NO_SERVICE_CONTROL"})
        return out
    try:
        st = target.service_status()
    except RemoteError as exc:
        out.update(reachable=False, error=exc.to_dict())
        out["service"]["state"] = "unreachable"
        return out
    out["reachable"] = True
    out["current"] = st.get("current")
    out["service"] = {
        "name": st.get("service"),
        "state": normalize_service_state(
            st.get("systemd_state"), st.get("stopped"), service=st.get("service")
        ),
        "systemd_state": st.get("systemd_state"),
        "stopped": st.get("stopped"),
    }
    hb = st.get("heartbeat") or None
    if hb:
        out["heartbeat"] = {
            k: hb.get(k)
            for k in (
                "deployment",
                "ts",
                "fps",
                "frames",
                "errors",
                "state",
                "last_detection_count",
            )
            if k in hb
        }
    try:
        hc = target.health_check()
        out["healthy"] = bool(hc.get("healthy"))
        out["health_reason"] = (hc.get("details") or {}).get("reason")
    except PipelineError as exc:
        out["healthy"] = False
        out["health_reason"] = exc.message
    project_name, dep_id = parse_ref(out["current"])
    view = _project_view(settings, project_name, dep_id)
    out.update(
        project=view["project"],
        deployment_id=view["deployment_id"],
        model_version=view["model_version"],
        export_id=view["export_id"],
        deployment_state=view["deployment_state"],
        project_active_deployment=view["project_active_deployment"],
    )
    if view.get("project_error"):
        out["project_error"] = view["project_error"]
    if out["current"] is None:
        out["consistent"] = None
    else:
        out["consistent"] = view["project_known"] and view["project_active_deployment"] == dep_id
    return out


def fleet_status(settings: Settings, *, names: list[str] | None = None) -> list[dict[str, Any]]:
    names = names or sorted(settings.infra.deployment_targets)
    return [device_status(settings, n) for n in names]


def resolve_target_name(project: Project, target_name: str | None) -> str:
    if target_name:
        return target_name
    active = project.pointers()["active_deployment"]
    if active and project.store.exists("deployments", active):
        return project.store.read("deployments", active)["target"]
    return project.config.deployment.target


def _wait_state(target: DeploymentTarget, *, want_active: bool, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        st = target.service_status()
        active = st.get("systemd_state") == "active"
        if active == want_active or time.monotonic() >= deadline:
            return st
        time.sleep(1.0)


def _record_event(project: Project, dep_id: str, event: dict[str, Any]) -> None:
    rec = project.store.read("deployments", dep_id)
    rec.setdefault("service_events", []).append(event)
    rec["timestamps"]["updated"] = utc_now_iso()
    project.store.write("deployments", dep_id, rec)


def service_action(
    project: Project,
    action: str,
    *,
    target_name: str | None = None,
    switch: bool = False,
) -> dict[str, Any]:
    """``stop`` / ``start`` / ``restart`` the inference service for this project's active
    deployment on its device. Refuses to act on another project's deployment (the device's
    ``current`` wins over pointers); ``switch`` lets ``start`` re-point the device to this
    project's installed deployment (approval-gated by policy, not by code)."""
    if action not in ("stop", "start", "restart"):
        raise ValueError(action)
    name = resolve_target_name(project, target_name)
    target = get_target(name, project.settings)
    if not _supports_service(target):
        raise OperationError(
            f"target {name} does not support service control",
            code="TARGET_NO_SERVICE_CONTROL",
            details={"target": name, "type": target.type},
        )
    pointers = project.pointers()
    active_id = pointers["active_deployment"]
    if not active_id:
        raise StateError(
            f"project {project.name} has no active deployment to {action}",
            code="NO_ACTIVE_DEPLOYMENT",
            details={"pointers": pointers},
            hint=f"`deploy start {project.name}` installs one; `target status --target {name}` "
            "shows what the device runs.",
        )
    rec = project.store.read("deployments", active_id)
    my_ref = _ref_of(project, active_id)
    st = target.service_status()
    if st.get("service") is None:
        raise OperationError(
            f"target {name} has no inference service configured",
            code="TARGET_NO_SERVICE",
            details={"target": name},
        )
    current = st.get("current")
    owner, owner_dep = parse_ref(current)
    before = normalize_service_state(
        st.get("systemd_state"), st.get("stopped"), service=st["service"]
    )
    blog = log.bind(project=project.name, target=name, deployment_id=active_id, action=action)
    result: dict[str, Any] = {
        "target": name,
        "action": action,
        "deployment_id": active_id,
        "target_ref": my_ref,
        "model_version": rec["model_version"],
        "export_id": rec["export_id"],
        "service": {"name": st["service"], "state_before": before},
        "changed": False,
        "switched_from": None,
    }

    def occupied(hint: str) -> StateError:
        return StateError(
            f"device {name} currently runs {current}, not {my_ref}",
            code="DEVICE_OCCUPIED",
            details={
                "target": name,
                "target_current": current,
                "current_project": owner,
                "current_deployment": owner_dep,
                "requested_ref": my_ref,
            },
            hint=hint,
        )

    hc = project.config.deployment.health
    if action == "stop":
        if current != my_ref:
            raise occupied(
                f"Stop it through its owner: `service stop {owner}`"
                if owner
                else "Nothing of this project is installed as current."
            )
        if before == "running":
            result["stop"] = target.service_stop()
            result["changed"] = True
        else:
            result["stop"] = {"previous_state": st.get("systemd_state"), "already": True}
        final = _wait_state(target, want_active=False, timeout_s=STOP_POLL_S)
        after = normalize_service_state(
            final.get("systemd_state"), final.get("stopped"), service=st["service"]
        )
        result["service"].update(state=after, systemd_state=final.get("systemd_state"))
        result["stopped"] = final.get("stopped")
        result["current_after"] = final.get("current")
        if after == "running":
            raise OperationError(
                f"service on {name} is still active after stop",
                code="SERVICE_STOP_FAILED",
                details=result,
            )
        blog.info("service stopped (was %s)", before)
    elif action == "start":
        if current == my_ref:
            if before == "running":
                result["start"] = {"already": True}
            else:
                result["start"] = target.service_start()
                result["changed"] = True
        elif current is None or switch:
            # Re-point `current` to this project's installed deployment and restart the
            # service: the target's rollback primitive, which checks the engine exists.
            try:
                result["start"] = target.rollback(my_ref)
            except (RemoteError, OperationError) as exc:
                raise StateError(
                    f"deployment {my_ref} is not installed on {name} (no engine); "
                    "a new `deploy start` is required",
                    code="DEPLOYMENT_NOT_INSTALLED",
                    details={"target": name, "target_ref": my_ref, **exc.details},
                ) from exc
            result["changed"] = True
            result["switched_from"] = current
        else:
            raise occupied(
                f"Another project's application is live. With approval: "
                f"`service start {project.name} --switch` re-points the device to {my_ref} "
                "(installed engine, no rebuild)."
            )
        health = wait_healthy(target, hc.timeout_s, hc.interval_s)
        result["health"] = health
        final = target.service_status()
        result["service"].update(
            state=normalize_service_state(
                final.get("systemd_state"), final.get("stopped"), service=st["service"]
            ),
            systemd_state=final.get("systemd_state"),
        )
        result["current_after"] = final.get("current")
        if not health.get("healthy") or final.get("current") != my_ref:
            raise OperationError(
                f"service on {name} did not become healthy for {my_ref}",
                code="SERVICE_START_FAILED",
                details=result,
                hint=f"`deploy status {project.name} {active_id} --full`; the device log is "
                "`journalctl --user -u <service>` on the device.",
            )
        blog.info("service running (was %s, switched_from=%s)", before, result["switched_from"])
    else:  # restart
        if current != my_ref:
            raise occupied(
                f"Restart it through its owner: `service restart {owner}`"
                if owner
                else "Nothing of this project is installed as current."
            )
        result["restart"] = target.service_restart()
        result["changed"] = True
        health = wait_healthy(target, hc.timeout_s, hc.interval_s)
        result["health"] = health
        final = target.service_status()
        result["service"].update(
            state=normalize_service_state(
                final.get("systemd_state"), final.get("stopped"), service=st["service"]
            ),
            systemd_state=final.get("systemd_state"),
        )
        result["current_after"] = final.get("current")
        if not health.get("healthy") or final.get("current") != my_ref:
            raise OperationError(
                f"service on {name} did not recover after restart for {my_ref}",
                code="SERVICE_RESTART_FAILED",
                details=result,
            )
        blog.info("service restarted")
    _record_event(
        project,
        active_id,
        {
            "action": action,
            "at": utc_now_iso(),
            "changed": result["changed"],
            "switched_from": result["switched_from"],
            "service_state": result["service"].get("state"),
        },
    )
    return result
