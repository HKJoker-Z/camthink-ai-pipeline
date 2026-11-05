"""RTSP application + live view of an installed deployment (M9).

One device runs one project; this module binds every operation to a *project* and refuses to
touch a device whose ``current`` deployment belongs to another project (``DEVICE_OCCUPIED``, same
semantics as ``service`` actions), so a request for project A can never configure or display
project B's results.

Two device processes are involved and are controlled separately:

* the **inference service** (M7 ``service start|stop|restart``): decodes the camera, runs
  TensorRT, publishes annotated frames + stats to shared memory. Its behaviour is driven by
  ``<root>/service/app.json`` — the only place camera credentials are stored (mode 0600).
* the **live-view server** (``app liveview start|stop|restart``): serves the webpage/MJPEG from
  shared memory. Stopping it never affects inference; starting it never rebuilds or restarts
  anything.

Credentials: RTSP URLs arrive on stdin, travel to the device by scp, and are masked
(``rtsp://***@host/path``) in every result, log line and status document.
"""

from __future__ import annotations

import copy
import time
import urllib.error
import urllib.request
from typing import Any

from ai_pipeline.core.deployment import _ref_of
from ai_pipeline.core.edge_service import (
    normalize_service_state,
    parse_ref,
    resolve_target_name,
)
from ai_pipeline.core.errors import ConfigError, OperationError, StateError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.edge_bundle.camthink_edge import appcfg
from ai_pipeline.targets import get_target
from ai_pipeline.targets.base import DeploymentTarget

log = get_logger("edge_app")

URL_CHECK_TIMEOUT_S = 4.0
STALE_STATS_S = 5.0

INFERENCE_KNOBS = {
    "inference_fps": ("inference", "fps", float),
    "width": ("inference", "width", int),
    "height": ("inference", "height", int),
    "confidence": ("inference", "confidence", float),
    "backend": ("inference", "backend", str),
    "latency_ms": ("inference", "latency_ms", int),
    "display_fps": ("liveview", "display_fps", float),
    "quality": ("liveview", "quality", int),
    "port": ("liveview", "port", int),
    "bind": ("liveview", "bind", str),
}


class AppContext:
    """Project + device resolved once per command, with the ownership check."""

    def __init__(self, project: Project, target_name: str | None) -> None:
        self.project = project
        self.target_name = resolve_target_name(project, target_name)
        self.target: DeploymentTarget = get_target(self.target_name, project.settings)
        if not callable(getattr(self.target, "app_config_read", None)):
            raise OperationError(
                f"target {self.target_name} does not support RTSP application control",
                code="TARGET_NO_APP_CONTROL",
                details={"target": self.target_name, "type": self.target.type},
            )
        pointers = project.pointers()
        self.active_id: str | None = pointers["active_deployment"]
        self.my_ref: str | None = _ref_of(project, self.active_id) if self.active_id else None
        self._service_status: dict[str, Any] | None = None

    # -- device truth ---------------------------------------------------------------------------

    def service_status(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._service_status is None or refresh:
            self._service_status = self.target.service_status()
        return self._service_status

    def ownership(self) -> dict[str, Any]:
        st = self.service_status()
        current = st.get("current")
        owner, owner_dep = parse_ref(current)
        return {
            "target": self.target_name,
            "project": self.project.name,
            "deployment_id": self.active_id,
            "target_ref": self.my_ref,
            "device_current": current,
            "device_project": owner,
            "device_deployment": owner_dep,
            "matches": bool(self.my_ref) and current == self.my_ref,
        }

    def require_ownership(self, *, what: str) -> dict[str, Any]:
        own = self.ownership()
        if not self.active_id:
            raise StateError(
                f"project {self.project.name} has no active deployment",
                code="NO_ACTIVE_DEPLOYMENT",
                details=own,
                hint=f"`deploy start {self.project.name}` installs one first.",
            )
        if not own["matches"]:
            other = own["device_project"]
            raise StateError(
                f"device {self.target_name} currently runs {own['device_current']}, "
                f"not {self.my_ref}; refusing to {what} for {self.project.name}",
                code="DEVICE_OCCUPIED",
                details=own,
                hint=(
                    f"The device belongs to project {other}. Switching it to "
                    f"{self.project.name} interrupts {other} and needs approval: "
                    f"`service start {self.project.name} --switch`."
                    if other
                    else f"`deploy start {self.project.name}` or `service start "
                    f"{self.project.name} --switch` makes it current."
                ),
            )
        return own

    def inference_state(self) -> str:
        st = self.service_status()
        return normalize_service_state(
            st.get("systemd_state"), st.get("stopped"), service=st.get("service")
        )

    # -- app config -----------------------------------------------------------------------------

    def read_config(self) -> tuple[dict[str, Any], bool]:
        raw = self.target.app_config_read()
        return appcfg.merge_defaults(raw or {}), raw is not None

    def write_config(self, cfg: dict[str, Any]) -> None:
        problems = appcfg.validate(cfg)
        if problems:
            raise ConfigError(
                "invalid application configuration: " + "; ".join(problems),
                code="APP_CONFIG_INVALID",
                details={"problems": problems, "config": appcfg.masked(cfg)},
            )
        tcfg = self.project.settings.infra.deployment_targets[self.target_name]
        cfg["device"] = {
            "name": self.target_name,
            "label": getattr(tcfg, "device_label", None),
        }
        self.target.app_config_write(cfg)


def _restart_hints(ctx: AppContext, restart: dict[str, bool], state: str) -> list[str]:
    ops: list[str] = []
    p = ctx.project.name
    if restart.get("inference"):
        ops.append(
            f"service restart {p}"
            if state == "running"
            else f"service start {p}"
            if state in ("stopped", "inactive", "failed")
            else f"service restart {p}"
        )
    if restart.get("liveview"):
        ops.append(f"app liveview restart {p}")
    return ops


def _config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    m = appcfg.masked(cfg)
    return {
        "sources": m.get("sources", []),
        "inference": m.get("inference", {}),
        "liveview": m.get("liveview", {}),
    }


# -- sources ---------------------------------------------------------------------------------------


def source_set(
    project: Project,
    url: str,
    *,
    name: str = "cam0",
    add: bool = False,
    target_name: str | None = None,
    test_seconds: float | None = 5.0,
    restart: bool = False,
) -> dict[str, Any]:
    """Set (or add) a camera source. ``url`` comes from stdin — never from argv. Optionally
    probe it on the device first and/or restart the inference service to apply it."""
    url = url.strip()
    if not url:
        raise ConfigError("empty source URL", code="APP_SOURCE_EMPTY")
    ctx = AppContext(project, target_name)
    own = ctx.require_ownership(what="change the camera source")
    cfg, _ = ctx.read_config()
    before = copy.deepcopy(cfg)
    masked_url = appcfg.mask_url(url)
    blog = log.bind(project=project.name, target=ctx.target_name, source=masked_url)

    result: dict[str, Any] = {
        "target": ctx.target_name,
        "ownership": own,
        "source": {"name": name, "url": masked_url},
        "probe": None,
        "changed": False,
        "restart_required": {"inference": False, "liveview": False},
        "inference": {"state_before": ctx.inference_state()},
        "restarted": False,
    }
    if test_seconds:
        probe = ctx.target.source_probe(seconds=test_seconds, url=url)
        result["probe"] = _mask_probe(probe)
        if not probe.get("ok"):
            raise OperationError(
                f"camera source {masked_url} could not be decoded on {ctx.target_name}",
                code="SOURCE_UNREACHABLE",
                details={"probe": result["probe"], "source": masked_url},
                hint="Check the camera address, credentials and that the device can reach it "
                "(`app source test <project>` after fixing). Use `--no-test` to store it anyway.",
            )
    new_sources = [s for s in cfg["sources"] if s.get("name") != name] if add else []
    new_sources.append({"name": name, "url": url})
    cfg["sources"] = new_sources
    ctx.write_config(cfg)
    result["changed"] = True
    result["config"] = _config_summary(cfg)
    result["restart_required"] = appcfg.restart_required(before, cfg)
    state = result["inference"]["state_before"]
    if state in ("stopped", "inactive", "failed", "none", "unknown"):
        # Nothing is running, so there is nothing to restart; a start picks the config up.
        result["restart_required"]["inference"] = False
        result["applied"] = state not in ("running",)
    else:
        result["applied"] = not result["restart_required"]["inference"]
    blog.info("source %s set (restart_required=%s)", name, result["restart_required"])
    if restart and result["restart_required"]["inference"]:
        result["restart"] = ctx.target.service_restart()
        result["restarted"] = True
        result["applied"] = True
        result["restart_required"]["inference"] = False
        ctx.service_status(refresh=True)
    result["inference"]["state"] = ctx.inference_state()
    result["next_operations"] = _restart_hints(ctx, result["restart_required"], state) + [
        f"app status {project.name}",
        f"app liveview start {project.name}",
    ]
    return result


def source_clear(project: Project, *, target_name: str | None = None) -> dict[str, Any]:
    """Remove all camera sources (service falls back to the sample-image loop on restart)."""
    ctx = AppContext(project, target_name)
    own = ctx.require_ownership(what="clear the camera sources")
    cfg, existed = ctx.read_config()
    before = copy.deepcopy(cfg)
    removed = [appcfg.mask_url(s.get("url")) for s in cfg["sources"]]
    cfg["sources"] = []
    ctx.write_config(cfg)
    state = ctx.inference_state()
    rr = appcfg.restart_required(before, cfg)
    if state != "running":
        rr["inference"] = False
    return {
        "target": ctx.target_name,
        "ownership": own,
        "removed": removed,
        "changed": bool(removed) or not existed,
        "config": _config_summary(cfg),
        "restart_required": rr,
        "inference": {"state": state},
        "next_operations": _restart_hints(ctx, rr, state) + [f"app status {project.name}"],
    }


def source_show(project: Project, *, target_name: str | None = None) -> dict[str, Any]:
    """The device's application config with credentials masked (read-only)."""
    ctx = AppContext(project, target_name)
    cfg, existed = ctx.read_config()
    return {
        "target": ctx.target_name,
        "ownership": ctx.ownership(),
        "configured": existed,
        "config": _config_summary(cfg),
        "problems": appcfg.validate(cfg),
        "inference": {"state": ctx.inference_state()},
        "next_operations": [f"app source test {project.name}", f"app status {project.name}"],
    }


def _mask_probe(probe: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(probe)
    for s in out.get("sources", []):
        if "url" in s:
            s["url"] = appcfg.mask_url(s["url"])
    return out


def source_test(
    project: Project,
    *,
    url: str | None = None,
    seconds: float = 5.0,
    target_name: str | None = None,
) -> dict[str, Any]:
    """Decode a few frames on the device from the configured sources (or a candidate ``url``
    from stdin). Opens a second, temporary RTSP session next to the running service."""
    ctx = AppContext(project, target_name)
    cfg, existed = ctx.read_config()
    if url is None and not cfg["sources"]:
        raise StateError(
            f"no camera source configured on {ctx.target_name}",
            code="NO_SOURCE",
            details={"target": ctx.target_name, "configured": existed},
            hint=f"`app source set {project.name}` (URL on stdin) configures one.",
        )
    probe = _mask_probe(ctx.target.source_probe(seconds=seconds, url=url))
    probe.update(target=ctx.target_name, ownership=ctx.ownership(), candidate=url is not None)
    probe["next_operations"] = (
        [f"app source set {project.name}"] if url is not None else [f"app status {project.name}"]
    )
    return probe


# -- settings --------------------------------------------------------------------------------------


def configure(project: Project, *, target_name: str | None = None, **knobs: Any) -> dict[str, Any]:
    """Change inference / live-view settings. ``inference_fps``, ``confidence``, ``display_fps``
    and ``quality`` are picked up live; the rest report ``restart_required``."""
    changes = {k: v for k, v in knobs.items() if v is not None}
    unknown = sorted(set(changes) - set(INFERENCE_KNOBS))
    if unknown:
        raise ValueError(f"unknown knobs: {unknown}")
    ctx = AppContext(project, target_name)
    own = ctx.require_ownership(what="change application settings")
    cfg, _ = ctx.read_config()
    before = copy.deepcopy(cfg)
    applied: dict[str, Any] = {}
    for key, value in changes.items():
        section, field, cast = INFERENCE_KNOBS[key]
        if key == "confidence" and value in ("default", "none", "null", ""):
            cfg[section][field] = None
        else:
            cfg[section][field] = cast(value)
        applied[f"{section}.{field}"] = cfg[section][field]
    if not applied:
        raise ConfigError("nothing to change", code="APP_CONFIG_NOOP")
    ctx.write_config(cfg)
    state = ctx.inference_state()
    rr = appcfg.restart_required(before, cfg)
    if state != "running":
        rr["inference"] = False
    hot = {k: v for k, v in applied.items() if k not in appcfg.INFERENCE_RESTART_KEYS}
    return {
        "target": ctx.target_name,
        "ownership": own,
        "changed": applied,
        "hot_reloaded": {k: v for k, v in hot.items() if k not in appcfg.LIVEVIEW_RESTART_KEYS}
        if state == "running"
        else {},
        "config": _config_summary(cfg),
        "restart_required": rr,
        "inference": {"state": state},
        "next_operations": _restart_hints(ctx, rr, state) + [f"app metrics {project.name}"],
    }


# -- live view -------------------------------------------------------------------------------------


def check_url(url: str | None, *, timeout: float = URL_CHECK_TIMEOUT_S) -> dict[str, Any]:
    """GET ``<url>healthz`` from this machine (the AI server). Only a verified check may be
    reported as 'working'; browsers on other LAN segments are not covered by it."""
    out: dict[str, Any] = {"url": url, "verified": False, "http_status": None, "error": None}
    if not url:
        out["error"] = "no LAN address known for the device (set liveview_host)"
        return out
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=timeout) as resp:
            out["http_status"] = resp.status
            out["verified"] = resp.status == 200
    except urllib.error.HTTPError as exc:
        out["http_status"] = exc.code
        out["error"] = f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        out["error"] = str(getattr(exc, "reason", exc))
    return out


def _liveview_view(ctx: AppContext, *, verify: bool) -> dict[str, Any]:
    lv = ctx.target.liveview_status()
    state = lv.get("systemd_state")
    out = {
        "service": lv.get("service"),
        "state": "running"
        if state == "active"
        else "stopped"
        if state in ("inactive", "failed", "deactivating")
        else state or "unknown",
        "systemd_state": state,
        "enabled": lv.get("enabled"),
        "port": lv.get("port"),
        "url": lv.get("url"),
        "url_check": None,
    }
    if verify and lv.get("service"):
        out["url_check"] = check_url(lv.get("url")) if state == "active" else None
    return out


def liveview_action(
    project: Project, action: str, *, target_name: str | None = None
) -> dict[str, Any]:
    """``start`` / ``stop`` / ``restart`` the live-view web server only. Inference is untouched:
    use ``service stop|start|restart`` for that."""
    if action not in ("start", "stop", "restart"):
        raise ValueError(action)
    ctx = AppContext(project, target_name)
    if not callable(getattr(ctx.target, "liveview_start", None)):
        raise OperationError(
            f"target {ctx.target_name} has no live-view support",
            code="TARGET_NO_LIVEVIEW",
            details={"target": ctx.target_name},
        )
    own = ctx.require_ownership(what=f"{action} the live view")
    cfg, _ = ctx.read_config()
    warnings: list[str] = []
    if action in ("start", "restart"):
        if not cfg["sources"]:
            raise StateError(
                f"no camera source configured on {ctx.target_name}; the live view would have "
                "nothing to show",
                code="NO_SOURCE",
                details={"target": ctx.target_name},
                hint=f"`app source set {project.name}` (RTSP URL on stdin), then "
                f"`service restart {project.name}` if the service is running.",
            )
        inf_state = ctx.inference_state()
        hb = ctx.service_status().get("heartbeat") or {}
        if inf_state != "running":
            warnings.append(
                f"inference service is {inf_state}: the page will show 'inference stopped' "
                f"until `service start {project.name}`"
            )
        elif hb.get("mode") not in (None, "rtsp"):
            warnings.append(
                "inference service runs in sample-image mode (started before the camera was "
                f"configured): `service restart {project.name}` switches it to the camera"
            )
    before = _liveview_view(ctx, verify=False)
    blog = log.bind(project=project.name, target=ctx.target_name, action=action)
    changed = False
    if action == "stop":
        if before["state"] == "running":
            op = ctx.target.liveview_stop()
            changed = True
        else:
            op = {"already": True, "previous_state": before["systemd_state"]}
    elif action == "start":
        if before["state"] == "running":
            op = {"already": True, "previous_state": before["systemd_state"]}
        else:
            op = ctx.target.liveview_start()
            changed = True
    else:
        op = ctx.target.liveview_restart()
        changed = True
    blog.info("liveview %s (changed=%s)", action, changed)
    after = _liveview_view(ctx, verify=action != "stop")
    if action != "stop" and after["state"] != "running":
        raise OperationError(
            f"live-view service did not come up on {ctx.target_name} ({after['systemd_state']})",
            code="LIVEVIEW_START_FAILED",
            details={"target": ctx.target_name, "liveview": after, "operation": op},
            hint="On the device: `journalctl --user -u <liveview service> -n 50`.",
        )
    if after.get("url_check") and not after["url_check"]["verified"]:
        warnings.append(
            f"service is running but {after['url']} did not answer from the AI server: "
            f"{after['url_check']['error']} — check firewall/LAN routing before sharing the URL"
        )
    next_ops = (
        [f"app liveview start {project.name}"]
        if action == "stop"
        else [f"app status {project.name}", f"app metrics {project.name}"]
    )
    return {
        "target": ctx.target_name,
        "action": action,
        "ownership": own,
        "changed": changed,
        "operation": op,
        "liveview": after,
        "inference": {"state": ctx.inference_state(), "affected": False},
        "url": after["url"] if action != "stop" else None,
        "warnings": warnings,
        "next_operations": next_ops,
    }


# -- status + metrics ------------------------------------------------------------------------------


def _stats_fresh(stats: dict[str, Any] | None) -> bool:
    if not stats or not isinstance(stats.get("ts_epoch"), (int, float)):
        return False
    return (time.time() - float(stats["ts_epoch"])) <= STALE_STATS_S


def _camera_view(stats: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg["sources"]:
        return {"configured": False, "state": "unconfigured", "channels": []}
    if not stats:
        return {
            "configured": True,
            "state": "unknown",
            "reason": "no live stats from the inference service (not running in camera mode)",
            "channels": [],
        }
    chans = []
    for ch in stats.get("channels", []):
        src = ch.get("source") or {}
        chans.append(
            {
                "channel": ch.get("channel"),
                "name": ch.get("name"),
                "connected": src.get("connected"),
                "decode_fps": src.get("decode_fps"),
                "reconnects": src.get("reconnects"),
                "last_frame_age_s": src.get("last_frame_age_s"),
                "disconnected_for_s": src.get("disconnected_for_s"),
                "error": src.get("error"),
                "native_size": src.get("native_size"),
                "size": src.get("size"),
                "backend": src.get("backend"),
            }
        )
    connected = [c["connected"] for c in chans]
    state = (
        "connected"
        if chans and all(connected)
        else "disconnected"
        if chans and not any(connected)
        else "degraded"
        if chans
        else "unknown"
    )
    return {"configured": True, "state": state, "channels": chans}


def status(project: Project, *, target_name: str | None = None) -> dict[str, Any]:
    """Everything about the application on this project's device in one read-only call:
    ownership, inference state/mode, camera state, live-view service + verified URL, config."""
    ctx = AppContext(project, target_name)
    own = ctx.ownership()
    st = ctx.service_status()
    hb = st.get("heartbeat") or {}
    cfg, configured = ctx.read_config()
    stats = ctx.target.live_stats() if own["matches"] else None
    fresh = _stats_fresh(stats)
    inf_state = ctx.inference_state()
    liveview = (
        _liveview_view(ctx, verify=True)
        if callable(getattr(ctx.target, "liveview_status", None))
        else None
    )
    out: dict[str, Any] = {
        "target": ctx.target_name,
        "ownership": own,
        "inference": {
            "state": inf_state,
            "mode": hb.get("mode") or ("rtsp" if stats else None),
            "deployment": hb.get("deployment"),
            "heartbeat_ts": hb.get("ts"),
            "fps": hb.get("fps"),
            "frames": hb.get("frames"),
            "errors": hb.get("errors"),
            "stopped": st.get("stopped"),
        },
        "camera": _camera_view(stats if fresh else None, cfg),
        "liveview": liveview,
        "url": liveview.get("url") if liveview and liveview["state"] == "running" else None,
        "url_verified": bool(liveview and (liveview.get("url_check") or {}).get("verified")),
        "config": _config_summary(cfg) if configured else None,
        "config_restart_required": (stats or {}).get("config_restart_required") if fresh else None,
        "stats_fresh": fresh,
        "warnings": [],
        "at": utc_now_iso(),
    }
    if not own["matches"] and own["device_current"]:
        out["warnings"].append(
            f"device runs {own['device_current']} (project {own['device_project']}), not "
            f"{project.name}; camera/metrics shown here are withheld"
        )
    if out["config_restart_required"] and any(out["config_restart_required"].values()):
        out["warnings"].append("configuration changed; a restart is required to apply it")
    out["next_operations"] = _next_for_status(project.name, out)
    return out


def _next_for_status(p: str, s: dict[str, Any]) -> list[str]:
    ops: list[str] = []
    own = s["ownership"]
    if not own["matches"]:
        return [f"target status {p}", f"service start {p} --switch"]
    if not (s["config"] or {}).get("sources"):
        ops.append(f"app source set {p}")
    if s["inference"]["state"] != "running":
        ops.append(f"service start {p}")
    elif s["camera"]["state"] in ("disconnected", "degraded"):
        ops += [f"app source test {p}", f"service restart {p}"]
    rr = s.get("config_restart_required") or {}
    if rr.get("inference"):
        ops.append(f"service restart {p}")
    if rr.get("liveview"):
        ops.append(f"app liveview restart {p}")
    if s["liveview"] and s["liveview"]["state"] != "running":
        ops.append(f"app liveview start {p}")
    ops.append(f"app metrics {p}")
    return ops


def _p(d: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def metrics(project: Project, *, target_name: str | None = None) -> dict[str, Any]:
    """Real-time metrics from the inference service's stats (null = unavailable; nothing is
    estimated). Refuses to report another project's numbers."""
    ctx = AppContext(project, target_name)
    own = ctx.require_ownership(what="read metrics")
    stats = ctx.target.live_stats()
    fresh = _stats_fresh(stats)
    inf_state = ctx.inference_state()
    cfg, _ = ctx.read_config()
    sysm = _p(stats, "system") or {}
    chans = (stats or {}).get("channels", []) if fresh else []
    out: dict[str, Any] = {
        "target": ctx.target_name,
        "ownership": own,
        "inference_state": inf_state,
        "available": fresh,
        "reason": None
        if fresh
        else (
            "inference service is not running"
            if inf_state != "running"
            else "no fresh stats (service not in camera mode or still starting)"
            if cfg["sources"]
            else "no camera source configured"
        ),
        "stats_ts": _p(stats, "ts") if fresh else None,
        "model": {
            "project": _p(stats, "model", "project"),
            "model_version": _p(stats, "model", "model_version"),
            "deployment": _p(stats, "deployment") or own["device_current"],
            "classes": _p(stats, "model", "classes"),
            "input_size": _p(stats, "model", "input_size"),
            "precision": _p(stats, "model", "precision"),
            "confidence": _p(stats, "model", "confidence"),
        }
        if fresh
        else None,
        "channels": [
            {
                "channel": c.get("channel"),
                "name": c.get("name"),
                "detection_fps": c.get("fps"),
                "target_fps": c.get("target_fps"),
                "input_fps": _p(c, "source", "decode_fps"),
                "inference_ms_p50": _p(c, "latency_ms", "inference_p50"),
                "inference_ms_p95": _p(c, "latency_ms", "inference_p95"),
                "e2e_ms_p50": _p(c, "latency_ms", "e2e_p50"),
                "e2e_ms_p95": _p(c, "latency_ms", "e2e_p95"),
                "frames": c.get("frames"),
                "errors": c.get("errors"),
                "skipped_frames": _p(c, "source", "frames_dropped"),
                "missed_slots": c.get("missed_slots"),
                "camera_connected": _p(c, "source", "connected"),
                "camera_reconnects": _p(c, "source", "reconnects"),
                "last_inference_age_s": c.get("result_age_s"),
                "detections": c.get("detections"),
                "detections_by_class": c.get("detections_by_class"),
            }
            for c in chans
        ],
        "system": {
            "cpu_pct": sysm.get("cpu_pct"),
            "gpu_pct": sysm.get("gpu_pct"),
            "ram_used_mb": sysm.get("ram_used_mb"),
            "ram_total_mb": sysm.get("ram_total_mb"),
            "temp_cpu_c": sysm.get("temp_cpu_c"),
            "temp_gpu_c": sysm.get("temp_gpu_c"),
            "temp_tj_c": sysm.get("temp_tj_c"),
            "gpu_mhz": sysm.get("gpu_mhz"),
            "power_total_mw": sysm.get("power_total_mw"),
            "power_mode": (stats or {}).get("power_mode"),
            "process_rss_mb": (stats or {}).get("process_rss_mb"),
            "backend": (stats or {}).get("system_backend"),
        }
        if fresh
        else None,
        "liveview": {
            "viewers": _p(stats, "live", "viewers"),
            "publishing": _p(stats, "live", "publishing"),
            "publish_fps": _p(stats, "live", "publish_fps"),
            "encode_ms_p50": _p(stats, "live", "encode_ms_p50"),
        }
        if fresh
        else None,
        "config_restart_required": _p(stats, "config_restart_required") if fresh else None,
        "at": utc_now_iso(),
    }
    out["next_operations"] = (
        [f"app status {project.name}"]
        if fresh
        else [f"app status {project.name}", f"service start {project.name}"]
        if inf_state != "running"
        else [f"app status {project.name}", f"service restart {project.name}"]
    )
    return out


__all__ = [
    "AppContext",
    "check_url",
    "configure",
    "liveview_action",
    "metrics",
    "source_clear",
    "source_set",
    "source_show",
    "source_test",
    "status",
]
