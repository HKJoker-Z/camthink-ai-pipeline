"""``ai-pipeline app ...`` — camera source, inference settings, live-view webpage and
real-time metrics of a project's installed deployment (M9).

Two device processes, controlled separately:

* inference service — camera decode + TensorRT; ``service start|stop|restart <project>`` (M7).
* live-view server — the webpage/MJPEG; ``app liveview start|stop|restart <project>``.
  Stopping it leaves inference running; starting it never rebuilds or restarts anything.

Camera URLs (with credentials) are read from **stdin only** so they never appear in shell
history, process lists or logs, and every output masks them (``rtsp://***@host/path``).
"""

from __future__ import annotations

import sys
from typing import Any

import typer

from ai_pipeline.cli.context import CliContext, run_command
from ai_pipeline.cli.output import Envelope
from ai_pipeline.core import edge_app
from ai_pipeline.core.errors import ConfigError
from ai_pipeline.core.project import Project

app = typer.Typer(
    help="Camera source, settings, live-view webpage and metrics of a deployed project.",
    no_args_is_help=True,
)
source_app = typer.Typer(help="Camera (RTSP) source of the deployed application.")
liveview_app = typer.Typer(help="The live-view web server (independent of inference).")
app.add_typer(source_app, name="source")
app.add_typer(liveview_app, name="liveview")

_TARGET = typer.Option(None, "--target", help="Target name (default: active deployment's).")
_NAME = typer.Argument(..., help="Project name.")


def _read_url_from_stdin() -> str:
    if sys.stdin is None or sys.stdin.isatty():
        raise ConfigError(
            "camera URL must be supplied on stdin (never as an argument)",
            code="APP_SOURCE_STDIN_REQUIRED",
            hint='echo "rtsp://user:pass@host:554/path" | ai-pipeline app source set <project>',
        )
    line = sys.stdin.readline()
    if not line.strip():
        raise ConfigError(
            "no camera URL on stdin",
            code="APP_SOURCE_EMPTY",
            hint='echo "rtsp://user:pass@host:554/path" | ai-pipeline app source set <project>',
        )
    return line.strip()


def _nl(items: list[str]) -> str:
    return "\n".join(items)


# -- source --------------------------------------------------------------------------------------


def _probe_text(p: dict[str, Any] | None) -> str:
    if not p:
        return ""
    lines = []
    for s in p.get("sources", []):
        if s.get("ok"):
            lines.append(
                f"  {s.get('name')}: OK {s.get('url')} {s.get('frames')} frames in "
                f"{s.get('seconds')}s ({s.get('decode_fps')} fps, first frame "
                f"{s.get('first_frame_s')}s, native {s.get('native_size')}, {s.get('backend')})"
            )
        else:
            lines.append(f"  {s.get('name')}: FAILED {s.get('url')}: {s.get('error')}")
    return _nl(lines)


@source_app.command("set")
def source_set_cmd(
    ctx: typer.Context,
    name: str = _NAME,
    target: str | None = _TARGET,
    source_name: str = typer.Option("cam0", "--name", help="Channel name."),
    add: bool = typer.Option(
        False, "--add", help="Add/replace this channel instead of replacing all sources."
    ),
    test: bool = typer.Option(
        True, "--test/--no-test", help="Decode a few frames on the device before saving."
    ),
    seconds: float = typer.Option(5.0, "--seconds", min=1.0, max=60.0, help="Probe length."),
    restart: bool = typer.Option(
        False,
        "--restart",
        help="Apply immediately by restarting the inference service (interrupts detection for "
        "a few seconds; needs operator approval). Default: only report restart_required.",
    ),
) -> None:
    """Set the camera source of this project's deployed application. The URL (with any
    credentials) is read from STDIN: `echo "rtsp://user:pw@cam/stream" | ... app source set P`.
    Stored only in the device's service/app.json (mode 0600); masked everywhere else. Refused
    (DEVICE_OCCUPIED) when the device runs another project."""

    def op(cli: CliContext) -> Envelope:
        url = _read_url_from_stdin()
        project = Project.load(name, cli.settings)
        r = edge_app.source_set(
            project,
            url,
            name=source_name,
            add=add,
            target_name=target,
            test_seconds=seconds if test else None,
            restart=restart,
        )
        rr = r["restart_required"]
        text = (
            f"source {r['source']['name']} = {r['source']['url']} saved on {r['target']}"
            + (f"\n{_probe_text(r.get('probe'))}" if r.get("probe") else "")
            + (
                f"\n  inference restarted ({r['inference']['state']})"
                if r["restarted"]
                else f"\n  inference service is {r['inference']['state']}: "
                + (
                    f"run `service restart {name}` to apply (interrupts detection briefly)"
                    if rr["inference"]
                    else "config applies on next start"
                    if r["inference"]["state"] != "running"
                    else "applied"
                )
            )
        )
        return Envelope(
            operation="app.source.set",
            project=name,
            result=r,
            next_operations=r["next_operations"],
            human_text=text,
        )

    run_command(ctx, "app.source.set", op, project=name)


@source_app.command("show")
def source_show_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Show the application config on the device with credentials masked (read-only)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.source_show(project, target_name=target)
        srcs = r["config"]["sources"]
        inf = r["config"]["inference"]
        conf = inf.get("confidence")
        conf_text = "model default" if conf is None else str(conf)
        text = (
            f"{r['target']}: "
            + (
                ", ".join(f"{s.get('name')}={s.get('url')}" for s in srcs)
                if srcs
                else "no camera source configured"
            )
            + f"\n  inference {inf.get('width')}x{inf.get('height')} @ {inf.get('fps')} fps, "
            f"confidence {conf_text}; service {r['inference']['state']}"
            + (f"\n  problems: {'; '.join(r['problems'])}" if r["problems"] else "")
        )
        return Envelope(
            operation="app.source.show",
            project=name,
            result=r,
            next_operations=r["next_operations"],
            human_text=text,
        )

    run_command(ctx, "app.source.show", op, project=name)


@source_app.command("test")
def source_test_cmd(
    ctx: typer.Context,
    name: str = _NAME,
    target: str | None = _TARGET,
    seconds: float = typer.Option(5.0, "--seconds", min=1.0, max=60.0),
    stdin: bool = typer.Option(
        False, "--stdin", help="Test a candidate URL from stdin instead of the saved sources."
    ),
) -> None:
    """Decode a few frames from the configured camera on the device (or from a candidate URL
    on stdin with --stdin) and report connectivity, decode FPS and native resolution. Opens a
    short extra RTSP session; the running service is not touched."""

    def op(cli: CliContext) -> Envelope:
        url = _read_url_from_stdin() if stdin else None
        project = Project.load(name, cli.settings)
        r = edge_app.source_test(project, url=url, seconds=seconds, target_name=target)
        text = f"{r['target']}: " + ("OK" if r.get("ok") else "FAILED") + "\n" + _probe_text(r)
        return Envelope(
            operation="app.source.test",
            project=name,
            status="ok" if r.get("ok") else "failed",
            result=r,
            error=None
            if r.get("ok")
            else {
                "code": "SOURCE_UNREACHABLE",
                "message": "camera could not be decoded on the device",
                "details": {"sources": r.get("sources")},
                "hint": "Check address/credentials/network; `app source set` replaces it.",
            },
            next_operations=r["next_operations"],
            human_text=text,
        )

    run_command(ctx, "app.source.test", op, project=name)


@source_app.command("clear")
def source_clear_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Remove all camera sources. After a restart the service falls back to the sample-image
    loop; the live view then has nothing to show."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.source_clear(project, target_name=target)
        return Envelope(
            operation="app.source.clear",
            project=name,
            result=r,
            next_operations=r["next_operations"],
            human_text=f"{r['target']}: removed {len(r['removed'])} source(s)"
            + (
                f"; `service restart {name}` to apply" if r["restart_required"]["inference"] else ""
            ),
        )

    run_command(ctx, "app.source.clear", op, project=name)


# -- configure -----------------------------------------------------------------------------------


@app.command("configure")
def configure_cmd(
    ctx: typer.Context,
    name: str = _NAME,
    target: str | None = _TARGET,
    inference_fps: float | None = typer.Option(
        None, "--inference-fps", min=0.1, max=60.0, help="Inference rate per camera (live)."
    ),
    confidence: str | None = typer.Option(
        None,
        "--confidence",
        help="Detection confidence threshold 0..1 (live), or 'default' for the model's own.",
    ),
    width: int | None = typer.Option(None, "--width", min=160, help="Decode width (restart)."),
    height: int | None = typer.Option(None, "--height", min=90, help="Decode height (restart)."),
    display_fps: float | None = typer.Option(
        None, "--display-fps", min=0.1, max=60.0, help="Max webpage stream FPS (live)."
    ),
    quality: int | None = typer.Option(
        None, "--quality", min=10, max=100, help="JPEG quality of the webpage stream (live)."
    ),
    port: int | None = typer.Option(
        None, "--port", min=1, max=65535, help="Live-view HTTP port (liveview restart)."
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Decode backend auto | cv2gst | cv2 (restart)."
    ),
    latency_ms: int | None = typer.Option(
        None, "--latency-ms", min=0, max=5000, help="RTSP jitter buffer (restart)."
    ),
) -> None:
    """Change inference / live-view settings of the deployed application. FPS, confidence,
    display FPS and quality are picked up by the running service within seconds; resolution,
    backend, latency and port report `restart_required` and are applied by the listed
    `service restart` / `app liveview restart` (which you must run explicitly)."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        conf: Any = confidence
        if conf is not None and conf not in ("default", "none", "null", ""):
            try:
                conf = float(conf)
            except ValueError as exc:
                raise ConfigError(
                    f"--confidence must be a number 0..1 or 'default', got {confidence!r}",
                    code="APP_CONFIG_INVALID",
                ) from exc
        r = edge_app.configure(
            project,
            target_name=target,
            inference_fps=inference_fps,
            confidence=conf,
            width=width,
            height=height,
            display_fps=display_fps,
            quality=quality,
            port=port,
            backend=backend,
            latency_ms=latency_ms,
        )
        rr = r["restart_required"]
        text = f"{r['target']}: changed " + ", ".join(f"{k}={v}" for k, v in r["changed"].items())
        if r["hot_reloaded"]:
            text += "\n  applied live: " + ", ".join(r["hot_reloaded"])
        if rr["inference"]:
            text += f"\n  restart required: `service restart {name}` (interrupts detection briefly)"
        if rr["liveview"]:
            text += f"\n  restart required: `app liveview restart {name}`"
        return Envelope(
            operation="app.configure",
            project=name,
            result=r,
            next_operations=r["next_operations"],
            human_text=text,
        )

    run_command(ctx, "app.configure", op, project=name)


# -- liveview ------------------------------------------------------------------------------------


def _liveview_envelope(name: str, r: dict[str, Any], operation: str) -> Envelope:
    lv = r["liveview"]
    text = (
        f"live view {r['action']} on {r['target']}: {lv['state']}"
        + ("" if r["changed"] else " (no change)")
        + f"; inference untouched ({r['inference']['state']})"
    )
    if r["url"]:
        chk = lv.get("url_check") or {}
        text += f"\n  URL: {r['url']}" + (
            " (verified from the AI server)"
            if chk.get("verified")
            else f" (NOT verified: {chk.get('error')})"
            if chk
            else " (not verified)"
        )
    for w in r.get("warnings", []):
        text += f"\n  warning: {w}"
    return Envelope(
        operation=operation,
        project=name,
        result=r,
        next_operations=r["next_operations"],
        human_text=text,
    )


@liveview_app.command("start")
def liveview_start_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Start the live-view web server for this project's deployed application and verify its
    URL from the AI server. Does not rebuild, redeploy or restart inference. Refused when the
    device runs another project (DEVICE_OCCUPIED) or no camera source is configured."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.liveview_action(project, "start", target_name=target)
        return _liveview_envelope(name, r, "app.liveview.start")

    run_command(ctx, "app.liveview.start", op, project=name)


@liveview_app.command("stop")
def liveview_stop_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Stop the webpage only. The inference service keeps running and detecting; use
    `service stop <project>` to stop detection."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.liveview_action(project, "stop", target_name=target)
        return _liveview_envelope(name, r, "app.liveview.stop")

    run_command(ctx, "app.liveview.stop", op, project=name)


@liveview_app.command("restart")
def liveview_restart_cmd(
    ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET
) -> None:
    """Restart the web server (e.g. after `app configure --port`). Inference is untouched."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.liveview_action(project, "restart", target_name=target)
        return _liveview_envelope(name, r, "app.liveview.restart")

    run_command(ctx, "app.liveview.restart", op, project=name)


@liveview_app.command("status")
def liveview_status_cmd(
    ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET
) -> None:
    """Alias of `app status <project>`."""
    status_cmd(ctx, name, target)


@liveview_app.command("url")
def liveview_url_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Print the live-view URL (null when the web server is not running) and whether it
    answered from the AI server."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.status(project, target_name=target)
        lv = r["liveview"] or {}
        result = {
            "target": r["target"],
            "url": r["url"],
            "url_verified": r["url_verified"],
            "url_check": lv.get("url_check"),
            "liveview_state": lv.get("state"),
            "inference_state": r["inference"]["state"],
            "ownership": r["ownership"],
        }
        text = (
            f"{r['url']} ({'verified from the AI server' if r['url_verified'] else 'NOT verified'})"
            if r["url"]
            else f"live view is {lv.get('state', 'unavailable')} on {r['target']}; "
            f"`app liveview start {name}`"
        )
        return Envelope(
            operation="app.liveview.url",
            project=name,
            result=result,
            next_operations=r["next_operations"],
            human_text=text,
        )

    run_command(ctx, "app.liveview.url", op, project=name)


# -- status / metrics ----------------------------------------------------------------------------


def _status_text(name: str, r: dict[str, Any]) -> str:
    own = r["ownership"]
    inf = r["inference"]
    cam = r["camera"]
    lv = r["liveview"] or {}
    lines = [
        f"{r['target']}: device runs {own['device_current']} "
        + (
            "(this project)" if own["matches"] else f"(project {own['device_project']}, NOT {name})"
        ),
        f"  inference: {inf['state']}"
        + (f", mode {inf['mode']}" if inf.get("mode") else "")
        + (f", {inf['fps']} fps" if inf.get("fps") is not None else ""),
        f"  camera: {cam['state']}"
        + (
            " — "
            + "; ".join(
                f"{c['name']} {'connected' if c['connected'] else 'disconnected'}"
                + (f" ({c['decode_fps']} fps in)" if c.get("decode_fps") else "")
                + (f" error: {c['error']}" if c.get("error") else "")
                for c in cam["channels"]
            )
            if cam["channels"]
            else (f" ({cam['reason']})" if cam.get("reason") else "")
        ),
        f"  live view: {lv.get('state', 'n/a')}"
        + (
            f" {r['url']} "
            + ("(verified)" if r["url_verified"] else "(NOT verified from the AI server)")
            if r["url"]
            else ""
        ),
    ]
    lines += [f"  warning: {w}" for w in r.get("warnings", [])]
    return _nl(lines)


@app.command("status")
def status_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Ownership, inference state/mode, camera connectivity, live-view service + URL
    (verified from the AI server), masked config and pending restarts. Read-only."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.status(project, target_name=target)
        return Envelope(
            operation="app.status",
            project=name,
            result=r,
            next_operations=r["next_operations"],
            human_text=_status_text(name, r),
        )

    run_command(ctx, "app.status", op, project=name)


def _metrics_text(r: dict[str, Any]) -> str:
    if not r["available"]:
        return f"{r['target']}: metrics unavailable — {r['reason']}"
    s = r["system"] or {}
    lines = [f"{r['target']} ({r['model']['project']} {r['model']['model_version']}):"]
    for c in r["channels"]:
        lines.append(
            f"  {c['name']}: detection {c['detection_fps']} fps (target {c['target_fps']}), "
            f"input {c['input_fps']} fps, inference p50 {c['inference_ms_p50']} ms "
            f"(p95 {c['inference_ms_p95']}), e2e p50 {c['e2e_ms_p50']} ms, "
            f"camera {'connected' if c['camera_connected'] else 'DISCONNECTED'}, "
            f"skipped {c['skipped_frames']}, last inference {c['last_inference_age_s']}s ago"
        )

    def u(v: Any, unit: str = "") -> str:
        return "unavailable" if v is None else f"{v}{unit}"

    lines.append(
        f"  system: CPU {u(s.get('cpu_pct'), '%')}, GPU {u(s.get('gpu_pct'), '%')}, "
        f"RAM {u(s.get('ram_used_mb'))}/{u(s.get('ram_total_mb'), ' MB')}, "
        f"temp {u(s.get('temp_tj_c'), 'C')}, power {u(s.get('power_total_mw'), ' mW')} "
        f"[{s.get('backend')}]"
    )
    lv = r["liveview"] or {}
    lines.append(
        f"  live view: {lv.get('viewers')} viewer(s), "
        f"{'publishing' if lv.get('publishing') else 'idle'} at {lv.get('publish_fps')} fps"
    )
    return _nl(lines)


@app.command("metrics")
def metrics_cmd(ctx: typer.Context, name: str = _NAME, target: str | None = _TARGET) -> None:
    """Real-time detection FPS, input FPS, inference latency, CPU/GPU/RAM/temperature, skipped
    frames, camera state and last inference time from the running service. Unavailable
    values are null, never estimated. Refused when the device runs another project."""

    def op(cli: CliContext) -> Envelope:
        project = Project.load(name, cli.settings)
        r = edge_app.metrics(project, target_name=target)
        return Envelope(
            operation="app.metrics",
            project=name,
            result=r,
            next_operations=r["next_operations"],
            human_text=_metrics_text(r),
        )

    run_command(ctx, "app.metrics", op, project=name)
