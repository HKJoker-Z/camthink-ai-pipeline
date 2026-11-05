"""DeploymentTarget protocol.

A target knows how to talk to one device/environment. It performs *phases*; the deployment
orchestrator (``core.deployment``) owns the state machine, acceptance policy, and rollback
decisions. Targets raise ``PipelineError`` subclasses on execution failures and return plain
dicts (persisted verbatim into the Deployment record) on success.

Phase contracts:

verify()                       -> {"reachable": bool, "environment": {...}, "current": dep|None,
                                   "staged": [dep, ...]}
stage(dep, artifact, meta)     -> {"path": str, "sha256": str, "size_bytes": int}
    meta carries "runtime_config" (written as runtime.json) and "samples" (local image paths)
prepare(dep, precision, params)-> {"engine_path": str, "engine_sha256": str, "build_params": {...},
                                   "environment": {...}, "duration_s": float}
smoke_test(dep)                -> {"passed": bool, "details": {...}}  (failed test => passed=False)
benchmark(dep)                 -> {"metrics": {latency_ms_p50, latency_ms_p95, fps, memory_mb,
                                   gpu_util_pct}, "iterations": int, "details": {...}}
activate(dep)                  -> {"previous": dep|None, "current": dep}
health_check()                 -> {"healthy": bool, "current": dep|None, "details": {...}}
current()                      -> dep | None
rollback(to)                   -> {"previous": dep|None, "current": dep|None}

Service control (optional; targets that run a long-lived inference service). None of these
touch deployment directories, engines or the ``current`` pointer — they only start/stop the
service process. A stop leaves a marker so that an intentional stop can be told apart from a
crash by ``health_check`` / ``service_status``:

service_status()               -> {"service": name|None, "systemd_state": "active"|"inactive"|
                                   "failed"|..., "current": dep|None, "stopped": marker|None,
                                   "heartbeat": {...}|None}
service_stop()                 -> {"previous_state": str, "state": str, "stopped": marker}
service_start()                -> {"previous_state": str, "state": str}
service_restart()              -> {"previous_state": str, "state": str}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DeploymentTarget(Protocol):
    name: str
    type: str

    def verify(self) -> dict[str, Any]: ...

    def stage(
        self, deployment_id: str, artifact: Path, metadata: dict[str, Any]
    ) -> dict[str, Any]: ...

    def prepare(
        self, deployment_id: str, *, precision: str, params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def smoke_test(self, deployment_id: str) -> dict[str, Any]: ...

    def benchmark(self, deployment_id: str) -> dict[str, Any]: ...

    def activate(self, deployment_id: str) -> dict[str, Any]: ...

    def health_check(self) -> dict[str, Any]: ...

    def current(self) -> str | None: ...

    def rollback(self, to_deployment_id: str | None) -> dict[str, Any]: ...

    # Optional extras used by the CLI (`target setup`, `deploy infer`):
    def setup_runtime(self) -> dict[str, Any]: ...

    def infer(self, images: list[Path], *, deployment_id: str | None = None) -> dict[str, Any]: ...

    def configure(self, **knobs: Any) -> None: ...

    # Optional service control (`service start|stop|restart`, `target status`):
    def service_status(self) -> dict[str, Any]: ...

    def service_stop(self) -> dict[str, Any]: ...

    def service_start(self) -> dict[str, Any]: ...

    def service_restart(self) -> dict[str, Any]: ...

    # Optional RTSP application + live view (`app ...`):
    def app_config_read(self) -> dict[str, Any] | None: ...

    def app_config_write(self, cfg: dict[str, Any]) -> None: ...

    def source_probe(self, *, seconds: float = 5.0, url: str | None = None) -> dict[str, Any]: ...

    def liveview_status(self) -> dict[str, Any]: ...

    def liveview_start(self) -> dict[str, Any]: ...

    def liveview_stop(self) -> dict[str, Any]: ...

    def liveview_restart(self) -> dict[str, Any]: ...

    def liveview_url(self) -> str | None: ...

    def live_stats(self) -> dict[str, Any] | None: ...
