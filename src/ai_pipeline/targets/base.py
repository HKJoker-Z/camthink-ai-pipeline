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
