"""Directory-based fake deployment target.

Simulates the phases the real NG4511 target will perform, with the same layout semantics::

    <root>/
      target.json                      simulated environment description
      deployments/<dep_id>/            one directory per staged deployment
          model.onnx                   staged artifact
          model.engine                 "built" runtime artifact (prepare)
          deployment.json              metadata handed over by the orchestrator
          smoke.json  benchmark.json
      current -> deployments/<dep_id>  the active deployment (atomic symlink swap)
      inject.json                      TEST ONLY: deterministic failure injection

``inject.json``::

    {
      "fail": "verify|stage|prepare|smoke_test|benchmark|activate|health_check|rollback",
      "health_fail_for": ["dep-002"],          # health fails only while that deployment is current
      "benchmark_metrics": {"fps": 9.0, ...}   # override the synthetic benchmark
    }

Nothing here is meant to be clever; it exists to verify deployment semantics (no change before
activation, automatic rollback after, explicit ROLLBACK_FAILED).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ai_pipeline.core.errors import OperationError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.store import read_json, write_json_atomic

DEFAULT_BENCHMARK = {
    "latency_ms_p50": 25.0,
    "latency_ms_p95": 41.0,
    "fps": 32.0,
    "memory_mb": 1850.0,
    "gpu_util_pct": 55.0,
}


class FakeTarget:
    type = "fake"

    def __init__(self, *, name: str, root: Path) -> None:
        self.name = name
        self.root = Path(root)
        self.deployments_dir = self.root / "deployments"
        self.current_link = self.root / "current"

    # -- helpers -------------------------------------------------------------------

    def _inject(self) -> dict[str, Any]:
        p = self.root / "inject.json"
        return read_json(p) if p.is_file() else {}

    def _maybe_fail(self, phase: str) -> None:
        if self._inject().get("fail") == phase:
            raise OperationError(
                f"fake target: injected failure during {phase}",
                code=f"TARGET_{phase.upper()}_FAILED",
                details={"target": self.name, "phase": phase, "injected": True},
                hint="Test-only failure from inject.json.",
            )

    def _dep_dir(self, deployment_id: str) -> Path:
        return self.deployments_dir / deployment_id

    def _ensure_root(self) -> None:
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        if not (self.root / "target.json").is_file():
            write_json_atomic(
                self.root / "target.json",
                {
                    "type": self.type,
                    "name": self.name,
                    "created_at": utc_now_iso(),
                    "environment": self._environment(),
                },
            )

    @staticmethod
    def _environment() -> dict[str, Any]:
        return {
            "device": "simulated-jetson-orin-nano",
            "runtime": "tensorrt",
            "tensorrt_version": "fake-10.x",
            "jetpack": "fake",
            "cuda": "fake",
        }

    def _set_current(self, deployment_id: str | None) -> None:
        """Atomic pointer swap: build a temp symlink and rename it over ``current``."""
        if deployment_id is None:
            self.current_link.unlink(missing_ok=True)
            return
        tmp = self.root / ".current.tmp"
        tmp.unlink(missing_ok=True)
        os.symlink(Path("deployments") / deployment_id, tmp)
        os.replace(tmp, self.current_link)

    # -- protocol -------------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        self._maybe_fail("verify")
        self._ensure_root()
        return {
            "reachable": True,
            "environment": self._environment(),
            "current": self.current(),
            "staged": sorted(p.name for p in self.deployments_dir.iterdir() if p.is_dir()),
            "root": str(self.root),
        }

    def stage(self, deployment_id: str, artifact: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        self._maybe_fail("stage")
        self._ensure_root()
        ddir = self._dep_dir(deployment_id)
        if ddir.exists():
            raise OperationError(
                f"deployment directory {ddir} already exists on target",
                code="TARGET_DEPLOYMENT_EXISTS",
                details={"path": str(ddir)},
                hint="Deployment ids are never reused; this indicates a stale target directory.",
            )
        ddir.mkdir(parents=True)
        dest = ddir / ("model" + Path(artifact).suffix)
        shutil.copyfile(artifact, dest)
        write_json_atomic(ddir / "deployment.json", {**metadata, "staged_at": utc_now_iso()})
        return {
            "path": str(dest),
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "size_bytes": dest.stat().st_size,
        }

    def prepare(
        self, deployment_id: str, *, precision: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        self._maybe_fail("prepare")
        ddir = self._dep_dir(deployment_id)
        onnx = ddir / "model.onnx"
        if not onnx.is_file():
            raise OperationError(
                "nothing staged to prepare", code="TARGET_NOT_STAGED", details={"path": str(onnx)}
            )
        onnx_sha = hashlib.sha256(onnx.read_bytes()).hexdigest()
        engine = ddir / "model.engine"
        engine.write_text(
            f"FAKE-TENSORRT-ENGINE\nsource_onnx_sha256={onnx_sha}\nprecision={precision}\n"
            f"params={json.dumps(params, sort_keys=True)}\n",
            encoding="utf-8",
        )
        return {
            "engine_path": str(engine),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "build_params": {"precision": precision, **params},
            "environment": self._environment(),
            "duration_s": 0.0,
        }

    def smoke_test(self, deployment_id: str) -> dict[str, Any]:
        self._maybe_fail("smoke_test")
        ddir = self._dep_dir(deployment_id)
        engine = ddir / "model.engine"
        passed = engine.is_file() and engine.read_text().startswith("FAKE-TENSORRT-ENGINE")
        result = {
            "passed": passed,
            "details": {
                "inferences": 3 if passed else 0,
                "engine": str(engine),
                "at": utc_now_iso(),
            },
        }
        write_json_atomic(ddir / "smoke.json", result)
        return result

    def benchmark(self, deployment_id: str) -> dict[str, Any]:
        self._maybe_fail("benchmark")
        ddir = self._dep_dir(deployment_id)
        metrics = {**DEFAULT_BENCHMARK, **(self._inject().get("benchmark_metrics") or {})}
        result = {
            "metrics": metrics,
            "iterations": 50,
            "details": {"runtime": "tensorrt-fake", "warmup": 5, "at": utc_now_iso()},
        }
        write_json_atomic(ddir / "benchmark.json", result)
        return result

    def activate(self, deployment_id: str) -> dict[str, Any]:
        self._maybe_fail("activate")
        if not (self._dep_dir(deployment_id) / "model.engine").is_file():
            raise OperationError(
                f"deployment {deployment_id} is not prepared", code="TARGET_NOT_PREPARED"
            )
        previous = self.current()
        self._set_current(deployment_id)
        return {"previous": previous, "current": deployment_id, "at": utc_now_iso()}

    def health_check(self) -> dict[str, Any]:
        self._maybe_fail("health_check")
        current = self.current()
        details: dict[str, Any] = {"at": utc_now_iso(), "current": current}
        if current is None:
            return {
                "healthy": False,
                "current": None,
                "details": {**details, "reason": "no active deployment"},
            }
        if current in (self._inject().get("health_fail_for") or []):
            return {
                "healthy": False,
                "current": current,
                "details": {**details, "reason": "injected: inference service unhealthy"},
            }
        engine = self._dep_dir(current) / "model.engine"
        healthy = engine.is_file()
        details.update(
            {"service": "running" if healthy else "missing-engine", "last_inference_age_s": 0.4}
        )
        return {"healthy": healthy, "current": current, "details": details}

    def current(self) -> str | None:
        if not self.current_link.is_symlink():
            return None
        return Path(os.readlink(self.current_link)).name

    def rollback(self, to_deployment_id: str | None) -> dict[str, Any]:
        self._maybe_fail("rollback")
        previous = self.current()
        if (
            to_deployment_id is not None
            and not (self._dep_dir(to_deployment_id) / "model.engine").is_file()
        ):
            raise OperationError(
                f"cannot roll back: deployment {to_deployment_id} is not present on the target",
                code="TARGET_ROLLBACK_MISSING",
                details={"to": to_deployment_id},
            )
        self._set_current(to_deployment_id)
        return {"previous": previous, "current": to_deployment_id, "at": utc_now_iso()}
