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
      service/app.json                 RTSP app config (same shape as on the device)
      liveview.json                    simulated live-view service state
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
import time
from pathlib import Path
from typing import Any

from ai_pipeline.core.errors import OperationError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.store import read_json, write_json_atomic


def _mask(url: str) -> str:
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
    return url


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

    # -- simulated inference service (service.json) --------------------------------------

    @property
    def _service_path(self) -> Path:
        return self.root / "service.json"

    def _service(self) -> dict[str, Any]:
        # Absent file = running: deployments made before service simulation existed stay healthy.
        if not self._service_path.is_file():
            return {"state": "active", "stopped": None}
        return read_json(self._service_path)

    def _set_service(self, state: str, stopped: dict[str, Any] | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._service_path, {"state": state, "stopped": stopped})

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
        meta = {k: v for k, v in metadata.items() if k != "samples"}
        write_json_atomic(ddir / "deployment.json", {**meta, "staged_at": utc_now_iso()})
        if metadata.get("runtime_config"):
            write_json_atomic(ddir / "runtime.json", metadata["runtime_config"])
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
        self._set_service("active")
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
        svc = self._service()
        details["service_state"] = svc["state"]
        if svc["state"] != "active":
            if svc.get("stopped"):
                details["stopped_intentionally"] = svc["stopped"]
            return {
                "healthy": False,
                "current": current,
                "details": {**details, "reason": f"service fake-inference is {svc['state']}"},
            }
        engine = self._dep_dir(current) / "model.engine"
        healthy = engine.is_file()
        details.update(
            {"service": "running" if healthy else "missing-engine", "last_inference_age_s": 0.4}
        )
        return {"healthy": healthy, "current": current, "details": details}

    # -- service control ---------------------------------------------------------------

    def service_status(self) -> dict[str, Any]:
        svc = self._service()
        current = self.current()
        return {
            "service": "fake-inference",
            "systemd_state": svc["state"],
            "current": current,
            "stopped": svc.get("stopped"),
            "heartbeat": {"deployment": current, "fps": 30.0, "frames": 1000, "errors": 0}
            if svc["state"] == "active" and current
            else None,
        }

    def service_stop(self) -> dict[str, Any]:
        self._maybe_fail("service_stop")
        before = self._service()["state"]
        marker = {
            "deployment": self.current(),
            "at": utc_now_iso(),
            "by": "ai-pipeline service stop",
        }
        self._set_service("inactive", marker)
        return {"previous_state": before, "state": "inactive", "stopped": marker}

    def service_start(self) -> dict[str, Any]:
        self._maybe_fail("service_start")
        before = self._service()["state"]
        self._set_service("active")
        return {"previous_state": before, "state": "active"}

    def service_restart(self) -> dict[str, Any]:
        self._maybe_fail("service_restart")
        before = self._service()["state"]
        self._set_service("active")
        return {"previous_state": before, "state": "active"}

    def current(self) -> str | None:
        if not self.current_link.is_symlink():
            return None
        return Path(os.readlink(self.current_link)).name

    def configure(self, **_: Any) -> None:
        return None

    # -- RTSP application + live view (M9) ------------------------------------------

    @property
    def app_config_path(self) -> Path:
        return self.root / "service" / "app.json"

    def app_config_read(self) -> dict[str, Any] | None:
        return read_json(self.app_config_path) if self.app_config_path.is_file() else None

    def app_config_write(self, cfg: dict[str, Any]) -> None:
        self.app_config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.app_config_path, cfg)
        os.chmod(self.app_config_path, 0o600)

    def source_probe(self, *, seconds: float = 5.0, url: str | None = None) -> dict[str, Any]:
        self._maybe_fail("source_probe")
        cfg = self.app_config_read() or {}
        urls = [url] if url is not None else [s["url"] for s in cfg.get("sources", [])]
        out = []
        for u in urls:
            bad = "unreachable" in u
            out.append(
                {
                    "name": "cam0",
                    "url": _mask(u),
                    "ok": not bad,
                    "backend": "fake",
                    "frames": 0 if bad else int(25 * seconds),
                    "decode_fps": 0.0 if bad else 25.0,
                    "first_frame_s": None if bad else 0.4,
                    "size": None if bad else [1280, 720],
                    "native_size": None if bad else [1920, 1080],
                    "error": "connection refused" if bad else None,
                    "seconds": seconds,
                }
            )
        return {"ok": all(s["ok"] for s in out) and bool(out), "sources": out}

    def _liveview(self) -> dict[str, Any]:
        p = self.root / "liveview.json"
        return read_json(p) if p.is_file() else {"state": "inactive", "enabled": "disabled"}

    def _set_liveview(self, state: str, enabled: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.root / "liveview.json", {"state": state, "enabled": enabled})

    def liveview_url(self) -> str | None:
        return "http://fake-device.lan:8080/"

    def liveview_status(self) -> dict[str, Any]:
        lv = self._liveview()
        return {
            "service": "fake-liveview",
            "systemd_state": lv["state"],
            "enabled": lv["enabled"],
            "port": 8080,
            "url": self.liveview_url(),
        }

    def liveview_start(self) -> dict[str, Any]:
        self._maybe_fail("liveview_start")
        before = self._liveview()["state"]
        self._set_liveview("active", "enabled")
        return {"previous_state": before, "state": "active"}

    def liveview_stop(self) -> dict[str, Any]:
        self._maybe_fail("liveview_stop")
        before = self._liveview()["state"]
        self._set_liveview("inactive", "disabled")
        return {"previous_state": before, "state": "inactive"}

    def liveview_restart(self) -> dict[str, Any]:
        self._maybe_fail("liveview_restart")
        before = self._liveview()["state"]
        self._set_liveview("active", "enabled")
        return {"previous_state": before, "state": "active"}

    def live_stats(self) -> dict[str, Any] | None:
        cfg = self.app_config_read()
        if not cfg or not cfg.get("sources") or self._service()["state"] != "active":
            return None
        inf = cfg.get("inference", {})
        current = self.current()
        return {
            "ts": utc_now_iso(),
            "ts_epoch": time.time(),
            "state": "running",
            "deployment": current,
            "camera_connected": True,
            "fps": float(inf.get("fps", 10.0)),
            "model": {
                "project": current.rsplit("-dep-", 1)[0] if current else None,
                "model_version": None,
                "classes": ["object"],
                "input_size": 640,
                "precision": "fp16",
                "confidence": inf.get("confidence") if inf.get("confidence") is not None else 0.25,
            },
            "channels": [
                {
                    "channel": i,
                    "name": s.get("name") or f"ch{i}",
                    "target_fps": float(inf.get("fps", 10.0)),
                    "fps": float(inf.get("fps", 10.0)),
                    "frames": 100,
                    "errors": 0,
                    "missed_slots": 0,
                    "latency_ms": {"inference_p50": 12.0, "e2e_p50": 20.0},
                    "source": {"connected": True, "decode_fps": 25.0, "reconnects": 0},
                    "detections": 1,
                    "detections_by_class": {"object": 1},
                }
                for i, s in enumerate(cfg["sources"])
            ],
            "system": {"cpu_pct": 20.0, "gpu_pct": 30.0, "ram_used_mb": 2000.0},
            "system_backend": "fake",
            "live": {"viewers": 0, "publishing": False, "publish_fps": 0.0},
            "config_restart_required": {"inference": False, "liveview": False},
        }

    def setup_runtime(self) -> dict[str, Any]:
        self._ensure_root()
        return {
            "root": str(self.root),
            "bundle": {"runtime_version": "fake", "updated": False},
            "warnings": [],
        }

    def infer(self, images: list[Path], *, deployment_id: str | None = None) -> dict[str, Any]:
        dep = deployment_id or self.current()
        if dep is None:
            raise OperationError(
                "no active deployment to run inference with", code="NO_ACTIVE_DEPLOYMENT"
            )
        cfg = (
            json.loads((self._dep_dir(dep) / "runtime.json").read_text())
            if (self._dep_dir(dep) / "runtime.json").is_file()
            else {}
        )
        cls = (cfg.get("classes") or ["object"])[0]
        return {
            "runtime_version": "fake",
            "classes": cfg.get("classes") or [cls],
            "images": [
                {
                    "image": str(p),
                    "detections": [
                        {
                            "class_id": 0,
                            "class_name": cls,
                            "confidence": 0.9,
                            "bbox_xyxy": [10.0, 10.0, 100.0, 100.0],
                        }
                    ],
                }
                for p in images
            ],
            "total_detections": len(images),
        }

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
        self._set_service("active" if to_deployment_id else "inactive")
        return {"previous": previous, "current": to_deployment_id, "at": utc_now_iso()}
