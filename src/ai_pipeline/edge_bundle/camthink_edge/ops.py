"""Runtime operations invoked by the deployment target over SSH. Each returns a JSON-able dict."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import VERSION, images, yolo

HEARTBEAT_MAX_AGE_S = 15.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rss_mb(pid: int | None = None) -> float | None:
    try:
        with open(f"/proc/{pid or 'self'}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def _system_mem() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable"):
                    out[k] = round(int(v.split()[0]) / 1024, 1)
    except OSError:
        pass
    return out


def load_runtime_config(path: Path) -> dict[str, Any]:
    return yolo.load_config(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- build


def build(
    onnx: Path, engine: Path, *, precision: str, trtexec: str, workspace_mb: int, timeout_s: int
) -> dict[str, Any]:
    """ONNX -> TensorRT engine using the device's trtexec. Engine is written atomically."""
    import tensorrt as trt

    if not Path(onnx).is_file():
        raise FileNotFoundError(f"onnx not found: {onnx}")
    tmp = Path(str(engine) + ".building")
    tmp.unlink(missing_ok=True)
    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={tmp}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--skipInference",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision != "fp32":
        raise ValueError(f"unsupported precision {precision!r} (fp16 | fp32)")
    log_path = Path(str(engine) + ".build.log")
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, check=False
        )
    duration = round(time.monotonic() - t0, 1)
    peak_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    if proc.returncode != 0 or not tmp.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise RuntimeError(f"trtexec failed (exit {proc.returncode}); log tail: {' | '.join(tail)}")
    os.replace(tmp, engine)
    return {
        "engine_path": str(engine),
        "engine_sha256": _sha256(Path(engine)),
        "engine_size_bytes": Path(engine).stat().st_size,
        "onnx_sha256": _sha256(Path(onnx)),
        "build_params": {
            "precision": precision,
            "workspace_mb": workspace_mb,
            "trtexec": trtexec,
            "command": cmd,
        },
        "environment": {"tensorrt": trt.__version__, "runtime_version": VERSION},
        "duration_s": duration,
        "builder_peak_rss_mb": peak_mb,
        "log": str(log_path),
    }


# --------------------------------------------------------------------------- inference


class Detector:
    def __init__(self, engine: Path, config: Path):
        from .engine import TrtEngine

        self.cfg = load_runtime_config(config)
        size = self.cfg["input_size"]
        self.engine = TrtEngine(engine, input_shape=(1, 3, size, size))
        got = self.engine.input_shape
        if tuple(got[-2:]) != (size, size):
            raise RuntimeError(f"engine input {got} does not match runtime.json input_size {size}")

    def detect(self, img_rgb: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
        t0 = time.perf_counter()
        x, lb = yolo.preprocess(img_rgb, self.cfg)
        t1 = time.perf_counter()
        out = self.engine.infer(x)
        t2 = time.perf_counter()
        dets = yolo.postprocess(out, lb, self.cfg)
        t3 = time.perf_counter()
        return dets, {
            "preprocess_ms": round((t1 - t0) * 1000, 2),
            "inference_ms": round((t2 - t1) * 1000, 2),
            "postprocess_ms": round((t3 - t2) * 1000, 2),
        }

    def close(self) -> None:
        self.engine.close()


def infer(engine: Path, config: Path, image_paths: list[str]) -> dict[str, Any]:
    det = Detector(engine, config)
    try:
        results = []
        for p in images.list_images(image_paths):
            img = images.read_rgb(p)
            dets, timing = det.detect(img)
            results.append(
                {
                    "image": str(p),
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "detections": dets,
                    "timing_ms": timing,
                }
            )
        return {
            "runtime_version": VERSION,
            "engine": det.engine.describe(),
            "classes": det.cfg["classes"],
            "image_backend": images.backend(),
            "images": results,
            "total_detections": sum(len(r["detections"]) for r in results),
        }
    finally:
        det.close()


def smoke(engine: Path, config: Path, samples: list[str], *, min_detections: int) -> dict[str, Any]:
    paths = images.list_images(samples)
    if not paths:
        return {"passed": False, "details": {"reason": "no sample images", "samples": samples}}
    try:
        result = infer(engine, config, [str(p) for p in paths])
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, it does not crash
        return {
            "passed": False,
            "details": {"reason": f"inference failed: {type(exc).__name__}: {exc}"},
        }
    finite = all(
        all(
            np.isfinite(d["bbox_xyxy"]).all() and 0 <= d["confidence"] <= 1 for d in r["detections"]
        )
        for r in result["images"]
    )
    total = result["total_detections"]
    passed = finite and total >= min_detections
    return {
        "passed": passed,
        "details": {
            "images": len(paths),
            "total_detections": total,
            "min_detections": min_detections,
            "finite_outputs": finite,
            "per_image": [
                {
                    "image": Path(r["image"]).name,
                    "detections": len(r["detections"]),
                    "top": (r["detections"][0] if r["detections"] else None),
                }
                for r in result["images"]
            ],
            "engine": result["engine"],
            "image_backend": result["image_backend"],
            "at": _now(),
        },
    }


# --------------------------------------------------------------------------- benchmark


class _Tegrastats:
    """Samples RAM / GPU utilisation from tegrastats while a benchmark runs (best effort)."""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.samples: list[str] = []
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self):
        exe = shutil.which("tegrastats")
        if not exe:
            return self
        try:
            self.proc = subprocess.Popen(
                [exe, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self.proc = None
            return self

        def pump():
            assert self.proc and self.proc.stdout
            for line in self.proc.stdout:
                self.samples.append(line.strip())

        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"available": False}
        ram_used, gpu = [], []
        for s in self.samples:
            m = re.search(r"RAM (\d+)/(\d+)MB", s)
            if m:
                ram_used.append(int(m.group(1)))
            g = re.search(r"GR3D_FREQ (\d+)%", s)
            if g:
                gpu.append(int(g.group(1)))
        return {
            "available": True,
            "samples": len(self.samples),
            "ram_used_mb_max": max(ram_used) if ram_used else None,
            "ram_total_mb": int(re.search(r"RAM \d+/(\d+)MB", self.samples[0]).group(1))
            if ram_used
            else None,
            "gpu_util_pct_max": max(gpu) if gpu else None,
            "gpu_util_pct_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
        }


def benchmark(
    engine: Path, config: Path, *, iterations: int, warmup: int, image: str | None
) -> dict[str, Any]:
    mem_before = _system_mem()
    det = Detector(engine, config)
    try:
        if image and Path(image).is_file():
            img = images.read_rgb(image)
        else:
            rng = np.random.default_rng(0)
            img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        x, _ = yolo.preprocess(img, det.cfg)
        with _Tegrastats() as ts:
            lat = det.engine.time_infer(x, iterations, warmup)
            # end-to-end including pre/post on the sample image
            e2e = []
            for _ in range(min(iterations, 20)):
                t0 = time.perf_counter()
                det.detect(img)
                e2e.append((time.perf_counter() - t0) * 1000)
        lat_arr = np.asarray(lat)
        p50, p95 = float(np.percentile(lat_arr, 50)), float(np.percentile(lat_arr, 95))
        mean = float(lat_arr.mean())
        tstats = ts.summary()
        rss = _rss_mb()
        return {
            "metrics": {
                "latency_ms_p50": round(p50, 2),
                "latency_ms_p95": round(p95, 2),
                "fps": round(1000.0 / mean, 1) if mean > 0 else None,
                "memory_mb": rss,
                "gpu_util_pct": tstats.get("gpu_util_pct_max"),
            },
            "iterations": iterations,
            "details": {
                "warmup": warmup,
                "latency_ms_mean": round(mean, 2),
                "latency_ms_min": round(float(lat_arr.min()), 2),
                "latency_ms_max": round(float(lat_arr.max()), 2),
                "end_to_end_ms_mean": round(float(np.mean(e2e)), 2) if e2e else None,
                "process_rss_mb": rss,
                "system_mem_before_mb": mem_before,
                "system_mem_after_mb": _system_mem(),
                "tegrastats": tstats,
                "engine": det.engine.describe(),
                "image": image,
                "runtime_version": VERSION,
                "at": _now(),
            },
        }
    finally:
        det.close()


# --------------------------------------------------------------------------- health / service


def current_deployment(root: Path) -> str | None:
    link = root / "current"
    return Path(os.readlink(link)).name if link.is_symlink() else None


def _service_active(name: str | None) -> str | None:
    if not name:
        return None
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", name],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unknown ({type(exc).__name__})"


def health(root: Path, *, service: str | None) -> dict[str, Any]:
    root = Path(root)
    current = current_deployment(root)
    details: dict[str, Any] = {"at": _now(), "current": current, "runtime_version": VERSION}
    if current is None:
        return {
            "healthy": False,
            "current": None,
            "details": {**details, "reason": "no active deployment"},
        }
    dep = root / "deployments" / current
    details["engine_present"] = (dep / "model.engine").is_file()
    details["config_present"] = (dep / "runtime.json").is_file()
    if not (details["engine_present"] and details["config_present"]):
        return {
            "healthy": False,
            "current": current,
            "details": {**details, "reason": "engine or runtime.json missing"},
        }
    if service:
        details["service"] = service
        details["service_state"] = _service_active(service)
        hb_path = root / "service" / "status.json"
        hb = json.loads(hb_path.read_text(encoding="utf-8")) if hb_path.is_file() else None
        details["heartbeat"] = hb
        if details["service_state"] != "active":
            return {
                "healthy": False,
                "current": current,
                "details": {
                    **details,
                    "reason": f"service {service} is {details['service_state']}",
                },
            }
        if not hb:
            return {
                "healthy": False,
                "current": current,
                "details": {**details, "reason": "no heartbeat yet"},
            }
        age = time.time() - float(hb.get("ts_epoch", 0))
        details["heartbeat_age_s"] = round(age, 1)
        if age > HEARTBEAT_MAX_AGE_S:
            return {
                "healthy": False,
                "current": current,
                "details": {**details, "reason": f"heartbeat stale ({age:.0f}s)"},
            }
        if hb.get("deployment") != current:
            return {
                "healthy": False,
                "current": current,
                "details": {
                    **details,
                    "reason": f"service runs {hb.get('deployment')}, current is {current}",
                },
            }
        if hb.get("error"):
            return {
                "healthy": False,
                "current": current,
                "details": {**details, "reason": f"service error: {hb['error']}"},
            }
        return {"healthy": True, "current": current, "details": details}
    # no service configured: one-shot inference on a sample proves the runtime works
    samples = sorted((dep / "samples").glob("*")) if (dep / "samples").is_dir() else []
    try:
        res = infer(
            dep / "model.engine", dep / "runtime.json", [str(samples[0])] if samples else []
        )
        details["oneshot"] = {"images": len(res["images"]), "detections": res["total_detections"]}
        return {"healthy": True, "current": current, "details": details}
    except Exception as exc:  # noqa: BLE001
        return {
            "healthy": False,
            "current": current,
            "details": {**details, "reason": f"one-shot inference failed: {exc}"},
        }


def serve(root: Path, *, source: str | None, fps_limit: float, heartbeat_s: float) -> int:
    """Long-running inference loop; writes ``<root>/service/status.json`` as a heartbeat.

    ``source``: RTSP/video URL (needs OpenCV) or None to loop over the deployment's sample
    images. Reloads when ``current`` changes (the target restarts the service anyway).
    """
    root = Path(root)
    status_path = root / "service" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def write_status(**fields):
        payload = {
            "ts": _now(),
            "ts_epoch": time.time(),
            "runtime_version": VERSION,
            "pid": os.getpid(),
            "rss_mb": _rss_mb(),
            **fields,
        }
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, status_path)

    current = current_deployment(root)
    if current is None:
        write_status(deployment=None, error="no active deployment")
        return 1
    dep = root / "deployments" / current
    try:
        det = Detector(dep / "model.engine", dep / "runtime.json")
    except Exception as exc:  # noqa: BLE001
        write_status(
            deployment=current, error=f"failed to load engine: {type(exc).__name__}: {exc}"
        )
        return 1

    cap = None
    samples: list[Path] = []
    if source:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            write_status(deployment=current, source=source, error=f"cannot open source {source}")
            return 1
    else:
        samples = images.list_images([str(dep / "samples")])
        if not samples:
            write_status(
                deployment=current, error="no samples to loop over and no source configured"
            )
            return 1

    frames, errors, idx = 0, 0, 0
    last_hb = 0.0
    window: list[float] = []  # wall-clock frame timestamps -> actual (capped) frame rate
    last_dets: list[dict[str, Any]] = []
    min_dt = 1.0 / fps_limit if fps_limit > 0 else 0.0
    write_status(
        deployment=current, source=source or "samples", frames=0, fps=0.0, state="starting"
    )
    while not stop["flag"]:
        t0 = time.perf_counter()
        try:
            if cap is not None:
                ok, bgr = cap.read()
                if not ok:
                    errors += 1
                    time.sleep(0.5)
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    continue
                img = bgr[:, :, ::-1]
            else:
                img = images.read_rgb(samples[idx % len(samples)])
                idx += 1
            last_dets, _ = det.detect(img)
            frames += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            write_status(
                deployment=current,
                source=source or "samples",
                frames=frames,
                errors=errors,
                error=f"{type(exc).__name__}: {exc}",
            )
            time.sleep(1.0)
            continue
        dt = time.perf_counter() - t0
        window.append(time.monotonic())
        window = window[-50:]
        if time.monotonic() - last_hb >= heartbeat_s:
            span = window[-1] - window[0]
            fps = round((len(window) - 1) / span, 1) if len(window) > 1 and span > 0 else 0.0
            write_status(
                deployment=current,
                source=source or "samples",
                frames=frames,
                errors=errors,
                fps=fps,
                state="running",
                last_detections=last_dets[:10],
                last_detection_count=len(last_dets),
            )
            last_hb = time.monotonic()
        if min_dt and dt < min_dt:
            time.sleep(min_dt - dt)
    write_status(
        deployment=current,
        source=source or "samples",
        frames=frames,
        errors=errors,
        state="stopped",
    )
    det.close()
    if cap is not None:
        cap.release()
    return 0
