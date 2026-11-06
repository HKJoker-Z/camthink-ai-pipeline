"""Multi-channel detection: N independent video channels, M inference workers sharing the GPU.

Each :class:`Channel` owns one decoder (:class:`stream.FrameSource`). An
:class:`InferenceWorker` owns one TensorRT execution context and serves a subset of channels
round-robin: whenever a channel is *due* (its inference interval has elapsed and a new frame
exists) the worker takes the latest frame, runs pre-process -> engine -> post-process and
stores the result on the channel. Frames the camera decoded in between are dropped by design
(latest-frame semantics) and counted.

Per-channel metrics (all measured, nothing derived from single-channel numbers):

* ``fps``               results completed per second (sliding 5 s window)
* ``latency_ms``        capture(decoded)->result, inference only, pre/post-processing
* ``missed_slots``      due slots not served within one interval (the scheduler fell behind)
* ``frames_dropped``    decoded frames never inferred (includes the intended rate reduction)

Detections are plain ``{class_id, class_name, confidence, bbox_xyxy}`` records; the model's
classes come from the deployment's ``runtime.json``, nothing here is task-specific.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .stream import FrameSource


def _pct(values, q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 2)


class Channel:
    def __init__(self, idx: int, name: str, source: FrameSource, fps_limit: float):
        self.idx = idx
        self.name = name
        self.source = source
        self.lock = threading.Lock()
        self.result: dict[str, Any] | None = None
        self.next_due = 0.0
        self.frames = 0
        self.errors = 0
        self.last_error: str | None = None
        self.missed_slots = 0
        self.last_result_wall: float | None = None  # time.time() of the last result
        self._done_times: deque = deque(maxlen=600)
        self._lat_e2e: deque = deque(maxlen=600)
        self._lat_inf: deque = deque(maxlen=600)
        self._lat_pre: deque = deque(maxlen=600)
        self._lat_post: deque = deque(maxlen=600)
        self._lag: deque = deque(maxlen=600)
        self._stats_since = time.monotonic()
        self.set_fps_limit(fps_limit)

    def set_fps_limit(self, fps_limit: float) -> None:
        self.fps_limit = float(fps_limit)
        self.min_dt = 1.0 / self.fps_limit if self.fps_limit > 0 else 0.0

    def reset_stats(self) -> None:
        with self.lock:
            self.frames = self.errors = self.missed_slots = 0
            for d in (
                self._done_times,
                self._lat_e2e,
                self._lat_inf,
                self._lat_pre,
                self._lat_post,
                self._lag,
            ):
                d.clear()
            self._stats_since = time.monotonic()
        src = self.source
        with src._lock:
            src._seq_at_reset = src._seq
            src._consumed_at_reset = src._consumed

    def due(self, now: float) -> bool:
        return now >= self.next_due and self.source.has_new()

    def record(
        self, result: dict[str, Any], t_capture: float, t_start: float, t_end: float, timing: dict
    ) -> None:
        with self.lock:
            self.result = result
            self.frames += 1
            self.last_result_wall = time.time()
            self._done_times.append(t_end)
            self._lat_e2e.append((t_end - t_capture) * 1000.0)
            self._lat_inf.append(timing.get("inference_ms", 0.0))
            self._lat_pre.append(timing.get("preprocess_ms", 0.0))
            self._lat_post.append(timing.get("postprocess_ms", 0.0))
            if self.min_dt and self.next_due:
                lag = t_start - self.next_due
                self._lag.append(max(lag, 0.0) * 1000.0)
                if lag > self.min_dt:
                    self.missed_slots += 1

    def snapshot(self) -> tuple[Any, dict[str, Any] | None]:
        """(latest decoded frame, latest result) without consuming the frame."""
        with self.lock:
            res = self.result
        return self.source._frame, res

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            done = [t for t in self._done_times if now - t <= 5.0]
            fps = (len(done) - 1) / max(done[-1] - done[0], 1e-6) if len(done) > 1 else 0.0
            elapsed = now - self._stats_since
            out: dict[str, Any] = {
                "channel": self.idx,
                "name": self.name,
                "target_fps": self.fps_limit,
                "fps": round(fps, 2),
                "fps_avg": round(self.frames / elapsed, 2) if elapsed > 0 else 0.0,
                "frames": self.frames,
                "errors": self.errors,
                "missed_slots": self.missed_slots,
                "latency_ms": {
                    "e2e_p50": _pct(self._lat_e2e, 50),
                    "e2e_p95": _pct(self._lat_e2e, 95),
                    "inference_p50": _pct(self._lat_inf, 50),
                    "inference_p95": _pct(self._lat_inf, 95),
                    "preprocess_p50": _pct(self._lat_pre, 50),
                    "postprocess_p50": _pct(self._lat_post, 50),
                    "schedule_lag_p95": _pct(self._lag, 95),
                },
                "last_error": self.last_error,
                "last_result_ts": self.last_result_wall,
            }
            res = self.result
        src = self.source.stats()
        base_seq = getattr(self.source, "_seq_at_reset", 0)
        base_cons = getattr(self.source, "_consumed_at_reset", 0)
        src["frames_decoded"] -= base_seq
        src["frames_consumed"] -= base_cons
        src["frames_dropped"] = max(src["frames_decoded"] - src["frames_consumed"], 0)
        out["source"] = src
        if res is not None:
            out["detections"] = len(res["detections"])
            out["result_age_s"] = round(now - res["t_end"], 2)
            counts: dict[str, int] = {}
            for d in res["detections"]:
                counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
            out["detections_by_class"] = counts
        else:
            out["detections"] = 0
            out["result_age_s"] = None
            out["detections_by_class"] = {}
        return out


class InferenceWorker(threading.Thread):
    """One engine/context, several channels, round-robin on *due* channels."""

    def __init__(
        self,
        idx: int,
        make_detector: Callable[[], Any],
        channels: list[Channel],
        *,
        idle_sleep_s: float = 0.002,
    ):
        super().__init__(name=f"infer-{idx}", daemon=True)
        self.idx = idx
        self.make_detector = make_detector
        self.channels = channels
        self.idle_sleep_s = idle_sleep_s
        self.detector = None
        self.ready = threading.Event()
        self.error: str | None = None
        self._stop_evt = threading.Event()
        self._rr = 0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        try:
            self.detector = self.make_detector()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
            return
        self.ready.set()
        n = len(self.channels)
        while not self._stop_evt.is_set():
            now = time.monotonic()
            served = False
            for k in range(n):
                ch = self.channels[(self._rr + k) % n]
                if not ch.due(now):
                    continue
                self._serve(ch)
                served = True
                now = time.monotonic()
            self._rr = (self._rr + 1) % max(n, 1)
            if not served:
                time.sleep(self.idle_sleep_s)
        with contextlib.suppress(Exception):
            self.detector.close()

    def _serve(self, ch: Channel) -> None:
        frame, seq, t_capture = ch.source.latest()
        t_start = time.monotonic()
        ch.next_due = t_start + ch.min_dt
        if frame is None:
            return
        try:
            dets, timing = self.detector.detect(frame, bgr=True)  # decoder frames are BGR(x)
            t_end = time.monotonic()
            ch.record(
                {
                    "seq": seq,
                    "t_capture": t_capture,
                    "t_end": t_end,
                    "detections": dets,
                    "timing_ms": timing,
                    "frame_size": [frame.shape[1], frame.shape[0]],
                    "frame": frame,  # the exact frame these boxes belong to (reference, no copy)
                },
                t_capture,
                t_start,
                t_end,
                timing,
            )
        except Exception as exc:  # noqa: BLE001
            ch.errors += 1
            ch.last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)


class ChannelSet:
    """Builds sources, channels and workers from plain arguments; owns their lifecycle."""

    def __init__(
        self,
        *,
        engine: Path,
        config: Path,
        sources: list[dict[str, Any]],
        width: int,
        height: int,
        fps_limit: float,
        workers: int = 1,
        backend: str = "auto",
        loop: bool = True,
        latency_ms: int = 200,
        confidence: float | None = None,
        make_detector: Callable[[], Any] | None = None,
    ):
        from .ops import Detector, load_runtime_config

        self.engine, self.config = Path(engine), Path(config)
        self.runtime_cfg = load_runtime_config(self.config)
        self.confidence_override = confidence
        self.sources = [
            FrameSource(
                s["url"],
                width,
                height,
                name=str(s.get("name") or f"cam{i}"),
                backend=backend,
                loop=loop,
                latency_ms=latency_ms,
            )
            for i, s in enumerate(sources)
        ]
        self.channels = [Channel(i, src.name, src, fps_limit) for i, src in enumerate(self.sources)]
        workers = max(1, min(int(workers), len(self.channels)))

        def _default_make():
            det = Detector(self.engine, self.config)
            if self.confidence_override is not None:
                det.cfg["postprocess"]["confidence"] = float(self.confidence_override)
            return det

        self.workers = [
            InferenceWorker(
                w,
                make_detector or _default_make,
                [c for c in self.channels if c.idx % workers == w],
            )
            for w in range(workers)
        ]

    def start(self, ready_timeout_s: float = 300) -> ChannelSet:
        for w in self.workers:
            w.start()
        for w in self.workers:
            w.ready.wait(ready_timeout_s)
            if w.error:
                raise RuntimeError(f"inference worker {w.idx}: {w.error}")
        for s in self.sources:
            s.start()
        return self

    def stop(self) -> None:
        for w in self.workers:
            w.stop()
        for s in self.sources:
            s.stop()
        for w in self.workers:
            if w.is_alive():
                w.join(timeout=10)

    def reset_stats(self) -> None:
        for c in self.channels:
            c.reset_stats()

    # -- live settings (no restart) ------------------------------------------------------------

    def set_fps_limit(self, fps_limit: float) -> None:
        for c in self.channels:
            c.set_fps_limit(fps_limit)

    def set_confidence(self, confidence: float | None) -> None:
        self.confidence_override = confidence
        value = (
            float(confidence)
            if confidence is not None
            else float(self.runtime_cfg["postprocess"]["confidence"])
        )
        for w in self.workers:
            det = w.detector
            if det is not None and hasattr(det, "cfg"):
                det.cfg["postprocess"]["confidence"] = value

    def effective_confidence(self) -> float:
        if self.confidence_override is not None:
            return float(self.confidence_override)
        return float(self.runtime_cfg["postprocess"]["confidence"])

    def describe(self) -> dict[str, Any]:
        det = self.workers[0].detector if self.workers else None
        eng = getattr(det, "engine", None)
        return {
            "engine": eng.describe() if eng is not None and hasattr(eng, "describe") else None,
            "classes": self.runtime_cfg["classes"],
            "input_size": self.runtime_cfg["input_size"],
            "precision": self.runtime_cfg.get("precision"),
            "model_version": self.runtime_cfg.get("model_version"),
            "export_id": self.runtime_cfg.get("export_id"),
            "project": self.runtime_cfg.get("project"),
            "confidence": self.effective_confidence(),
            "confidence_default": float(self.runtime_cfg["postprocess"]["confidence"]),
            "iou": self.runtime_cfg["postprocess"].get("iou"),
            "workers": len(self.workers),
            "channels": len(self.channels),
            "decode_backend": self.sources[0].backend if self.sources else None,
        }
