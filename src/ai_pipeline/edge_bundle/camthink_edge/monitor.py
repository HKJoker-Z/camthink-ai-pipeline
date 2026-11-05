"""RTSP monitoring mode of the inference service (``camthink_edge serve`` with sources).

One process: camera channels -> TensorRT detection -> results. It publishes for the separate
live-view server (:mod:`liveview`) through a shared-memory directory (:func:`appcfg.live_dir`):

    stats.json       every second: channels, model, system usage, config (masked), state
    chN.jpg / chN.json   the last inferred frame of channel N with its boxes drawn + metadata -
                     written only while a viewer is connected (``viewers.json`` is fresh) and
                     at most at ``liveview.display_fps``; browsers never trigger decoding or
                     inference, they only cost one JPEG encode per displayed frame.

The heartbeat ``<root>/service/status.json`` keeps the fields ``camthink_edge health`` checks,
so ``ai-pipeline health`` / ``service`` / ``target status`` work unchanged.

Hot reload: ``inference.fps`` and ``inference.confidence`` from ``app.json`` are applied
without a restart; changes to sources / decode size are reported as ``restart_required``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from . import VERSION, appcfg, overlay
from .channels import ChannelSet
from .sysmon import SystemMonitor

VIEWER_FRESH_S = 4.0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def viewers_active(live: Path) -> int:
    """Number of connected live-view clients, 0 if the marker is missing or stale."""
    try:
        v = json.loads((live / "viewers.json").read_text(encoding="utf-8"))
        if time.time() - float(v.get("ts", 0)) <= VIEWER_FRESH_S:
            return int(v.get("count", 0))
    except (OSError, ValueError, TypeError):
        pass
    return 0


class Publisher(threading.Thread):
    """Encodes each channel's newest inferred frame (with boxes) into the live directory."""

    def __init__(self, cs: ChannelSet, live: Path, cfg_ref: dict[str, Any]):
        super().__init__(name="publisher", daemon=True)
        self.cs = cs
        self.live = live
        self.cfg_ref = cfg_ref  # {"cfg": app config}; re-read on every frame (hot settings)
        self._stop = threading.Event()
        self._last_seq: dict[int, int] = {}
        self._jpg: dict[int, tuple[int, float]] = {}
        self._last_pub = 0.0
        self.published = 0
        self.skipped_no_viewer = 0
        self._enc_ms: deque = deque(maxlen=200)
        self._pub_times: deque = deque(maxlen=200)
        self.viewers = 0

    def stop(self) -> None:
        self._stop.set()

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        recent = [t for t in self._pub_times if now - t <= 5.0]
        fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6) if len(recent) > 1 else 0.0
        enc = list(self._enc_ms)
        return {
            "viewers": self.viewers,
            "publishing": self.viewers > 0,
            "published_frames": self.published,
            "publish_fps": round(fps, 1),
            "encode_ms_p50": round(float(np.percentile(enc, 50)), 2) if enc else None,
            "dir": str(self.live),
        }

    def _meta(self, ch: Any, res: dict[str, Any], *, jpg: bool) -> dict[str, Any]:
        """Boxes + timing of the newest inference; ``jpg_seq``/``jpg_ts`` say which frame the
        JPEG next to it shows (older than ``seq`` while nobody is watching)."""
        now = time.time()
        if jpg:
            self._jpg[ch.idx] = (res["seq"], now)
        jpg_seq, jpg_ts = self._jpg.get(ch.idx, (None, None))
        return {
            "channel": ch.idx,
            "name": ch.name,
            "seq": res["seq"],
            "ts": now,
            "jpg_seq": jpg_seq,
            "jpg_ts": jpg_ts,
            "frame_size": res["frame_size"],
            "detections": res["detections"],
            "timing_ms": res["timing_ms"],
        }

    def run(self) -> None:
        last_meta = 0.0
        while not self._stop.is_set():
            lv = self.cfg_ref["cfg"].get("liveview", {})
            period = 1.0 / max(float(lv.get("display_fps", 10.0)), 0.5)
            quality = int(lv.get("quality", 70))
            self.viewers = viewers_active(self.live)
            now = time.monotonic()
            if self.viewers <= 0:
                # Nobody watching: no JPEG work, but keep the detection metadata (boxes only,
                # a few hundred bytes) fresh at 2 Hz for /api/detections and the CLI.
                if now - last_meta >= 0.5:
                    for ch in self.cs.channels:
                        with ch.lock:
                            res = ch.result
                        if res is not None and self._last_seq.get(ch.idx) != res["seq"]:
                            _write_json_atomic(
                                self.live / f"ch{ch.idx}.json", self._meta(ch, res, jpg=False)
                            )
                            self._last_seq[ch.idx] = res["seq"]
                    last_meta = now
                self._stop.wait(0.25)
                continue
            if now - self._last_pub < period:
                self._stop.wait(min(period - (now - self._last_pub), 0.05))
                continue
            did = False
            for ch in self.cs.channels:
                with ch.lock:
                    res = ch.result
                if res is None or self._last_seq.get(ch.idx) == res["seq"]:
                    continue
                frame = res.get("frame")
                if frame is None:
                    continue
                st = ch.stats()
                lat = st["latency_ms"]
                hdr = (
                    f"{ch.name}  video {st['source']['decode_fps']} fps  detect {st['fps']} fps  "
                    f"infer {lat['inference_p50'] or 0} ms  e2e {lat['e2e_p50'] or 0} ms"
                )
                t0 = time.perf_counter()
                img = overlay.draw(frame, res["detections"], hdr)
                data = overlay.encode_jpeg(img, quality)
                self._enc_ms.append((time.perf_counter() - t0) * 1000.0)
                if not data:
                    continue
                _write_bytes_atomic(self.live / f"ch{ch.idx}.jpg", data)
                _write_json_atomic(self.live / f"ch{ch.idx}.json", self._meta(ch, res, jpg=True))
                self._last_seq[ch.idx] = res["seq"]
                self.published += 1
                did = True
            if did:
                self._last_pub = time.monotonic()
                self._pub_times.append(self._last_pub)
            else:
                self._stop.wait(0.01)


def run_monitor(
    root: Path,
    cfg: dict[str, Any],
    *,
    heartbeat_s: float = 2.0,
    stop: threading.Event | None = None,
) -> int:
    from .ops import current_deployment

    root = Path(root)
    status_path = root / "service" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    live = appcfg.live_dir(root)
    live.mkdir(parents=True, exist_ok=True)
    os.chmod(live, 0o700)
    started = time.time()
    sources_masked = [appcfg.mask_url(s["url"]) for s in cfg["sources"]]

    def heartbeat(**fields: Any) -> None:
        payload = {
            "ts": _now_iso(),
            "ts_epoch": time.time(),
            "runtime_version": VERSION,
            "pid": os.getpid(),
            "rss_mb": _rss_mb(),
            "mode": "rtsp",
            "source": sources_masked,
            "live_dir": str(live),
            **fields,
        }
        _write_json_atomic(status_path, payload)

    current = current_deployment(root)
    if current is None:
        heartbeat(deployment=None, error="no active deployment")
        return 1
    dep = root / "deployments" / current
    heartbeat(deployment=current, frames=0, fps=0.0, state="starting")
    inf = cfg["inference"]
    try:
        cs = ChannelSet(
            engine=dep / "model.engine",
            config=dep / "runtime.json",
            sources=cfg["sources"],
            width=int(inf["width"]),
            height=int(inf["height"]),
            fps_limit=float(inf["fps"]),
            workers=1,
            backend=str(inf.get("backend", "auto")),
            loop=True,
            latency_ms=int(inf.get("latency_ms", 200)),
            confidence=inf.get("confidence"),
        )
        cs.start()
    except Exception as exc:  # noqa: BLE001
        heartbeat(deployment=current, error=f"failed to load engine: {type(exc).__name__}: {exc}")
        return 1

    sysmon = SystemMonitor().start()
    cfg_ref = {"cfg": cfg}
    pub = Publisher(cs, live, cfg_ref)
    pub.start()
    stop = stop or threading.Event()

    def _sig(*_: Any) -> None:
        stop.set()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sig)
        signal.signal(signal.SIGINT, _sig)

    cfg_path = appcfg.app_config_path(root)
    cfg_mtime = cfg_path.stat().st_mtime if cfg_path.is_file() else 0.0
    restart_required = {"inference": False, "liveview": False}
    boot_cfg = cfg

    def reload_config() -> None:
        nonlocal cfg_mtime
        try:
            m = cfg_path.stat().st_mtime
        except OSError:
            return
        if m == cfg_mtime:
            return
        cfg_mtime = m
        try:
            new = appcfg.load(root)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"config_reload_error": str(exc)}), flush=True)
            return
        if appcfg.validate(new):
            print(json.dumps({"config_reload_error": appcfg.validate(new)}), flush=True)
            return
        cs.set_fps_limit(float(new["inference"]["fps"]))
        cs.set_confidence(new["inference"].get("confidence"))
        req = appcfg.restart_required(boot_cfg, new)
        restart_required["inference"] = req["inference"]
        cfg_ref["cfg"] = new

    def stats_payload(state: str) -> dict[str, Any]:
        chans = [c.stats() for c in cs.channels]
        c0 = chans[0] if chans else {}
        cur = cfg_ref["cfg"]
        return {
            "ts": _now_iso(),
            "ts_epoch": time.time(),
            "state": state,
            "runtime_version": VERSION,
            "pid": os.getpid(),
            "process_rss_mb": _rss_mb(),
            "uptime_s": round(time.time() - started, 1),
            "deployment": current,
            "model": cs.describe(),
            "channels": chans,
            "camera_connected": all(c["source"]["connected"] for c in chans) if chans else False,
            "fps": c0.get("fps", 0.0),
            "system": sysmon.snapshot(),
            "system_backend": sysmon.backend,
            "power_mode": sysmon.power_mode,
            "live": pub.stats(),
            "config": {
                "sources": [
                    {"name": n, "url": u}
                    for n, u in zip(  # noqa: B905 - py3.8 runtime
                        appcfg.source_names(cur), sources_masked
                    )
                ],
                "inference": cur["inference"],
                "liveview": cur["liveview"],
            },
            "config_restart_required": dict(restart_required),
        }

    print(
        json.dumps(
            {
                "monitor": "started",
                "deployment": current,
                "channels": len(cs.channels),
                "sources": sources_masked,
                "model": cs.describe(),
                "live_dir": str(live),
            },
            default=str,
        ),
        flush=True,
    )
    last_hb = 0.0
    last_stats = 0.0
    last_reload = 0.0
    try:
        while not stop.is_set():
            now = time.monotonic()
            if now - last_reload >= 2.0:
                reload_config()
                last_reload = now
            if now - last_stats >= 1.0:
                _write_json_atomic(live / "stats.json", stats_payload("running"))
                last_stats = now
            if now - last_hb >= heartbeat_s:
                chans = cs.channels
                res = chans[0].result if chans else None
                heartbeat(
                    deployment=current,
                    frames=sum(c.frames for c in chans),
                    errors=sum(c.errors for c in chans),
                    fps=chans[0].stats()["fps"] if chans else 0.0,
                    state="running",
                    camera_connected=all(c.source.stats()["connected"] for c in chans),
                    channels=len(chans),
                    last_detections=(res or {}).get("detections", [])[:10],
                    last_detection_count=len((res or {}).get("detections", [])),
                    config_restart_required=restart_required["inference"],
                )
                last_hb = now
            stop.wait(0.2)
    finally:
        pub.stop()
        cs.stop()
        sysmon.stop()
        with contextlib.suppress(Exception):
            _write_json_atomic(live / "stats.json", stats_payload("stopped"))
        heartbeat(
            deployment=current,
            frames=sum(c.frames for c in cs.channels),
            errors=sum(c.errors for c in cs.channels),
            state="stopped",
        )
    return 0
