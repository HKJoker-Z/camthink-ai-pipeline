"""Video sources with *latest-frame* semantics (one decoder per channel).

``FrameSource`` keeps only the most recent decoded frame; consumers call :meth:`latest` and
decide themselves how often to run inference. Frames decoded but never consumed are counted as
dropped, which is the honest number for a detector that runs slower than the camera.

Backends (all in-process through OpenCV, so the RTSP URL - which may carry credentials -
never appears in a process argument list):

``cv2gst``  OpenCV's GStreamer backend driving NVIDIA's hardware decoder
            (``rtspsrc ! parsebin ! nvv4l2decoder ! nvvidconv ! BGRx@WxH ! appsink``): no CPU
            decoding, frames arrive already scaled. Needs a GStreamer-enabled OpenCV and the
            JetPack plugins (verified on the NG4511, JetPack 6).
``cv2``     OpenCV/FFmpeg software decoding (any machine); frames are resized on the CPU.

Sources: ``rtsp://`` URLs and local video files (``loop=True`` restarts a file at EOF).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


def is_rtsp(url: str) -> bool:
    return url.lower().startswith(("rtsp://", "rtsps://"))


def _cv2_has_gstreamer() -> bool:
    try:
        import cv2  # type: ignore

        return (
            "GStreamer:" in cv2.getBuildInformation()
            and "YES" in (cv2.getBuildInformation().split("GStreamer:", 1)[1].splitlines()[0])
        )
    except Exception:  # noqa: BLE001
        return False


def _nvdec_available() -> bool:
    if not shutil.which("gst-inspect-1.0"):
        return False
    try:
        rc = subprocess.run(
            ["gst-inspect-1.0", "nvv4l2decoder"], capture_output=True, timeout=20
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return False
    return rc == 0


_BACKEND_CACHE: dict[str, str] = {}


def pick_backend(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if "auto" not in _BACKEND_CACHE:
        _BACKEND_CACHE["auto"] = "cv2gst" if _cv2_has_gstreamer() and _nvdec_available() else "cv2"
    return _BACKEND_CACHE["auto"]


def gst_pipeline(url: str, width: int, height: int, *, latency_ms: int = 200) -> str:
    """Hardware decode -> BGRx at width x height -> appsink (for ``cv2.CAP_GSTREAMER``).

    ``parsebin`` autoplugs the RTP depayloader + parser for H.264/H.265 so the stream always
    lands in ``nvv4l2decoder``. ``drop=true max-buffers=1`` keeps the latest frame only.
    """
    loc = url.replace('"', '\\"')
    if is_rtsp(url):
        src = (
            f'rtspsrc location="{loc}" protocols=tcp latency={int(latency_ms)} drop-on-latency=true'
        )
    else:
        src = f'filesrc location="{Path(url).resolve()}"'
    return (
        f"{src} ! parsebin ! nvv4l2decoder ! nvvidconv ! "
        f"video/x-raw,format=BGRx,width={int(width)},height={int(height)} ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class FrameSource:
    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        *,
        name: str = "cam",
        backend: str = "auto",
        latency_ms: int = 200,
        loop: bool = False,
        reconnect_s: float = 2.0,
    ) -> None:
        self.url, self.width, self.height, self.name = url, int(width), int(height), name
        self.loop = loop
        self.latency_ms = latency_ms
        self.reconnect_s = reconnect_s
        self.backend = pick_backend(backend)
        self._lock = threading.Lock()
        self._frame: Any = None
        self._seq = 0  # decoded frames
        self._consumed_seq = 0  # seq of the last frame handed out
        self._consumed = 0  # distinct frames handed out
        self._ts = 0.0
        self._times: deque = deque(maxlen=120)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.reconnects = -1
        self.error: str | None = None
        self.started_at = 0.0
        self.connected_at: float | None = None
        self.disconnected_at: float | None = None
        self.native_size: list[int] | None = None  # as reported by the decoder, if known

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> FrameSource:
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"src-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8)

    # -- consumer API ------------------------------------------------------------------------

    def latest(self) -> tuple[Any, int, float]:
        """(frame, seq, capture_ts). Marks the frame consumed (for drop accounting)."""
        with self._lock:
            if self._frame is not None and self._seq != self._consumed_seq:
                self._consumed_seq = self._seq
                self._consumed += 1
            return self._frame, self._seq, self._ts

    def has_new(self) -> bool:
        return self._seq != self._consumed_seq

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            times = list(self._times)
            seq, consumed, ts = self._seq, self._consumed, self._ts
        fps = 0.0
        recent = [t for t in times if now - t <= 5.0]
        if len(recent) > 1:
            fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
        live = bool(ts) and self.connected and (now - ts) < 5.0
        return {
            "name": self.name,
            "backend": self.backend,
            "connected": live,
            "decode_fps": round(fps, 1),
            "frames_decoded": seq,
            "frames_consumed": consumed,
            "frames_dropped": max(seq - consumed, 0),
            "reconnects": max(self.reconnects, 0),
            "last_frame_age_s": round(now - ts, 2) if ts else None,
            "size": [self.width, self.height],
            "native_size": self.native_size,
            "error": self.error,
            "connected_for_s": (
                round(now - self.connected_at, 1) if live and self.connected_at else None
            ),
            "disconnected_for_s": (
                round(now - (self.disconnected_at or self.started_at), 1)
                if not live and self.started_at
                else None
            ),
        }

    # -- producer thread -------------------------------------------------------------------

    def _publish(self, frame) -> None:
        now = time.monotonic()
        with self._lock:
            self._frame = frame
            self._seq += 1
            self._ts = now
            self._times.append(now)
        if not self.connected:
            self.connected_at = now
        self.connected = True

    def _run(self) -> None:
        while not self._stop.is_set():
            self.reconnects += 1
            try:
                if not self.url.strip():
                    raise ValueError("empty source URL")
                if not is_rtsp(self.url) and "://" not in self.url and not Path(self.url).is_file():
                    raise FileNotFoundError(self.url)
                self._run_cv2(gst=self.backend == "cv2gst")
            except Exception as exc:  # noqa: BLE001 - keep the channel alive, report
                self.error = f"{type(exc).__name__}: {exc}"
            if self.connected:
                self.disconnected_at = time.monotonic()
            self.connected = False
            if self._stop.is_set():
                break
            if not self.loop or is_rtsp(self.url):
                self._stop.wait(self.reconnect_s)

    def _run_cv2(self, *, gst: bool) -> None:
        import cv2  # type: ignore

        if gst:
            cap = cv2.VideoCapture(
                gst_pipeline(self.url, self.width, self.height, latency_ms=self.latency_ms),
                cv2.CAP_GSTREAMER,
            )
        else:
            if is_rtsp(self.url):
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.error = "cannot open source" if is_rtsp(self.url) else f"cannot open {self.url}"
            return
        try:
            if not gst:
                w, h = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                if w and h:
                    self.native_size = [w, h]
            misses = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.loop and not is_rtsp(self.url):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        misses += 1
                        if misses > 5:
                            return  # unreadable file: reopen
                        continue
                    self.error = "stream ended" if self.connected else "no frames received"
                    return  # EOF / decoder gone -> reconnect
                misses = 0
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(
                        frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                    )
                self.error = None
                self._publish(frame)
        finally:
            cap.release()


def probe(
    url: str, *, width: int, height: int, backend: str = "auto", seconds: float = 5.0
) -> dict[str, Any]:
    """Open a source, read frames for ``seconds``; report connectivity without the URL."""
    src = FrameSource(url, width, height, name="probe", backend=backend, loop=False)
    t0 = time.monotonic()
    src.start()
    first: float | None = None
    deadline = t0 + seconds
    while time.monotonic() < deadline:
        if first is None and src._seq > 0:
            first = round(time.monotonic() - t0, 2)
        time.sleep(0.1)
    st = src.stats()
    src.stop()
    return {
        "ok": st["frames_decoded"] > 0,
        "backend": src.backend,
        "frames": st["frames_decoded"],
        "decode_fps": st["decode_fps"],
        "first_frame_s": first,
        "size": st["size"],
        "native_size": st["native_size"],
        "reconnect_attempts": st["reconnects"],
        "error": None if st["frames_decoded"] > 0 else (src.error or "no frames received"),
        "seconds": seconds,
    }
