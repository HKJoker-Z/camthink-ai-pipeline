"""Edge runtime M9: app config, file/RTSP frame source, channel scheduling, the RTSP monitor
mode of `serve` and the live-view HTTP server — all on a synthetic clip with a fake detector
(no TensorRT, no camera)."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

BUNDLE_PARENT = Path(__file__).resolve().parents[2] / "src" / "ai_pipeline" / "edge_bundle"
if str(BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_PARENT))

from camthink_edge import appcfg, overlay  # noqa: E402

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

RTSP = "rtsp://admin:S3cret%21@192.168.5.78:554/main"


# -- appcfg ----------------------------------------------------------------------------------------


def test_mask_and_masked_copy():
    assert appcfg.mask_url(RTSP) == "rtsp://***@192.168.5.78:554/main"
    assert appcfg.mask_url("rtsp://192.168.5.78/main") == "rtsp://192.168.5.78/main"
    assert appcfg.mask_url("/tmp/clip.mp4") == "/tmp/clip.mp4"
    assert appcfg.mask_url(None) is None
    cfg = appcfg.merge_defaults({"sources": [{"name": "cam0", "url": RTSP}]})
    m = appcfg.masked(cfg)
    assert m["sources"][0]["url"] == "rtsp://***@192.168.5.78:554/main"
    assert cfg["sources"][0]["url"] == RTSP  # original untouched
    assert m["inference"]["fps"] == 10.0 and m["liveview"]["port"] == 8080
    assert m["device"] == {"name": None, "label": None}


def test_validate_and_restart_required():
    cfg = appcfg.merge_defaults({"sources": [{"name": "cam0", "url": RTSP}]})
    assert appcfg.validate(cfg) == []
    bad = appcfg.merge_defaults(
        {
            "sources": [{"name": "a b", "url": "ftp://x"}, {"url": ""}],
            "inference": {"fps": 0, "width": 333, "confidence": 2, "backend": "ffmpeg"},
            "liveview": {"port": 70000, "quality": 5},
        }
    )
    problems = appcfg.validate(bad)
    assert len(problems) >= 7
    new = json.loads(json.dumps(cfg))
    new["inference"]["fps"] = 3
    new["inference"]["confidence"] = 0.6
    new["liveview"]["display_fps"] = 2
    assert appcfg.restart_required(cfg, new) == {"inference": False, "liveview": False}
    new["inference"]["width"] = 640
    assert appcfg.restart_required(cfg, new) == {"inference": True, "liveview": False}
    new = json.loads(json.dumps(cfg))
    new["liveview"]["port"] = 8081
    assert appcfg.restart_required(cfg, new) == {"inference": False, "liveview": True}
    new = json.loads(json.dumps(cfg))
    new["sources"].append({"name": "cam1", "url": "rtsp://x/y"})
    assert appcfg.restart_required(cfg, new)["inference"] is True


def test_save_is_private_and_load_merges_defaults(tmp_path: Path):
    root = tmp_path / "root"
    appcfg.save(root, {"sources": [{"name": "cam0", "url": RTSP}]})
    p = appcfg.app_config_path(root)
    assert p == root / "service" / "app.json"
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"
    loaded = appcfg.load(root)
    assert loaded["sources"][0]["url"] == RTSP and loaded["inference"]["width"] == 1280
    assert appcfg.load(tmp_path / "nowhere")["sources"] == []
    assert appcfg.source_names({"sources": [{"url": "a"}, {"name": "x", "url": "b"}]}) == [
        "cam0",
        "x",
    ]
    assert appcfg.live_dir(root).name.startswith("camthink-ai-")


# -- overlay ---------------------------------------------------------------------------------------


def test_overlay_draws_and_encodes():
    frame = np.zeros((120, 160, 4), np.uint8)  # BGRx as produced by nvvidconv
    dets = [{"class_id": 1, "class_name": "dog", "confidence": 0.77, "bbox_xyxy": [10, 10, 60, 80]}]
    out = overlay.draw(frame, dets, "cam0  9.8 fps")
    assert out.shape[2] == 3 and out.any()
    jpg = overlay.encode_jpeg(out, 70)
    assert jpg[:2] == b"\xff\xd8"
    assert overlay.class_colour(1) != overlay.class_colour(2)
    assert overlay.class_colour_hex(0).startswith("#") and len(overlay.class_colour_hex(0)) == 7
    ph = overlay.placeholder(160, 120, "camera disconnected")
    assert ph.shape == (120, 160, 3)


# -- end to end: monitor + liveview on a clip ------------------------------------------------------


CLIP_W, CLIP_H, CLIP_FPS = 320, 180, 25
# Encoders to try, most portable first. Which ones exist depends on the OpenCV build (pip
# wheels bundle FFmpeg; distro builds may not), so the clip is verified by reading it back.
CLIP_CODECS = (("MJPG", ".avi"), ("mp4v", ".mp4"), ("XVID", ".avi"), ("MJPG", ".mkv"))


def _clip_frame(i: int) -> np.ndarray:
    f = np.zeros((CLIP_H, CLIP_W, 3), np.uint8)
    x = 10 + (i * 3) % 200
    cv2.rectangle(f, (x, 50), (x + 60, 120), (0, 200, 255), -1)
    return f


def _write_clip(directory: Path, frames: int = 150) -> Path:
    """Deterministic synthetic clip (moving orange square the FakeDetector finds). Returns
    the path of the first encoding this OpenCV build can both write and read back."""
    attempts = []
    for fourcc, ext in CLIP_CODECS:
        path = directory / f"clip{ext}"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), CLIP_FPS, (CLIP_W, CLIP_H))
        if not vw.isOpened():
            attempts.append(f"{fourcc}{ext}: writer not opened")
            continue
        for i in range(frames):
            vw.write(_clip_frame(i))
        vw.release()
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if ok and frame is not None and frame.shape[:2] == (CLIP_H, CLIP_W) and n >= frames // 2:
            return path
        attempts.append(f"{fourcc}{ext}: read back ok={ok} frames={n} size={path.stat().st_size}")
        path.unlink(missing_ok=True)
    raise AssertionError(
        "no usable video encoder in this OpenCV build "
        f"({cv2.__version__}); tried: {'; '.join(attempts)}"
    )


class FakeDetector:
    """Finds the orange square by colour; mimics `ops.Detector`."""

    def __init__(self, engine, config):
        from camthink_edge.ops import load_runtime_config

        self.cfg = load_runtime_config(config)
        self.calls = 0

    def detect(self, img, *, bgr=False):
        self.calls += 1
        t = time.perf_counter()
        mask = cv2.inRange(img[:, :, :3], (0, 150, 200), (10, 255, 255))
        ys, xs = np.where(mask > 0)
        dets = []
        if len(xs) and float(self.cfg["postprocess"]["confidence"]) <= 0.9:
            dets = [
                {
                    "class_id": 0,
                    "class_name": self.cfg["classes"][0],
                    "confidence": 0.9,
                    "bbox_xyxy": [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max()),
                        float(ys.max()),
                    ],
                }
            ]
        return dets, {
            "preprocess_ms": 0.5,
            "inference_ms": round((time.perf_counter() - t) * 1000, 3),
            "postprocess_ms": 0.1,
        }

    def close(self):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def device_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import camthink_edge.ops as ops

    monkeypatch.setattr(ops, "Detector", FakeDetector)
    root = tmp_path / "camthink-ai"
    dep = root / "deployments" / "demo-dep-001"
    dep.mkdir(parents=True)
    (root / "service").mkdir()
    _write_clip(root)
    (dep / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project": "demo",
                "model_version": "mv-001",
                "export_id": "exp-001",
                "classes": ["cat", "dog"],
                "input_size": 640,
                "precision": "fp16",
                "postprocess": {"confidence": 0.25},
            }
        )
    )
    (dep / "deployment.json").write_text(json.dumps({"deployment_id": "dep-001"}))
    (dep / "model.engine").write_bytes(b"fake")
    os.symlink(Path("deployments") / "demo-dep-001", root / "current")
    # keep the shared-memory dir inside tmp so parallel test runs never collide
    monkeypatch.setattr(appcfg, "live_dir", lambda r: tmp_path / "shm")
    return root


def _clip(root: Path) -> Path:
    return next(p for p in root.iterdir() if p.name.startswith("clip"))


def test_frame_source_reads_file_and_reports_stats(device_root: Path):
    from camthink_edge import stream

    clip = _clip(device_root)
    src = stream.FrameSource(str(clip), 320, 180, name="t", backend="cv2", loop=True)
    src.start()
    try:
        deadline = time.time() + 10
        while not src.has_new() and time.time() < deadline:
            time.sleep(0.05)
        frame, seq, ts = src.latest()
        assert frame is not None and frame.shape[:2] == (180, 320)
        st = src.stats()
        assert st["connected"] is True and st["frames_decoded"] >= 1
        assert st["backend"] == "cv2" and st["size"] == [320, 180]
    finally:
        src.stop()
    assert src.stats()["connected"] is False
    bad = stream.probe("/nonexistent/clip.mp4", width=320, height=180, backend="cv2", seconds=1)
    assert bad["ok"] is False and bad["frames"] == 0 and bad["error"]
    good = stream.probe(str(clip), width=320, height=180, backend="cv2", seconds=1)
    assert good["ok"] is True and good["frames"] > 5 and good["native_size"] == [320, 180]
    assert stream.is_rtsp(RTSP) and not stream.is_rtsp("/x.mp4")
    assert stream.pick_backend("cv2") == "cv2"
    assert "nvv4l2decoder" in stream.gst_pipeline(RTSP, 1280, 720, latency_ms=200)


def _get(port: int, path: str, *, binary: bool = False):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        data = r.read()
    return data if binary else json.loads(data)


class Service:
    """Runs `run_monitor` / `run_liveview` in a thread, keeps its exception, stops it on exit."""

    def __init__(self, name: str, target, **kwargs):
        self.name = name
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.result = None

        def body():
            try:
                self.result = target(stop=self.stop, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced by Harness.check()
                self.error = exc

        self.thread = threading.Thread(target=body, name=name, daemon=True)

    def start(self) -> Service:
        self.thread.start()
        return self

    def shutdown(self, timeout: float = 15.0) -> None:
        self.stop.set()
        self.thread.join(timeout)

    def describe(self) -> str:
        state = "alive" if self.thread.is_alive() else f"exited (result={self.result!r})"
        return f"{self.name}: {state}" + (f" error={self.error!r}" if self.error else "")


class Harness:
    """Monitor + live-view services on a device root, with diagnostics for every wait."""

    def __init__(self, root: Path, port: int, *, heartbeat_s: float = 0.5):
        from camthink_edge import liveview, monitor

        self.root, self.port = root, port
        self.live = appcfg.live_dir(root)
        self.services = [
            Service(
                "monitor",
                monitor.run_monitor,
                root=root,
                cfg=appcfg.load(root),
                heartbeat_s=heartbeat_s,
            ),
            Service("liveview", liveview.run_liveview, root=root),
        ]

    def __enter__(self) -> Harness:
        for svc in self.services:
            svc.start()
        return self

    def __exit__(self, *exc) -> None:
        for svc in self.services:
            svc.shutdown()
        for svc in self.services:
            assert not svc.thread.is_alive(), f"{svc.name} did not stop: {svc.describe()}"
            assert svc.error is None, f"{svc.name} crashed: {svc.error!r}"

    def check(self) -> None:
        """Fail immediately (instead of waiting for a timeout) when a service died."""
        for svc in self.services:
            if svc.error is not None or not svc.thread.is_alive():
                raise AssertionError(f"service stopped unexpectedly: {self.diagnostics()}")

    def _read(self, path: Path) -> str:
        try:
            return path.read_text()[:1500]
        except OSError as exc:
            return f"<{exc.__class__.__name__}>"

    def diagnostics(self) -> str:
        lines = [svc.describe() for svc in self.services]
        lines.append(f"live dir {self.live}: {sorted(p.name for p in self.live.glob('*'))}")
        lines.append("status.json: " + self._read(self.root / "service" / "status.json"))
        lines.append("stats.json: " + self._read(self.live / "stats.json"))
        try:
            lines.append("/api/status: " + json.dumps(self.get("/api/status"))[:1500])
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            lines.append(f"/api/status: <{exc!r}>")
        return "\n".join(lines)

    def get(self, path: str, *, binary: bool = False):
        return _get(self.port, path, binary=binary)

    def wait(self, what: str, pred, timeout: float = 20.0, step: float = 0.1):
        """Poll `pred` (exceptions count as "not yet") until truthy; rich failure message."""
        deadline = time.time() + timeout
        last: object = None
        while time.time() < deadline:
            self.check()
            try:
                last = pred()
            except (urllib.error.URLError, OSError, KeyError, IndexError, TypeError) as exc:
                last = exc
            else:
                if last:
                    return last
            time.sleep(step)
        raise AssertionError(
            f"timed out after {timeout}s waiting for {what}\nlast value: {last!r}\n"
            + self.diagnostics()
        )

    def status(self):
        return self.get("/api/status")

    def heartbeat(self) -> dict:
        return json.loads((self.root / "service" / "status.json").read_text())

    def mjpeg_parts(self, ch: int = 0, *, parts: int = 5, timeout: float = 10.0) -> bytes:
        sk = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sk.sendall(f"GET /stream.mjpg?ch={ch} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            buf = b""
            deadline = time.time() + timeout
            while buf.count(b"--frame") < parts and time.time() < deadline:
                self.check()
                chunk = sk.recv(65536)
                if not chunk:
                    break
                buf += chunk
        finally:
            sk.close()
        assert buf.count(b"--frame") >= parts, (
            f"only {buf.count(b'--frame')} MJPEG parts in {timeout}s\n" + self.diagnostics()
        )
        return buf


def test_monitor_and_liveview_end_to_end(device_root: Path):
    root = device_root
    clip = _clip(root)
    port = _free_port()
    appcfg.save(
        root,
        appcfg.merge_defaults(
            {
                "sources": [{"name": "cam0", "url": str(clip)}],
                "inference": {"fps": 10, "width": 320, "height": 180, "backend": "cv2"},
                "liveview": {"port": port, "bind": "127.0.0.1", "display_fps": 8},
                "device": {"name": "fake-lab", "label": "Test box"},
            }
        ),
    )
    with Harness(root, port) as h:
        s = h.wait(
            "inference running with a channel",
            lambda: (
                lambda d: d if d["inference"]["state"] == "running" and d["channels"] else None
            )(h.status()),
        )
        assert s["model"]["project"] == "demo" and s["model"]["model_version"] == "mv-001"
        assert s["model"]["deployment_id"] == "dep-001" and s["model"]["classes"] == ["cat", "dog"]
        assert s["model"]["confidence"] == 0.25 and s["device"]["name"] == "fake-lab"
        assert s["inference"]["mode"] == "rtsp"
        assert s["config"]["sources"][0]["url"] == str(clip)
        ch = h.wait(
            "5 inferred frames",
            lambda: (lambda c: c if c["frames"] >= 5 else None)(h.status()["channels"][0]),
        )
        assert ch["source"]["connected"] is True and ch["source"]["backend"] == "cv2"
        assert ch["target_fps"] == 10.0 and ch["detections_by_class"] == {"cat": 1}
        # nobody watching: no JPEG encoding
        assert s["monitor"]["live"]["publishing"] is False
        assert s["monitor"]["live"]["published_frames"] == 0

        # health + stats + detections endpoints
        assert h.get("/healthz")["ok"] is True
        # /stats.json is the same document as /api/status; the system sampler publishes its
        # first /proc sample one interval (1 s) after start, so wait for it rather than
        # asserting on whichever stats.json write happens to be current.
        st = h.wait(
            "system sample with cpu_pct in /stats.json",
            lambda: (lambda d: d if "cpu_pct" in (d["system"] or {}) else None)(
                h.get("/stats.json")
            ),
        )
        assert st["inference"]["state"] == "running" and st["monitor"]["camera_connected"] is True
        assert st["system"]["ram_total_mb"] > 0 and st["system_backend"] in (
            "proc",
            "jtop",
            "tegrastats",
        )
        # detection metadata is published (2 Hz) even without a viewer
        dets = h.wait("detections metadata", lambda: h.get("/api/detections?ch=0")["detections"])
        assert dets[0]["class_name"] == "cat"

        # snapshot => the viewers marker is nudged, publishing starts
        snap = h.get("/snapshot.jpg?ch=0", binary=True)
        assert snap[:2] == b"\xff\xd8"
        # MJPEG stream delivers several parts quickly, capped by display_fps
        buf = h.mjpeg_parts(0, parts=5)
        assert b"Content-Type: image/jpeg" in buf
        live = h.wait(
            "published frames in stats",
            lambda: (lambda d: d if d["published_frames"] > 0 else None)(
                h.status()["monitor"]["live"]
            ),
        )
        assert live["encode_ms_p50"] is not None
        page = h.get("/", binary=True)
        assert b"S3cret" not in page and b"CamThink" in page  # page has no credentials

        # heartbeat for the pipeline's health/status machinery
        hb = h.wait(
            "heartbeat with camera connected",
            lambda: (lambda d: d if d["camera_connected"] else None)(h.heartbeat()),
        )
        assert hb["mode"] == "rtsp" and hb["deployment"] == "demo-dep-001"
        assert hb["state"] == "running" and hb["source"] == [str(clip)]

        # hot reload: confidence + fps apply without restart; width needs one
        cfg = appcfg.load(root)
        cfg["inference"]["confidence"] = 0.95  # above the fake detector's 0.9 => no detections
        cfg["inference"]["fps"] = 3
        appcfg.save(root, cfg)
        s = h.wait(
            "confidence 0.95 hot-reloaded",
            lambda: (lambda d: d if d["model"]["confidence"] == 0.95 else None)(h.status()),
        )
        assert s["channels"][0]["target_fps"] == 3.0
        assert s["monitor"]["config_restart_required"] == {"inference": False, "liveview": False}
        assert s["inference"]["config_restart_required"] is False
        h.wait("no detections at 0.95", lambda: h.get("/api/detections?ch=0")["detections"] == [])

        cfg["inference"]["width"] = 640
        appcfg.save(root, cfg)
        # Both documents the pipeline reads must agree: stats.json (live view) and the
        # heartbeat status.json (health / target status) are flushed together on reload.
        s = h.wait(
            "restart_required in stats.json and the heartbeat",
            lambda: (
                lambda d: (
                    d
                    if d["monitor"]["config_restart_required"]["inference"]
                    and d["inference"]["config_restart_required"]
                    else None
                )
            )(h.status()),
        )
        assert s["monitor"]["config_restart_required"] == {"inference": True, "liveview": False}
        assert h.heartbeat()["config_restart_required"] is True
        # ... and the live view never shows the pending restart before the heartbeat does
        assert not (
            s["monitor"]["config_restart_required"]["inference"]
            and not s["inference"]["config_restart_required"]
        )
    hb = h.heartbeat()
    assert hb["state"] == "stopped"
    stats = json.loads((appcfg.live_dir(root) / "stats.json").read_text())
    assert stats["state"] == "stopped"


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def test_liveview_without_inference_shows_reason(device_root: Path):
    from camthink_edge import liveview

    root = device_root
    port = _free_port()
    appcfg.save(
        root,
        appcfg.merge_defaults(
            {
                "sources": [{"name": "cam0", "url": RTSP}],
                "liveview": {"port": port, "bind": "127.0.0.1"},
            }
        ),
    )
    svc = Service("liveview", liveview.run_liveview, root=root).start()
    try:
        deadline = time.time() + 20
        while not _port_open(port) and time.time() < deadline:
            assert svc.thread.is_alive(), svc.describe()
            time.sleep(0.1)
        assert _port_open(port), f"live view did not listen on {port}: {svc.describe()}"
        s = _get(port, "/api/status")
        assert s["inference"]["state"] in ("stopped", "unknown")
        assert s["config"]["sources"][0]["url"] == "rtsp://***@192.168.5.78:554/main"
        assert s["channels"] == [] and s["system"] is None
        assert s["model"]["project"] == "demo"  # model info comes from the deployment itself
        page = _get(port, "/", binary=True).decode()
        assert "S3cret" not in page
        assert "S3cret" not in json.dumps(s)
        snap = _get(port, "/snapshot.jpg?ch=0", binary=True)
        assert snap[:2] == b"\xff\xd8"  # placeholder with the reason, not an error
        assert _get(port, "/healthz")["ok"] is True
    finally:
        svc.shutdown()
    assert not svc.thread.is_alive() and svc.error is None, svc.describe()
