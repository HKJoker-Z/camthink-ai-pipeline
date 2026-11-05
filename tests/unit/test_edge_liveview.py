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


def _write_clip(path: Path, frames: int = 150) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    assert vw.isOpened()
    for i in range(frames):
        f = np.zeros((180, 320, 3), np.uint8)
        x = 10 + (i * 3) % 200
        cv2.rectangle(f, (x, 50), (x + 60, 120), (0, 200, 255), -1)
        vw.write(f)
    vw.release()


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
    _write_clip(root / "clip.mp4")
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


def test_frame_source_reads_file_and_reports_stats(device_root: Path):
    from camthink_edge import stream

    src = stream.FrameSource(
        str(device_root / "clip.mp4"), 320, 180, name="t", backend="cv2", loop=True
    )
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
    good = stream.probe(
        str(device_root / "clip.mp4"), width=320, height=180, backend="cv2", seconds=1
    )
    assert good["ok"] is True and good["frames"] > 5 and good["native_size"] == [320, 180]
    assert stream.is_rtsp(RTSP) and not stream.is_rtsp("/x.mp4")
    assert stream.pick_backend("cv2") == "cv2"
    assert "nvv4l2decoder" in stream.gst_pipeline(RTSP, 1280, 720, latency_ms=200)


def _get(port: int, path: str, *, binary: bool = False):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        data = r.read()
    return data if binary else json.loads(data)


def _wait(pred, timeout: float = 15.0, step: float = 0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not met in time")


def test_monitor_and_liveview_end_to_end(device_root: Path):
    from camthink_edge import liveview, monitor

    root = device_root
    port = _free_port()
    appcfg.save(
        root,
        appcfg.merge_defaults(
            {
                "sources": [{"name": "cam0", "url": str(root / "clip.mp4")}],
                "inference": {"fps": 10, "width": 320, "height": 180, "backend": "cv2"},
                "liveview": {"port": port, "bind": "127.0.0.1", "display_fps": 8},
                "device": {"name": "fake-lab", "label": "Test box"},
            }
        ),
    )
    stop_m, stop_l = threading.Event(), threading.Event()
    tm = threading.Thread(
        target=lambda: monitor.run_monitor(root, appcfg.load(root), heartbeat_s=0.5, stop=stop_m),
        daemon=True,
    )
    tl = threading.Thread(target=lambda: liveview.run_liveview(root, stop=stop_l), daemon=True)
    tm.start()
    tl.start()
    try:
        s = _wait(
            lambda: (
                (lambda d: d if d["inference"]["state"] == "running" and d["channels"] else None)(
                    _get(port, "/api/status")
                )
                if _port_open(port)
                else None
            )
        )
        assert s["model"]["project"] == "demo" and s["model"]["model_version"] == "mv-001"
        assert s["model"]["deployment_id"] == "dep-001" and s["model"]["classes"] == ["cat", "dog"]
        assert s["model"]["confidence"] == 0.25 and s["device"]["name"] == "fake-lab"
        assert s["inference"]["mode"] == "rtsp"
        assert s["config"]["sources"][0]["url"] == str(root / "clip.mp4")
        ch = _wait(
            lambda: (lambda c: c if c["frames"] >= 5 else None)(
                _get(port, "/api/status")["channels"][0]
            )
        )
        assert ch["source"]["connected"] is True and ch["source"]["backend"] == "cv2"
        assert ch["target_fps"] == 10.0 and ch["detections_by_class"] == {"cat": 1}
        # nobody watching: no JPEG encoding
        assert s["monitor"]["live"]["publishing"] is False
        assert s["monitor"]["live"]["published_frames"] == 0

        # health + stats + detections endpoints
        assert _get(port, "/healthz")["ok"] is True
        st = _get(port, "/stats.json")  # same document as /api/status
        assert st["inference"]["state"] == "running" and st["monitor"]["camera_connected"] is True
        assert st["system"] is not None and "cpu_pct" in st["system"]
        dets = _get(port, "/api/detections?ch=0")["detections"]
        assert dets and dets[0]["class_name"] == "cat"

        # snapshot => the viewers marker is nudged, publishing starts
        snap = _get(port, "/snapshot.jpg?ch=0", binary=True)
        assert snap[:2] == b"\xff\xd8"
        # MJPEG stream delivers several parts quickly, capped by display_fps
        sk = socket.create_connection(("127.0.0.1", port), timeout=5)
        sk.sendall(b"GET /stream.mjpg?ch=0 HTTP/1.1\r\nHost: x\r\n\r\n")
        buf = b""
        t0 = time.time()
        while buf.count(b"--frame") < 5 and time.time() - t0 < 10:
            buf += sk.recv(65536)
        sk.close()
        assert buf.count(b"--frame") >= 5
        assert b"Content-Type: image/jpeg" in buf
        live = _wait(
            lambda: (lambda d: d if d["published_frames"] > 0 else None)(
                _get(port, "/api/status")["monitor"]["live"]
            )
        )
        assert live["encode_ms_p50"] is not None
        assert "S3cret" not in _get(port, "/", binary=True).decode()  # page has no credentials
        assert b"CamThink" in _get(port, "/", binary=True)

        # heartbeat for the pipeline's health/status machinery
        hb = json.loads((root / "service" / "status.json").read_text())
        assert hb["mode"] == "rtsp" and hb["deployment"] == "demo-dep-001"
        assert hb["state"] == "running" and hb["camera_connected"] is True
        assert hb["source"] == [str(root / "clip.mp4")]

        # hot reload: confidence + fps apply without restart; width needs one
        cfg = appcfg.load(root)
        cfg["inference"]["confidence"] = 0.95  # above the fake detector's 0.9 => no detections
        cfg["inference"]["fps"] = 3
        appcfg.save(root, cfg)
        s = _wait(
            lambda: (lambda d: d if d["model"]["confidence"] == 0.95 else None)(
                _get(port, "/api/status")
            )
        )
        assert s["channels"][0]["target_fps"] == 3.0
        assert s["monitor"]["config_restart_required"] == {"inference": False, "liveview": False}
        _wait(lambda: _get(port, "/api/detections?ch=0")["detections"] == [])
        cfg["inference"]["width"] = 640
        appcfg.save(root, cfg)
        s = _wait(
            lambda: (lambda d: d if d["monitor"]["config_restart_required"]["inference"] else None)(
                _get(port, "/api/status")
            )
        )
        assert s["inference"]["config_restart_required"] is True
    finally:
        stop_m.set()
        stop_l.set()
        tm.join(10)
        tl.join(10)
    hb = json.loads((root / "service" / "status.json").read_text())
    assert hb["state"] == "stopped"


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
    stop = threading.Event()
    t = threading.Thread(target=lambda: liveview.run_liveview(root, stop=stop), daemon=True)
    t.start()
    try:
        _wait(lambda: _port_open(port))
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
        stop.set()
        t.join(10)
