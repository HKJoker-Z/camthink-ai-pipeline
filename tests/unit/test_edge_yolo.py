"""Edge runtime pre/post-processing (numpy only; no TensorRT needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE_PARENT = Path(__file__).resolve().parents[2] / "src" / "ai_pipeline" / "edge_bundle"
if str(BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_PARENT))

from camthink_edge import VERSION, yolo  # noqa: E402
from camthink_edge.__main__ import main  # noqa: E402

CFG = yolo.load_config(
    {"classes": ["person", "dog"], "input_size": 64, "postprocess": {"confidence": 0.5, "iou": 0.5}}
)


def test_load_config_defaults_and_validation():
    assert CFG["preprocess"] == {
        "letterbox": True,
        "pad_value": 114,
        "scale": 1 / 255.0,
        "channels": "rgb",
    }
    assert CFG["postprocess"]["max_detections"] == 300 and CFG["postprocess"]["type"] == "yolo"
    with pytest.raises(ValueError):
        yolo.load_config({"classes": []})
    with pytest.raises(ValueError):
        yolo.load_config({"classes": ["a"], "postprocess": {"type": "ssd"}})
    with pytest.raises(ValueError):
        yolo.load_config({"classes": ["a"], "schema_version": 9})


def test_letterbox_geometry_and_preprocess():
    img = np.zeros((30, 60, 3), dtype=np.uint8)  # wide image
    img[:, :, 0] = 200
    canvas, lb = yolo.letterbox(img, 64, 114)
    assert canvas.shape == (64, 64, 3)
    assert lb.gain_x == lb.gain_y == pytest.approx(64 / 60)
    assert lb.orig_w == 60 and lb.orig_h == 30
    assert lb.pad_x == 0 and lb.pad_y == 16  # 30*64/60 = 32 rows -> 16 padding top/bottom
    assert (canvas[:16] == 114).all() and (canvas[16:48, :, 0] == 200).all()
    x, lb2 = yolo.preprocess(img, CFG)
    assert x.shape == (1, 3, 64, 64) and x.dtype == np.float32
    assert x.max() <= 1.0 and x[0, 0, 32, 32] == pytest.approx(200 / 255)
    assert lb2 == lb


def test_stretch_mode_inverts_with_separate_gains():
    cfg = yolo.load_config({"classes": ["a"], "input_size": 64, "preprocess": {"letterbox": False}})
    img = np.zeros((32, 128, 3), dtype=np.uint8)
    x, lb = yolo.preprocess(img, cfg)
    assert x.shape == (1, 3, 64, 64)
    assert lb.gain_x == 0.5 and lb.gain_y == 2.0 and lb.pad_x == lb.pad_y == 0


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = yolo.nms(boxes, scores, 0.5)
    assert keep.tolist() == [0, 2]
    assert yolo.nms(np.empty((0, 4)), np.empty((0,)), 0.5).size == 0


def _raw_output(
    dets: list[tuple[float, float, float, float, list[float]]], n: int = 20
) -> np.ndarray:
    """Build a (1, 4+nc, n) tensor like the ultralytics ONNX head."""
    nc = len(dets[0][4])
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, (cx, cy, w, h, scores) in enumerate(dets):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4:, i] = scores
    return out


def test_postprocess_decodes_filters_nms_and_rescales():
    # input 64x64 letterboxed from a 60x30 image: gain 64/60, pad_y 16
    _, lb = yolo.letterbox(np.zeros((30, 60, 3), dtype=np.uint8), 64, 114)
    raw = _raw_output(
        [
            (32, 32, 20, 10, [0.9, 0.1]),  # person, centre
            (33, 33, 20, 10, [0.8, 0.0]),  # duplicate of the above -> suppressed
            (32, 32, 20, 10, [0.1, 0.95]),  # dog at the same place -> kept (class-aware NMS)
            (10, 10, 4, 4, [0.3, 0.2]),  # below confidence
        ]
    )
    dets = yolo.postprocess({"output0": raw}, lb, CFG)
    assert [d["class_name"] for d in dets] == ["dog", "person"]  # sorted by confidence
    person = dets[1]
    x1, y1, x2, y2 = person["bbox_xyxy"]
    # cx 32 -> (32-0)/gain = 30 ; cy 32 -> (32-16)/gain = 15 ; w 20 -> 18.75 ; h 10 -> 9.375
    assert (x1, x2) == pytest.approx((30 - 18.75 / 2, 30 + 18.75 / 2), abs=0.1)
    assert (y1, y2) == pytest.approx((15 - 9.375 / 2, 15 + 9.375 / 2), abs=0.1)
    assert person["class_id"] == 0 and person["confidence"] == 0.9
    # transposed layout (1, N, 4+nc) is accepted too
    assert yolo.postprocess({"o": np.transpose(raw, (0, 2, 1))}, lb, CFG) == dets
    # wrong class count is a clear error
    with pytest.raises(ValueError):
        yolo.postprocess(
            {"o": raw}, lb, yolo.load_config({"classes": ["only_one"], "input_size": 64})
        )


def test_postprocess_clips_to_image_and_respects_max_detections():
    _, lb = yolo.letterbox(np.zeros((64, 64, 3), dtype=np.uint8), 64, 114)
    raw = _raw_output(
        [(2, 2, 20, 20, [0.9])] + [(40 + i * 3, 40, 4, 4, [0.6]) for i in range(5)], n=10
    )
    cfg = yolo.load_config(
        {
            "classes": ["p"],
            "input_size": 64,
            "postprocess": {"confidence": 0.5, "iou": 0.99, "max_detections": 3},
        }
    )
    dets = yolo.postprocess({"o": raw}, lb, cfg)
    assert len(dets) == 3
    assert dets[0]["bbox_xyxy"][0] == 0.0 and dets[0]["bbox_xyxy"][1] == 0.0  # clipped at 0


def test_cli_info_and_smoke_without_tensorrt(tmp_path, capsys):
    rc = main(["info"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["runtime_version"] == VERSION and out["numpy"]
    # smoke never crashes: missing engine -> passed False with a reason
    (tmp_path / "runtime.json").write_text(json.dumps({"classes": ["p"], "input_size": 64}))
    (tmp_path / "a.jpg").write_bytes(b"not an image")
    rc = main(
        [
            "smoke",
            "--engine",
            str(tmp_path / "missing.engine"),
            "--config",
            str(tmp_path / "runtime.json"),
            str(tmp_path / "a.jpg"),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["passed"] is False and "inference failed" in out["details"]["reason"]
    rc = main(
        [
            "smoke",
            "--engine",
            "x",
            "--config",
            str(tmp_path / "runtime.json"),
            str(tmp_path / "nothing"),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["passed"] is False and out["details"]["reason"] == "no sample images"


def test_health_without_current(tmp_path, capsys):
    rc = main(["health", "--root", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert (
        rc == 0 and out["healthy"] is False and out["details"]["reason"] == "no active deployment"
    )


def test_build_reuses_identical_engine_without_trtexec(tmp_path, monkeypatch):
    """Engine cache: same ONNX sha + precision + TensorRT version -> copy, no trtexec run."""
    import hashlib
    import sys
    import types

    from camthink_edge import ops

    trt = types.ModuleType("tensorrt")
    trt.__version__ = "10.3.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt", trt)

    deployments = tmp_path / "deployments"
    old = deployments / "demo-dep-001"
    old.mkdir(parents=True)
    onnx_bytes = b"onnx-model-bytes"
    (old / "model.onnx").write_bytes(onnx_bytes)
    (old / "model.engine").write_bytes(b"ENGINE")
    sidecar = {
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "precision": "fp16",
        "tensorrt": "10.3.0",
        "engine_sha256": hashlib.sha256(b"ENGINE").hexdigest(),
    }
    (old / "model.engine.json").write_text(json.dumps(sidecar))

    new = deployments / "demo-dep-002"
    new.mkdir()
    (new / "model.onnx").write_bytes(onnx_bytes)
    res = ops.build(
        new / "model.onnx",
        new / "model.engine",
        precision="fp16",
        trtexec="/nonexistent/trtexec",
        workspace_mb=512,
        timeout_s=10,
        reuse_scan=deployments,
    )
    assert res["reused_from"] == "demo-dep-001" and (new / "model.engine").read_bytes() == b"ENGINE"
    assert json.loads((new / "model.engine.json").read_text())["reused_from"] == "demo-dep-001"
    assert res["engine_sha256"] == sidecar["engine_sha256"] and res["builder_peak_rss_mb"] is None

    # different precision / TRT version / ONNX -> no reuse (trtexec would run and fail here)
    assert (
        ops.find_reusable_engine(
            deployments, onnx_sha256=sidecar["onnx_sha256"], precision="fp32", tensorrt="10.3.0"
        )
        is None
    )
    assert (
        ops.find_reusable_engine(
            deployments, onnx_sha256=sidecar["onnx_sha256"], precision="fp16", tensorrt="10.4.0"
        )
        is None
    )
    assert (
        ops.find_reusable_engine(
            deployments, onnx_sha256="0" * 64, precision="fp16", tensorrt="10.3.0"
        )
        is None
    )
    # the reused copy is itself a valid cache source; tampered engines are never reused
    assert ops.find_reusable_engine(
        deployments, onnx_sha256=sidecar["onnx_sha256"], precision="fp16", tensorrt="10.3.0"
    ) in (old / "model.engine", new / "model.engine")
    (old / "model.engine").write_bytes(b"CORRUPT")
    (new / "model.engine").write_bytes(b"CORRUPT")
    assert (
        ops.find_reusable_engine(
            deployments, onnx_sha256=sidecar["onnx_sha256"], precision="fp16", tensorrt="10.3.0"
        )
        is None
    )
    with pytest.raises(RuntimeError):
        ops.build(
            new / "model.onnx",
            tmp_path / "x.engine",
            precision="fp16",
            trtexec="/nonexistent/trtexec",
            workspace_mb=512,
            timeout_s=10,
            reuse_scan=deployments,
        )
