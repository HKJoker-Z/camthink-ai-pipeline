"""YOLO (ultralytics v8/11 style) pre- and post-processing, driven by ``runtime.json``.

runtime.json (written by the pipeline per deployment)::

    {
      "schema_version": 1,
      "task": "object_detection",
      "classes": ["person"],
      "input_size": 640,                         # square network input
      "preprocess":  {"letterbox": true, "pad_value": 114, "scale": 0.00392156862745098,
                      "channels": "rgb"},
      "postprocess": {"type": "yolo", "confidence": 0.25, "iou": 0.45, "max_detections": 300}
    }

The ONNX exported by ultralytics outputs ``(1, 4 + nc, N)`` with ``cx, cy, w, h`` in input
pixels followed by per-class scores (no objectness). Both that layout and ``(1, N, 4 + nc)``
are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import images

SCHEMA_VERSION = 1


@dataclass
class Letterbox:
    """Geometry needed to map network-input coordinates back to the original image."""

    gain_x: float
    gain_y: float
    pad_x: float
    pad_y: float
    orig_w: int
    orig_h: int


def load_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and validate the parts the runtime needs."""
    if cfg.get("schema_version", 1) != SCHEMA_VERSION:
        raise ValueError(f"unsupported runtime.json schema_version {cfg.get('schema_version')}")
    classes = list(cfg.get("classes") or [])
    if not classes:
        raise ValueError("runtime.json: classes must be a non-empty list")
    size = int(cfg.get("input_size", 640))
    pre = {"letterbox": True, "pad_value": 114, "scale": 1 / 255.0, "channels": "rgb"}
    pre.update(cfg.get("preprocess") or {})
    post = {"type": "yolo", "confidence": 0.25, "iou": 0.45, "max_detections": 300}
    post.update(cfg.get("postprocess") or {})
    if post["type"] != "yolo":
        raise ValueError(f"unsupported postprocess type {post['type']!r}")
    return {**cfg, "classes": classes, "input_size": size, "preprocess": pre, "postprocess": post}


def letterbox(img: np.ndarray, size: int, pad_value: int) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to ``size`` x ``size`` (centre padding, like
    ultralytics)."""
    h, w = img.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = max(1, round(w * gain)), max(1, round(h * gain))
    resized = images.resize_rgb(img, new_w, new_h) if (new_w, new_h) != (w, h) else img
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    canvas = np.full((size, size, img.shape[2]), pad_value, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, Letterbox(gain, gain, left, top, w, h)


def preprocess(
    img: np.ndarray, cfg: dict[str, Any], *, bgr: bool = False
) -> tuple[np.ndarray, Letterbox]:
    """HxWx3 uint8 (RGB, or BGR with ``bgr=True``; a 4th channel such as BGRx padding is
    ignored) -> (1, 3, S, S) float32 input + geometry.

    Resize first, swap channels afterwards (on the small canvas), and scale each channel
    straight into the NCHW buffer: on a Jetson CPU this is ~3x faster than
    ``astype * scale`` + ``transpose`` + ``ascontiguousarray`` and avoids resizing a
    negative-stride channel-swapped view (which costs ~10 ms at 720p)."""
    size = cfg["input_size"]
    pre = cfg["preprocess"]
    if pre["letterbox"]:
        canvas, lb = letterbox(img, size, int(pre["pad_value"]))
    else:  # plain stretch to the square input
        h, w = img.shape[:2]
        canvas = images.resize_rgb(img, size, size)
        lb = Letterbox(size / w, size / h, 0.0, 0.0, w, h)
    if canvas.shape[2] == 4:
        canvas = canvas[:, :, :3]
    if bgr != (pre["channels"] == "bgr"):
        canvas = canvas[:, :, ::-1]
    scale = np.float32(pre["scale"])
    x = np.empty((1, 3, size, size), dtype=np.float32)
    for c in range(3):
        np.multiply(canvas[:, :, c], scale, out=x[0, c], casting="unsafe")
    return x, lb


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes. Returns indices to keep (descending score)."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    outputs: dict[str, np.ndarray], lb: Letterbox, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Decode the YOLO head output into detections in original-image pixel coordinates."""
    classes = cfg["classes"]
    nc = len(classes)
    post = cfg["postprocess"]
    out = next(iter(outputs.values()))
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] == 4 + nc and out.shape[1] != 4 + nc:
        out = out.T  # (N, 4+nc)
    if out.shape[1] != 4 + nc:
        raise ValueError(
            f"output shape {out.shape} does not match 4 + {nc} classes; check runtime.json classes"
        )
    boxes_cxcywh = out[:, :4]
    scores_all = out[:, 4:]
    cls = scores_all.argmax(axis=1)
    conf = scores_all[np.arange(len(cls)), cls]
    mask = conf >= float(post["confidence"])
    boxes_cxcywh, cls, conf = boxes_cxcywh[mask], cls[mask], conf[mask]
    if boxes_cxcywh.size == 0:
        return []
    cx, cy, w, h = boxes_cxcywh.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # class-aware NMS via per-class coordinate offsets
    offset = cls.astype(np.float32)[:, None] * (cfg["input_size"] * 2)
    keep = nms(xyxy + offset, conf, float(post["iou"]))[: int(post["max_detections"])]
    dets = []
    for i in keep:
        x1 = (xyxy[i, 0] - lb.pad_x) / lb.gain_x
        y1 = (xyxy[i, 1] - lb.pad_y) / lb.gain_y
        x2 = (xyxy[i, 2] - lb.pad_x) / lb.gain_x
        y2 = (xyxy[i, 3] - lb.pad_y) / lb.gain_y
        x1, x2 = float(np.clip(x1, 0, lb.orig_w)), float(np.clip(x2, 0, lb.orig_w))
        y1, y2 = float(np.clip(y1, 0, lb.orig_h)), float(np.clip(y2, 0, lb.orig_h))
        dets.append(
            {
                "class_id": int(cls[i]),
                "class_name": classes[int(cls[i])],
                "confidence": round(float(conf[i]), 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return dets
