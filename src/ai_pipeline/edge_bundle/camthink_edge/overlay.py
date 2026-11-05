"""Draw detections on a frame: one stable colour per class (by class index), label + score."""

from __future__ import annotations

from typing import Any

import numpy as np

# BGR palette; class i uses PALETTE[i % len(PALETTE)] so colours are stable per project.
PALETTE = [
    (80, 200, 80),  # green
    (60, 120, 255),  # orange
    (255, 160, 0),  # azure
    (200, 60, 220),  # magenta
    (0, 220, 255),  # yellow
    (255, 90, 90),  # blue
    (90, 255, 220),  # lime-ish
    (160, 160, 255),  # pink
    (255, 255, 120),  # cyan-ish
    (60, 60, 230),  # red
]


def class_colour(class_id: int) -> tuple[int, int, int]:
    return PALETTE[int(class_id) % len(PALETTE)]


def class_colour_hex(class_id: int) -> str:
    b, g, r = class_colour(class_id)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw(frame: np.ndarray, detections: list[dict[str, Any]], header: str = "") -> np.ndarray:
    """Return a BGR copy of ``frame`` (BGR or BGRx) with boxes, labels and an optional header."""
    import cv2  # type: ignore

    if frame.ndim == 3 and frame.shape[2] == 4:
        img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        img = np.array(frame, copy=True)
    h, w = img.shape[:2]
    thick = 2 if w >= 960 else 1
    scale = 0.55 if w >= 960 else 0.45
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox_xyxy"])
        col = class_colour(d.get("class_id", 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        label = f"{d.get('class_name', '?')} {float(d.get('confidence', 0)):.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), col, -1)
        cv2.putText(
            img, label, (x1 + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA
        )
    if header:
        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(
            img, header, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def placeholder(width: int, height: int, text: str) -> np.ndarray:
    import cv2  # type: ignore

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 20, 16)
    cv2.putText(
        canvas,
        text,
        (16, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def encode_jpeg(img: np.ndarray, quality: int = 70) -> bytes | None:
    import cv2  # type: ignore

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
