"""Image IO with whichever library the device has (OpenCV preferred, Pillow fallback).

All functions produce/consume HxWx3 uint8 arrays in RGB order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - depends on the device
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # noqa: BLE001
    cv2 = None
    _HAS_CV2 = False


def backend() -> str:
    return "opencv" if _HAS_CV2 else "pillow"


def read_rgb(path: str | Path) -> np.ndarray:
    if _HAS_CV2:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read image {path}")
        return bgr[:, :, ::-1]
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_rgb(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def list_images(paths: list[str]) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(sorted(x for x in pp.iterdir() if x.suffix.lower() in exts))
        elif pp.is_file():
            out.append(pp)
    return out
