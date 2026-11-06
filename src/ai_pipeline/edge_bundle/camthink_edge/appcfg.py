"""``<root>/service/app.json`` - the device-side application configuration.

Written by the pipeline (``ai-pipeline app source set / app configure``) over SSH, read by the
inference service (``camthink_edge serve``) and the live-view server (``camthink_edge
liveview``). It is the **only** place the camera URL (with credentials) is stored; the file is
created with mode 0600. Everything that leaves the device (CLI output, heartbeat, the web page)
goes through :func:`mask_url`.

Layout (all keys optional; :data:`DEFAULTS` fills the rest)::

    {"schema_version": 1,
     "sources": [{"name": "cam0", "url": "rtsp://user:pw@host:554/main"}],
     "inference": {"fps": 10, "width": 1280, "height": 720, "confidence": null,
                   "backend": "auto", "latency_ms": 200},
     "liveview": {"enabled": true, "port": 8080, "bind": "0.0.0.0",
                  "display_fps": 10, "quality": 70}}

``inference.confidence`` overrides ``runtime.json`` (the deployment's threshold) when set;
``inference.fps`` and ``confidence`` and every ``liveview`` display setting are re-read live by
the running processes, the rest needs a service restart (see :func:`restart_required`).
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DEFAULTS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "sources": [],
    "inference": {
        "fps": 10.0,
        "width": 1280,
        "height": 720,
        "confidence": None,
        "backend": "auto",
        "latency_ms": 200,
    },
    "liveview": {
        "enabled": True,
        "port": 8080,
        "bind": "0.0.0.0",
        "display_fps": 10.0,
        "quality": 70,
    },
    "device": {"name": None, "label": None},  # written by the pipeline for the page header
}

# Changing these requires restarting the inference service; everything else is hot-reloaded.
INFERENCE_RESTART_KEYS = (
    "sources",
    "inference.width",
    "inference.height",
    "inference.backend",
    "inference.latency_ms",
)
# Changing these requires restarting the live-view server.
LIVEVIEW_RESTART_KEYS = ("liveview.port", "liveview.bind")

_CRED_RE = re.compile(r"^(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<cred>[^/@]*@)", re.IGNORECASE)
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def mask_url(url: str | None) -> str | None:
    """``rtsp://user:pw@host/path`` -> ``rtsp://***@host/path``; file paths unchanged."""
    if not url:
        return url
    return _CRED_RE.sub(lambda m: m.group("scheme") + "***@", url)


def masked(cfg: dict[str, Any]) -> dict[str, Any]:
    """A copy safe to print: credentials replaced in every source URL."""
    out = copy.deepcopy(cfg)
    for s in out.get("sources", []):
        s["url"] = mask_url(s.get("url"))
    return out


def merge_defaults(cfg: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(DEFAULTS)
    for key, val in (cfg or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key].update(val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def validate(cfg: dict[str, Any]) -> list[str]:
    """Return human-readable problems (empty list = valid)."""
    problems: list[str] = []
    srcs = cfg.get("sources")
    if not isinstance(srcs, list):
        problems.append("sources must be a list")
        srcs = []
    names = set()
    for i, s in enumerate(srcs):
        if not isinstance(s, dict) or not isinstance(s.get("url"), str) or not s["url"].strip():
            problems.append(f"sources[{i}]: needs a non-empty 'url'")
            continue
        name = s.get("name") or f"cam{i}"
        if not _NAME_RE.match(str(name)):
            problems.append(f"sources[{i}]: invalid name {name!r} (letters, digits, _ -)")
        if name in names:
            problems.append(f"sources[{i}]: duplicate name {name!r}")
        names.add(name)
        u = s["url"].strip().lower()
        if not (u.startswith(("rtsp://", "rtsps://", "http://", "https://", "/", "./", "~"))):
            problems.append(f"sources[{i}]: url must be rtsp(s)://, http(s):// or a file path")
    inf = cfg.get("inference", {})
    if not (0 < float(inf.get("fps", 1)) <= 60):
        problems.append("inference.fps must be in (0, 60]")
    for k, lo in (("width", 160), ("height", 90)):
        v = inf.get(k)
        if not isinstance(v, int) or v < lo or v % 2:
            problems.append(f"inference.{k} must be an even integer >= {lo}")
    c = inf.get("confidence")
    if c is not None and not (0.0 <= float(c) <= 1.0):
        problems.append("inference.confidence must be between 0 and 1 (or null)")
    if inf.get("backend") not in ("auto", "cv2gst", "cv2"):
        problems.append("inference.backend must be auto | cv2gst | cv2")
    lv = cfg.get("liveview", {})
    p = lv.get("port")
    if not isinstance(p, int) or not (1 <= p <= 65535):
        problems.append("liveview.port must be 1..65535")
    if not (0 < float(lv.get("display_fps", 1)) <= 60):
        problems.append("liveview.display_fps must be in (0, 60]")
    q = lv.get("quality")
    if not isinstance(q, int) or not (10 <= q <= 100):
        problems.append("liveview.quality must be 10..100")
    return problems


def app_config_path(root: Path) -> Path:
    return Path(root) / "service" / "app.json"


def load(root: Path) -> dict[str, Any]:
    """Config with defaults applied; an absent file means 'no sources' (legacy behaviour)."""
    p = app_config_path(root)
    if not p.is_file():
        return merge_defaults(None)
    return merge_defaults(json.loads(p.read_text(encoding="utf-8")))


def save(root: Path, cfg: dict[str, Any]) -> Path:
    """Atomic write with mode 0600 (the file holds camera credentials)."""
    p = app_config_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".app-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def source_names(cfg: dict[str, Any]) -> list[str]:
    return [str(s.get("name") or f"cam{i}") for i, s in enumerate(cfg.get("sources", []))]


def live_dir(root: Path) -> Path:
    """Shared-memory directory where the inference service publishes frames and stats.

    tmpfs keeps the per-frame JPEG writes off the eMMC/SD card. Derived from the install root
    so two instances on one device (e.g. production and a test root) never collide.
    """
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(Path(root).expanduser().resolve())).strip("_")
    return base / f"camthink-ai-{tag}"


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def restart_required(old: dict[str, Any], new: dict[str, Any]) -> dict[str, bool]:
    """Which services must restart for ``new`` to take full effect."""
    return {
        "inference": any(_get(old, k) != _get(new, k) for k in INFERENCE_RESTART_KEYS),
        "liveview": any(_get(old, k) != _get(new, k) for k in LIVEVIEW_RESTART_KEYS),
    }
