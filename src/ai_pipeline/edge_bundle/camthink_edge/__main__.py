"""``python3 -m camthink_edge <command>`` — one JSON document on stdout, logs on stderr."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import VERSION, ops


def _emit(payload: dict, code: int = 0) -> int:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="camthink_edge", description="CamThink edge inference runtime")
    p.add_argument("--version", action="version", version=VERSION)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="ONNX -> TensorRT engine via trtexec")
    b.add_argument("--onnx", required=True)
    b.add_argument("--engine", required=True)
    b.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    b.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    b.add_argument("--workspace-mb", type=int, default=1024)
    b.add_argument("--timeout-s", type=int, default=1800)
    b.add_argument(
        "--reuse-scan",
        default=None,
        help="directory whose */model.engine.json sidecars may be reused",
    )

    i = sub.add_parser("infer", help="run detection on images")
    i.add_argument("--engine", required=True)
    i.add_argument("--config", required=True)
    i.add_argument("images", nargs="+")

    s = sub.add_parser("smoke", help="sanity-check a deployment on its sample images")
    s.add_argument("--engine", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--min-detections", type=int, default=1)
    s.add_argument("samples", nargs="+")

    bm = sub.add_parser("benchmark", help="latency / fps / memory")
    bm.add_argument("--engine", required=True)
    bm.add_argument("--config", required=True)
    bm.add_argument("--iterations", type=int, default=100)
    bm.add_argument("--warmup", type=int, default=20)
    bm.add_argument("--image", default=None)

    h = sub.add_parser("health", help="health of the active deployment")
    h.add_argument("--root", required=True)
    h.add_argument("--service", default=None)

    sv = sub.add_parser("serve", help="long-running inference loop with heartbeat")
    sv.add_argument("--root", required=True)
    sv.add_argument("--source", default=None, help="RTSP/video URL; omit to loop over samples")
    sv.add_argument("--fps-limit", type=float, default=5.0)
    sv.add_argument("--heartbeat-s", type=float, default=2.0)

    lv = sub.add_parser("liveview", help="live-view web server for the monitoring service")
    lv.add_argument("--root", required=True)
    lv.add_argument("--port", type=int, default=None, help="override liveview.port of app.json")
    lv.add_argument("--bind", default=None, help="override liveview.bind of app.json")

    pr = sub.add_parser("probe", help="test the configured camera source(s); read-only")
    pr.add_argument("--root", required=True)
    pr.add_argument("--seconds", type=float, default=5.0)
    pr.add_argument("--source", default=None, help="probe this URL instead of app.json's")
    pr.add_argument("--source-stdin", action="store_true", help="read the URL from stdin")

    ac = sub.add_parser("app-config", help="print the application config (credentials masked)")
    ac.add_argument("--root", required=True)

    sub.add_parser("info", help="runtime version and environment")

    args = p.parse_args(argv)
    try:
        if args.cmd == "build":
            return _emit(
                ops.build(
                    Path(args.onnx),
                    Path(args.engine),
                    precision=args.precision,
                    trtexec=args.trtexec,
                    workspace_mb=args.workspace_mb,
                    timeout_s=args.timeout_s,
                    reuse_scan=Path(args.reuse_scan) if args.reuse_scan else None,
                )
            )
        if args.cmd == "infer":
            return _emit(ops.infer(Path(args.engine), Path(args.config), args.images))
        if args.cmd == "smoke":
            return _emit(
                ops.smoke(
                    Path(args.engine),
                    Path(args.config),
                    args.samples,
                    min_detections=args.min_detections,
                )
            )
        if args.cmd == "benchmark":
            return _emit(
                ops.benchmark(
                    Path(args.engine),
                    Path(args.config),
                    iterations=args.iterations,
                    warmup=args.warmup,
                    image=args.image,
                )
            )
        if args.cmd == "health":
            return _emit(ops.health(Path(args.root), service=args.service))
        if args.cmd == "serve":
            return ops.serve(
                Path(args.root),
                source=args.source,
                fps_limit=args.fps_limit,
                heartbeat_s=args.heartbeat_s,
            )
        if args.cmd == "liveview":
            from .liveview import run_liveview

            return run_liveview(Path(args.root), port=args.port, bind=args.bind)
        if args.cmd == "probe":
            return _emit(_probe(args))
        if args.cmd == "app-config":
            from . import appcfg

            cfg = appcfg.load(Path(args.root))
            return _emit(
                {
                    "path": str(appcfg.app_config_path(Path(args.root))),
                    "exists": appcfg.app_config_path(Path(args.root)).is_file(),
                    "config": appcfg.masked(cfg),
                    "problems": appcfg.validate(cfg),
                    "live_dir": str(appcfg.live_dir(Path(args.root))),
                }
            )
        if args.cmd == "info":
            return _emit(_info())
    except Exception as exc:  # noqa: BLE001 - report as JSON, non-zero exit
        traceback.print_exc(file=sys.stderr)
        return _emit({"error": {"type": type(exc).__name__, "message": str(exc)}}, 1)
    return 2


def _probe(args) -> dict:
    from . import appcfg, stream

    root = Path(args.root)
    cfg = appcfg.load(root)
    inf = cfg["inference"]
    if args.source_stdin:
        sources = [{"name": "stdin", "url": sys.stdin.readline().strip()}]
    elif args.source:
        sources = [{"name": "arg", "url": args.source}]
    else:
        sources = cfg["sources"]
    if not sources:
        return {"sources": [], "ok": False, "error": "no source configured (app source set)"}
    results = []
    for i, src in enumerate(sources):
        res = stream.probe(
            src["url"],
            width=int(inf["width"]),
            height=int(inf["height"]),
            backend=str(inf.get("backend", "auto")),
            seconds=float(args.seconds),
        )
        res["name"] = src.get("name") or f"cam{i}"
        res["url"] = appcfg.mask_url(src["url"])
        results.append(res)
    return {"ok": all(r["ok"] for r in results), "sources": results}


def _info() -> dict:
    import platform

    info: dict = {
        "runtime_version": VERSION,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    for mod in ("numpy", "tensorrt", "cv2", "PIL"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception as exc:  # noqa: BLE001
            info[mod] = f"missing ({type(exc).__name__})"
    try:
        from . import stream

        info["decode_backend"] = stream.pick_backend("auto")
    except Exception as exc:  # noqa: BLE001
        info["decode_backend"] = f"unavailable ({exc})"
    try:
        from . import cudart

        info["cudart_version"] = cudart.runtime_version()
    except Exception as exc:  # noqa: BLE001
        info["cudart_version"] = f"unavailable ({exc})"
    return info


if __name__ == "__main__":
    sys.exit(main())
