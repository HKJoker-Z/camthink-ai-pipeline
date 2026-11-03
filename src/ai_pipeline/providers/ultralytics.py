"""Ultralytics (YOLO) provider. Imported only inside the job runner.

Maps ultralytics' metric names to the object-detection task vocabulary and keeps every
YOLO-specific call in this one module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_pipeline.core.jobs.context import JobContext

# ultralytics results_dict keys -> task metric names
_METRIC_KEYS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}


def _yolo():
    from ultralytics import YOLO  # heavy import, deliberately local

    return YOLO


def _absolute_data_config(ctx: JobContext, data_config: Path) -> Path:
    """Ultralytics resolves a relative ``path:`` against the CWD, not the YAML's directory.
    Dataset versions keep ``path: .`` to stay portable, so write a job-local copy with the
    absolute dataset directory."""
    import yaml

    data = yaml.safe_load(data_config.read_text(encoding="utf-8")) or {}
    root = (data_config.parent / str(data.get("path", "."))).resolve()
    data["path"] = str(root)
    out = ctx.out_dir / "data.resolved.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _device(params: dict[str, Any]) -> str:
    """Honour the configured device but fall back to CPU when CUDA is unavailable, so the same
    project config works on a CPU-only compute host (e.g. integration tests)."""
    requested = str(params.get("device", "0"))
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        return requested if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch always present with ultralytics
        return "cpu"


def _collect_metrics(results, classes: list[str]) -> dict[str, Any]:
    rd = getattr(results, "results_dict", {}) or {}
    metrics: dict[str, Any] = {}
    for k, name in _METRIC_KEYS.items():
        if k in rd:
            metrics[name] = round(float(rd[k]), 6)
    p, r = metrics.get("precision"), metrics.get("recall")
    if p is not None and r is not None:
        metrics["f1"] = round(2 * p * r / (p + r), 6) if (p + r) > 0 else 0.0
    box = getattr(results, "box", None)
    per_class: dict[str, Any] = {}
    if box is not None:
        try:
            maps = list(box.maps)  # mAP50-95 per class index
            ap50 = list(box.ap50) if hasattr(box, "ap50") else []
            idx = (
                list(box.ap_class_index)
                if hasattr(box, "ap_class_index")
                else list(range(len(maps)))
            )
            for pos, ci in enumerate(idx):
                if 0 <= ci < len(classes):
                    entry = {"mAP50-95": round(float(maps[ci]), 6)}
                    if pos < len(ap50):
                        entry["mAP50"] = round(float(ap50[pos]), 6)
                    per_class[classes[ci]] = entry
        except Exception:  # per-class details are best-effort
            per_class = {}
    metrics["per_class"] = per_class
    return metrics


class UltralyticsProvider:
    name = "ultralytics"

    def describe_environment(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        try:
            import ultralytics

            info["ultralytics"] = ultralytics.__version__
        except Exception:
            pass
        try:
            import torch

            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda
            info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return info

    def train(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        YOLO = _yolo()
        candidate = params["candidate"]
        classes = list(params.get("classes", []))
        data_config = _absolute_data_config(ctx, ctx.resolve(inputs["data_config"]))
        device = _device(params)
        ctx.log.info("ultralytics train %s on %s device=%s", candidate, data_config, device)
        model = YOLO(candidate["weights"])
        extra = dict(params.get("provider_args") or {})
        results = model.train(
            data=str(data_config),
            epochs=int(params["epochs"]),
            imgsz=int(params.get("image_size", 640)),
            batch=int(params.get("batch_size", 16)),
            device=device,
            project=str(ctx.out_dir),
            name="train",
            exist_ok=True,
            verbose=False,
            plots=False,
            **extra,
        )
        weights = (
            Path(model.trainer.best)
            if getattr(model, "trainer", None)
            else ctx.out_dir / "train/weights/best.pt"
        )
        if not weights.is_file():
            raise FileNotFoundError(f"training produced no best.pt at {weights}")
        metrics = _collect_metrics(results, classes)
        metrics_file = ctx.out_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "artifacts": {"weights": ctx.relpath(weights), "metrics": ctx.relpath(metrics_file)},
            "metrics": metrics,
            "extra": {"save_dir": ctx.relpath(Path(model.trainer.save_dir))}
            if getattr(model, "trainer", None)
            else {},
        }

    def evaluate(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        YOLO = _yolo()
        classes = list(params.get("classes", []))
        weights = ctx.resolve(inputs["weights"])
        data_config = _absolute_data_config(ctx, ctx.resolve(inputs["data_config"]))
        split = inputs.get("split", "test")
        device = _device(params)
        ctx.log.info("ultralytics val %s split=%s device=%s", weights, split, device)
        model = YOLO(str(weights))
        results = model.val(
            data=str(data_config),
            split=split,
            imgsz=int(params.get("image_size", 640)),
            device=device,
            project=str(ctx.out_dir),
            name="val",
            exist_ok=True,
            verbose=False,
            plots=False,
            **dict(params.get("provider_args") or {}),
        )
        metrics = _collect_metrics(results, classes)
        metrics["split"] = split
        metrics_file = ctx.out_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return {"artifacts": {"metrics": ctx.relpath(metrics_file)}, "metrics": metrics}

    def autolabel(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        YOLO = _yolo()
        images = [
            ln.strip()
            for ln in ctx.resolve(inputs["images_list"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        conf = float(params.get("confidence", 0.25))
        device = _device(params)
        model = YOLO(params["weights"])
        names = model.names  # {idx: name}
        out = ctx.out_dir / "detections.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as fh:
            for start in range(0, len(images), 16):
                batch = images[start : start + 16]
                results = model.predict(
                    batch,
                    conf=conf,
                    imgsz=int(params.get("image_size", 640)),
                    device=device,
                    verbose=False,
                    **dict(params.get("provider_args") or {}),
                )
                for img, res in zip(batch, results, strict=False):
                    boxes = res.boxes
                    if boxes is None:
                        continue
                    xyxyn = boxes.xyxyn.tolist()
                    for box, cls, c in zip(
                        xyxyn, boxes.cls.tolist(), boxes.conf.tolist(), strict=False
                    ):
                        fh.write(
                            json.dumps(
                                {
                                    "image": img,
                                    "class_name": names[int(cls)],
                                    "bbox": [round(v, 6) for v in box],
                                    "confidence": round(float(c), 6),
                                }
                            )
                            + "\n"
                        )
                        n += 1
        return {
            "artifacts": {"detections": ctx.relpath(out)},
            "metrics": {"images": len(images), "detections": n},
            "extra": {"source_classes": [names[i] for i in sorted(names)]},
        }
