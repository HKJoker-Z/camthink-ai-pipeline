"""Deterministic fake provider for tests and dry runs. No ML libraries.

Behaviour knobs via ``params["provider_args"]``:
    fake_metrics: {...}      metrics to report (train and evaluate); default derived from epochs
    fake_fail: "train" | "evaluate" | "autolabel"   raise in that operation
    fake_sleep_s: float      sleep before finishing (to observe RUNNING / test cancel)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ai_pipeline.core.jobs.context import JobContext


class FakeProvider:
    name = "fake"

    def describe_environment(self) -> dict[str, Any]:
        return {"provider_version": "fake-1", "device": "cpu"}

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _args(params: dict[str, Any]) -> dict[str, Any]:
        return dict(params.get("provider_args") or {})

    @staticmethod
    def _maybe_fail(args: dict[str, Any], op: str) -> None:
        if args.get("fake_fail") == op:
            raise RuntimeError(f"fake provider was asked to fail during {op}")

    @staticmethod
    def _maybe_sleep(args: dict[str, Any]) -> None:
        s = float(args.get("fake_sleep_s", 0) or 0)
        if s > 0:
            time.sleep(s)

    @staticmethod
    def _default_metrics(epochs: int, classes: list[str]) -> dict[str, Any]:
        map50 = round(min(0.95, 0.5 + 0.05 * epochs), 4)
        p, r = 0.8, 0.75
        m = {
            "precision": p,
            "recall": r,
            "f1": round(2 * p * r / (p + r), 4),
            "mAP50": map50,
            "mAP50-95": round(map50 * 0.7, 4),
        }
        m["per_class"] = {c: {"precision": p, "recall": r, "mAP50": map50} for c in classes}
        return m

    # -- operations ----------------------------------------------------------------------

    def train(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        args = self._args(params)
        self._maybe_fail(args, "train")
        data_config = ctx.resolve(inputs["data_config"])
        epochs = int(params.get("epochs", 1))
        classes = list(params.get("classes", []))
        ctx.log.info(
            "fake training %s for %d epochs on %s", params.get("candidate"), epochs, data_config
        )
        for epoch in range(1, epochs + 1):
            ctx.report_progress(epoch=epoch, total_epochs=epochs)
        self._maybe_sleep(args)
        metrics = dict(args.get("fake_metrics") or self._default_metrics(epochs, classes))

        weights_dir = ctx.out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights = weights_dir / "best.pt"
        payload = {
            "fake_weights": True,
            "candidate": params.get("candidate"),
            "epochs": epochs,
            "data_config_sha256": hashlib.sha256(data_config.read_bytes()).hexdigest(),
            "metrics": metrics,
        }
        weights.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        metrics_file = ctx.out_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "artifacts": {"weights": ctx.relpath(weights), "metrics": ctx.relpath(metrics_file)},
            "metrics": metrics,
            "extra": {"epochs_completed": epochs},
        }

    def evaluate(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        args = self._args(params)
        self._maybe_fail(args, "evaluate")
        self._maybe_sleep(args)
        weights = ctx.resolve(inputs["weights"])
        if not weights.is_file():
            raise FileNotFoundError(f"weights not found: {weights}")
        stored = json.loads(weights.read_text(encoding="utf-8"))
        metrics = dict(args.get("fake_metrics") or stored.get("metrics") or {})
        metrics["split"] = inputs.get("split", "test")
        metrics_file = ctx.out_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return {"artifacts": {"metrics": ctx.relpath(metrics_file)}, "metrics": metrics}

    def autolabel(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        args = self._args(params)
        self._maybe_fail(args, "autolabel")
        self._maybe_sleep(args)
        images = [
            ln.strip()
            for ln in ctx.resolve(inputs["images_list"]).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        class_map: dict[str, str] = params.get("class_map") or {}
        source_classes = list(class_map) or ["object"]
        conf = float(params.get("confidence", 0.25))
        out = ctx.out_dir / "detections.jsonl"
        n = 0
        with out.open("w", encoding="utf-8") as fh:
            for img in images:
                stem = Path(img).stem
                # Deterministic per image: one centred box, a second one for "odd" stems.
                rows = [[source_classes[0], [0.3, 0.3, 0.7, 0.7], 0.9]]
                if stem and stem[-1] in "13579bdf":
                    rows.append([source_classes[-1], [0.05, 0.05, 0.25, 0.25], 0.6])
                for cls, box, c in rows:
                    if c >= conf:
                        fh.write(
                            json.dumps(
                                {"image": img, "class_name": cls, "bbox": box, "confidence": c}
                            )
                            + "\n"
                        )
                        n += 1
        return {
            "artifacts": {"detections": ctx.relpath(out)},
            "metrics": {"images": len(images), "detections": n},
            "extra": {"source_classes": source_classes},
        }
