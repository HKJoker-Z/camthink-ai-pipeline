"""Object detection task module (the only fully implemented task in V1)."""

from __future__ import annotations


class ObjectDetectionTask:
    name = "object_detection"
    label_formats: tuple[str, ...] = ("yolo",)
    # Scalar metrics an evaluation may report; per-class metrics are nested under these names.
    metric_names: tuple[str, ...] = ("precision", "recall", "f1", "mAP50", "mAP50-95")
