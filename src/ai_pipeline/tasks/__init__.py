"""Layer B: task modules (object_detection in V1). Registry by ``task_type`` name."""

from __future__ import annotations

from ai_pipeline.tasks.base import TaskModule
from ai_pipeline.tasks.object_detection import ObjectDetectionTask

_REGISTRY: dict[str, TaskModule] = {
    ObjectDetectionTask.name: ObjectDetectionTask(),
}


def get_task(task_type: str) -> TaskModule:
    try:
        return _REGISTRY[task_type]
    except KeyError:
        raise KeyError(task_type) from None


def task_types() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["TaskModule", "get_task", "task_types"]
