"""TaskModule protocol.

A task module owns what is specific to a computer-vision task type and nothing else:
which label formats exist, which metrics are meaningful, how a dataset is validated, and how
the provider's data description is generated. It never references a model library.

M0 declares only the static vocabulary (``label_formats``, ``metric_names``) used for
configuration cross-validation; dataset methods are added in M1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskModule(Protocol):
    name: str
    label_formats: tuple[str, ...]
    metric_names: tuple[str, ...]
