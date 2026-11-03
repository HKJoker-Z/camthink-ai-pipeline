"""ModelProvider protocol and the per-kind input/output contracts.

Providers run **only inside the job runner** on the compute host. They receive plain dicts
(from ``job.json``) and return plain dicts; the runner wraps the return value in a
``JobResult``. Every returned path must be relative to the job directory (use
``ctx.relpath``).

Contracts (all paths in ``inputs`` are resolvable via ``ctx.resolve``):

train
    inputs : data_config (provider data file, e.g. data.yaml), dataset_dir, dataset_version
    params : candidate {name, weights}, epochs, image_size, batch_size, device,
             provider_args {}, classes [..]
    returns: artifacts {weights, metrics}, metrics {<task metric names>, per_class {...}}

evaluate
    inputs : weights, data_config, split ("val" | "test")
    params : classes [..], image_size, device, provider_args {}
    returns: artifacts {metrics}, metrics {<task metric names>, per_class {...}}

autolabel
    inputs : images_list (text file, one image path per line)
    params : weights, confidence, class_map {source -> project}, image_size, device,
             provider_args {}
    returns: artifacts {detections}  — JSONL rows
             {image, class_name (source vocabulary), bbox [x_min,y_min,x_max,y_max] normalised,
              confidence}
             metrics {images, detections}

Metric names in ``metrics`` must be the task module's canonical names (object detection:
precision, recall, f1, mAP50, mAP50-95); mapping from library-specific names is the
provider's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ai_pipeline.core.jobs.context import JobContext


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def train(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def evaluate(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def autolabel(
        self, ctx: JobContext, *, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]: ...

    def describe_environment(self) -> dict[str, Any]:
        """Library versions / device info recorded on the run for reproducibility."""
        ...
