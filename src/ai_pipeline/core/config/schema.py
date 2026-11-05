"""Pydantic schemas for ``project.yaml`` and ``infrastructure.yaml``.

Every model forbids unknown keys so typos fail early with a path-qualified error.
The schema is versioned via ``schema_version`` so it can evolve without silent breakage.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Metrics produced by the on-device benchmark; deployment.acceptance keys must be a subset.
BENCHMARK_METRICS: tuple[str, ...] = (
    "latency_ms_p50",
    "latency_ms_p95",
    "fps",
    "memory_mb",
    "gpu_util_pct",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- shared


class Rule(_Strict):
    """Acceptance rule for a single metric. At least one bound is required."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> Rule:
        if self.min is None and self.max is None:
            raise ValueError("rule needs at least one of 'min' or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("'min' must not exceed 'max'")
        return self

    def check(self, value: float) -> bool:
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)


# --------------------------------------------------------------------------- project.yaml


class ProjectMeta(_Strict):
    name: str
    task_type: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not PROJECT_NAME_RE.match(v):
            raise ValueError(
                "must match ^[a-z][a-z0-9_]{0,63}$ (lowercase letters, digits, underscore)"
            )
        return v


class LocalImagesSource(_Strict):
    type: Literal["local_images"]
    path: str


class RtspCaptureSource(_Strict):
    type: Literal["rtsp_capture"]
    url: str
    frames: int = Field(default=100, ge=1)
    interval_s: float = Field(default=1.0, gt=0)


DataSource = Annotated[LocalImagesSource | RtspCaptureSource, Field(discriminator="type")]


class SplitConfig(_Strict):
    train: float = Field(ge=0, le=1)
    val: float = Field(ge=0, le=1)
    test: float = Field(default=0.0, ge=0, le=1)
    seed: int = 42

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {total:.4f}")
        if self.train <= 0 or self.val <= 0:
            raise ValueError("train and val fractions must be > 0")
        return self


class DatasetValidationConfig(_Strict):
    min_images: int = Field(default=1, ge=1)
    min_instances_per_class: int = Field(default=1, ge=0)
    max_empty_label_ratio: float = Field(default=1.0, ge=0, le=1)
    # Warnings never invalidate a dataset unless the project says so.
    fail_on_warnings: bool = False


class DatasetConfig(_Strict):
    classes: list[str] = Field(min_length=1)
    label_format: str = "yolo"
    sources: list[DataSource] = Field(min_length=1)
    split: SplitConfig
    validation: DatasetValidationConfig = DatasetValidationConfig()

    @field_validator("classes")
    @classmethod
    def _unique_classes(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            dupes = sorted({c for c in v if v.count(c) > 1})
            raise ValueError(f"class names must be unique; duplicates: {dupes}")
        if any(not c.strip() for c in v):
            raise ValueError("class names must be non-empty")
        return v


class AutoLabelConfig(_Strict):
    provider: str
    weights: str
    class_map: dict[str, str] = Field(min_length=1)
    confidence: float = Field(default=0.25, ge=0, le=1)
    backend: str


class LabelingConfig(_Strict):
    auto: AutoLabelConfig | None = None


class Candidate(_Strict):
    name: str
    weights: str


class ModelConfig(_Strict):
    provider: str
    candidates: list[Candidate] = Field(min_length=1)

    @field_validator("candidates")
    @classmethod
    def _unique_names(cls, v: list[Candidate]) -> list[Candidate]:
        names = [c.name for c in v]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        return v


DEVICE_RE = re.compile(r"^(cpu|auto|cuda|cuda:\d+(,\d+)*|\d+(,\d+)*)$")


class TrainingConfig(_Strict):
    backend: str
    epochs: int = Field(ge=1)
    image_size: int = Field(default=640, ge=32)
    batch_size: int = Field(default=16, ge=1)
    # "cpu" | "auto" | explicit GPU ("cuda", "cuda:N", "N", "N,M"). Explicit GPU requests
    # never fall back to CPU (see providers.device).
    device: str = "0"
    provider_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("device")
    @classmethod
    def _device_format(cls, v: str) -> str:
        if not DEVICE_RE.match(v.strip().lower()):
            raise ValueError('must be "cpu", "auto", "cuda", "cuda:N", "N" or "N,M"')
        return v.strip()


class SelectionConfig(_Strict):
    primary_metric: str


class EvaluationConfig(_Strict):
    split: Literal["val", "test"] = "test"
    metrics: list[str] = Field(min_length=1)
    acceptance: dict[str, Rule] = Field(default_factory=dict)
    selection: SelectionConfig

    @model_validator(mode="after")
    def _consistent(self) -> EvaluationConfig:
        if self.selection.primary_metric not in self.metrics:
            raise ValueError(
                f"selection.primary_metric {self.selection.primary_metric!r} "
                f"must be listed in evaluation.metrics"
            )
        unknown = sorted(set(self.acceptance) - set(self.metrics))
        if unknown:
            raise ValueError(f"acceptance refers to metrics not in evaluation.metrics: {unknown}")
        return self


class ExportConfig(_Strict):
    format: Literal["onnx"] = "onnx"
    opset: int = Field(default=17, ge=11)
    image_size: int = Field(default=640, ge=32)
    dynamic: bool = False
    provider_args: dict[str, Any] = Field(default_factory=dict)


class HealthConfig(_Strict):
    # After activation the service needs time to load the engine; poll until healthy.
    timeout_s: float = Field(default=120, gt=0)
    interval_s: float = Field(default=5, gt=0)


class InferenceConfig(_Strict):
    """Runtime post-processing parameters, written to the device as runtime.json."""

    confidence: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.45, ge=0, le=1)
    max_detections: int = Field(default=300, ge=1)


class SmokeConfig(_Strict):
    samples: int = Field(default=3, ge=1)  # test-split images shipped with the deployment
    min_detections: int = Field(default=1, ge=0)


class BenchmarkConfig(_Strict):
    iterations: int = Field(default=100, ge=1)
    warmup: int = Field(default=20, ge=0)


class DeploymentConfig(_Strict):
    target: str
    runtime: Literal["tensorrt"] = "tensorrt"
    precision: Literal["fp16", "fp32"] = "fp16"
    acceptance: dict[str, Rule] = Field(default_factory=dict)
    inference: InferenceConfig = InferenceConfig()
    smoke: SmokeConfig = SmokeConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    health: HealthConfig = HealthConfig()
    keep_versions: int = Field(default=3, ge=1)

    @field_validator("acceptance")
    @classmethod
    def _known_benchmark_metrics(cls, v: dict[str, Rule]) -> dict[str, Rule]:
        unknown = sorted(set(v) - set(BENCHMARK_METRICS))
        if unknown:
            raise ValueError(
                f"unknown benchmark metrics {unknown}; known: {list(BENCHMARK_METRICS)}"
            )
        return v


class ProjectConfig(_Strict):
    schema_version: Literal[1]
    project: ProjectMeta
    dataset: DatasetConfig
    labeling: LabelingConfig = LabelingConfig()
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    export: ExportConfig = ExportConfig()
    deployment: DeploymentConfig

    @property
    def name(self) -> str:
        return self.project.name


# --------------------------------------------------------------------------- infrastructure.yaml


class LocalBackendConfig(_Strict):
    type: Literal["local"]
    # Interpreter used to run jobs; it must have the provider extra installed.
    # None = the interpreter running the CLI (fine when control host == compute host).
    python: str | None = None


class SshBackendConfig(_Strict):
    type: Literal["ssh"]
    ssh_host: str
    workdir: str
    python: str = "python"


ComputeBackendConfig = Annotated[LocalBackendConfig | SshBackendConfig, Field(discriminator="type")]


class FakeTargetConfig(_Strict):
    type: Literal["fake"]
    root: str = "./.fake-target"


class CamThinkNG4511TargetConfig(_Strict):
    type: Literal["camthink_ng4511"]
    ssh_host: str
    install_root: str = "~/camthink-ai"  # user-owned; no sudo needed
    service_name: str | None = "camthink-ai-inference"  # systemd --user unit; None = no service
    inference_source: str = ""  # RTSP/video URL for the service; "" = loop over sample images
    python: str = "python3"
    trtexec: str = "/usr/src/tensorrt/bin/trtexec"
    workspace_mb: int = Field(default=1024, ge=64)
    build_timeout_s: int = Field(default=1800, ge=60)
    fps_limit: float = Field(default=5.0, gt=0)
    connect_timeout_s: int = Field(default=10, ge=1)
    # Live view (M9): a second systemd --user unit serving the detection webpage from the
    # frames the inference service publishes. None = no live view on this device.
    liveview_service_name: str | None = "camthink-ai-liveview"
    liveview_port: int = Field(default=8080, ge=1, le=65535)
    # LAN address browsers use to reach the device. None = the SSH alias's HostName.
    liveview_host: str | None = None
    device_label: str | None = None  # human label shown on the live-view page


DeploymentTargetConfig = Annotated[
    FakeTargetConfig | CamThinkNG4511TargetConfig, Field(discriminator="type")
]


class InfrastructureConfig(_Strict):
    schema_version: Literal[1]
    projects_root: str = "./projects"
    artifact_root: str = "./artifacts"
    compute_backends: dict[str, ComputeBackendConfig] = Field(min_length=1)
    deployment_targets: dict[str, DeploymentTargetConfig] = Field(min_length=1)
