"""Strict GPU device policy, tested against a stubbed ``torch`` module (no real torch needed)."""

from __future__ import annotations

import sys
import types

import pytest

from ai_pipeline.core.errors import OperationError
from ai_pipeline.providers.device import resolve_device


@pytest.fixture
def fake_torch(monkeypatch):
    def install(*, available: bool, names: list[str]):
        torch = types.ModuleType("torch")
        cuda = types.SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(names) if available else 0,
            get_device_name=lambda i: names[i],
        )
        torch.cuda = cuda  # type: ignore[attr-defined]
        torch.version = types.SimpleNamespace(cuda="12.4" if available else None)  # type: ignore[attr-defined]
        torch.__version__ = "2.x-fake"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)
        return torch

    return install


def test_cpu_and_auto_never_fail(fake_torch):
    fake_torch(available=False, names=[])
    assert resolve_device("cpu")["used"] == "cpu"
    auto = resolve_device("auto")
    assert auto["used"] == "cpu" and auto["policy"] == "auto-fallback"
    fake_torch(available=True, names=["NVIDIA A10"])
    auto = resolve_device("auto")
    assert auto["used"] == "0" and auto["cuda_available"] and auto["device_names"] == ["NVIDIA A10"]


@pytest.mark.parametrize("requested", ["0", "cuda", "cuda:0", 0, None])
def test_explicit_gpu_without_cuda_fails_with_structured_error(fake_torch, requested):
    fake_torch(available=False, names=[])
    with pytest.raises(OperationError) as ei:
        resolve_device(requested)
    err = ei.value
    assert err.code == "GPU_DEVICE_UNAVAILABLE"
    assert err.details["cuda_available"] is False and err.details["device_count"] == 0
    assert err.details["requested"] == ("0" if requested is None else str(requested))
    assert err.details["device_names"] == [] and err.details["torch_cuda"] is None
    assert "nvidia-smi" in err.hint


def test_explicit_gpu_index_out_of_range_fails(fake_torch):
    fake_torch(available=True, names=["GPU0"])
    with pytest.raises(OperationError) as ei:
        resolve_device("cuda:1")
    assert ei.value.code == "GPU_DEVICE_UNAVAILABLE" and ei.value.details["missing_indices"] == [1]
    with pytest.raises(OperationError) as ei:
        resolve_device("0,1")
    assert ei.value.details["missing_indices"] == [1]


def test_explicit_gpu_resolves_when_present(fake_torch):
    fake_torch(available=True, names=["GPU0", "GPU1"])
    assert resolve_device("0")["used"] == "0"
    assert resolve_device("cuda")["used"] == "0"
    assert resolve_device("cuda:1")["used"] == "1"
    info = resolve_device("0,1")
    assert info["used"] == "0,1" and info["policy"] == "explicit-gpu"
    assert info["device_names"] == ["GPU0", "GPU1"] and info["torch_cuda"] == "12.4"


def test_invalid_spec(fake_torch):
    fake_torch(available=True, names=["GPU0"])
    for bad in ("gpu", "cuda:x", "mps", "0;1"):
        with pytest.raises(OperationError) as ei:
            resolve_device(bad)
        assert ei.value.code == "DEVICE_SPEC_INVALID"


def test_config_rejects_bad_device():
    import copy

    import yaml
    from pydantic import ValidationError

    from ai_pipeline.core.config import ProjectConfig
    from ai_pipeline.core.project import render_template

    data = yaml.safe_load(
        render_template("default", project_name="demo", task_type="object_detection")[
            "project.yaml"
        ]
    )
    for ok in ("cpu", "auto", "cuda", "cuda:1", "0", "0,1"):
        d = copy.deepcopy(data)
        d["training"]["device"] = ok
        ProjectConfig.model_validate(d)
    for bad in ("gpu", "cuda:a", "mps"):
        d = copy.deepcopy(data)
        d["training"]["device"] = bad
        with pytest.raises(ValidationError):
            ProjectConfig.model_validate(d)
