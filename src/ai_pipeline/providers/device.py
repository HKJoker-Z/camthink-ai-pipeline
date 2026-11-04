"""Strict compute-device resolution for providers that run on torch.

Semantics (``training.device`` in project.yaml):

    "cpu"                     CPU, always allowed
    "auto"                    first CUDA device if available, otherwise CPU (explicit opt-in)
    "cuda" | "cuda:N" | "N" | "N,M"   explicit GPU request: every named device must exist,
                              otherwise GPU_DEVICE_UNAVAILABLE — never a silent CPU fallback

The resolved form is what ultralytics accepts: ``"cpu"``, ``"0"`` or ``"0,1"``.
"""

from __future__ import annotations

import re
from typing import Any

from ai_pipeline.core.errors import OperationError

_INDEX_LIST = re.compile(r"^\d+(,\d+)*$")


def _torch_cuda_state() -> dict[str, Any]:
    import torch  # heavy import, deliberately local

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    return {
        "cuda_available": available,
        "device_count": count,
        "device_names": names,
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
    }


def _parse_indices(requested: str) -> list[int] | None:
    """Return the GPU indices named by ``requested`` or None for non-GPU values."""
    r = requested.strip().lower()
    if r == "cuda":
        return [0]
    if r.startswith("cuda:"):
        rest = r[len("cuda:") :]
        if _INDEX_LIST.match(rest):
            return [int(x) for x in rest.split(",")]
        raise OperationError(
            f"invalid device specification {requested!r}",
            code="DEVICE_SPEC_INVALID",
            details={"requested": requested},
            hint='Use "cpu", "auto", "cuda", "cuda:N", "N" or "N,M".',
        )
    if _INDEX_LIST.match(r):
        return [int(x) for x in r.split(",")]
    raise OperationError(
        f"invalid device specification {requested!r}",
        code="DEVICE_SPEC_INVALID",
        details={"requested": requested},
        hint='Use "cpu", "auto", "cuda", "cuda:N", "N" or "N,M".',
    )


def resolve_device(requested: str | int | None) -> dict[str, Any]:
    """Resolve the configured device strictly.

    Returns ``{"requested", "used", "policy", "cuda_available", "device_count",
    "device_names", "torch", "torch_cuda"}``. Raises GPU_DEVICE_UNAVAILABLE when an explicit
    GPU request cannot be honoured.
    """
    req = "0" if requested is None else str(requested)
    state = _torch_cuda_state()
    info = {"requested": req, **state}

    if req.strip().lower() == "cpu":
        return {**info, "used": "cpu", "policy": "explicit-cpu"}
    if req.strip().lower() == "auto":
        used = "0" if state["cuda_available"] else "cpu"
        return {**info, "used": used, "policy": "auto-fallback"}

    indices = _parse_indices(req)
    missing = [i for i in indices if i >= state["device_count"]]
    if not state["cuda_available"] or missing:
        raise OperationError(
            f"requested GPU device {req!r} is not available "
            f"(cuda_available={state['cuda_available']}, device_count={state['device_count']})",
            code="GPU_DEVICE_UNAVAILABLE",
            details={
                "requested": req,
                "missing_indices": missing,
                "cuda_available": state["cuda_available"],
                "device_count": state["device_count"],
                "device_names": state["device_names"],
                "torch": state["torch"],
                "torch_cuda": state["torch_cuda"],
            },
            hint=(
                "Check `nvidia-smi` and that the job interpreter has a CUDA-enabled torch "
                "(torch.version.cuda is None for CPU-only wheels). Set training.device to an "
                'existing index, or to "cpu" / "auto" only if CPU execution is intended.'
            ),
        )
    return {**info, "used": ",".join(str(i) for i in indices), "policy": "explicit-gpu"}
