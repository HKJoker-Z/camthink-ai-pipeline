"""Compute backends: where and how a job directory gets executed."""

from __future__ import annotations

from ai_pipeline.backends.base import ComputeBackend
from ai_pipeline.core.config.schema import InfrastructureConfig, LocalBackendConfig
from ai_pipeline.core.errors import ConfigError, OperationError


def get_backend(name: str, infra: InfrastructureConfig) -> ComputeBackend:
    try:
        cfg = infra.compute_backends[name]
    except KeyError:
        raise ConfigError(
            f"compute backend {name!r} is not defined in infrastructure.yaml",
            code="BACKEND_UNKNOWN",
            details={"backend": name, "defined": sorted(infra.compute_backends)},
        ) from None
    if isinstance(cfg, LocalBackendConfig):
        from ai_pipeline.backends.local import LocalBackend

        return LocalBackend(name=name, python=cfg.python)
    raise OperationError(
        f"compute backend type {cfg.type!r} is not implemented yet (planned for M4)",
        code="BACKEND_NOT_IMPLEMENTED",
        details={"backend": name, "type": cfg.type},
        hint="Use a `type: local` backend for now.",
    )


__all__ = ["ComputeBackend", "get_backend"]
