"""Deployment targets (the device-specific seam)."""

from __future__ import annotations

from pathlib import Path

from ai_pipeline.core.config.schema import FakeTargetConfig
from ai_pipeline.core.errors import ConfigError, OperationError
from ai_pipeline.core.settings import Settings
from ai_pipeline.targets.base import DeploymentTarget


def get_target(name: str, settings: Settings) -> DeploymentTarget:
    try:
        cfg = settings.infra.deployment_targets[name]
    except KeyError:
        raise ConfigError(
            f"deployment target {name!r} is not defined in infrastructure.yaml",
            code="TARGET_UNKNOWN",
            details={"target": name, "defined": sorted(settings.infra.deployment_targets)},
        ) from None
    if isinstance(cfg, FakeTargetConfig):
        from ai_pipeline.targets.fake import FakeTarget

        root = Path(cfg.root)
        if not root.is_absolute():  # relative to the infrastructure file, like other roots
            root = (settings.infra_path.parent / root).resolve()
        return FakeTarget(name=name, root=root)
    raise OperationError(
        f"deployment target type {cfg.type!r} is not implemented yet (planned for M5)",
        code="TARGET_NOT_IMPLEMENTED",
        details={"target": name, "type": cfg.type},
        hint="Use a `type: fake` target for now.",
    )


__all__ = ["DeploymentTarget", "get_target"]
