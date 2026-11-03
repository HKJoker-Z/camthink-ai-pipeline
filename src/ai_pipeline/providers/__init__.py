"""Model providers (the model-specific seam).

The control plane only ever touches :data:`PROVIDER_MODULES` (names). Implementations are
imported lazily by :func:`load_provider`, which is called from the job runner on the compute
host — the only place where ultralytics/torch are allowed to load.
"""

from __future__ import annotations

import importlib

from ai_pipeline.core.errors import ConfigError, OperationError

# name -> "module:ClassName"
PROVIDER_MODULES: dict[str, str] = {
    "fake": "ai_pipeline.providers.fake:FakeProvider",
    "ultralytics": "ai_pipeline.providers.ultralytics:UltralyticsProvider",
}


def provider_names() -> list[str]:
    return sorted(PROVIDER_MODULES)


def load_provider(name: str):
    """Import and instantiate a provider. Heavy imports happen here, on purpose."""
    try:
        target = PROVIDER_MODULES[name]
    except KeyError:
        raise ConfigError(
            f"unknown provider {name!r}",
            code="PROVIDER_UNKNOWN",
            details={"provider": name, "known": provider_names()},
        ) from None
    module_name, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise OperationError(
            f"provider {name!r} is not installed in this environment: {exc}",
            code="PROVIDER_NOT_INSTALLED",
            details={"provider": name, "module": module_name, "error": str(exc)},
            hint=f"Install the provider extra on the compute host, e.g. "
            f"`pip install -e '.[provider-{name}]'`.",
        ) from exc
    return getattr(module, class_name)()
