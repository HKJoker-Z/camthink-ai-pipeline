"""Model providers (the model-specific seam).

Only the names are registered here so configuration can be validated on the control host
without importing ML libraries. Implementations are loaded lazily by the job runner (M2).
"""

from __future__ import annotations

PROVIDER_NAMES: tuple[str, ...] = ("ultralytics", "fake")


def provider_names() -> list[str]:
    return list(PROVIDER_NAMES)
