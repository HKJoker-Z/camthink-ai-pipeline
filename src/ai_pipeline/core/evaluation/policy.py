"""Acceptance rules: ``metric -> {min, max}`` from project configuration applied to a metrics
dict. No policy is hard-coded here; the project decides which metrics matter and how much."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai_pipeline.core.config.schema import Rule


def apply_rules(metrics: Mapping[str, Any], rules: Mapping[str, Rule]) -> dict[str, Any]:
    checks = []
    for metric, rule in sorted(rules.items()):
        value = metrics.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            checks.append(
                {
                    "metric": metric,
                    "value": value,
                    "rule": rule.model_dump(exclude_none=True),
                    "passed": False,
                    "reason": "metric missing or non-numeric",
                }
            )
            continue
        ok = rule.check(float(value))
        checks.append(
            {
                "metric": metric,
                "value": value,
                "rule": rule.model_dump(exclude_none=True),
                "passed": ok,
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
