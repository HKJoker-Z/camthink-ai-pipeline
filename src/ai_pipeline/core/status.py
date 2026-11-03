"""Derived project status: entity counts, pointers, latest run, and next valid operations.

Nothing here is persisted; it is computed from the store so it can never drift from the
entity records.
"""

from __future__ import annotations

from typing import Any

from ai_pipeline.core.project import Project
from ai_pipeline.core.state import TERMINAL_RUN_STATES, RunState
from ai_pipeline.core.store import KINDS


def project_status(project: Project) -> dict[str, Any]:
    store = project.store
    record = project.record()
    counts = {kind: store.count(kind) for kind in KINDS}
    pointers = project.pointers()

    runs = store.list("runs")
    latest_run = max(runs, key=lambda r: r.get("created_at", ""), default=None)
    active_runs = [
        r["run_id"]
        for r in runs
        if r.get("state") and RunState(r["state"]) not in TERMINAL_RUN_STATES
    ]

    summary = {
        "project": project.name,
        "task_type": record.get("task_type"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "config_path": str(project.config_path),
        "artifact_dir": str(project.artifact_dir),
        "pointers": pointers,
        "counts": counts,
        "latest_run": (
            {k: latest_run.get(k) for k in ("run_id", "kind", "state", "created_at")}
            if latest_run
            else None
        ),
        "active_runs": active_runs,
    }
    summary["next_operations"] = next_operations(project.name, counts, pointers, active_runs)
    return summary


def next_operations(
    name: str,
    counts: dict[str, int],
    pointers: dict[str, str | None],
    active_runs: list[str],
) -> list[str]:
    """Suggest the operations that make sense now. Advisory only; never enforced."""
    ops: list[str] = []
    if active_runs:
        ops.extend(f"train status {name} {rid}" for rid in active_runs)
        return ops
    if counts["imports"] == 0:
        return [f"data import {name} <path>"]
    if counts["labels"] == 0:
        return [f"label auto {name}", f"label import {name} <path>"]
    if counts["datasets"] == 0:
        return [f"dataset validate {name}", f"dataset build {name}"]
    if counts["runs"] == 0:
        return [f"train start {name}"]
    if counts["models"] == 0:
        return [f"train list {name}", f"model register {name} <run_id>"]
    if pointers.get("selected_model_version") is None:
        return [f"model select {name}"]
    if pointers.get("active_deployment") is None:
        return [f"export run {name}", f"deploy run {name}"]
    return [f"health {name}", f"deploy status {name}", f"train start {name}"]
