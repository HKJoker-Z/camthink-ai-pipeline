"""Derived project status: entity counts, pointers, latest run, and next valid operations.

Nothing here is persisted; it is computed from the store so it can never drift from the
entity records.
"""

from __future__ import annotations

from typing import Any

from ai_pipeline.core.data.assets import AssetStore
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import TERMINAL_RUN_STATES, ModelState, RunState
from ai_pipeline.core.store import KINDS


def project_status(project: Project) -> dict[str, Any]:
    store = project.store
    record = project.record()
    counts = {kind: store.count(kind) for kind in KINDS}
    counts["assets"] = AssetStore(project.artifact_dir / "assets").count()
    pointers = project.pointers()

    runs = store.list("runs")
    latest_run = max(
        runs,
        key=lambda r: r["timestamps"]["created"] if "timestamps" in r else r.get("created_at", ""),
        default=None,
    )
    active_runs = [
        r["run_id"]
        for r in runs
        if r.get("state") and RunState(r["state"]) not in TERMINAL_RUN_STATES
    ]
    models = store.list("models")
    model_states = {m["model_version"]: m["state"] for m in models}
    completed_train_unregistered = [
        r["run_id"]
        for r in runs
        if r.get("kind") == "train" and r.get("state") == "COMPLETED" and not r.get("model_version")
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
            {
                "run_id": latest_run["run_id"],
                "kind": latest_run.get("kind"),
                "state": latest_run.get("state"),
                "created_at": (latest_run.get("timestamps") or {}).get(
                    "created", latest_run.get("created_at")
                ),
            }
            if latest_run
            else None
        ),
        "active_runs": active_runs,
        "model_states": model_states,
        "deployment_states": {d["deployment_id"]: d["state"] for d in store.list("deployments")},
    }
    deployment_states = summary["deployment_states"]
    summary["next_operations"] = next_operations(
        project.name,
        counts,
        pointers,
        active_runs,
        model_states=model_states,
        unregistered_train_runs=completed_train_unregistered,
        exported_models={
            e["model_version"] for e in store.list("exports") if e.get("state") == "READY"
        },
        deployment_states=deployment_states,
    )
    return summary


def next_operations(
    name: str,
    counts: dict[str, int],
    pointers: dict[str, str | None],
    active_runs: list[str],
    *,
    model_states: dict[str, str] | None = None,
    unregistered_train_runs: list[str] | None = None,
    exported_models: set[str] | None = None,
    deployment_states: dict[str, str] | None = None,
) -> list[str]:
    """Suggest the operations that make sense now. Advisory only; never enforced."""
    model_states = model_states or {}
    exported_models = exported_models or set()
    deployment_states = deployment_states or {}
    broken = [d for d, s in deployment_states.items() if s == "ROLLBACK_FAILED"]
    if broken and pointers.get("active_deployment") is None:
        # The target's live state is unknown: inspect before deploying anything else.
        return [
            f"health {name}",
            f"deploy status {name} {broken[-1]}",
            f"rollback {name} --to <dep>",
        ]
    if active_runs:
        return [f"run status {name} {rid}" for rid in active_runs]
    if counts["imports"] == 0:
        return [f"data import {name} <path>"]
    if counts["labels"] == 0:
        return [f"label import {name} <path>", f"label auto {name}"]
    if pointers.get("current_dataset_version") is None:
        return [f"dataset validate {name}", f"dataset build {name}"]
    if unregistered_train_runs:
        return [f"model register {name} {rid}" for rid in unregistered_train_runs]
    if counts["models"] == 0:
        return [f"train start {name}"]
    to_evaluate = [mv for mv, s in model_states.items() if s == ModelState.TRAINED.value]
    if to_evaluate:
        return [f"evaluate run {name} {mv}" for mv in to_evaluate]
    accepted = [mv for mv, s in model_states.items() if s == ModelState.ACCEPTED.value]
    selected = pointers.get("selected_model_version")
    if selected is None:
        if accepted:
            return [f"model select {name}"]
        return [f"train start {name}", f"model list {name}"]
    if not exported_models or selected not in exported_models:
        return [f"export start {name}"]
    if pointers.get("active_deployment") is None:
        return [f"deploy start {name}"]
    return [f"health {name}", f"deploy status {name}", f"train start {name}"]
