"""Model versions (``artifacts/<p>/models/<mv>.json``).

A model version is created from a COMPLETED train run and references — never copies — the
run's artifacts. Its lifecycle is about model *quality* only (``core.state.MODEL_TRANSITIONS``)::

    TRAINED → EVALUATED → ACCEPTED | REJECTED   (re-evaluation always allowed)

Exports (``exports/exp-NNN``) and deployments (``deployments/dep-NNN``) are separate entities
that reference a model version; the model keeps ``exports`` / ``deployments`` id lists for
traceability but its state never changes because of them.
"""

from __future__ import annotations

from typing import Any

from ai_pipeline.core.config.schema import Rule
from ai_pipeline.core.data.assets import sha256_file
from ai_pipeline.core.errors import NotFoundError, StateError
from ai_pipeline.core.evaluation.policy import apply_rules
from ai_pipeline.core.ids import next_sequential_id, utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import MODEL_TRANSITIONS, ModelState, transition

log = get_logger("registry")
MODEL_ID_PREFIX = "mv-"


def _history(record: dict[str, Any], state: ModelState, note: str, **extra: Any) -> None:
    record.setdefault("history", []).append(
        {"at": utc_now_iso(), "state": state.value, "note": note, **extra}
    )


def set_model_state(
    project: Project, record: dict[str, Any], new: ModelState, note: str, **extra
) -> None:
    current = ModelState(record["state"])
    if current is not new:
        transition(current, new, MODEL_TRANSITIONS, entity=f"model {record['model_version']}")
    record["state"] = new.value
    record["updated_at"] = utc_now_iso()
    _history(record, new, note, **extra)
    project.store.write("models", record["model_version"], record)


def register_from_run(project: Project, run: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Create a TRAINED model version from a completed train run. Idempotent per run."""
    if run.get("kind") != "train":
        raise StateError(
            f"run {run['run_id']} is a {run.get('kind')} run, not a train run",
            code="RUN_NOT_TRAIN",
        )
    if run.get("state") != "COMPLETED":
        raise StateError(
            f"run {run['run_id']} is {run.get('state')}, not COMPLETED",
            code="RUN_NOT_COMPLETED",
            hint="Wait for `run status <project> <run_id>` to report COMPLETED.",
        )
    if run.get("model_version"):
        return project.store.read("models", run["model_version"]), False

    weights_rel = (run.get("outputs") or {}).get("weights")
    if not weights_rel:
        raise NotFoundError(
            f"run {run['run_id']} produced no weights artifact",
            code="RUN_NO_WEIGHTS",
            details={"outputs": run.get("outputs")},
        )
    weights_path = project.settings.artifact_root / weights_rel
    if not weights_path.is_file():
        raise NotFoundError(
            f"weights file missing: {weights_path}",
            code="WEIGHTS_MISSING",
            details={"path": str(weights_path)},
        )

    mv = next_sequential_id(MODEL_ID_PREFIX, project.store.list_ids("models"))
    now = utc_now_iso()
    record = {
        "model_version": mv,
        "state": ModelState.TRAINED.value,
        "project": project.name,
        "task_type": run["task_type"],
        "provider": run["provider"],
        "candidate": run.get("candidate"),
        "training_run_id": run["run_id"],
        "dataset_version": (run.get("inputs") or {}).get("dataset_version"),
        "training_params": run.get("params"),
        "backend": run.get("backend"),
        "environment": run.get("environment"),
        "created_at": now,
        "updated_at": now,
        "artifacts": {
            "weights": {
                "path": weights_rel,
                "sha256": sha256_file(weights_path),
                "size_bytes": weights_path.stat().st_size,
            }
        },
        "metrics": {"train": run.get("metrics") or {}, "eval": None},
        "evaluation": None,
        "exports": [],
        "deployments": [],
        "history": [],
    }
    _history(record, ModelState.TRAINED, f"registered from run {run['run_id']}")
    project.store.create("models", mv, record)
    project.store.update("runs", run["run_id"], model_version=mv)
    run["model_version"] = mv
    log.bind(project=project.name, model_version=mv, run_id=run["run_id"]).info("registered")
    return record, True


def apply_evaluation(
    project: Project,
    model_version: str,
    *,
    run_id: str,
    metrics: dict[str, Any],
    rules: dict[str, Rule],
    dataset_version: str | None,
    split: str,
) -> dict[str, Any]:
    """Record evaluation metrics and apply the project's acceptance policy."""
    record = project.store.read("models", model_version)
    verdict = apply_rules(metrics, rules)
    record["metrics"]["eval"] = metrics
    record["evaluation"] = {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "split": split,
        "at": utc_now_iso(),
        "acceptance": verdict,
    }
    set_model_state(project, record, ModelState.EVALUATED, f"evaluated by run {run_id}")
    final = ModelState.ACCEPTED if verdict["passed"] else ModelState.REJECTED
    failed = [c["metric"] for c in verdict["checks"] if not c["passed"]]
    set_model_state(
        project,
        record,
        final,
        "all acceptance rules passed" if verdict["passed"] else f"failed: {failed}",
    )
    return record


def list_models(project: Project) -> list[dict[str, Any]]:
    return [
        {
            "model_version": r["model_version"],
            "state": r["state"],
            "candidate": (r.get("candidate") or {}).get("name"),
            "dataset_version": r.get("dataset_version"),
            "training_run_id": r.get("training_run_id"),
            "created_at": r["created_at"],
            "eval_metrics": {
                k: v for k, v in (r["metrics"].get("eval") or {}).items() if k != "per_class"
            },
            "accepted": (r.get("evaluation") or {}).get("acceptance", {}).get("passed"),
        }
        for r in project.store.list("models")
    ]


def select_model(project: Project, model_version: str | None) -> dict[str, Any]:
    """Explicit selection must be ACCEPTED; implicit selection picks the ACCEPTED model with
    the highest ``evaluation.selection.primary_metric``."""
    selectable = {ModelState.ACCEPTED.value}
    if model_version is not None:
        record = project.store.read("models", model_version)
        if record["state"] not in selectable:
            raise StateError(
                f"model {model_version} is {record['state']}; only accepted models can be selected",
                code="MODEL_NOT_ACCEPTED",
                details={"model_version": model_version, "state": record["state"]},
                hint="Run `evaluate run <project> <model_version>` and make sure it passes "
                "evaluation.acceptance.",
            )
    else:
        metric = project.config.evaluation.selection.primary_metric
        candidates = [
            r
            for r in project.store.list("models")
            if r["state"] in selectable
            and isinstance((r["metrics"].get("eval") or {}).get(metric), (int, float))
        ]
        if not candidates:
            raise StateError(
                "no accepted model with evaluation metrics to select from",
                code="NO_ACCEPTED_MODEL",
                hint="Train and evaluate a model first; check `model list <project>`.",
            )
        record = max(candidates, key=lambda r: (r["metrics"]["eval"][metric], r["model_version"]))
    project.set_pointer("selected_model_version", record["model_version"])
    return record


def link_model(project: Project, model_version: str, field: str, entity_id: str) -> None:
    """Append an export/deployment id to the model's traceability list (idempotent)."""
    record = project.store.read("models", model_version)
    ids = record.setdefault(field, [])
    if entity_id not in ids:
        ids.append(entity_id)
        record["updated_at"] = utc_now_iso()
        project.store.write("models", model_version, record)
