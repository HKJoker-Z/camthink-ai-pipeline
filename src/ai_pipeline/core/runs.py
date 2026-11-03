"""Runs: the control-plane record of a job (train / evaluate / autolabel).

``artifacts/<p>/runs/<run_id>.json`` is the generic, backend-agnostic record; the job directory
``artifacts/<p>/runs/<run_id>/`` holds what the compute host sees and produces. The record never
contains backend-specific concepts — the backend's opaque ``handle`` is the only thing it
carries from that side.

Lifecycle::

    CREATED ──submit──▶ SUBMITTED ──runner starts──▶ RUNNING ──▶ COMPLETED | FAILED | CANCELLED

State changes are observed by :func:`sync_run` (called by ``run status``, ``--wait`` and every
other command that needs fresh state). On COMPLETED a kind-specific post hook turns the job
output into pipeline records:

    train     → ModelVersion (TRAINED)              core.registry.register_from_run
    evaluate  → ModelVersion metrics + acceptance   core.registry.apply_evaluation
    autolabel → LabelSet (source.kind = model)      _autolabel_to_label_set
    export    → ExportArtifact (READY)              core.exports.export_from_run

Hooks are idempotent (``post_processed`` flag) and a hook failure leaves the run COMPLETED with
``post_error`` set so it can be retried.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ai_pipeline.backends import get_backend
from ai_pipeline.core.data.assets import AssetInfo
from ai_pipeline.core.data.importer import asset_store, scoped_asset_infos
from ai_pipeline.core.dataset.build import dataset_dir
from ai_pipeline.core.errors import NotFoundError, OperationError, StateError
from ai_pipeline.core.ids import new_event_id, utc_now_iso
from ai_pipeline.core.jobs.spec import (
    TERMINAL_JOB_STATES,
    JobSpec,
    JobState,
    JobStatus,
    read_result,
    read_status,
    write_spec,
)
from ai_pipeline.core.labeling.annotations import Annotation, AnnotationSource, BoundingBox
from ai_pipeline.core.labeling.labelsets import create_label_set
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import (
    RUN_TRANSITIONS,
    TERMINAL_RUN_STATES,
    DatasetState,
    RunState,
    transition,
)

log = get_logger("runs")
RUN_ID_PREFIX = "run"

_JOB_TO_RUN = {
    JobState.PENDING: RunState.SUBMITTED,
    JobState.RUNNING: RunState.RUNNING,
    JobState.COMPLETED: RunState.COMPLETED,
    JobState.FAILED: RunState.FAILED,
    JobState.CANCELLED: RunState.CANCELLED,
}


def job_dir(project: Project, run_id: str) -> Path:
    return project.store.root / "runs" / run_id


def _rel(project: Project, path: Path) -> str:
    return str(Path(path).resolve().relative_to(project.settings.artifact_root.resolve()))


# --------------------------------------------------------------------------- creation


def create_run(
    project: Project,
    *,
    kind: str,
    provider: str,
    backend: str,
    inputs: dict[str, Any],
    params: dict[str, Any],
    candidate: dict[str, Any] | None = None,
    job_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the run record (CREATED) and the job spec. Does not launch anything."""
    get_backend(backend, project.settings.infra)  # fail early on unknown/unimplemented backend
    run_id = new_event_id(RUN_ID_PREFIX)
    jdir = job_dir(project, run_id)
    jdir.mkdir(parents=True, exist_ok=False)
    now = utc_now_iso()
    spec = JobSpec(
        run_id=run_id,
        kind=kind,  # type: ignore[arg-type]
        project=project.name,
        task_type=project.config.project.task_type,
        provider=provider,
        inputs=job_inputs if job_inputs is not None else inputs,
        params=params,
        created_at=now,
    )
    write_spec(jdir, spec)
    record = {
        "run_id": run_id,
        "kind": kind,
        "project": project.name,
        "task_type": project.config.project.task_type,
        "state": RunState.CREATED.value,
        "backend": backend,
        "provider": provider,
        "candidate": candidate,
        "inputs": inputs,
        "params": params,
        "job_dir": _rel(project, jdir),
        "timestamps": {"created": now, "submitted": None, "started": None, "finished": None},
        "progress": {},
        "outputs": {},
        "metrics": {},
        "environment": {},
        "handle": {},
        "error": None,
        "model_version": None,
        "label_set_id": None,
        "export_id": None,
        "post_processed": False,
        "post_error": None,
    }
    project.store.create("runs", run_id, record)
    return record


def submit_run(project: Project, run_id: str) -> dict[str, Any]:
    run = project.store.read("runs", run_id)
    if run["state"] != RunState.CREATED.value:
        raise StateError(
            f"run {run_id} is {run['state']}; only CREATED runs can be submitted",
            code="RUN_NOT_CREATED",
        )
    backend = get_backend(run["backend"], project.settings.infra)
    try:
        status = backend.submit(job_dir(project, run_id))
    except OperationError as exc:
        run["state"] = RunState.FAILED.value
        run["timestamps"]["finished"] = utc_now_iso()
        run["error"] = {"type": exc.code, "message": exc.message, **exc.details}
        project.store.write("runs", run_id, run)
        raise
    run["state"] = transition(
        RunState.CREATED, RunState.SUBMITTED, RUN_TRANSITIONS, entity=run_id
    ).value
    run["timestamps"]["submitted"] = status.submitted_at or utc_now_iso()
    run["handle"] = status.handle
    project.store.write("runs", run_id, run)
    log.bind(project=project.name, run_id=run_id).info("submitted %s run", run["kind"])
    return run


def start_training(
    project: Project,
    *,
    dataset_version: str | None = None,
    candidate_name: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    cfg = project.config
    ds_version = dataset_version or project.pointers()["current_dataset_version"]
    if not ds_version:
        raise StateError(
            "no dataset version to train on",
            code="NO_DATASET",
            hint="Run `dataset build <project>` first or pass --dataset.",
        )
    ds = project.store.read("datasets", ds_version)
    if ds["state"] != DatasetState.READY.value:
        raise StateError(
            f"dataset {ds_version} is {ds['state']}, not READY",
            code="DATASET_NOT_READY",
            details={"dataset_version": ds_version, "state": ds["state"]},
        )
    candidates = {c.name: c for c in cfg.model.candidates}
    if candidate_name is None:
        cand = cfg.model.candidates[0]
    elif candidate_name in candidates:
        cand = candidates[candidate_name]
    else:
        raise StateError(
            f"unknown candidate {candidate_name!r}",
            code="CANDIDATE_UNKNOWN",
            details={"candidate": candidate_name, "known": sorted(candidates)},
        )
    ddir = dataset_dir(project, ds_version)
    data_config = ddir / (ds["files"].get("data_config") or "data.yaml")
    params = {
        "candidate": cand.model_dump(),
        "epochs": cfg.training.epochs,
        "image_size": cfg.training.image_size,
        "batch_size": cfg.training.batch_size,
        "device": cfg.training.device,
        "provider_args": cfg.training.provider_args,
        "classes": list(cfg.dataset.classes),
    }
    run = create_run(
        project,
        kind="train",
        provider=cfg.model.provider,
        backend=backend or cfg.training.backend,
        inputs={
            "dataset_version": ds_version,
            "dataset_content_hash": ds["content_hash"],
            "dataset_dir": str(ddir),
            "data_config": str(data_config),
        },
        params=params,
        candidate=cand.model_dump(),
    )
    return submit_run(project, run["run_id"])


def start_evaluation(
    project: Project,
    model_version: str,
    *,
    dataset_version: str | None = None,
    split: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    cfg = project.config
    model = project.store.read("models", model_version)
    ds_version = dataset_version or model.get("dataset_version")
    if not ds_version:
        raise StateError(
            f"model {model_version} has no dataset version and none was given",
            code="NO_DATASET",
            hint="Pass --dataset <ds_version>.",
        )
    ds = project.store.read("datasets", ds_version)
    weights = project.settings.artifact_root / model["artifacts"]["weights"]["path"]
    data_config = dataset_dir(project, ds_version) / (ds["files"].get("data_config") or "data.yaml")
    split = split or cfg.evaluation.split
    run = create_run(
        project,
        kind="evaluate",
        provider=model["provider"],
        backend=backend or cfg.training.backend,
        inputs={
            "model_version": model_version,
            "dataset_version": ds_version,
            "split": split,
            "weights": str(weights),
            "data_config": str(data_config),
        },
        params={
            "classes": list(cfg.dataset.classes),
            "image_size": cfg.training.image_size,
            "device": cfg.training.device,
            "provider_args": cfg.training.provider_args,
        },
        candidate=model.get("candidate"),
    )
    return submit_run(project, run["run_id"])


def start_autolabel(
    project: Project,
    *,
    import_ids: list[str] | None = None,
    weights: str | None = None,
    confidence: float | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    cfg = project.config
    auto = cfg.labeling.auto
    if auto is None:
        raise StateError(
            "project has no labeling.auto configuration",
            code="NO_AUTOLABEL_CONFIG",
            hint="Add a `labeling.auto` section to project.yaml.",
        )
    assets = scoped_asset_infos(project, import_ids)
    if not assets:
        raise StateError(
            "no imported assets to label",
            code="NO_ASSETS",
            hint="Run `data import <project> <source>` first.",
        )
    store = asset_store(project)
    # The job dir does not exist yet; write the image list after create_run.
    params = {
        "weights": weights or auto.weights,
        "confidence": auto.confidence if confidence is None else confidence,
        "class_map": dict(auto.class_map),
        "image_size": cfg.training.image_size,
        "device": cfg.training.device,
        "provider_args": {},
    }
    run = create_run(
        project,
        kind="autolabel",
        provider=auto.provider,
        backend=backend or auto.backend,
        inputs={"import_ids": import_ids or [], "asset_count": len(assets)},
        params=params,
        job_inputs={"images_list": "inputs/images.txt", "import_ids": import_ids or []},
    )
    lst = job_dir(project, run["run_id"]) / "inputs" / "images.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("".join(f"{store.file_path(a)}\n" for a in assets), encoding="utf-8")
    return submit_run(project, run["run_id"])


# --------------------------------------------------------------------------- observation


def _advance(run: dict[str, Any], target: RunState, status: JobStatus) -> None:
    current = RunState(run["state"])
    if current is target:
        return
    # A fast job can be terminal before we ever observed RUNNING; walk through it.
    if (
        current is RunState.SUBMITTED
        and target in TERMINAL_RUN_STATES
        and target is not RunState.CANCELLED
    ):
        transition(current, RunState.RUNNING, RUN_TRANSITIONS, entity=run["run_id"])
        current = RunState.RUNNING
    transition(current, target, RUN_TRANSITIONS, entity=run["run_id"])
    run["state"] = target.value


def sync_run(project: Project, run_id: str) -> dict[str, Any]:
    """Reconcile the run record with the job's status.json; run post hooks on completion."""
    run = project.store.read("runs", run_id)
    if RunState(run["state"]) in TERMINAL_RUN_STATES:
        if run["state"] == RunState.COMPLETED.value and not run.get("post_processed"):
            _post_process(project, run)
        return run
    jdir = job_dir(project, run_id)
    if run["state"] == RunState.CREATED.value:
        # Never submitted — unless the backend recorded a failed submit attempt.
        status = read_status(jdir)
        if status is not None and status.state in (JobState.FAILED, JobState.CANCELLED):
            run["state"] = _JOB_TO_RUN[status.state].value
            run["timestamps"]["finished"] = status.finished_at or utc_now_iso()
            run["error"] = status.error
            project.store.write("runs", run_id, run)
        return run

    backend = get_backend(run["backend"], project.settings.infra)
    status = backend.status(jdir)
    if status.state in TERMINAL_JOB_STATES:
        backend.fetch(jdir)

    target = _JOB_TO_RUN[status.state]
    _advance(run, target, status)
    ts = run["timestamps"]
    ts["started"] = ts["started"] or status.started_at
    ts["finished"] = ts["finished"] or status.finished_at
    run["progress"] = status.progress
    if status.handle:
        run["handle"] = status.handle
    if status.state in (JobState.FAILED, JobState.CANCELLED):
        run["error"] = status.error
    if status.state is JobState.COMPLETED:
        result = read_result(jdir)
        if result is None:
            run["state"] = RunState.FAILED.value
            run["error"] = {
                "type": "ResultMissing",
                "message": "job COMPLETED but out/result.json is missing",
            }
        else:
            run["outputs"] = {k: _rel(project, jdir / v) for k, v in result.artifacts.items()}
            run["outputs"]["result"] = _rel(project, jdir / "out/result.json")
            run["metrics"] = result.metrics
            run["environment"] = result.environment
    project.store.write("runs", run_id, run)
    if run["state"] == RunState.COMPLETED.value:
        _post_process(project, run)
    return run


def wait_run(
    project: Project, run_id: str, *, timeout_s: float = 3600, interval_s: float = 0.5
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        run = sync_run(project, run_id)
        if RunState(run["state"]) in TERMINAL_RUN_STATES:
            return run
        if time.monotonic() >= deadline:
            raise OperationError(
                f"run {run_id} still {run['state']} after {timeout_s:.0f}s",
                code="RUN_WAIT_TIMEOUT",
                details={"run_id": run_id, "state": run["state"]},
                hint="Check again later with `run status <project> <run_id>`.",
            )
        time.sleep(interval_s)


def cancel_run(project: Project, run_id: str) -> dict[str, Any]:
    run = project.store.read("runs", run_id)
    if RunState(run["state"]) in TERMINAL_RUN_STATES:
        return run
    if run["state"] == RunState.CREATED.value:
        run["state"] = RunState.CANCELLED.value
        run["timestamps"]["finished"] = utc_now_iso()
        project.store.write("runs", run_id, run)
        return run
    backend = get_backend(run["backend"], project.settings.infra)
    backend.cancel(job_dir(project, run_id))
    return sync_run(project, run_id)


def run_logs(project: Project, run_id: str, *, tail: int | None = None) -> str:
    run = project.store.read("runs", run_id)
    backend = get_backend(run["backend"], project.settings.infra)
    return backend.logs(job_dir(project, run_id), tail=tail)


def list_runs(project: Project, kind: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for r in project.store.list("runs"):
        if kind and r["kind"] != kind:
            continue
        rows.append(
            {
                "run_id": r["run_id"],
                "kind": r["kind"],
                "state": r["state"],
                "backend": r["backend"],
                "provider": r["provider"],
                "candidate": (r.get("candidate") or {}).get("name"),
                "dataset_version": (r.get("inputs") or {}).get("dataset_version"),
                "model_version": r.get("model_version"),
                "label_set_id": r.get("label_set_id"),
                "export_id": r.get("export_id"),
                "created": r["timestamps"]["created"],
                "finished": r["timestamps"]["finished"],
            }
        )
    return sorted(rows, key=lambda r: r["created"])


# --------------------------------------------------------------------------- post hooks


def _post_process(project: Project, run: dict[str, Any]) -> None:
    try:
        if run["kind"] == "train":
            from ai_pipeline.core.registry import register_from_run

            register_from_run(project, run)
        elif run["kind"] == "evaluate":
            from ai_pipeline.core.registry import apply_evaluation

            apply_evaluation(
                project,
                run["inputs"]["model_version"],
                run_id=run["run_id"],
                metrics=run["metrics"],
                rules=project.config.evaluation.acceptance,
                dataset_version=run["inputs"].get("dataset_version"),
                split=run["inputs"].get("split", "test"),
            )
        elif run["kind"] == "autolabel":
            _autolabel_to_label_set(project, run)
        elif run["kind"] == "export":
            from ai_pipeline.core.exports import export_from_run

            export_from_run(project, run)
        run["post_processed"] = True
        run["post_error"] = None
    except Exception as exc:
        run["post_processed"] = False
        run["post_error"] = {"type": type(exc).__name__, "message": str(exc)}
        project.store.write("runs", run["run_id"], run)
        raise OperationError(
            f"run {run['run_id']} completed but post-processing failed: {exc}",
            code="RUN_POST_PROCESS_FAILED",
            details={"run_id": run["run_id"], "kind": run["kind"]},
            hint="Fix the cause and re-run `run status <project> <run_id>`; the hook is retried.",
        ) from exc
    project.store.write("runs", run["run_id"], run)


def _autolabel_to_label_set(project: Project, run: dict[str, Any]) -> dict[str, Any]:
    if run.get("label_set_id"):
        return project.store.read("labels", run["label_set_id"])
    det_rel = run["outputs"].get("detections")
    if not det_rel:
        raise NotFoundError(
            "autolabel run produced no detections artifact", code="RUN_NO_DETECTIONS"
        )
    det_path = project.settings.artifact_root / det_rel
    import_ids = run["inputs"].get("import_ids") or None
    scope: list[AssetInfo] = scoped_asset_infos(project, import_ids)
    by_id = {a.asset_id: a for a in scope}
    class_map: dict[str, str] = run["params"]["class_map"]
    label_set_id = new_event_id("lbl")
    source = AnnotationSource(
        kind="model",
        ref=run["params"]["weights"],
        label_set_id=label_set_id,
        run_id=run["run_id"],
        provider=run["provider"],
    )
    annotations: list[Annotation] = []
    labeled: set[str] = set()
    dropped_unmapped = 0
    unknown_assets = 0
    with det_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            asset_id = Path(row["image"]).stem
            if asset_id not in by_id:
                unknown_assets += 1
                continue
            target = class_map.get(row["class_name"])
            if target is None:
                dropped_unmapped += 1
                continue
            x1, y1, x2, y2 = row["bbox"]
            annotations.append(
                Annotation(
                    asset_id=asset_id,
                    class_name=target,
                    bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                    confidence=row.get("confidence"),
                    source=source,
                    metadata={"source_class": row["class_name"]},
                )
            )
            labeled.add(asset_id)
    record = create_label_set(
        project,
        label_set_id=label_set_id,
        annotations=annotations,
        source={
            "kind": "model",
            "provider": run["provider"],
            "weights": run["params"]["weights"],
            "confidence": run["params"]["confidence"],
            "class_map": class_map,
            "run_id": run["run_id"],
            "dropped_unmapped": dropped_unmapped,
            "unknown_assets": unknown_assets,
        },
        scope_assets=scope,
        import_ids=import_ids,
        labeled_asset_ids=labeled,
        run_id=run["run_id"],
    )
    run["label_set_id"] = label_set_id
    project.store.write("runs", run["run_id"], run)
    log.bind(project=project.name, run_id=run["run_id"]).info(
        "autolabel -> label set %s (%d annotations, %d unmapped dropped)",
        label_set_id,
        len(annotations),
        dropped_unmapped,
    )
    return record
