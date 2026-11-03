"""Exports: a model version → an immutable, portable artifact (ONNX in V1).

``artifacts/<p>/exports/exp-NNN.json``::

    {
      "export_id": "exp-001", "state": "READY", "project": "...",
      "model_version": "mv-001", "format": "onnx", "provider": "ultralytics",
      "export_run_id": "run-…",
      "source_weights": {"path": "…/best.pt", "sha256": "…"},
      "export_params": {"opset": 17, "image_size": 640, "dynamic": false},
      "environment": {...},                       # from the job result
      "artifact": {"path": "<rel to artifact root>", "sha256": "…", "size_bytes": N},
      "details": {...},                           # exporter extras (input/output names, …)
      "created_at": "…"
    }

Export is a compute job (``kind: export``) so the control plane never imports the ML runtime.
An export never changes the model version's state; the model only gains the export id in its
``exports`` list. Records are never overwritten: ``start_export`` either reuses a READY export
with an identical *request* (model, weights hash, format, params) or creates a new one.
"""

from __future__ import annotations

from typing import Any

from ai_pipeline.core.data.assets import sha256_file
from ai_pipeline.core.errors import NotFoundError, StateError
from ai_pipeline.core.ids import next_sequential_id, utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.state import ExportState

log = get_logger("exports")
EXPORT_ID_PREFIX = "exp-"


def export_params(project: Project) -> dict[str, Any]:
    e = project.config.export
    return {"opset": e.opset, "image_size": e.image_size, "dynamic": e.dynamic}


def resolve_model_version(project: Project, model_version: str | None) -> dict[str, Any]:
    mv = model_version or project.pointers()["selected_model_version"]
    if not mv:
        raise StateError(
            "no model version given and no selected model",
            code="NO_SELECTED_MODEL",
            hint="Run `model select <project>` or pass an explicit model version.",
        )
    return project.store.read("models", mv)


def find_identical_export(
    project: Project, *, model_version: str, fmt: str, weights_sha256: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    for rec in project.store.list("exports"):
        if (
            rec.get("state") == ExportState.READY.value
            and rec["model_version"] == model_version
            and rec["format"] == fmt
            and rec["source_weights"]["sha256"] == weights_sha256
            and rec["export_params"] == params
        ):
            return rec
    return None


def start_export(
    project: Project,
    *,
    model_version: str | None = None,
    backend: str | None = None,
    rerun: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Returns ``(run, existing_export)``: exactly one is not None. ``existing_export`` means an
    identical READY export already exists and no job was started (unless ``rerun``)."""
    from ai_pipeline.core.runs import create_run, submit_run

    cfg = project.config
    model = resolve_model_version(project, model_version)
    weights = model["artifacts"]["weights"]
    fmt = cfg.export.format
    params = export_params(project)
    if not rerun:
        existing = find_identical_export(
            project,
            model_version=model["model_version"],
            fmt=fmt,
            weights_sha256=weights["sha256"],
            params=params,
        )
        if existing is not None:
            return None, existing
    weights_path = project.settings.artifact_root / weights["path"]
    if not weights_path.is_file():
        raise NotFoundError(
            f"weights file missing: {weights_path}",
            code="WEIGHTS_MISSING",
            details={"path": str(weights_path)},
        )
    run = create_run(
        project,
        kind="export",
        provider=model["provider"],
        backend=backend or cfg.training.backend,
        inputs={
            "model_version": model["model_version"],
            "format": fmt,
            "weights": str(weights_path),
            "weights_sha256": weights["sha256"],
        },
        params={
            **params,
            "classes": list(cfg.dataset.classes),
            "provider_args": cfg.export.provider_args,
        },
        candidate=model.get("candidate"),
    )
    return submit_run(project, run["run_id"]), None


def export_from_run(project: Project, run: dict[str, Any]) -> dict[str, Any]:
    """Post hook of a COMPLETED export run: create the immutable ExportArtifact record."""
    from ai_pipeline.core.registry import link_model

    if run.get("export_id"):
        return project.store.read("exports", run["export_id"])
    model_rel = (run.get("outputs") or {}).get("model")
    if not model_rel:
        raise NotFoundError("export run produced no model artifact", code="RUN_NO_EXPORT_ARTIFACT")
    path = project.settings.artifact_root / model_rel
    if not path.is_file():
        raise NotFoundError(
            f"export artifact missing: {path}",
            code="EXPORT_ARTIFACT_MISSING",
            details={"path": str(path)},
        )
    export_id = next_sequential_id(EXPORT_ID_PREFIX, project.store.list_ids("exports"))
    inputs = run["inputs"]
    params = {k: run["params"][k] for k in ("opset", "image_size", "dynamic") if k in run["params"]}
    record = {
        "export_id": export_id,
        "state": ExportState.READY.value,
        "project": project.name,
        "model_version": inputs["model_version"],
        "format": inputs["format"],
        "provider": run["provider"],
        "export_run_id": run["run_id"],
        "source_weights": {"path": None, "sha256": inputs.get("weights_sha256")},
        "export_params": params,
        "environment": run.get("environment") or {},
        "artifact": {
            "path": model_rel,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
        "details": {},
        "created_at": utc_now_iso(),
    }
    model = project.store.read("models", inputs["model_version"])
    record["source_weights"]["path"] = model["artifacts"]["weights"]["path"]
    from ai_pipeline.core.jobs.spec import read_result
    from ai_pipeline.core.runs import job_dir

    result = read_result(job_dir(project, run["run_id"]))
    if result is not None:
        record["details"] = result.extra
    project.store.create("exports", export_id, record)
    link_model(project, inputs["model_version"], "exports", export_id)
    run["export_id"] = export_id
    project.store.write("runs", run["run_id"], run)
    log.bind(
        project=project.name, run_id=run["run_id"], model_version=inputs["model_version"]
    ).info(
        "export %s ready (%s, %d bytes)",
        export_id,
        inputs["format"],
        record["artifact"]["size_bytes"],
    )
    return record


def latest_export_for(
    project: Project, model_version: str, fmt: str = "onnx"
) -> dict[str, Any] | None:
    candidates = [
        r
        for r in project.store.list("exports")
        if r["model_version"] == model_version
        and r["format"] == fmt
        and r["state"] == ExportState.READY.value
    ]
    return max(candidates, key=lambda r: r["created_at"], default=None)


def list_exports(project: Project) -> list[dict[str, Any]]:
    return [
        {
            "export_id": r["export_id"],
            "state": r["state"],
            "model_version": r["model_version"],
            "format": r["format"],
            "export_run_id": r["export_run_id"],
            "size_bytes": r["artifact"]["size_bytes"],
            "sha256": r["artifact"]["sha256"][:12],
            "created_at": r["created_at"],
        }
        for r in project.store.list("exports")
    ]
