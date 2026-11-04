"""Deployments: a first-class entity recording one attempt to put an ExportArtifact on a target.

``artifacts/<p>/deployments/dep-NNN.json`` carries everything an operator or agent needs to
judge the attempt without reading logs: every phase result, the benchmark, the deployment
acceptance verdict, activation/rollback outcomes and errors. Pointers on the project record
(``active_deployment``, ``last_known_good_deployment``) name what is live.

Phase machine (``core.state.DEPLOYMENT_TRANSITIONS``)::

    CREATED → VERIFYING → STAGING → PREPARING → SMOKE_TESTING → BENCHMARKING
            → READY_TO_ACTIVATE → ACTIVATING → HEALTH_CHECKING → ACTIVE

    any pre-activation phase ──error──▶ FAILED        (active deployment untouched)
    BENCHMARKING ──policy──▶ REJECTED                 (not activated; exit code 5)
    ACTIVATING | HEALTH_CHECKING ──error──▶ ROLLING_BACK → ROLLED_BACK | ROLLBACK_FAILED
    ACTIVE → SUPERSEDED (a newer deployment activated) ; SUPERSEDED → ACTIVE (restored)
    ACTIVE → ROLLING_BACK → ROLLED_BACK                (manual `rollback`)

Two acceptance concepts stay separate: *model* acceptance (evaluation metrics, on the
ModelVersion) gates whether a model may be deployed at all; *deployment* acceptance (benchmark
metrics vs ``deployment.acceptance``) gates activation on a particular target.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_pipeline.core.errors import OperationError, PipelineError, PolicyRejected, StateError
from ai_pipeline.core.evaluation.policy import apply_rules
from ai_pipeline.core.exports import latest_export_for, resolve_model_version
from ai_pipeline.core.ids import next_sequential_id, utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.project import Project
from ai_pipeline.core.registry import link_model
from ai_pipeline.core.state import (
    DEPLOYMENT_TRANSITIONS,
    DeploymentState,
    ExportState,
    ModelState,
    transition,
)
from ai_pipeline.targets import get_target
from ai_pipeline.targets.base import DeploymentTarget

log = get_logger("deployment")
DEPLOYMENT_ID_PREFIX = "dep-"
_D = DeploymentState


class _Phase:
    """Persist state changes and phase results as they happen."""

    def __init__(self, project: Project, record: dict[str, Any]) -> None:
        self.project = project
        self.record = record

    def to(self, new: _D, **fields: Any) -> None:
        rec = self.record
        current = _D(rec["state"])
        if current is not new:
            transition(current, new, DEPLOYMENT_TRANSITIONS, entity=rec["deployment_id"])
            rec["state"] = new.value
        rec.update(fields)
        rec["timestamps"]["updated"] = utc_now_iso()
        rec["phases"].append({"state": new.value, "at": rec["timestamps"]["updated"]})
        self.project.store.write("deployments", rec["deployment_id"], rec)

    def finish(self, new: _D, **fields: Any) -> None:
        self.to(new, **fields)
        self.record["timestamps"]["finished"] = utc_now_iso()
        self.project.store.write("deployments", self.record["deployment_id"], self.record)


def _error(exc: BaseException, phase: str) -> dict[str, Any]:
    base = (
        exc.to_dict()
        if isinstance(exc, PipelineError)
        else {
            "code": "TARGET_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
            "details": {},
            "hint": None,
        }
    )
    return {**base, "phase": phase}


def start_deployment(
    project: Project,
    *,
    model_version: str | None = None,
    export_id: str | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    cfg = project.config
    model = resolve_model_version(project, model_version)
    mv = model["model_version"]
    if model["state"] != ModelState.ACCEPTED.value:
        raise StateError(
            f"model {mv} is {model['state']}; only ACCEPTED models can be deployed",
            code="MODEL_REJECTED"
            if model["state"] == ModelState.REJECTED.value
            else "MODEL_NOT_ACCEPTED",
            details={"model_version": mv, "state": model["state"]},
            hint="Evaluate the model (`evaluate run`) and make sure evaluation.acceptance passes.",
        )
    if export_id is not None:
        export = project.store.read("exports", export_id)
        if export["model_version"] != mv:
            raise StateError(
                f"export {export_id} belongs to {export['model_version']}, not {mv}",
                code="EXPORT_MODEL_MISMATCH",
            )
        if export["state"] != ExportState.READY.value:
            raise StateError(f"export {export_id} is {export['state']}", code="EXPORT_NOT_READY")
    else:
        export = latest_export_for(project, mv, cfg.export.format)
        if export is None:
            raise StateError(
                f"model {mv} has no READY {cfg.export.format} export",
                code="NO_EXPORT",
                details={"model_version": mv, "format": cfg.export.format},
                hint=f"Run `export start {project.name} {mv}` first.",
            )
    target_name = target_name or cfg.deployment.target
    target = get_target(target_name, project.settings)
    artifact_path = project.settings.artifact_root / export["artifact"]["path"]
    if not artifact_path.is_file():
        raise OperationError(
            f"export artifact missing: {artifact_path}",
            code="EXPORT_ARTIFACT_MISSING",
            details={"path": str(artifact_path)},
        )

    pointers = project.pointers()
    dep_id = next_sequential_id(DEPLOYMENT_ID_PREFIX, project.store.list_ids("deployments"))
    ref = target_ref(project.name, dep_id)
    now = utc_now_iso()
    record: dict[str, Any] = {
        "deployment_id": dep_id,
        # Device-side name (directory, `current` link): unique across projects sharing a target.
        "target_ref": ref,
        "state": _D.CREATED.value,
        "project": project.name,
        "model_version": mv,
        "export_id": export["export_id"],
        "export_artifact": export["artifact"],
        "target": target_name,
        "target_type": target.type,
        "runtime": cfg.deployment.runtime,
        "precision": cfg.deployment.precision,
        "acceptance_rules": {
            k: v.model_dump(exclude_none=True) for k, v in cfg.deployment.acceptance.items()
        },
        "previous_deployment": pointers["active_deployment"],
        "timestamps": {"created": now, "updated": now, "finished": None},
        "phases": [{"state": _D.CREATED.value, "at": now}],
        "target_info": None,
        "staged": None,
        "runtime_artifact": None,
        "runtime_config": None,
        "samples": None,
        "smoke": None,
        "benchmark": None,
        "acceptance": None,
        "activation": None,
        "health": None,
        "rollback": None,
        "error": None,
    }
    project.store.create("deployments", dep_id, record)
    link_model(project, mv, "deployments", dep_id)
    blog = log.bind(
        project=project.name, deployment_id=dep_id, model_version=mv, target=target_name
    )
    ph = _Phase(project, record)

    def pre_activation(state: _D, phase: str, fn: Callable[[], dict[str, Any]], field: str) -> bool:
        """Run a pre-activation phase; on error finish FAILED and return False."""
        ph.to(state)
        try:
            record[field] = fn()
        except Exception as exc:  # noqa: BLE001 - any target failure must be recorded
            blog.error("%s failed: %s", phase, exc)
            ph.finish(_D.FAILED, error=_error(exc, phase))
            return False
        project.store.write("deployments", dep_id, record)
        return True

    metadata = {
        "deployment_id": dep_id,
        "target_ref": ref,
        "project": project.name,
        "model_version": mv,
        "export_id": export["export_id"],
        "export_sha256": export["artifact"]["sha256"],
        "precision": cfg.deployment.precision,
        "runtime": cfg.deployment.runtime,
        "runtime_config": runtime_config(project, model_version=mv, export_id=export["export_id"]),
        "samples": [
            str(p)
            for p in sample_images(
                project, model.get("dataset_version"), cfg.deployment.smoke.samples
            )
        ],
    }
    record["runtime_config"] = metadata["runtime_config"]
    record["samples"] = metadata["samples"]
    configure = getattr(target, "configure", None)
    if configure:
        configure(
            smoke_min_detections=cfg.deployment.smoke.min_detections,
            bench_iterations=cfg.deployment.benchmark.iterations,
            bench_warmup=cfg.deployment.benchmark.warmup,
        )
    if not pre_activation(_D.VERIFYING, "verify", target.verify, "target_info"):
        return record
    if not pre_activation(
        _D.STAGING, "stage", lambda: target.stage(ref, artifact_path, metadata), "staged"
    ):
        return record
    if not pre_activation(
        _D.PREPARING,
        "prepare",
        lambda: target.prepare(
            ref, precision=cfg.deployment.precision, params={"runtime": cfg.deployment.runtime}
        ),
        "runtime_artifact",
    ):
        return record
    if not pre_activation(_D.SMOKE_TESTING, "smoke_test", lambda: target.smoke_test(ref), "smoke"):
        return record
    if not record["smoke"].get("passed"):
        ph.finish(
            _D.FAILED,
            error={
                "code": "SMOKE_TEST_FAILED",
                "message": "smoke test did not pass on the target",
                "details": record["smoke"].get("details", {}),
                "hint": "Inspect deployment.smoke; the active deployment was not changed.",
                "phase": "smoke_test",
            },
        )
        return record
    if not pre_activation(_D.BENCHMARKING, "benchmark", lambda: target.benchmark(ref), "benchmark"):
        return record
    # Benchmark *execution* above is separate from the acceptance *policy* below: results are
    # persisted even when no thresholds are configured.
    verdict = apply_rules(record["benchmark"]["metrics"], cfg.deployment.acceptance)
    record["acceptance"] = verdict
    if not verdict["passed"]:
        failed = [c["metric"] for c in verdict["checks"] if not c["passed"]]
        ph.finish(
            _D.REJECTED,
            error={
                "code": "POLICY_REJECTED",
                "message": f"deployment acceptance failed for {failed}",
                "details": {"checks": verdict["checks"]},
                "hint": "Not activated; the active deployment is unchanged. Adjust the model, "
                "precision, or deployment.acceptance.",
                "phase": "benchmark_policy",
            },
        )
        return record
    ph.to(_D.READY_TO_ACTIVATE)

    # ---- activation: from here on failures roll back -------------------------------------
    ph.to(_D.ACTIVATING)
    try:
        record["activation"] = target.activate(ref)
    except Exception as exc:  # noqa: BLE001
        blog.error("activation failed: %s", exc)
        _rollback(project, target, record, reason=_error(exc, "activate"))
        return record
    previous = record["previous_deployment"]
    if previous and project.store.exists("deployments", previous):
        prev_rec = project.store.read("deployments", previous)
        if prev_rec["state"] == _D.ACTIVE.value:
            _Phase(project, prev_rec).to(_D.SUPERSEDED, superseded_by=dep_id)
    project.store.write("deployments", dep_id, record)

    ph.to(_D.HEALTH_CHECKING)
    try:
        record["health"] = wait_healthy(
            target, cfg.deployment.health.timeout_s, cfg.deployment.health.interval_s
        )
    except Exception as exc:  # noqa: BLE001
        _rollback(project, target, record, reason=_error(exc, "health_check"))
        return record
    if not record["health"].get("healthy"):
        _rollback(
            project,
            target,
            record,
            reason={
                "code": "HEALTH_CHECK_FAILED",
                "message": "health check failed after activation",
                "details": record["health"].get("details", {}),
                "hint": None,
                "phase": "health_check",
            },
        )
        return record

    ph.finish(_D.ACTIVE)
    project.set_pointer("active_deployment", dep_id)
    project.set_pointer("last_known_good_deployment", dep_id)
    blog.info("deployment %s ACTIVE (previous %s)", dep_id, previous)
    return record


def _rollback(
    project: Project, target: DeploymentTarget, record: dict[str, Any], *, reason: dict[str, Any]
) -> None:
    """Automatic rollback after a post-activation failure."""
    dep_id = record["deployment_id"]
    to = project.pointers()["last_known_good_deployment"]
    if to == dep_id:  # cannot happen before ACTIVE, but never roll back onto ourselves
        to = record["previous_deployment"]
    ph = _Phase(project, record)
    ph.to(_D.ROLLING_BACK, error=reason)
    _perform_rollback(project, target, record, to=to, reason=reason["code"], trigger="automatic")


def _perform_rollback(
    project: Project,
    target: DeploymentTarget,
    record: dict[str, Any],
    *,
    to: str | None,
    reason: str,
    trigger: str,
) -> None:
    """Shared by automatic and manual rollback. ``record`` is already ROLLING_BACK."""
    dep_id = record["deployment_id"]
    ph = _Phase(project, record)
    blog = log.bind(project=project.name, deployment_id=dep_id, target=record["target"])
    outcome: dict[str, Any] = {"trigger": trigger, "reason": reason, "to": to, "at": utc_now_iso()}
    try:
        outcome["result"] = target.rollback(_ref_of(project, to) if to else None)
        if to is not None:
            hc = project.config.deployment.health
            health = wait_healthy(target, hc.timeout_s, hc.interval_s)
            outcome["health"] = health
            if not health.get("healthy"):
                raise OperationError(
                    f"restored deployment {to} is not healthy",
                    code="ROLLBACK_HEALTH_FAILED",
                    details=health.get("details", {}),
                )
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = _error(exc, "rollback")
        outcome["target_current"] = _safe_current(target)
        ph.finish(_D.ROLLBACK_FAILED, rollback=outcome)
        # We no longer know what is live; say so instead of pretending.
        project.set_pointer("active_deployment", None)
        blog.error("ROLLBACK FAILED for %s: %s — target state must be inspected", dep_id, exc)
        return
    ph.finish(_D.ROLLED_BACK, rollback=outcome)
    if to is not None and project.store.exists("deployments", to):
        prev = project.store.read("deployments", to)
        if prev["state"] == _D.SUPERSEDED.value:
            _Phase(project, prev).to(_D.ACTIVE, restored_by=dep_id)
    project.set_pointer("active_deployment", to)
    blog.warning("rolled back %s to %s (%s)", dep_id, to, reason)


def target_ref(project_name: str, deployment_id: str) -> str:
    """Device-side name of a deployment; a target may host deployments of several projects."""
    return f"{project_name}-{deployment_id}"


def _ref_of(project: Project, deployment_id: str) -> str:
    rec = project.store.read("deployments", deployment_id)
    return rec.get("target_ref") or target_ref(project.name, deployment_id)


def wait_healthy(target: DeploymentTarget, timeout_s: float, interval_s: float) -> dict[str, Any]:
    """Poll the target until healthy or the timeout elapses; returns the last result (with the
    number of attempts). A freshly restarted service needs time to load its engine."""
    deadline = time.monotonic() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        result = target.health_check()
        result.setdefault("details", {})["attempts"] = attempts
        if result.get("healthy") or time.monotonic() >= deadline:
            return result
        time.sleep(interval_s)


def runtime_config(project: Project, *, model_version: str, export_id: str) -> dict[str, Any]:
    """The configuration the edge runtime needs; everything model-specific lives here."""
    cfg = project.config
    inf = cfg.deployment.inference
    return {
        "schema_version": 1,
        "task": cfg.project.task_type,
        "project": project.name,
        "model_version": model_version,
        "export_id": export_id,
        "classes": list(cfg.dataset.classes),
        "input_size": cfg.export.image_size,
        "precision": cfg.deployment.precision,
        "preprocess": {"letterbox": True, "pad_value": 114, "scale": 1 / 255.0, "channels": "rgb"},
        "postprocess": {
            "type": "yolo",
            "confidence": inf.confidence,
            "iou": inf.iou,
            "max_detections": inf.max_detections,
        },
    }


def sample_images(project: Project, dataset_version: str | None, n: int) -> list[Path]:
    """First ``n`` test-split images of the dataset version (by asset id) for smoke tests."""
    if not dataset_version or n <= 0:
        return []
    from ai_pipeline.core.dataset.build import dataset_dir

    ddir = dataset_dir(project, dataset_version)
    manifest = ddir / "manifest.jsonl"
    if not manifest.is_file():
        return []
    out: list[Path] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") == "test" and (ddir / row["image"]).is_file():
            out.append(ddir / row["image"])
            if len(out) >= n:
                break
    if not out:  # tiny datasets may have an empty test split: fall back to val
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if line.strip() else None
            if row and row.get("split") == "val" and (ddir / row["image"]).is_file():
                out.append(ddir / row["image"])
                if len(out) >= n:
                    break
    return out


def setup_target(project: Project, target_name: str | None = None) -> dict[str, Any]:
    name = target_name or project.config.deployment.target
    target = get_target(name, project.settings)
    setup = getattr(target, "setup_runtime", None)
    if setup is None:
        raise OperationError(f"target {name} does not support setup", code="TARGET_NO_SETUP")
    return {"target": name, **setup()}


def infer_on_target(
    project: Project,
    images: list[Path],
    *,
    target_name: str | None = None,
    deployment_id: str | None = None,
) -> dict[str, Any]:
    pointers = project.pointers()
    dep_id = deployment_id or pointers["active_deployment"]
    name = target_name
    if name is None and dep_id and project.store.exists("deployments", dep_id):
        name = project.store.read("deployments", dep_id)["target"]
    name = name or project.config.deployment.target
    target = get_target(name, project.settings)
    missing = [str(p) for p in images if not p.is_file()]
    if missing:
        raise OperationError(
            f"image(s) not found: {missing}", code="IMAGE_NOT_FOUND", details={"missing": missing}
        )
    ref = _ref_of(project, deployment_id) if deployment_id else None
    result = target.infer(images, deployment_id=ref)
    return {"target": name, "deployment_id": dep_id, **result}


def _safe_current(target: DeploymentTarget) -> str | None:
    try:
        return target.current()
    except Exception:  # noqa: BLE001
        return None


def rollback_deployment(project: Project, *, to: str | None = None) -> dict[str, Any]:
    """Operator-initiated rollback of the active deployment to the previous known-good one.

    With no active deployment (e.g. after ROLLBACK_FAILED) an explicit ``to`` restores that
    deployment directly — the recovery path."""
    active_id = project.pointers()["active_deployment"]
    if not active_id:
        if to is None:
            raise StateError(
                "no active deployment to roll back",
                code="NO_ACTIVE_DEPLOYMENT",
                hint="Check `deploy list <project>`; after ROLLBACK_FAILED pass --to <deployment>.",
            )
        return _recover(project, to)
    active = project.store.read("deployments", active_id)
    to = to or active.get("previous_deployment")
    if not to:
        raise StateError(
            f"deployment {active_id} has no previous deployment to roll back to",
            code="NO_PREVIOUS_DEPLOYMENT",
            details={"active_deployment": active_id},
        )
    prev = project.store.read("deployments", to)
    if prev["state"] not in (_D.SUPERSEDED.value, _D.ACTIVE.value):
        raise StateError(
            f"cannot roll back to {to}: it is {prev['state']}, not a previously active deployment",
            code="ROLLBACK_TARGET_INVALID",
            details={"to": to, "state": prev["state"]},
        )
    target = get_target(active["target"], project.settings)
    ph = _Phase(project, active)
    ph.to(_D.ROLLING_BACK)
    _perform_rollback(project, target, active, to=to, reason="manual", trigger="manual")
    if active["state"] == _D.ROLLED_BACK.value:
        project.set_pointer("last_known_good_deployment", to)
    return active


def _recover(project: Project, to: str) -> dict[str, Any]:
    prev = project.store.read("deployments", to)
    if prev["state"] not in (_D.SUPERSEDED.value, _D.ACTIVE.value):
        raise StateError(
            f"cannot restore {to}: it is {prev['state']}, not a previously active deployment",
            code="ROLLBACK_TARGET_INVALID",
            details={"to": to, "state": prev["state"]},
        )
    target = get_target(prev["target"], project.settings)
    result = target.rollback(_ref_of(project, to))
    health_result = target.health_check()
    if not health_result.get("healthy"):
        raise OperationError(
            f"restored {to} but the target is not healthy",
            code="ROLLBACK_HEALTH_FAILED",
            details=health_result.get("details", {}),
            hint="Inspect the target manually; no deployment is marked active.",
        )
    if prev["state"] == _D.SUPERSEDED.value:
        _Phase(project, prev).to(_D.ACTIVE, restored_by="manual-recovery")
    prev.setdefault("recoveries", []).append(
        {"at": utc_now_iso(), "result": result, "health": health_result}
    )
    project.store.write("deployments", to, prev)
    project.set_pointer("active_deployment", to)
    project.set_pointer("last_known_good_deployment", to)
    return prev


def health(project: Project, *, target_name: str | None = None) -> dict[str, Any]:
    pointers = project.pointers()
    active_id = pointers["active_deployment"]
    name = target_name
    if name is None and active_id:
        name = project.store.read("deployments", active_id)["target"]
    name = name or project.config.deployment.target
    target = get_target(name, project.settings)
    result = target.health_check()
    active_ref = _ref_of(project, active_id) if active_id else None
    return {
        "target": name,
        "healthy": bool(result.get("healthy")),
        "target_current": result.get("current"),
        "active_deployment": active_id,
        "active_target_ref": active_ref,
        "last_known_good_deployment": pointers["last_known_good_deployment"],
        "consistent": result.get("current") == active_ref,
        "details": result.get("details", {}),
    }


def deployment_summary(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "deployment_id": rec["deployment_id"],
        "target_ref": rec.get("target_ref"),
        "state": rec["state"],
        "model_version": rec["model_version"],
        "export_id": rec["export_id"],
        "target": rec["target"],
        "precision": rec["precision"],
        "previous_deployment": rec.get("previous_deployment"),
        "benchmark": (rec.get("benchmark") or {}).get("metrics"),
        "acceptance": rec.get("acceptance"),
        "health": (rec.get("health") or {}).get("healthy"),
        "rollback": rec.get("rollback"),
        "error": rec.get("error"),
        "timestamps": rec["timestamps"],
    }


def list_deployments(project: Project) -> list[dict[str, Any]]:
    return [deployment_summary(r) for r in project.store.list("deployments")]


def raise_for_state(record: dict[str, Any]) -> None:
    """Translate a finished deployment into the right exception/exit code for the CLI."""
    state = _D(record["state"])
    err = record.get("error") or {}
    if state is _D.ACTIVE:
        return
    if state is _D.REJECTED:
        raise PolicyRejected(
            err.get("message", "deployment acceptance failed"),
            code="POLICY_REJECTED",
            details={"deployment_id": record["deployment_id"], **(err.get("details") or {})},
            hint=err.get("hint"),
        )
    code = {
        _D.FAILED: "DEPLOYMENT_FAILED",
        _D.ROLLED_BACK: "DEPLOYMENT_ROLLED_BACK",
        _D.ROLLBACK_FAILED: "DEPLOYMENT_ROLLBACK_FAILED",
    }.get(state, "DEPLOYMENT_NOT_ACTIVE")
    message = f"deployment {record['deployment_id']} ended {state.value}"
    if err.get("message"):
        message += f": {err['message']}"
    raise OperationError(
        message,
        code=code,
        details={
            "deployment_id": record["deployment_id"],
            "state": state.value,
            "phase": err.get("phase"),
            "cause": err.get("code"),
            "rollback": record.get("rollback"),
        },
        hint=err.get("hint")
        or f"Inspect `deploy status {record['project']} {record['deployment_id']}`.",
    )
