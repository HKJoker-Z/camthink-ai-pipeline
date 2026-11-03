"""Entity state machines.

There is deliberately no single project-wide state: a project holds many dataset versions,
runs, model versions and deployments at once. Each entity kind has its own small state
machine; ``ai-pipeline status`` derives a project summary from them.

Transitions are enforced through :func:`transition`; an illegal transition raises
:class:`~ai_pipeline.core.errors.StateError` (exit code 3).
"""

from __future__ import annotations

from enum import Enum

from ai_pipeline.core.errors import StateError


class LabelSetState(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"


class DatasetState(str, Enum):
    BUILDING = "BUILDING"
    READY = "READY"
    FAILED = "FAILED"


class RunState(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelState(str, Enum):
    """Model *quality* lifecycle only. Exports and deployments are separate entities
    (ExportArtifact, Deployment) and never change a model's state."""

    TRAINED = "TRAINED"
    EVALUATED = "EVALUATED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"


class ExportState(str, Enum):
    READY = "READY"
    FAILED = "FAILED"


class DeploymentState(str, Enum):
    """Phase machine of one deployment attempt on one target.

    Pre-activation phases fail into FAILED (or REJECTED for benchmark policy) without touching
    the active deployment. Post-activation failures roll back automatically.
    """

    CREATED = "CREATED"
    VERIFYING = "VERIFYING"
    STAGING = "STAGING"
    PREPARING = "PREPARING"
    SMOKE_TESTING = "SMOKE_TESTING"
    BENCHMARKING = "BENCHMARKING"
    READY_TO_ACTIVATE = "READY_TO_ACTIVATE"
    ACTIVATING = "ACTIVATING"
    HEALTH_CHECKING = "HEALTH_CHECKING"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"  # was ACTIVE; a later deployment is now current
    FAILED = "FAILED"  # failed before activation; active deployment untouched
    REJECTED = "REJECTED"  # benchmark did not satisfy deployment.acceptance; not activated
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"  # previous known-good restored (or nothing active if none)
    ROLLBACK_FAILED = "ROLLBACK_FAILED"  # high severity: target state must be inspected


_T = dict[Enum, frozenset[Enum]]

LABEL_SET_TRANSITIONS: _T = {
    LabelSetState.PENDING: frozenset({LabelSetState.READY, LabelSetState.FAILED}),
    LabelSetState.READY: frozenset(),
    LabelSetState.FAILED: frozenset(),
}

DATASET_TRANSITIONS: _T = {
    DatasetState.BUILDING: frozenset({DatasetState.READY, DatasetState.FAILED}),
    DatasetState.READY: frozenset(),
    DatasetState.FAILED: frozenset(),
}

RUN_TRANSITIONS: _T = {
    RunState.CREATED: frozenset({RunState.SUBMITTED, RunState.FAILED, RunState.CANCELLED}),
    RunState.SUBMITTED: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.RUNNING: frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}

MODEL_TRANSITIONS: _T = {
    ModelState.TRAINED: frozenset({ModelState.EVALUATED, ModelState.ARCHIVED}),
    ModelState.EVALUATED: frozenset(
        {ModelState.ACCEPTED, ModelState.REJECTED, ModelState.ARCHIVED}
    ),
    # Re-evaluation is always allowed; exporting/deploying never blocks it.
    ModelState.REJECTED: frozenset({ModelState.EVALUATED, ModelState.ARCHIVED}),
    ModelState.ACCEPTED: frozenset({ModelState.EVALUATED, ModelState.ARCHIVED}),
    ModelState.ARCHIVED: frozenset(),
}

_D = DeploymentState
_PRE_ACTIVATION_FAIL = frozenset({_D.FAILED})
DEPLOYMENT_TRANSITIONS: _T = {
    _D.CREATED: frozenset({_D.VERIFYING}) | _PRE_ACTIVATION_FAIL,
    _D.VERIFYING: frozenset({_D.STAGING}) | _PRE_ACTIVATION_FAIL,
    _D.STAGING: frozenset({_D.PREPARING}) | _PRE_ACTIVATION_FAIL,
    _D.PREPARING: frozenset({_D.SMOKE_TESTING}) | _PRE_ACTIVATION_FAIL,
    _D.SMOKE_TESTING: frozenset({_D.BENCHMARKING}) | _PRE_ACTIVATION_FAIL,
    _D.BENCHMARKING: frozenset({_D.READY_TO_ACTIVATE, _D.REJECTED}) | _PRE_ACTIVATION_FAIL,
    _D.READY_TO_ACTIVATE: frozenset({_D.ACTIVATING}) | _PRE_ACTIVATION_FAIL,
    # From here on a failure means the target may have been changed: roll back.
    _D.ACTIVATING: frozenset({_D.HEALTH_CHECKING, _D.ROLLING_BACK}),
    _D.HEALTH_CHECKING: frozenset({_D.ACTIVE, _D.ROLLING_BACK}),
    _D.ACTIVE: frozenset({_D.SUPERSEDED, _D.ROLLING_BACK}),
    _D.SUPERSEDED: frozenset({_D.ACTIVE}),  # restored by a rollback
    _D.ROLLING_BACK: frozenset({_D.ROLLED_BACK, _D.ROLLBACK_FAILED}),
    _D.FAILED: frozenset(),
    _D.REJECTED: frozenset(),
    _D.ROLLED_BACK: frozenset(),
    _D.ROLLBACK_FAILED: frozenset(),
}

PRE_ACTIVATION_STATES = frozenset(
    {
        _D.CREATED,
        _D.VERIFYING,
        _D.STAGING,
        _D.PREPARING,
        _D.SMOKE_TESTING,
        _D.BENCHMARKING,
        _D.READY_TO_ACTIVATE,
    }
)
TERMINAL_DEPLOYMENT_STATES = frozenset(
    {_D.ACTIVE, _D.SUPERSEDED, _D.FAILED, _D.REJECTED, _D.ROLLED_BACK, _D.ROLLBACK_FAILED}
)

TERMINAL_RUN_STATES = frozenset({RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED})


def transition(current: Enum, new: Enum, table: _T, *, entity: str = "") -> Enum:
    """Validate ``current -> new`` against ``table`` and return ``new``."""
    allowed = table.get(current)
    if allowed is None:
        raise StateError(
            f"Unknown state {current!r} for {entity or 'entity'}",
            code="UNKNOWN_STATE",
            details={"entity": entity, "state": getattr(current, "value", str(current))},
        )
    if new not in allowed:
        raise StateError(
            f"Illegal state transition {current.value} -> {new.value}"
            + (f" for {entity}" if entity else ""),
            code="ILLEGAL_TRANSITION",
            details={
                "entity": entity,
                "from": current.value,
                "to": new.value,
                "allowed": sorted(s.value for s in allowed),
            },
            hint="Inspect the entity record; terminal states cannot be changed. "
            "Start a new run/deployment instead of mutating a finished one.",
        )
    return new
