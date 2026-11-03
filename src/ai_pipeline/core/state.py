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
    TRAINED = "TRAINED"
    EVALUATED = "EVALUATED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPORTED = "EXPORTED"
    BENCHMARKED = "BENCHMARKED"
    DEPLOYED = "DEPLOYED"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"


class DeploymentState(str, Enum):
    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    PREPARED = "PREPARED"
    SMOKE_PASSED = "SMOKE_PASSED"
    BENCHMARKED = "BENCHMARKED"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


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
    # Re-evaluation is allowed until the model has been exported.
    ModelState.REJECTED: frozenset({ModelState.EVALUATED, ModelState.ARCHIVED}),
    ModelState.ACCEPTED: frozenset(
        {ModelState.EVALUATED, ModelState.EXPORTED, ModelState.ARCHIVED}
    ),
    ModelState.EXPORTED: frozenset(
        {ModelState.BENCHMARKED, ModelState.DEPLOYED, ModelState.ARCHIVED}
    ),
    ModelState.BENCHMARKED: frozenset(
        {ModelState.DEPLOYED, ModelState.REJECTED, ModelState.ARCHIVED}
    ),
    ModelState.DEPLOYED: frozenset({ModelState.SUPERSEDED, ModelState.ARCHIVED}),
    # A superseded model is redeployed on rollback.
    ModelState.SUPERSEDED: frozenset({ModelState.DEPLOYED, ModelState.ARCHIVED}),
    ModelState.ARCHIVED: frozenset(),
}

DEPLOYMENT_TRANSITIONS: _T = {
    DeploymentState.PENDING: frozenset({DeploymentState.UPLOADED, DeploymentState.FAILED}),
    DeploymentState.UPLOADED: frozenset({DeploymentState.PREPARED, DeploymentState.FAILED}),
    DeploymentState.PREPARED: frozenset({DeploymentState.SMOKE_PASSED, DeploymentState.FAILED}),
    DeploymentState.SMOKE_PASSED: frozenset(
        {DeploymentState.BENCHMARKED, DeploymentState.ACTIVE, DeploymentState.FAILED}
    ),
    DeploymentState.BENCHMARKED: frozenset({DeploymentState.ACTIVE, DeploymentState.FAILED}),
    DeploymentState.ACTIVE: frozenset({DeploymentState.ROLLED_BACK, DeploymentState.FAILED}),
    DeploymentState.FAILED: frozenset(),
    DeploymentState.ROLLED_BACK: frozenset(),
}

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
