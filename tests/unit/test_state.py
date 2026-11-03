import pytest

from ai_pipeline.core.errors import ExitCode, StateError
from ai_pipeline.core.state import (
    DEPLOYMENT_TRANSITIONS,
    MODEL_TRANSITIONS,
    RUN_TRANSITIONS,
    DeploymentState,
    ModelState,
    RunState,
    transition,
)


def test_legal_run_lifecycle():
    s = RunState.CREATED
    for nxt in (RunState.SUBMITTED, RunState.RUNNING, RunState.COMPLETED):
        s = transition(s, nxt, RUN_TRANSITIONS, entity="run")
    assert s is RunState.COMPLETED


def test_terminal_states_cannot_move():
    with pytest.raises(StateError) as ei:
        transition(RunState.COMPLETED, RunState.RUNNING, RUN_TRANSITIONS, entity="run")
    err = ei.value
    assert err.code == "ILLEGAL_TRANSITION"
    assert err.exit_code == ExitCode.STATE
    assert err.details["from"] == "COMPLETED"
    assert err.details["allowed"] == []


def test_model_lifecycle_paths():
    assert transition(ModelState.TRAINED, ModelState.EVALUATED, MODEL_TRANSITIONS)
    assert transition(ModelState.EVALUATED, ModelState.REJECTED, MODEL_TRANSITIONS)
    assert transition(ModelState.SUPERSEDED, ModelState.DEPLOYED, MODEL_TRANSITIONS)  # rollback
    with pytest.raises(StateError):
        transition(ModelState.TRAINED, ModelState.DEPLOYED, MODEL_TRANSITIONS)
    with pytest.raises(StateError):
        transition(ModelState.REJECTED, ModelState.ACCEPTED, MODEL_TRANSITIONS)


def test_deployment_can_skip_benchmark_but_not_smoke():
    assert transition(DeploymentState.SMOKE_PASSED, DeploymentState.ACTIVE, DEPLOYMENT_TRANSITIONS)
    with pytest.raises(StateError):
        transition(DeploymentState.PREPARED, DeploymentState.ACTIVE, DEPLOYMENT_TRANSITIONS)


def test_every_non_terminal_state_can_fail_or_finish():
    for table, failed in (
        (RUN_TRANSITIONS, RunState.FAILED),
        (DEPLOYMENT_TRANSITIONS, DeploymentState.FAILED),
    ):
        for state, allowed in table.items():
            if allowed:
                assert failed in allowed, f"{state} must be able to reach {failed}"


def test_unknown_state_is_reported():
    with pytest.raises(StateError) as ei:
        transition(DeploymentState.ACTIVE, RunState.FAILED, RUN_TRANSITIONS, entity="x")
    assert ei.value.code == "UNKNOWN_STATE"
