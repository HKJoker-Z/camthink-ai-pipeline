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


def test_model_lifecycle_is_quality_only():
    assert transition(ModelState.TRAINED, ModelState.EVALUATED, MODEL_TRANSITIONS)
    assert transition(ModelState.EVALUATED, ModelState.REJECTED, MODEL_TRANSITIONS)
    # re-evaluation is always possible; export/deploy never lock a model
    assert transition(ModelState.ACCEPTED, ModelState.EVALUATED, MODEL_TRANSITIONS)
    assert transition(ModelState.REJECTED, ModelState.EVALUATED, MODEL_TRANSITIONS)
    with pytest.raises(StateError):
        transition(ModelState.TRAINED, ModelState.ACCEPTED, MODEL_TRANSITIONS)
    with pytest.raises(StateError):
        transition(ModelState.REJECTED, ModelState.ACCEPTED, MODEL_TRANSITIONS)
    assert {s.value for s in ModelState} == {
        "TRAINED",
        "EVALUATED",
        "ACCEPTED",
        "REJECTED",
        "ARCHIVED",
    }


def test_deployment_phase_machine():
    D = DeploymentState
    happy = [
        D.CREATED,
        D.VERIFYING,
        D.STAGING,
        D.PREPARING,
        D.SMOKE_TESTING,
        D.BENCHMARKING,
        D.READY_TO_ACTIVATE,
        D.ACTIVATING,
        D.HEALTH_CHECKING,
        D.ACTIVE,
    ]
    for a, b in zip(happy, happy[1:], strict=False):
        assert transition(a, b, DEPLOYMENT_TRANSITIONS) is b
    # pre-activation phases fail into FAILED, never ROLLING_BACK
    for s in happy[:7]:
        assert transition(s, D.FAILED, DEPLOYMENT_TRANSITIONS)
        with pytest.raises(StateError):
            transition(s, D.ROLLING_BACK, DEPLOYMENT_TRANSITIONS)
    assert transition(D.BENCHMARKING, D.REJECTED, DEPLOYMENT_TRANSITIONS)
    # post-activation failures roll back, never plain FAILED
    for s in (D.ACTIVATING, D.HEALTH_CHECKING):
        assert transition(s, D.ROLLING_BACK, DEPLOYMENT_TRANSITIONS)
        with pytest.raises(StateError):
            transition(s, D.FAILED, DEPLOYMENT_TRANSITIONS)
    assert transition(D.ROLLING_BACK, D.ROLLED_BACK, DEPLOYMENT_TRANSITIONS)
    assert transition(D.ROLLING_BACK, D.ROLLBACK_FAILED, DEPLOYMENT_TRANSITIONS)
    assert transition(D.ACTIVE, D.SUPERSEDED, DEPLOYMENT_TRANSITIONS)
    assert transition(D.SUPERSEDED, D.ACTIVE, DEPLOYMENT_TRANSITIONS)
    with pytest.raises(StateError):
        transition(D.PREPARING, D.ACTIVE, DEPLOYMENT_TRANSITIONS)
    for terminal in (D.FAILED, D.REJECTED, D.ROLLED_BACK, D.ROLLBACK_FAILED):
        assert DEPLOYMENT_TRANSITIONS[terminal] == frozenset()


def test_every_non_terminal_run_state_can_fail():
    for state, allowed in RUN_TRANSITIONS.items():
        if allowed:
            assert RunState.FAILED in allowed, f"{state} must be able to reach FAILED"


def test_unknown_state_is_reported():
    with pytest.raises(StateError) as ei:
        transition(DeploymentState.ACTIVE, RunState.FAILED, RUN_TRANSITIONS, entity="x")
    assert ei.value.code == "UNKNOWN_STATE"
