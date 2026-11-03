from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ai_pipeline.backends import get_backend
from ai_pipeline.backends.local import LocalBackend, _pid_alive
from ai_pipeline.core.config import InfrastructureConfig
from ai_pipeline.core.errors import ConfigError, OperationError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.jobs import JobSpec, JobState, read_result, read_status, write_spec
from ai_pipeline.core.jobs.runner import run_job
from ai_pipeline.providers import load_provider, provider_names


def _spec(job_dir: Path, kind="train", provider="fake", **params) -> JobSpec:
    job_dir.mkdir(parents=True, exist_ok=True)
    data = job_dir / "inputs" / "data.yaml"
    data.parent.mkdir(exist_ok=True)
    data.write_text("path: .\nnames: {0: person}\n")
    (job_dir / "inputs" / "images.txt").write_text("/x/aaa0.jpg\n/x/bbb1.jpg\n")
    spec = JobSpec(
        run_id="run-test",
        kind=kind,
        project="demo",
        task_type="object_detection",
        provider=provider,
        inputs={
            "data_config": "inputs/data.yaml",
            "weights": "inputs/weights.pt",
            "split": "test",
            "images_list": "inputs/images.txt",
        },
        params={
            "candidate": {"name": "n", "weights": "n.pt"},
            "epochs": 2,
            "classes": ["person"],
            **params,
        },
        created_at=utc_now_iso(),
    )
    write_spec(job_dir, spec)
    return spec


# --- runner (in-process) -------------------------------------------------------------------------


def test_runner_train_writes_status_result_and_log(tmp_path):
    jd = tmp_path / "job"
    _spec(jd)
    assert run_job(jd) == 0
    st = read_status(jd)
    assert st.state is JobState.COMPLETED and st.started_at and st.finished_at
    assert st.handle["pid"] and st.result_file == "out/result.json"
    assert st.progress["epoch"] == 2 and st.progress["total_epochs"] == 2
    res = read_result(jd)
    assert res.kind == "train" and res.provider == "fake"
    assert (jd / res.artifacts["weights"]).is_file() and (jd / res.artifacts["metrics"]).is_file()
    assert res.metrics["mAP50"] == 0.6  # 0.5 + 0.05 * 2
    assert res.environment["provider"] == "fake" and res.environment["python"]
    log = (jd / "runner.log").read_text()
    assert "starting" in log and "completed" in log


def test_runner_failure_is_recorded_and_exit_code_1(tmp_path):
    jd = tmp_path / "job"
    _spec(jd, provider_args={"fake_fail": "train"})
    assert run_job(jd) == 1
    st = read_status(jd)
    assert st.state is JobState.FAILED
    assert st.error["type"] == "RuntimeError" and "asked to fail" in st.error["message"]
    assert "Traceback" in st.error["traceback"]
    assert read_result(jd) is None


def test_runner_evaluate_and_autolabel(tmp_path):
    jd = tmp_path / "job"
    _spec(jd)
    run_job(jd)
    weights = jd / "out/weights/best.pt"

    jd2 = tmp_path / "job2"
    _spec(jd2, kind="evaluate")
    (jd2 / "inputs" / "weights.pt").write_bytes(weights.read_bytes())
    assert run_job(jd2) == 0
    assert read_result(jd2).metrics["split"] == "test" and read_result(jd2).metrics["mAP50"] == 0.6

    jd3 = tmp_path / "job3"
    _spec(jd3, kind="autolabel", class_map={"person": "person"}, confidence=0.5)
    assert run_job(jd3) == 0
    res = read_result(jd3)
    rows = [
        json.loads(line) for line in (jd3 / res.artifacts["detections"]).read_text().splitlines()
    ]
    # aaa0 -> one box (conf .9); bbb1 -> two boxes but the second (.6) passes conf .5 too
    assert res.metrics == {"images": 2, "detections": 3}
    assert all(r["class_name"] == "person" and len(r["bbox"]) == 4 for r in rows)


def test_unknown_provider_fails_in_runner(tmp_path):
    jd = tmp_path / "job"
    _spec(jd)
    spec = json.loads((jd / "job.json").read_text())
    spec["provider"] = "fake"  # valid for the pydantic spec; patch the registry instead
    (jd / "job.json").write_text(json.dumps(spec))
    import ai_pipeline.providers as prov

    prov.PROVIDER_MODULES["broken"] = "ai_pipeline.providers.does_not_exist:X"
    try:
        spec["provider"] = "broken"
        (jd / "job.json").write_text(json.dumps(spec))
        assert run_job(jd) == 1
        assert read_status(jd).error["type"] == "OperationError"
    finally:
        del prov.PROVIDER_MODULES["broken"]


# --- provider loading boundary -------------------------------------------------------------------


def test_provider_registry_and_lazy_loading():
    assert provider_names() == ["fake", "ultralytics"]
    assert load_provider("fake").name == "fake"
    with pytest.raises(ConfigError) as ei:
        load_provider("nope")
    assert ei.value.code == "PROVIDER_UNKNOWN"


def test_control_plane_does_not_import_ml_libraries():
    """The CLI and core must be importable without torch/ultralytics ever being loaded."""
    import subprocess

    code = (
        "import sys, ai_pipeline.cli.main, ai_pipeline.core.runs, ai_pipeline.core.registry, "
        "ai_pipeline.providers, ai_pipeline.backends; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'ultralytics', 'cv2')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


# --- local backend (detached subprocess) --------------------------------------------------------


def test_local_backend_submit_status_logs(tmp_path):
    jd = tmp_path / "job"
    _spec(jd, provider_args={"fake_sleep_s": 0.3})
    backend = LocalBackend()
    st = backend.submit(jd)
    assert st.state is JobState.PENDING and st.handle["pid"] and st.submitted_at
    import time

    deadline = time.monotonic() + 20
    while backend.status(jd).state not in (JobState.COMPLETED, JobState.FAILED):
        assert time.monotonic() < deadline, backend.logs(jd)
        time.sleep(0.05)
    st = backend.status(jd)
    assert st.state is JobState.COMPLETED, st
    assert "completed" in backend.logs(jd, tail=2)
    deadline = time.monotonic() + 5  # the interpreter may still be shutting down
    while _pid_alive(st.handle["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_alive(st.handle["pid"])


def test_local_backend_cancel(tmp_path):
    jd = tmp_path / "job"
    _spec(jd, provider_args={"fake_sleep_s": 30})
    backend = LocalBackend()
    st = backend.submit(jd)
    import time

    deadline = time.monotonic() + 20
    while backend.status(jd).state is JobState.PENDING:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    st = backend.cancel(jd)
    assert st.state is JobState.CANCELLED
    assert not _pid_alive(st.handle["pid"])
    assert backend.status(jd).state is JobState.CANCELLED  # terminal, stays


def test_local_backend_detects_dead_runner(tmp_path):
    jd = tmp_path / "job"
    _spec(jd)
    backend = LocalBackend()
    from ai_pipeline.core.jobs import JobStatus, write_status

    write_status(
        jd, JobStatus(run_id="run-test", state=JobState.RUNNING, handle={"pid": 2**22 + 12345})
    )
    st = backend.status(jd)
    assert st.state is JobState.FAILED and st.error["type"] == "RunnerDied"


def test_local_backend_bad_python(tmp_path):
    jd = tmp_path / "job"
    _spec(jd)
    backend = LocalBackend(python="/nonexistent/python")
    with pytest.raises(OperationError) as ei:
        backend.submit(jd)
    assert ei.value.code == "RUNNER_START_FAILED"
    assert read_status(jd).state is JobState.FAILED


def test_get_backend_factory():
    infra = InfrastructureConfig.model_validate(
        {
            "schema_version": 1,
            "compute_backends": {
                "local": {"type": "local"},
                "remote": {"type": "ssh", "ssh_host": "h", "workdir": "/w"},
            },
            "deployment_targets": {"fake": {"type": "fake"}},
        }
    )
    b = get_backend("local", infra)
    assert isinstance(b, LocalBackend) and b.python == sys.executable
    with pytest.raises(OperationError) as ei:
        get_backend("remote", infra)
    assert ei.value.code == "BACKEND_NOT_IMPLEMENTED"
    with pytest.raises(ConfigError):
        get_backend("nope", infra)
