"""Real ultralytics provider through the real job runner, on CPU.

Opt-in: needs the `provider-ultralytics` extra (torch + ultralytics) in the interpreter that
runs the jobs and network access for the pretrained weights. Skipped otherwise.

    AIP_ML_PYTHON=/path/to/venv-ml/bin/python pytest -m slow tests/integration
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from ai_pipeline.core.data.importer import import_images
from ai_pipeline.core.dataset.build import build_dataset, plan_dataset
from ai_pipeline.core.labeling.labelsets import import_labels, load_label_set
from ai_pipeline.core.project import Project
from ai_pipeline.core.runs import start_autolabel, start_evaluation, start_training, wait_run
from tests.conftest import make_image_set, write_yolo_labels

pytestmark = pytest.mark.slow


def _ml_python() -> str | None:
    explicit = os.environ.get("AIP_ML_PYTHON")
    if explicit and Path(explicit).exists():
        return explicit
    try:
        import ultralytics  # noqa: F401

        return sys.executable
    except ImportError:
        return None


@pytest.mark.skipif(_ml_python() is None, reason="ultralytics not installed (set AIP_ML_PYTHON)")
def test_ultralytics_train_evaluate_autolabel_on_cpu(workspace: Path, tmp_path: Path):
    infra = workspace / "infrastructure.yaml"
    infra.write_text(
        infra.read_text().replace(
            "  local:\n    type: local\n",
            f"  local:\n    type: local\n    python: {_ml_python()}\n",
        )
    )
    from ai_pipeline.core.settings import Settings

    settings = Settings.resolve(cwd=workspace)
    p = Project.create("demo", settings)
    cfg = p.config_path
    cfg.write_text(
        cfg.read_text()
        .replace("min_images: 20", "min_images: 4")
        .replace("min_instances_per_class: 10", "min_instances_per_class: 1")
        .replace("epochs: 5", "epochs: 1")
        .replace("image_size: 640", "image_size: 64")
        .replace("batch_size: 16", "batch_size: 8")
        .replace('device: "0"', 'device: "cpu"')
    )
    p = Project.load("demo", settings)

    src = tmp_path / "src"
    imgs = make_image_set(src, 12)
    write_yolo_labels(tmp_path / "labels", [i.stem for i in imgs])
    import_images(p, [src])
    import_labels(p, tmp_path / "labels", fmt="yolo")
    build_dataset(p, plan_dataset(p))

    run = wait_run(p, start_training(p)["run_id"], timeout_s=900)
    assert run["state"] == "COMPLETED", run.get("error")
    assert run["environment"]["ultralytics"] and run["environment"]["torch"]
    assert set(run["metrics"]) >= {"precision", "recall", "mAP50", "mAP50-95", "f1", "per_class"}
    weights = p.settings.artifact_root / run["outputs"]["weights"]
    assert weights.is_file() and weights.stat().st_size > 1_000_000
    mv = run["model_version"]
    assert p.store.read("models", mv)["state"] == "TRAINED"

    ev = wait_run(p, start_evaluation(p, mv)["run_id"], timeout_s=600)
    assert ev["state"] == "COMPLETED", ev.get("error")
    assert ev["metrics"]["split"] == "test"
    assert p.store.read("models", mv)["state"] in ("ACCEPTED", "REJECTED")

    al = wait_run(p, start_autolabel(p, confidence=0.001)["run_id"], timeout_s=600)
    assert al["state"] == "COMPLETED", al.get("error")
    rec, anns = load_label_set(p, al["label_set_id"])
    assert rec["source"]["kind"] == "model" and rec["source"]["provider"] == "ultralytics"
    assert all(a.source.kind == "model" for a in anns)

    # real ONNX export through the job runner (ultralytics installs onnx as an export dep)
    from ai_pipeline.core.exports import start_export

    ex_run, existing = start_export(p, model_version=mv)
    assert existing is None
    ex_run = wait_run(p, ex_run["run_id"], timeout_s=600)
    assert ex_run["state"] == "COMPLETED", ex_run.get("error")
    export = p.store.read("exports", ex_run["export_id"])
    onnx_path = p.settings.artifact_root / export["artifact"]["path"]
    assert onnx_path.is_file() and onnx_path.read_bytes()[:4] != b"FAKE"
    assert (
        export["artifact"]["size_bytes"] > 1_000_000
        and export["details"]["exporter"] == "ultralytics"
    )
    assert export["environment"]["ultralytics"]
    train_dir = p.settings.artifact_root / run["outputs"]["weights"]
    assert not train_dir.with_suffix(".onnx").exists(), "export must not write into the train run"

    shutil.rmtree(p.artifact_dir, ignore_errors=True)
