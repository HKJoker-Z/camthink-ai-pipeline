from __future__ import annotations

from pathlib import Path

import pytest

from ai_pipeline.core.settings import Settings

INFRA_YAML = """\
schema_version: 1
projects_root: ./projects
artifact_root: ./artifacts
compute_backends:
  local:
    type: local
  remote:
    type: ssh
    ssh_host: ai-server
    workdir: /data/aip
deployment_targets:
  fake:
    type: fake
    root: ./.fake-target
  ng4511-lab:
    type: camthink_ng4511
    ssh_host: ng4511
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated working directory containing an infrastructure.yaml."""
    (tmp_path / "infrastructure.yaml").write_text(INFRA_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for var in ("AIP_INFRA_FILE", "AIP_PROJECTS_ROOT", "AIP_ARTIFACT_ROOT"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings.resolve(cwd=workspace)


def make_image(path: Path, seed: int, size: tuple[int, int] = (64, 48)) -> Path:
    """Write a tiny deterministic image whose bytes differ per ``seed``."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    colour = ((seed * 37) % 256, (seed * 59) % 256, (seed * 83) % 256)
    im = Image.new("RGB", size, colour)
    im.putpixel((0, 0), (seed % 256, 0, 0))
    im.save(path, format="PNG" if path.suffix == ".png" else "JPEG", quality=95)
    return path


def make_image_set(root: Path, n: int, *, prefix: str = "img", ext: str = ".jpg") -> list[Path]:
    return [make_image(root / f"{prefix}_{i:03d}{ext}", seed=i + 1) for i in range(n)]


def write_yolo_labels(root: Path, stems: list[str], *, boxes_for: callable = None) -> Path:
    """Default: one centred box of class 0 per stem, two for even indices."""
    root.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        if boxes_for is not None:
            lines = boxes_for(i, stem)
        else:
            lines = ["0 0.5 0.5 0.4 0.4"] + (["0 0.2 0.3 0.1 0.2"] if i % 2 == 0 else [])
        (root / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return root


@pytest.fixture
def demo_project(settings: Settings):
    """A created project with relaxed validation thresholds suitable for tiny fixtures."""
    from ai_pipeline.core.project import Project

    p = Project.create("demo", settings)
    text = p.config_path.read_text()
    text = (
        text.replace("min_images: 20", "min_images: 4")
        .replace("min_instances_per_class: 10", "min_instances_per_class: 1")
        .replace("timeout_s: 120", "timeout_s: 0.5")  # health polling: keep fake tests fast
        .replace("interval_s: 5", "interval_s: 0.1")
    )
    p.config_path.write_text(text)
    return Project.load("demo", settings)


def export_model(p, mv: str | None = None, **kw):
    """Run (or reuse) an export for a model through the job framework and return the record."""
    from ai_pipeline.core.exports import start_export
    from ai_pipeline.core.runs import wait_run

    run, existing = start_export(p, model_version=mv, **kw)
    if existing is not None:
        return existing
    run = wait_run(p, run["run_id"], timeout_s=30)
    assert run["state"] == "COMPLETED", run.get("error")
    return p.store.read("exports", run["export_id"])


@pytest.fixture
def accepted_project(demo_project, tmp_path: Path):
    """Project with a fake provider, a READY dataset, an ACCEPTED + selected model."""
    from ai_pipeline.core.data.importer import import_images
    from ai_pipeline.core.dataset.build import build_dataset, plan_dataset
    from ai_pipeline.core.labeling.labelsets import import_labels
    from ai_pipeline.core.project import Project
    from ai_pipeline.core.registry import select_model
    from ai_pipeline.core.runs import start_evaluation, start_training, wait_run

    cfg = demo_project.config_path
    cfg.write_text(cfg.read_text().replace("provider: ultralytics", "provider: fake"))
    p = Project.load("demo", demo_project.settings)
    src = tmp_path / "src"
    imgs = make_image_set(src, 12)
    write_yolo_labels(tmp_path / "labels", [i.stem for i in imgs])
    import_images(p, [src])
    import_labels(p, tmp_path / "labels", fmt="yolo")
    build_dataset(p, plan_dataset(p))
    run = wait_run(p, start_training(p)["run_id"], timeout_s=30)
    wait_run(p, start_evaluation(p, run["model_version"])["run_id"], timeout_s=30)
    select_model(p, None)
    return p


@pytest.fixture
def exported_project(accepted_project):
    export_model(accepted_project)
    return accepted_project
