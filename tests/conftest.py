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
    text = text.replace("min_images: 20", "min_images: 4").replace(
        "min_instances_per_class: 10", "min_instances_per_class: 1"
    )
    p.config_path.write_text(text)
    return Project.load("demo", settings)
