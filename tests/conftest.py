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
