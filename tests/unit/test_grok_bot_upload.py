"""The restricted-SSH ``upload`` helper: archive on stdin -> projects/<p>/data/uploads/<name>/."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from tests.conftest import make_image_set

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "grok_bot" / "upload.py"


def _run(repo: Path, args: list[str], data: bytes | None) -> tuple[int, dict]:
    env = {**os.environ, "AIP_REPO": str(repo), "AIP_UPLOAD_MAX_MB": "1"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=data if data is not None else b"",
        capture_output=True,
        env=env,
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout)


def _project(repo: Path, name: str = "demo") -> Path:
    d = repo / "projects" / name
    d.mkdir(parents=True)
    (d / "project.yaml").write_text("schema_version: 1\n")
    return d


def _dataset_zip(tmp_path: Path) -> Path:
    src = tmp_path / "ds"
    make_image_set(src / "images", 3)
    (src / "labels").mkdir()
    (src / "labels" / "img_000.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (src / "classes.txt").write_text("cat\ndog\n")
    return Path(shutil.make_archive(str(tmp_path / "ds"), "zip", tmp_path, "ds"))


def test_upload_extracts_and_describes_dataset(tmp_path: Path):
    repo = tmp_path / "repo"
    pdir = _project(repo)
    archive = _dataset_zip(tmp_path)

    code, out = _run(repo, ["demo", "ds.zip"], archive.read_bytes())
    assert code == 0, out
    res = out["result"]
    dest = pdir / "data" / "uploads" / "ds"
    assert res["path"] == str(dest) and res["files_extracted"] == 5
    assert res["images"] == 3 and res["label_files"] == 1
    assert res["image_root"] == str(dest / "images") and res["label_root"] == str(dest / "labels")
    assert res["class_name_files"] == [str(dest / "classes.txt")]
    assert out["next_operations"] == [
        f"data import demo {dest / 'images'}",
        f"label import demo {dest / 'labels'} --format yolo --classes-file {dest / 'classes.txt'}",
    ]
    assert (dest / "images" / "img_000.jpg").is_file()

    # re-upload replaces the previous extraction instead of merging into it
    (dest / "stale.txt").write_text("x")
    code, out = _run(repo, ["demo", "ds.zip"], archive.read_bytes())
    assert code == 0 and not (dest / "stale.txt").exists()


def test_upload_rejects_bad_input(tmp_path: Path):
    repo = tmp_path / "repo"
    _project(repo)

    code, out = _run(repo, ["demo"], b"x")
    assert code == 2 and out["error"]["code"] == "USAGE"
    code, out = _run(repo, ["nope", "ds.zip"], b"x")
    assert code == 3 and out["error"]["code"] == "PROJECT_NOT_FOUND"
    code, out = _run(repo, ["demo", "../ds.zip"], b"x")
    assert code == 2 and out["error"]["code"] == "ARCHIVE_NAME_INVALID"
    code, out = _run(repo, ["Demo", "ds.zip"], b"x")
    assert code == 2 and out["error"]["code"] == "PROJECT_NAME_INVALID"
    code, out = _run(repo, ["demo", "ds.zip"], b"")
    assert code == 2 and out["error"]["code"] == "EMPTY_UPLOAD"
    code, out = _run(repo, ["demo", "ds.zip"], b"not an archive")
    assert out["error"]["code"] == "ARCHIVE_INVALID"
    code, out = _run(repo, ["demo", "big.zip"], b"\0" * (1024 * 1024 + 1))
    assert code == 1 and out["error"]["code"] == "UPLOAD_TOO_LARGE"

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", "x")
    code, out = _run(repo, ["demo", "evil.zip"], evil.read_bytes())
    assert out["error"]["code"] == "ARCHIVE_INVALID"
    assert not (repo / "projects" / "demo" / "data" / "escape.txt").exists()
    assert not (repo / "projects" / "demo" / "data" / "uploads" / "evil").exists()


def test_upload_normalises_windows_separators(tmp_path: Path):
    """Windows PowerShell 5.1 Compress-Archive stores ``labels\\frame.txt``; it must unpack as a
    ``labels/`` folder, not as a file literally named ``labels\\frame.txt``."""
    repo = tmp_path / "repo"
    pdir = _project(repo)
    archive = tmp_path / "reviewed.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("review\\labels\\cam0_t0001s.txt", "0 0.5 0.5 0.1 0.1\n")
        z.writestr("review\\labels\\cam0_t0002s.txt", "")
        z.writestr("review\\classes.txt", "rodent\n")

    code, out = _run(repo, ["demo", "reviewed.zip"], archive.read_bytes())
    assert code == 0, out
    dest = pdir / "data" / "uploads" / "reviewed"
    assert sorted(p.name for p in (dest / "labels").iterdir()) == [
        "cam0_t0001s.txt",
        "cam0_t0002s.txt",
    ]
    assert out["result"]["label_root"] == str(dest / "labels")
    assert out["result"]["class_name_files"] == [str(dest / "classes.txt")]


def test_upload_rejects_windows_style_traversal(tmp_path: Path):
    repo = tmp_path / "repo"
    pdir = _project(repo)
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("..\\..\\outside.txt", "x")
    code, out = _run(repo, ["demo", "evil.zip"], archive.read_bytes())
    assert code == 1 and out["error"]["code"] == "ARCHIVE_INVALID"
    assert not (pdir / "data" / "outside.txt").exists() and not (repo / "outside.txt").exists()
