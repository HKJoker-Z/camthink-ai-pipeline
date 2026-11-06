"""Receive a dataset archive on stdin and unpack it under the project's data directory.

The restricted SSH key cannot scp; this is the one write path it has, and it only ever
writes below ``projects/<project>/data/uploads/<name>/``:

    ssh ai-server upload <project> <archive-name>.zip  < local.zip      (.zip, .tar, .tar.gz, .tgz)

Prints one JSON document describing what arrived (files, images, label files, class-name
files, the directories to pass to ``data import`` / ``label import``). Re-uploading the same
name replaces the previous extraction. Archive members with absolute paths or ``..`` are
rejected; the archive is capped at ``AIP_UPLOAD_MAX_MB`` (default 4096).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AIP_REPO") or Path(__file__).resolve().parents[2])
PROJECT_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_MB = int(os.environ.get("AIP_UPLOAD_MAX_MB", "4096"))


def fail(code: str, message: str, exit_code: int = 2, **details) -> int:
    print(
        json.dumps(
            {
                "operation": "grok_bot.upload",
                "status": "failed",
                "result": {},
                "error": {"code": code, "message": message, "details": details, "hint": None},
            },
            indent=2,
        )
    )
    return exit_code


def _safe_member(name: str) -> bool:
    p = Path(name)
    return not (p.is_absolute() or ".." in p.parts or name.startswith("/"))


def _extract(archive: Path, dest: Path) -> int:
    n = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            for m in members:
                # Windows PowerShell's Compress-Archive writes "\" separators; on Linux they
                # would become part of the file name ("images\frame.jpg").
                m.filename = m.filename.replace("\\", "/")
                if not _safe_member(m.filename):
                    raise ValueError(f"unsafe member {m.filename!r}")
            z.extractall(dest, members=members)
            n = sum(1 for m in members if not m.is_dir())
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = t.getmembers()
            for m in members:
                if not _safe_member(m.name) or m.issym() or m.islnk():
                    raise ValueError(f"unsafe member {m.name!r}")
            kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            t.extractall(dest, members=members, **kw)
            n = sum(1 for m in members if m.isfile())
    else:
        raise ValueError("not a zip or tar archive")
    return n


def _summarise(root: Path) -> dict:
    images: dict[str, int] = {}
    labels: dict[str, int] = {}
    names_files: list[str] = []
    yaml_files: list[str] = []
    other = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_dir = str(p.parent.relative_to(root)) or "."
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            images[rel_dir] = images.get(rel_dir, 0) + 1
        elif ext == ".txt":
            if p.name in ("names.txt", "classes.txt", "obj.names"):
                names_files.append(str(p.relative_to(root)))
            else:
                labels[rel_dir] = labels.get(rel_dir, 0) + 1
        elif ext in (".yaml", ".yml"):
            yaml_files.append(str(p.relative_to(root)))
        else:
            other += 1

    def _top(dirs: dict[str, int]) -> list[str]:
        # Common ancestor directories to pass to the CLI (recursive import handles subfolders).
        tops: list[str] = []
        for d in sorted(dirs):
            if not any(d == t or d.startswith(t + "/") for t in tops):
                tops.append(d)
        return tops

    def _root(dirs: list[str]) -> str | None:
        # One directory covering all of them (images/{train,val} -> images), unless that
        # would be the upload root itself, which mixes images and labels.
        if not dirs:
            return None
        common = os.path.commonpath(dirs) if len(dirs) > 1 else dirs[0]
        return str(root / common) if common not in ("", ".") else str(root / dirs[0])

    image_dirs, label_dirs = _top(images), _top(labels)
    return {
        "images": sum(images.values()),
        "label_files": sum(labels.values()),
        "image_dirs": [str(root / d) for d in image_dirs],
        "label_dirs": [str(root / d) for d in label_dirs],
        "image_root": _root(image_dirs),
        "label_root": _root(label_dirs),
        "class_name_files": [str(root / f) for f in names_files],
        "yaml_files": [str(root / f) for f in yaml_files],
        "other_files": other,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return fail("USAGE", "usage: upload <project> <archive-name>  (archive on stdin)")
    project, name = argv
    if not PROJECT_RE.match(project):
        return fail("PROJECT_NAME_INVALID", f"invalid project name {project!r}")
    if not NAME_RE.match(name) or name in (".", ".."):
        return fail("ARCHIVE_NAME_INVALID", f"invalid archive name {name!r}")
    project_dir = REPO / "projects" / project
    if not (project_dir / "project.yaml").is_file():
        return fail("PROJECT_NOT_FOUND", f"project {project!r} does not exist; create it first", 3)
    if sys.stdin.isatty():
        return fail("NO_STDIN", "pipe the archive into the command: ... < dataset.zip")

    uploads = project_dir / "data" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"\.(zip|tar|tgz|tar\.gz)$", "", name, flags=re.IGNORECASE)
    archive = uploads / name
    dest = uploads / stem

    limit = MAX_MB * 1024 * 1024
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=uploads, prefix=".upload-", delete=False) as tmp:
        src = sys.stdin.buffer
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                return fail("UPLOAD_TOO_LARGE", f"archive exceeds {MAX_MB} MB", 1)
            digest.update(chunk)
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    if size == 0:
        tmp_path.unlink(missing_ok=True)
        return fail("EMPTY_UPLOAD", "no bytes received on stdin")
    os.replace(tmp_path, archive)

    if dest.exists():
        shutil.rmtree(dest)
    extract_tmp = Path(tempfile.mkdtemp(dir=uploads, prefix=".extract-"))
    try:
        n = _extract(archive, extract_tmp)
    except (ValueError, zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        shutil.rmtree(extract_tmp, ignore_errors=True)
        archive.unlink(missing_ok=True)
        return fail("ARCHIVE_INVALID", f"cannot extract {name}: {exc}", 1)
    # A single top-level folder inside the archive is flattened away.
    entries = list(extract_tmp.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        os.replace(entries[0], dest)
        shutil.rmtree(extract_tmp, ignore_errors=True)
    else:
        os.replace(extract_tmp, dest)
    archive.unlink(missing_ok=True)

    summary = _summarise(dest)
    next_ops = []
    if summary["image_root"]:
        next_ops.append(f"data import {project} {summary['image_root']}")
    if summary["label_root"]:
        cf = (
            f" --classes-file {summary['class_name_files'][0]}"
            if summary["class_name_files"]
            else ""
        )
        next_ops.append(f"label import {project} {summary['label_root']} --format yolo{cf}")
    elif summary["image_root"]:
        next_ops.append(f"label auto {project}")
    print(
        json.dumps(
            {
                "operation": "grok_bot.upload",
                "project": project,
                "status": "ok",
                "result": {
                    "archive": name,
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                    "files_extracted": n,
                    "path": str(dest),
                    **summary,
                },
                "error": None,
                "next_operations": next_ops,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
