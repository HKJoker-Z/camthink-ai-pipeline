"""Send a generated label-export ZIP to stdout: the restricted key's only read path for files.

    ssh ai-server download <project> <export_id>            > review.zip   (raw bytes)
    ssh ai-server download <project> <export_id> --base64   > review.b64   (text-safe)
    ssh ai-server download <project> <export_id> --info                    (JSON, no bytes)

Only archives recorded by ``label export`` (``artifacts/<p>/label_exports/<export_id>.json``)
can be sent; the archive path comes from that record and must lie in the export's own
directory. Size and SHA-256 are re-checked against the record before any byte is written.
Errors and the success summary are JSON on **stderr**, so stdout carries nothing but the file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AIP_REPO") or Path(__file__).resolve().parents[2])
PROJECT_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
EXPORT_RE = re.compile(r"^lex-\d{8}-\d{6}-[0-9a-f]{4}$")
CHUNK = 57 * 16384  # multiple of 57 bytes = whole 76-character base64 lines
MODES = ("--base64", "--info")


def _emit(stream, doc: dict) -> None:
    doc["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stream.write(json.dumps(doc, indent=2) + "\n")
    stream.flush()


def fail(code: str, message: str, exit_code: int = 2, **details) -> int:
    _emit(
        sys.stderr,
        {
            "operation": "grok_bot.download",
            "status": "failed",
            "result": {},
            "error": {"code": code, "message": message, "details": details, "hint": None},
        },
    )
    return exit_code


def _artifact_root() -> tuple[Path, Path]:
    from ai_pipeline.core.settings import Settings

    s = Settings.resolve(cwd=REPO)
    return s.projects_root, s.artifact_root


def main(argv: list[str]) -> int:
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2 or len(flags) > 1 or any(f not in MODES for f in flags):
        return fail("USAGE", "usage: download <project> <export_id> [--base64 | --info]")
    project, export_id = args
    mode = flags[0] if flags else "raw"
    if not PROJECT_RE.match(project):
        return fail("PROJECT_NAME_INVALID", f"invalid project name {project!r}")
    if not EXPORT_RE.match(export_id):
        return fail(
            "EXPORT_ID_INVALID",
            f"invalid export id {export_id!r}; expected lex-YYYYMMDD-HHMMSS-xxxx",
        )
    try:
        projects_root, artifact_root = _artifact_root()
    except Exception as exc:  # noqa: BLE001
        return fail("SETTINGS_INVALID", f"cannot resolve pipeline settings: {exc}", 1)
    if not (projects_root / project / "project.yaml").is_file():
        return fail("PROJECT_NOT_FOUND", f"project {project!r} does not exist", 3)

    exports = (artifact_root / project / "label_exports").resolve()
    rec_path = exports / f"{export_id}.json"
    if not rec_path.is_file():
        return fail(
            "LABEL_EXPORT_NOT_FOUND",
            f"label export {export_id!r} not found; run `label exports {project}`",
            3,
        )
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        archive = (artifact_root / project / rec["archive"]).resolve()
        size, sha = int(rec["bytes"]), str(rec["sha256"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return fail("EXPORT_RECORD_INVALID", f"unreadable export record: {exc}", 1)
    if archive.parent != exports / export_id or archive.suffix != ".zip":
        return fail("EXPORT_PATH_INVALID", "export record points outside its directory", 1)

    summary = {
        "project": project,
        "export_id": export_id,
        "label_set_id": rec.get("label_set_id"),
        "archive_name": archive.name,
        "bytes": size,
        "sha256": sha,
        "encoding": "base64" if mode == "--base64" else "binary",
    }
    try:
        fh = archive.open("rb")
    except OSError as exc:
        return fail("EXPORT_FILE_MISSING", f"export archive unavailable: {exc}", 1)
    with fh:
        h = hashlib.sha256()
        n = 0
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
            n += len(chunk)
        if n != size or h.hexdigest() != sha:
            return fail(
                "EXPORT_INTEGRITY_FAILED",
                "archive does not match its record; re-run `label export`",
                1,
                expected_bytes=size,
                actual_bytes=n,
            )
        if mode == "--info":
            _emit(
                sys.stdout,
                {
                    "operation": "grok_bot.download",
                    "project": project,
                    "status": "ok",
                    "result": {**summary, "counts": rec.get("counts")},
                    "error": None,
                    "next_operations": [
                        f"download {project} {export_id}",
                        f"download {project} {export_id} --base64",
                    ],
                },
            )
            return 0
        fh.seek(0)
        out = sys.stdout.buffer
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            out.write(base64.encodebytes(chunk) if mode == "--base64" else chunk)
        out.flush()
    _emit(
        sys.stderr,
        {"operation": "grok_bot.download", "status": "ok", "result": summary, "error": None},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
