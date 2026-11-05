"""Run one synchronous ``ai-pipeline`` command detached from the SSH session and read it back.

``deploy start``, ``rollback`` and ``target setup`` block for minutes (TensorRT build, service
restart) and are not resumable if the CLI process dies with the session. This helper starts the
command in its own session, keeps stdout (the JSON envelope), stderr and the exit code under
``artifacts/grok_bot/<handle>/`` and refuses to start a second command while one is running.

    detach.py start <ai-pipeline args...>   -> {"handle", "dir", "pid", "command", "poll"}
    detach.py status <handle>               -> {"handle", "state": running|finished|lost,
                                                "exit_code", "envelope", "stderr_tail", ...}

Stdlib only; invoked through ``scripts/grok_bot/aip``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("AIP_REPO") or Path(__file__).resolve().parents[2])
CLI = REPO / ".venv" / "bin" / "ai-pipeline"
ROOT = REPO / "artifacts" / "grok_bot"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _status(handle: str) -> dict:
    d = ROOT / handle
    if not d.is_dir():
        return {"handle": handle, "state": "unknown", "error": "no such handle"}
    meta = json.loads(_read(d / "command.json") or "{}")
    code_txt = _read(d / "exit_code")
    pid = meta.get("pid")
    if code_txt is not None:
        state, exit_code = "finished", int(code_txt.strip())
    elif pid and _alive(pid):
        state, exit_code = "running", None
    else:
        state, exit_code = "lost", None
    envelope = None
    out = _read(d / "stdout.json")
    if out and out.strip():
        try:
            envelope = json.loads(out)
        except json.JSONDecodeError:
            envelope = {"raw_stdout_tail": out[-2000:]}
    stderr = (_read(d / "stderr.log") or "").splitlines()[-20:]
    return {
        "handle": handle,
        "state": state,
        "exit_code": exit_code,
        "command": meta.get("command"),
        "started_at": meta.get("started_at"),
        "finished_at": _read(d / "finished_at"),
        "envelope": envelope,
        "stderr_tail": stderr,
        "dir": str(d),
    }


def start(args: list[str]) -> int:
    if not args:
        print(json.dumps({"error": "no ai-pipeline arguments given"}), file=sys.stderr)
        return 2
    ROOT.mkdir(parents=True, exist_ok=True)
    running = [p.name for p in sorted(ROOT.iterdir()) if _status(p.name)["state"] == "running"]
    if running:
        print(
            json.dumps(
                {
                    "error": "DETACHED_COMMAND_RUNNING",
                    "message": "another detached command is still running; wait for it",
                    "running": running,
                }
            )
        )
        return 3
    slug = re.sub(r"[^a-z0-9]+", "-", " ".join(args[:3]).lower()).strip("-")[:40]
    handle = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{slug}"
    d = ROOT / handle
    d.mkdir()
    script = (
        f'"$0" -o json "$@" > {json.dumps(str(d / "stdout.json"))} '
        f"2> {json.dumps(str(d / 'stderr.log'))}; code=$?; "
        f"date -u +%Y-%m-%dT%H:%M:%SZ > {json.dumps(str(d / 'finished_at'))}; "
        f"echo $code > {json.dumps(str(d / 'exit_code'))}"
    )
    proc = subprocess.Popen(  # noqa: S603 - fixed CLI path, args passed as a list
        ["bash", "-c", script, str(CLI), *args],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    meta = {"command": ["ai-pipeline", "-o", "json", *args], "pid": proc.pid, "started_at": _now()}
    (d / "command.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "handle": handle,
                "dir": str(d),
                "pid": proc.pid,
                "command": meta["command"],
                "poll": f"aip detach-status {handle}",
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "start":
        return start(argv[1:])
    if len(argv) == 2 and argv[0] == "status":
        st = _status(argv[1])
        print(json.dumps(st, indent=2))
        return 0 if st["state"] in ("running", "finished") else 3
    print("usage: detach.py start <ai-pipeline args...> | status <handle>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
