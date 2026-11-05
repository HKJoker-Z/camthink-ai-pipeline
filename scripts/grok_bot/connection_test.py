"""Read-only connection test for Grok Bot: proves the SSH path, the CLI, the projects and the
devices are usable without changing anything. Prints one JSON document; exit 0 when the CLI
works (device reachability is reported, not required).

Invoked as ``aip connection-test`` (see scripts/grok_bot/aip)."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(os.environ.get("AIP_REPO") or Path(__file__).resolve().parents[2])
CLI = REPO / ".venv" / "bin" / "ai-pipeline"


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(  # noqa: S603 - fixed commands, no shell
            cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return p.returncode, p.stdout, p.stderr


def cli_json(*args: str) -> tuple[int, dict | None]:
    code, out, _ = run([str(CLI), "-o", "json", *args], timeout=120)
    try:
        return code, json.loads(out)
    except json.JSONDecodeError:
        return code, None


def main() -> int:
    report: dict = {
        "operation": "grok_bot.connection_test",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "server": {
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "repo": str(REPO),
            "ssh_forced_command": bool(os.environ.get("SSH_ORIGINAL_COMMAND")),
        },
        "checks": {},
        "warnings": [],
    }
    checks = report["checks"]

    code, out, _ = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    code2, sha, _ = run(["git", "rev-parse", "--short", "HEAD"])
    checks["git"] = {"ok": code == 0, "branch": out.strip() or None, "commit": sha.strip() or None}

    code, out, err = run([str(CLI), "--version"])
    checks["cli"] = {
        "ok": code == 0,
        "path": str(CLI),
        "version": out.strip() or err.strip()[-200:],
    }
    if code != 0:
        report["status"] = "failed"
        report["error"] = {"code": "CLI_UNAVAILABLE", "message": err.strip()[-500:]}
        print(json.dumps(report, indent=2))
        return 1

    code, env = cli_json("project", "list")
    projects = (env or {}).get("result", {}).get("projects", []) if code == 0 else []
    checks["projects"] = {"ok": code == 0, "list": projects}
    pointers: dict = {}
    for p in projects:
        if not p.get("valid"):
            continue
        c, e = cli_json("status", p["project"])
        if c == 0 and e:
            r = e["result"]
            pointers[p["project"]] = {
                "task_type": r.get("task_type"),
                "pointers": r.get("pointers"),
                "active_runs": r.get("active_runs"),
                "next_operations": e.get("next_operations"),
            }
    checks["project_status"] = pointers

    if shutil.which("nvidia-smi"):
        code, out, _ = run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
        checks["gpu"] = {"ok": code == 0, "devices": [ln.strip() for ln in out.splitlines()]}
    else:
        checks["gpu"] = {"ok": False, "devices": []}
        report["warnings"].append("nvidia-smi not found on the AI server")

    targets: dict = {}
    infra = REPO / "infrastructure.yaml"
    if infra.is_file():
        cfg = yaml.safe_load(infra.read_text(encoding="utf-8")) or {}
        for name, t in (cfg.get("deployment_targets") or {}).items():
            host = (t or {}).get("ssh_host")
            if not host:
                targets[name] = {"type": (t or {}).get("type"), "ssh": None}
                continue
            code, _, err = run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "true"], timeout=20
            )
            targets[name] = {
                "type": t.get("type"),
                "ssh_host": host,
                "reachable": code == 0,
                "error": None if code == 0 else err.strip()[-200:],
            }
            if code != 0:
                report["warnings"].append(f"target {name} ({host}) unreachable from the AI server")
    checks["targets"] = targets

    report["status"] = "ok"
    report["error"] = None
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
