"""Remote execution over the system ``ssh``/``scp`` binaries.

Deliberately thin: the operator's ``~/.ssh/config`` supplies host, user, key and the pinned host
key; we only add non-interactive flags. Failures become ``RemoteError`` (exit code 4) with the
command, exit status and stderr tail so they are actionable without re-running by hand.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ai_pipeline.core.errors import RemoteError
from ai_pipeline.core.logging import get_logger

log = get_logger("remote")


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _tail(text: str, n: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


class SshHost:
    def __init__(self, host: str, *, connect_timeout: int = 10) -> None:
        self.host = host
        self.connect_timeout = connect_timeout

    def _opts(self) -> list[str]:
        return [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ServerAliveInterval=15",
        ]

    def run(
        self,
        command: str,
        *,
        timeout: float = 120,
        check: bool = True,
        stdin: str | None = None,
    ) -> CommandResult:
        """Run ``command`` through the remote user's ``sh -c``. ``stdin`` (if given) is piped to
        the remote process — the way to pass secrets without putting them on a command line."""
        argv = ["ssh", *self._opts(), self.host, "--", f"sh -c {shlex.quote(command)}"]
        if stdin is None:
            argv.insert(1, "-n")
        log.debug("ssh %s: %s", self.host, command[:200])
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                input=stdin,
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(
                f"command on {self.host} timed out after {timeout:.0f}s",
                code="REMOTE_TIMEOUT",
                details={"host": self.host, "command": command[:500], "timeout_s": timeout},
                hint="Increase the timeout in infrastructure.yaml or inspect the device.",
            ) from exc
        result = CommandResult(proc.returncode, proc.stdout, proc.stderr)
        if check and proc.returncode != 0:
            code = "REMOTE_UNREACHABLE" if proc.returncode == 255 else "REMOTE_COMMAND_FAILED"
            raise RemoteError(
                f"ssh {self.host} exited {proc.returncode}: {_tail(proc.stderr, 3) or _tail(proc.stdout, 3)}",
                code=code,
                details={
                    "host": self.host,
                    "command": command[:500],
                    "exit_code": proc.returncode,
                    "stderr_tail": _tail(proc.stderr),
                    "stdout_tail": _tail(proc.stdout),
                },
                hint=(
                    f"Check `ssh -o BatchMode=yes {self.host} true` works for this user "
                    "(key, known_hosts entry, network)."
                    if proc.returncode == 255
                    else "See stderr_tail; rerun the command on the device to reproduce."
                ),
            )
        return result

    def run_json(self, command: str, *, timeout: float = 120, stdin: str | None = None) -> dict:
        """Run a command that prints one JSON document as its last stdout line."""
        res = self.run(command, timeout=timeout, check=False, stdin=stdin)
        lines = [ln for ln in res.stdout.strip().splitlines() if ln.strip()]
        payload = None
        if lines:
            try:
                payload = json.loads(lines[-1])
            except json.JSONDecodeError:
                payload = None
        if payload is None or not isinstance(payload, dict):
            raise RemoteError(
                f"command on {self.host} did not return JSON (exit {res.returncode})",
                code="REMOTE_BAD_OUTPUT" if res.returncode == 0 else "REMOTE_COMMAND_FAILED",
                details={
                    "host": self.host,
                    "command": command[:500],
                    "exit_code": res.returncode,
                    "stdout_tail": _tail(res.stdout),
                    "stderr_tail": _tail(res.stderr),
                },
            )
        if res.returncode != 0 or "error" in payload:
            err = payload.get("error") or {}
            raise RemoteError(
                f"remote operation failed on {self.host}: {err.get('type', 'error')}: "
                f"{err.get('message', _tail(res.stderr, 3))}",
                code="REMOTE_OPERATION_FAILED",
                details={
                    "host": self.host,
                    "command": command[:500],
                    "exit_code": res.returncode,
                    "error": err,
                    "stderr_tail": _tail(res.stderr),
                },
            )
        return payload

    def resolved_hostname(self) -> str | None:
        """The ``HostName`` the local ssh client resolves this alias to (``ssh -G``), i.e. the
        address other LAN machines can most likely use too. None if ssh cannot tell."""
        try:
            proc = subprocess.run(
                ["ssh", "-G", self.host],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in proc.stdout.splitlines():
            key, _, value = line.strip().partition(" ")
            if key.lower() == "hostname" and value:
                return value
        return None

    def upload(
        self,
        local: Path | list[Path],
        remote: str,
        *,
        recursive: bool = False,
        timeout: float = 600,
    ) -> None:
        locals_ = [str(p) for p in (local if isinstance(local, list) else [local])]
        argv = [
            "scp",
            "-q",
            *self._opts(),
            *(["-r"] if recursive else []),
            *locals_,
            f"{self.host}:{remote}",
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise RemoteError(
                f"scp to {self.host} timed out", code="REMOTE_TIMEOUT", details={"remote": remote}
            ) from exc
        if proc.returncode != 0:
            raise RemoteError(
                f"scp to {self.host}:{remote} failed (exit {proc.returncode}): {_tail(proc.stderr, 3)}",
                code="REMOTE_TRANSFER_FAILED",
                details={
                    "host": self.host,
                    "remote": remote,
                    "files": locals_[:10],
                    "stderr_tail": _tail(proc.stderr),
                },
            )

    def download(
        self, remote: str, local: Path, *, recursive: bool = False, timeout: float = 600
    ) -> None:
        argv = [
            "scp",
            "-q",
            *self._opts(),
            *(["-r"] if recursive else []),
            f"{self.host}:{remote}",
            str(local),
        ]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            raise RemoteError(
                f"scp from {self.host}:{remote} failed (exit {proc.returncode}): {_tail(proc.stderr, 3)}",
                code="REMOTE_TRANSFER_FAILED",
                details={"host": self.host, "remote": remote, "stderr_tail": _tail(proc.stderr)},
            )
