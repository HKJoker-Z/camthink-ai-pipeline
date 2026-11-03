"""Error hierarchy. Every error carries a machine-readable code, a hint, and an exit code.

Exit codes (see docs/cli.md):
    0 success
    1 operation failed
    2 usage / configuration error
    3 precondition / state error
    4 remote / connectivity error
    5 policy rejection (metrics below acceptance)
"""

from __future__ import annotations

from typing import Any


class ExitCode:
    OK = 0
    OPERATION_FAILED = 1
    CONFIG = 2
    STATE = 3
    REMOTE = 4
    POLICY_REJECTED = 5
    INTERNAL = 1


class PipelineError(Exception):
    """Base class for all expected pipeline failures."""

    exit_code: int = ExitCode.OPERATION_FAILED
    default_code: str = "OPERATION_FAILED"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "hint": self.hint,
        }


class ConfigError(PipelineError):
    exit_code = ExitCode.CONFIG
    default_code = "CONFIG_INVALID"


class StateError(PipelineError):
    exit_code = ExitCode.STATE
    default_code = "INVALID_STATE"


class NotFoundError(StateError):
    default_code = "NOT_FOUND"


class AlreadyExistsError(StateError):
    default_code = "ALREADY_EXISTS"


class RemoteError(PipelineError):
    exit_code = ExitCode.REMOTE
    default_code = "REMOTE_ERROR"


class OperationError(PipelineError):
    exit_code = ExitCode.OPERATION_FAILED
    default_code = "OPERATION_FAILED"


class PolicyRejected(PipelineError):
    exit_code = ExitCode.POLICY_REJECTED
    default_code = "POLICY_REJECTED"
