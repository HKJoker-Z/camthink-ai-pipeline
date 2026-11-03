"""Task-agnostic validation report types. Rules live in task modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    asset_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.asset_id:
            d["asset_id"] = self.asset_id
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def error(self, code: str, message: str, **kw: Any) -> None:
        self.issues.append(Issue(code, Severity.ERROR, message, kw.pop("asset_id", None), kw))

    def warning(self, code: str, message: str, **kw: Any) -> None:
        self.issues.append(Issue(code, Severity.WARNING, message, kw.pop("asset_id", None), kw))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def is_valid(self, *, fail_on_warnings: bool = False) -> bool:
        if self.errors:
            return False
        return not (fail_on_warnings and self.warnings)

    def summary_by_code(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.issues:
            out[i.code] = out.get(i.code, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(
        self, *, fail_on_warnings: bool = False, max_issues: int | None = None
    ) -> dict[str, Any]:
        issues = self.issues if max_issues is None else self.issues[:max_issues]
        return {
            "valid": self.is_valid(fail_on_warnings=fail_on_warnings),
            "fail_on_warnings": fail_on_warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues_by_code": self.summary_by_code(),
            "issues": [i.to_dict() for i in issues],
            "issues_truncated": max_issues is not None and len(self.issues) > max_issues,
            "stats": self.stats,
        }
