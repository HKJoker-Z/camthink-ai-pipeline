"""Structured command output.

With ``--output json`` stdout contains exactly one JSON document (the envelope below) and
nothing else; logs go to stderr. Human mode renders the same data as text. Exit codes are
derived from the envelope, never from parsing text.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

from ai_pipeline.core.ids import utc_now_iso

ENVELOPE_SCHEMA_VERSION = 1
OutputMode = Literal["human", "json"]


@dataclass
class Envelope:
    operation: str
    project: str | None = None
    status: Literal["ok", "failed", "rejected"] = "ok"
    result: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    next_operations: list[str] = field(default_factory=list)
    human_text: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "project": self.project,
            "operation": self.operation,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "next_operations": self.next_operations,
            "timestamp": utc_now_iso(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    def to_human(self) -> str:
        lines: list[str] = []
        head = f"{self.operation}: {self.status.upper()}"
        if self.project:
            head += f"  (project {self.project})"
        lines.append(head)
        if self.error:
            lines.append(f"error [{self.error.get('code')}]: {self.error.get('message')}")
            details = self.error.get("details") or {}
            if details:
                lines.append(_yaml_block(details, indent=2))
            if self.error.get("hint"):
                lines.append(f"hint: {self.error['hint']}")
        if self.human_text is not None:
            lines.append(self.human_text)
        elif self.result:
            lines.append(_yaml_block(self.result))
        if self.next_operations:
            lines.append("next: " + " | ".join(f"ai-pipeline {op}" for op in self.next_operations))
        return "\n".join(lines)


def _yaml_block(data: Any, indent: int = 0) -> str:
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    pad = " " * indent
    return "\n".join(pad + line for line in text.rstrip("\n").splitlines())


def emit(envelope: Envelope, mode: OutputMode) -> None:
    text = envelope.to_json() if mode == "json" else envelope.to_human()
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
