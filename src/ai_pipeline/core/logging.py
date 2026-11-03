"""Context-aware logging.

Logs always go to stderr (stdout is reserved for command output so ``--output json`` stays
parseable). Optionally a per-operation log file is added under ``artifacts/<project>/logs/``.
Every record carries whatever context has been bound (project, operation, run_id, ...).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from ai_pipeline.core.ids import utc_now_iso

ROOT_LOGGER = "ai_pipeline"
CONTEXT_KEYS = (
    "project",
    "operation",
    "run_id",
    "dataset_version",
    "model_version",
    "deployment_id",
    "target",
)


class ContextFilter(logging.Filter):
    """Ensure every record has all context attributes so formatters never KeyError."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in CONTEXT_KEYS:
            if not hasattr(record, key):
                setattr(record, key, None)
        return True


class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = " ".join(
            f"{k}={getattr(record, k)}" for k in CONTEXT_KEYS if getattr(record, k, None)
        )
        base = f"{utc_now_iso()} {record.levelname:<7} {record.name}"
        msg = record.getMessage()
        line = f"{base} [{ctx}] {msg}" if ctx else f"{base} {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k in CONTEXT_KEYS:
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, verbosity: int = 0, quiet: bool = False, json_logs: bool = False) -> None:
    """Configure the ``ai_pipeline`` logger once per process. Idempotent."""
    logger = logging.getLogger(ROOT_LOGGER)
    for h in list(logger.handlers):
        if getattr(h, "_aip_stderr", False):
            logger.removeHandler(h)

    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logger.setLevel(logging.DEBUG)  # handlers filter; file handler may want everything
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter() if json_logs else HumanFormatter())
    handler.addFilter(ContextFilter())
    handler._aip_stderr = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


def add_file_handler(path: Path) -> logging.Handler:
    """Attach a DEBUG-level file handler (human format). Caller removes it when done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(HumanFormatter())
    handler.addFilter(ContextFilter())
    logging.getLogger(ROOT_LOGGER).addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    logging.getLogger(ROOT_LOGGER).removeHandler(handler)
    handler.close()


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a logger adapter that injects ``context`` into every record."""
    base = logging.getLogger(f"{ROOT_LOGGER}.{name}" if not name.startswith(ROOT_LOGGER) else name)
    return _ContextAdapter(base, {k: v for k, v in context.items() if v is not None})


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.pop("extra", {}) or {})
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **context: Any) -> _ContextAdapter:
        merged = dict(self.extra or {})
        merged.update({k: v for k, v in context.items() if v is not None})
        return _ContextAdapter(self.logger, merged)
