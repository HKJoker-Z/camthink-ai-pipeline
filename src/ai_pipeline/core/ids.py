"""Identifier and timestamp helpers.

Two id styles are used deliberately:
- time-based ids for things that happen (runs, deployments): unique, sortable, never reused;
- sequential ids for things that are versions (dataset versions, model versions): short,
  human-friendly, and only ever allocated by scanning what already exists.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable
from datetime import datetime, timezone

_SEQ_RE = re.compile(r"^(?P<prefix>.+?)(?P<num>\d+)$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with second precision and a trailing Z."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def new_event_id(prefix: str, now: datetime | None = None) -> str:
    """e.g. ``run-20261103-153300-7f3a``. The random suffix guards against same-second retries."""
    ts = (now or utc_now()).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}-{secrets.token_hex(2)}"


def next_sequential_id(prefix: str, existing: Iterable[str], width: int = 3) -> str:
    """Return ``<prefix><N+1>`` where N is the highest number among ``existing`` ids
    sharing the prefix. Ids that do not match are ignored. Starts at 1."""
    highest = 0
    for item in existing:
        if not item.startswith(prefix):
            continue
        m = _SEQ_RE.match(item)
        if m and m.group("prefix") == prefix:
            highest = max(highest, int(m.group("num")))
    return f"{prefix}{highest + 1:0{width}d}"
