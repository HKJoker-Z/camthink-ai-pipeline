import re
from datetime import datetime, timezone

from ai_pipeline.core.ids import new_event_id, next_sequential_id, utc_now_iso


def test_utc_now_iso_format():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", utc_now_iso())
    a, b = utc_now_iso(), utc_now_iso()
    assert a <= b, "timestamps must sort chronologically as strings"


def test_new_event_id_is_prefixed_timestamped_and_unique():
    now = datetime(2026, 11, 3, 15, 33, 0, tzinfo=timezone.utc)
    a = new_event_id("run", now)
    b = new_event_id("run", now)
    assert a.startswith("run-20261103-153300-")
    assert a != b
    assert re.fullmatch(r"run-\d{8}-\d{6}-[0-9a-f]{4}", a)


def test_next_sequential_id_starts_at_one_and_ignores_other_prefixes():
    assert next_sequential_id("ds-v", []) == "ds-v001"
    assert next_sequential_id("ds-v", ["ds-v001", "ds-v007", "mv-003", "junk"]) == "ds-v008"
    assert next_sequential_id("mv-", ["mv-003"]) == "mv-004"


def test_next_sequential_id_handles_width_overflow():
    assert next_sequential_id("v", ["v999"]) == "v1000"
