import json

import pytest

from ai_pipeline.core.errors import AlreadyExistsError, NotFoundError
from ai_pipeline.core.store import Store, write_json_atomic


def test_create_read_update_list(tmp_path):
    store = Store(tmp_path / "proj")
    assert store.list_ids("runs") == []
    store.create("runs", "run-1", {"run_id": "run-1", "state": "CREATED"})
    store.create("runs", "run-0", {"run_id": "run-0", "state": "CREATED"})
    assert store.list_ids("runs") == ["run-0", "run-1"]
    assert store.count("runs") == 2
    rec = store.update("runs", "run-1", state="RUNNING")
    assert rec["state"] == "RUNNING"
    assert store.read("runs", "run-1")["state"] == "RUNNING"
    assert [r["run_id"] for r in store.list("runs")] == ["run-0", "run-1"]


def test_create_refuses_overwrite(tmp_path):
    store = Store(tmp_path)
    store.create("models", "mv-001", {"a": 1})
    with pytest.raises(AlreadyExistsError) as ei:
        store.create("models", "mv-001", {"a": 2})
    assert ei.value.code == "MODEL_EXISTS"
    assert store.read("models", "mv-001") == {"a": 1}


def test_missing_record_is_not_found_with_hint(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(NotFoundError) as ei:
        store.read("datasets", "ds-v009")
    assert ei.value.code == "DATASET_NOT_FOUND"
    assert ei.value.hint
    with pytest.raises(NotFoundError):
        store.read_project()


def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        Store(tmp_path).list_ids("widgets")


def test_atomic_write_leaves_no_temp_files_and_is_pretty(tmp_path):
    path = tmp_path / "x" / "y.json"
    write_json_atomic(path, {"b": 1, "a": [1, 2]})
    assert json.loads(path.read_text()) == {"a": [1, 2], "b": 1}
    assert path.read_text().endswith("\n")
    assert [p.name for p in path.parent.iterdir()] == ["y.json"]


def test_temp_files_are_ignored_by_listing(tmp_path):
    store = Store(tmp_path)
    store.create("runs", "run-1", {})
    (tmp_path / "runs" / ".run-2.json.abc.tmp").write_text("{}")
    assert store.list_ids("runs") == ["run-1"]
