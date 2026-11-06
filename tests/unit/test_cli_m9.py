"""M9: `app source|configure|liveview|status|metrics` against the fake target."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.unit.test_cli_m3 import runner
from tests.unit.test_cli_m7 import _deployed

from ai_pipeline.cli.main import app

RTSP = "rtsp://admin:S3cret!@192.168.5.78:554/main"
MASKED = "rtsp://***@192.168.5.78:554/main"


def invoke(*args: str, stdin: str | None = None):
    result = runner.invoke(app, ["-o", "json", *args], input=stdin)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:  # pragma: no cover
        raise AssertionError(f"stdout is not JSON:\n{result.stdout}") from None
    return result, payload


def _fake_root(workspace: Path) -> Path:
    return workspace / ".fake-target"


def _assert_no_secret(obj) -> None:
    assert "S3cret" not in json.dumps(obj)


def test_source_set_masks_and_stores_only_on_device(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path)
    r, p = invoke("app", "source", "set", "demo")  # no stdin
    assert r.exit_code != 0
    assert p["error"]["code"] in ("APP_SOURCE_STDIN_REQUIRED", "APP_SOURCE_EMPTY")

    r, p = invoke("app", "source", "set", "demo", stdin=RTSP + "\n")
    assert r.exit_code == 0, p
    res = p["result"]
    _assert_no_secret(p)
    assert res["source"] == {"name": "cam0", "url": MASKED}
    assert res["probe"]["ok"] and res["probe"]["sources"][0]["url"] == MASKED
    assert res["config"]["sources"] == [{"name": "cam0", "url": MASKED}]
    # the fake service is "running" -> a restart is required to pick up the new source
    assert res["restart_required"] == {"inference": True, "liveview": False}
    assert res["restarted"] is False
    assert "service restart demo" in res["next_operations"]

    # credentials exist only in the device-side app.json, mode 0600
    app_json = _fake_root(workspace) / "service" / "app.json"
    data = json.loads(app_json.read_text())
    assert data["sources"][0]["url"] == RTSP
    assert oct(os.stat(app_json).st_mode & 0o777) == "0o600"
    assert data["device"]["name"] == "fake"
    # nothing under the project (git-tracked side) knows the URL
    for path in (workspace / "projects" / "demo").rglob("*"):
        if path.is_file():
            assert "S3cret" not in path.read_text(errors="ignore")

    r, p = invoke("app", "source", "show", "demo")
    assert r.exit_code == 0 and p["result"]["config"]["sources"][0]["url"] == MASKED
    _assert_no_secret(p)

    r, p = invoke("app", "source", "test", "demo")
    assert r.exit_code == 0 and p["result"]["ok"]
    _assert_no_secret(p)

    # same URL again: nothing changes, so nothing to restart
    r, p = invoke("app", "source", "set", "demo", "--restart", stdin=RTSP + "\n")
    assert r.exit_code == 0 and p["result"]["restarted"] is False
    assert p["result"]["restart_required"]["inference"] is False
    r, p = invoke("app", "source", "set", "demo", "--restart", stdin=RTSP + "2\n")
    assert r.exit_code == 0 and p["result"]["restarted"] is True
    assert p["result"]["restart_required"]["inference"] is False
    assert p["result"]["source"]["url"] == MASKED + "2"


def test_source_set_probe_failure_keeps_old_config(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path)
    r, p = invoke("app", "source", "set", "demo", stdin=RTSP + "\n")
    assert r.exit_code == 0
    r, p = invoke("app", "source", "set", "demo", stdin="rtsp://u:pw@unreachable.lan:554/x\n")
    assert r.exit_code != 0 and p["error"]["code"] == "SOURCE_UNREACHABLE"
    assert ":pw@" not in json.dumps(p)
    r, p = invoke("app", "source", "show", "demo")
    assert p["result"]["config"]["sources"][0]["url"] == MASKED
    # --no-test stores it anyway
    r, p = invoke(
        "app",
        "source",
        "set",
        "demo",
        "--no-test",
        "--name",
        "cam1",
        "--add",
        stdin="rtsp://u:pw@unreachable.lan:554/x\n",
    )
    assert r.exit_code == 0 and len(p["result"]["config"]["sources"]) == 2
    r, p = invoke("app", "source", "test", "demo")
    assert r.exit_code != 0 and p["status"] == "failed"
    assert [s["ok"] for s in p["result"]["sources"]] == [True, False]
    r, p = invoke("app", "source", "clear", "demo")
    assert r.exit_code == 0 and len(p["result"]["removed"]) == 2
    r, p = invoke("app", "source", "show", "demo")
    assert p["result"]["config"]["sources"] == []


def test_configure_hot_vs_restart(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path)
    invoke("app", "source", "set", "demo", stdin=RTSP + "\n")
    r, p = invoke("app", "configure", "demo", "--confidence", "0.5", "--inference-fps", "4")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["changed"] == {"inference.confidence": 0.5, "inference.fps": 4.0}
    assert res["restart_required"] == {"inference": False, "liveview": False}
    assert set(res["hot_reloaded"]) == {"inference.confidence", "inference.fps"}

    r, p = invoke("app", "configure", "demo", "--width", "640", "--height", "360", "--port", "8090")
    assert r.exit_code == 0
    assert p["result"]["restart_required"] == {"inference": True, "liveview": True}
    assert "service restart demo" in p["result"]["next_operations"]
    assert "app liveview restart demo" in p["result"]["next_operations"]

    r, p = invoke("app", "configure", "demo", "--confidence", "default")
    assert r.exit_code == 0 and p["result"]["config"]["inference"]["confidence"] is None
    r, p = invoke("app", "configure", "demo", "--confidence", "abc")
    assert r.exit_code != 0 and p["error"]["code"] == "APP_CONFIG_INVALID"
    r, p = invoke("app", "configure", "demo", "--width", "333")
    assert r.exit_code != 0 and p["error"]["code"] == "APP_CONFIG_INVALID"
    r, p = invoke("app", "configure", "demo")
    assert r.exit_code != 0 and p["error"]["code"] == "APP_CONFIG_NOOP"


def test_liveview_lifecycle_is_independent_of_inference(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path)
    r, p = invoke("app", "liveview", "start", "demo")
    assert r.exit_code != 0 and p["error"]["code"] == "NO_SOURCE"

    invoke("app", "source", "set", "demo", stdin=RTSP + "\n")
    r, p = invoke("app", "liveview", "start", "demo")
    assert r.exit_code == 0, p
    res = p["result"]
    assert res["liveview"]["state"] == "running" and res["changed"] is True
    assert res["url"] == "http://fake-device.lan:8080/"
    assert res["inference"] == {"state": "running", "affected": False}
    # the fake URL is not reachable from the test host: that must be reported, not claimed
    assert res["liveview"]["url_check"]["verified"] is False
    assert any("did not answer" in w for w in res["warnings"])

    r, p = invoke("app", "liveview", "url", "demo")
    assert r.exit_code == 0 and p["result"]["url_verified"] is False
    assert p["result"]["url"] == "http://fake-device.lan:8080/"

    r, p = invoke("app", "status", "demo")
    assert r.exit_code == 0
    st = p["result"]
    assert st["ownership"]["matches"] and st["inference"]["state"] == "running"
    assert st["camera"]["state"] == "connected" and st["liveview"]["state"] == "running"
    assert st["config"]["sources"][0]["url"] == MASKED
    _assert_no_secret(p)

    r, p = invoke("app", "metrics", "demo")
    assert r.exit_code == 0 and p["result"]["available"] is True
    ch = p["result"]["channels"][0]
    assert ch["detection_fps"] == 10.0 and ch["camera_connected"] is True
    assert p["result"]["system"]["cpu_pct"] == 20.0

    # stopping the webpage leaves inference running
    r, p = invoke("app", "liveview", "stop", "demo")
    assert r.exit_code == 0 and p["result"]["liveview"]["state"] == "stopped"
    assert p["result"]["inference"]["state"] == "running" and p["result"]["url"] is None
    r, p = invoke("target", "status", "demo")
    assert p["result"]["service"]["state"] == "running"

    # stopping inference leaves the webpage service alone, and metrics become unavailable
    invoke("app", "liveview", "start", "demo")
    r, p = invoke("service", "stop", "demo")
    assert r.exit_code == 0
    r, p = invoke("app", "status", "demo")
    assert p["result"]["inference"]["state"] == "stopped"
    assert p["result"]["liveview"]["state"] == "running"
    assert "service start demo" in p["result"]["next_operations"]
    r, p = invoke("app", "metrics", "demo")
    assert r.exit_code == 0 and p["result"]["available"] is False
    assert p["result"]["system"] is None and p["result"]["channels"] == []
    assert "not running" in p["result"]["reason"]
    # a restart while stopped: the webpage only
    r, p = invoke("app", "liveview", "restart", "demo")
    assert r.exit_code == 0 and any(
        "inference service is stopped" in w for w in p["result"]["warnings"]
    )


def test_app_refuses_other_projects_device(workspace: Path, tmp_path: Path):
    _deployed(workspace, tmp_path, "alpha")
    _deployed(workspace, tmp_path, "beta")  # device now runs beta
    invoke("app", "source", "set", "beta", stdin=RTSP + "\n")
    for args in (
        ("app", "source", "set", "alpha"),
        ("app", "source", "clear", "alpha"),
        ("app", "configure", "alpha", "--confidence", "0.4"),
        ("app", "liveview", "start", "alpha"),
        ("app", "liveview", "stop", "alpha"),
        ("app", "metrics", "alpha"),
    ):
        r, p = invoke(*args, stdin=RTSP + "\n")
        assert r.exit_code == 3 and p["error"]["code"] == "DEVICE_OCCUPIED", args
        assert p["error"]["details"]["device_project"] == "beta"
        assert "--switch" in p["error"]["hint"]
    # read-only status says so instead of showing beta's numbers
    r, p = invoke("app", "status", "alpha")
    assert r.exit_code == 0 and p["result"]["ownership"]["matches"] is False
    assert p["result"]["camera"]["channels"] == []
    assert any("not alpha" in w for w in p["result"]["warnings"])
    assert "service start alpha --switch" in p["result"]["next_operations"]
    # beta unaffected
    r, p = invoke("app", "status", "beta")
    assert p["result"]["ownership"]["matches"] is True
