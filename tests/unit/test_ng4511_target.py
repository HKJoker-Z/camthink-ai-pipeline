"""CamThinkNG4511Target against a scripted fake SSH host (no network).

The fake host keeps a tiny model of the device filesystem (deployments, current symlink,
runtime bundle) and answers the `camthink_edge` commands with JSON, so the whole M3 lifecycle
can run through the real target code and the real orchestrator.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from ai_pipeline.core.config.schema import CamThinkNG4511TargetConfig
from ai_pipeline.core.deployment import infer_on_target, setup_target, start_deployment
from ai_pipeline.core.errors import OperationError, RemoteError
from ai_pipeline.core.remote import CommandResult
from ai_pipeline.targets.camthink_ng4511 import CamThinkNG4511Target, bundle_version

HOME = "/home/julian"


class FakeSsh:
    """Minimal device simulator keyed on the commands the target issues."""

    def __init__(self, *, fail: str | None = None, healthy: bool = True, bench: dict | None = None):
        self.fail = fail
        self.healthy = healthy
        self.bench = bench or {
            "latency_ms_p50": 12.0,
            "latency_ms_p95": 15.0,
            "fps": 80.0,
            "memory_mb": 420.0,
            "gpu_util_pct": 60,
        }
        self.runtime_version: str | None = None
        self.deployments: dict[str, set[str]] = {}  # dep -> files
        self.current: str | None = None
        self.service_state = "inactive"
        self.stop_marker: str | None = None  # contents of service/stopped.json
        self.commands: list[str] = []
        self.uploads: list[tuple[list[str], str]] = []
        self.unit_installed = False
        self.units: dict[str, str] = {}  # unit file name -> contents
        self.app_json: str | None = None  # contents of service/app.json
        self.app_json_tmp: str | None = None
        self.liveview_state = "inactive"
        self.liveview_enabled = "disabled"
        self.live_stats: str | None = None  # contents of /dev/shm/.../stats.json
        self.stdin_seen: list[str] = []

    # -- SshHost API -------------------------------------------------------------------

    def run(
        self, command: str, *, timeout: float = 120, check: bool = True, stdin: str | None = None
    ) -> CommandResult:
        self.commands.append(command)
        if stdin is not None:
            self.stdin_seen.append(stdin)
        out, rc = self._handle(command, stdin=stdin)
        if check and rc != 0:
            raise RemoteError(
                f"exit {rc}", code="REMOTE_COMMAND_FAILED", details={"command": command}
            )
        return CommandResult(rc, out, "")

    def run_json(self, command: str, *, timeout: float = 120, stdin: str | None = None) -> dict:
        res = self.run(command, timeout=timeout, check=False, stdin=stdin)
        payload = json.loads(res.stdout.strip().splitlines()[-1])
        if "error" in payload:
            raise RemoteError(
                payload["error"]["message"],
                code="REMOTE_OPERATION_FAILED",
                details=payload["error"],
            )
        return payload

    def upload(self, local, remote: str, *, recursive=False, timeout=600) -> None:
        files = [str(p) for p in (local if isinstance(local, list) else [local])]
        self.uploads.append((files, remote))
        if remote.endswith("runtime.new/"):
            self.runtime_version = bundle_version()
        m = re.search(r"/deployments/([\w-]*dep-\d+)/", remote)
        if m:
            self.deployments.setdefault(m.group(1), set()).update(Path(f).name for f in files)
        if remote.endswith(".service"):
            self.unit_installed = True
            self.units[remote.rsplit("/", 1)[1]] = Path(files[0]).read_text()
        if remote.endswith("service/app.json.tmp"):
            self.app_json_tmp = Path(files[0]).read_text()

    def download(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def resolved_hostname(self) -> str | None:
        return "192.168.5.222"

    # -- command handling ----------------------------------------------------------------------

    def _handle(self, c: str, *, stdin: str | None = None) -> tuple[str, int]:
        if c == "echo $HOME":
            return HOME + "\n", 0
        # -- app config / live view (M9) ------------------------------------------------
        if c.startswith("cat ") and "service/app.json" in c:
            return (self.app_json, 0) if self.app_json else ("", 0)
        if c.startswith("chmod 600 ") and "app.json.tmp" in c:
            self.app_json, self.app_json_tmp = self.app_json_tmp, None
            return "", 0
        if c.startswith("cat ") and "/dev/shm/" in c:
            return (self.live_stats, 0) if self.live_stats else ("", 0)
        if "__EN__" in c:
            return f"{self.liveview_state}\n__EN__\n{self.liveview_enabled}\n", 0
        if "camthink-ai-liveview" in c and "systemctl --user" in c:
            for part in c.replace(">/dev/null 2>&1", "").split(";"):
                action = part.split("systemctl --user ")[1].split()[0]
                if action in ("start", "restart"):
                    self.liveview_state = "active"
                elif action == "stop":
                    self.liveview_state = "inactive"
                elif action == "enable":
                    self.liveview_enabled = "enabled"
                elif action == "disable":
                    self.liveview_enabled = "disabled"
            return "", 0
        if "grep '^VERSION'" in c:
            return (f'VERSION = "{self.runtime_version}"\n', 0) if self.runtime_version else ("", 1)
        if c.startswith("echo machine="):
            cur = self.current or ""
            return (
                f"machine=aarch64\npython=3.10.12\nl4t=# R36 (release), REVISION: 4.3\ntrtexec=yes\nroot_exists=yes\n"
                f"current={cur}\nstaged={','.join(sorted(self.deployments))}\ndisk_free_mb=170000\nmem_total_mb=7620\nmem_avail_mb=5300\n",
                0,
            )
        if "-m camthink_edge" in c:
            return self._edge(c, stdin)
        # -- service control ------------------------------------------------------------
        if "__STATE__" in c:  # service_status(): one composed round trip
            cur = f"deployments/{self.current}" if self.current else ""
            hb = (
                json.dumps({"deployment": self.current, "fps": 5.0, "frames": 42, "errors": 0})
                if self.service_state == "active"
                else ""
            )
            return (
                f"__STATE__\n{self.service_state}\n__CURRENT__\n{cur}\n__STOPPED__\n"
                f"{self.stop_marker or ''}\n__HEARTBEAT__\n{hb}\n",
                0,
            )
        if "stopped.json" in c and c.startswith("cat "):
            return (self.stop_marker, 0) if self.stop_marker else ("", 1)
        if "stopped.json" in c and "printf" in c:  # service_stop(): write marker, then stop
            self.stop_marker = shlex.split(c.split("printf '%s' ", 1)[1].split(" > ")[0])[0]
            self.service_state = "inactive"
            return "", 0
        if "stopped.json" in c and c.startswith("rm -f"):  # marker cleared, then start/restart
            self.stop_marker = None
            rest = c.split(";", 1)[1] if ";" in c else c.split("&&", 1)[1]
            return self._handle(rest.strip())
        if c.startswith("test -e ") and "/deployments/" in c:
            dep = c.split("/deployments/")[1].split()[0].strip("'")
            return "", 0 if dep in self.deployments else 1
        if c.startswith("mkdir -p"):
            return "", 0
        if c.startswith("mv ") and "model.onnx" in c:
            src, dst = shlex.split(c)[1:3]
            if src == dst:  # real mv refuses this; the bug the first device run hit
                return "", 1
            dep = c.split("/deployments/")[1].split("/")[0]
            self.deployments[dep].add("model.onnx")
            return "", 0
        if c.startswith("sha256sum"):
            return "a" * 64 + "  model.onnx\n174\n", 0
        if c.startswith("ls ") and "/samples" in c:
            return "img.jpg\n", 0
        if "ln -sfn deployments/" in c:
            if self.fail == "activate" and "mv -T" in c and self.current is not None:
                return "", 1
            dep = c.split("ln -sfn deployments/")[1].split()[0].strip("'")
            if dep not in self.deployments or "model.engine" not in self.deployments[dep]:
                return "", 1
            self.current = dep
            return "", 0
        if c.startswith("rm -f") and "/current" in c:
            self.current = None
            return "", 0
        if "systemctl --user" in c:
            action = c.split("systemctl --user ")[1].split()[0]
            if action == "is-active":
                return self.service_state + "\n", 0 if self.service_state == "active" else 3
            if action in ("restart", "start"):
                if (
                    self.fail == "rollback"
                    and self.current
                    and not self.current.endswith("dep-002")
                ):
                    return "", 1
                self.service_state = "active"
            elif action == "stop":
                self.service_state = "inactive"
            return "", 0
        if c.startswith("loginctl"):
            return "Linger=no\n", 0
        if c.startswith("readlink"):
            return (f"deployments/{self.current}\n" if self.current else "", 0)
        if c.startswith("rm -rf") or c.startswith("test -f") or c.startswith("cat "):
            return "", 0
        raise AssertionError(f"unexpected command: {c}")

    def _edge(self, c: str, stdin: str | None = None) -> tuple[str, int]:
        args = shlex.split(c.split("-m camthink_edge ", 1)[1])
        cmd = args[0]
        if cmd == "probe":
            if "--source-stdin" in args:
                urls = [(stdin or "").strip()]
            else:
                urls = [s["url"] for s in json.loads(self.app_json or "{}").get("sources", [])]
            return json.dumps(
                {
                    "ok": all("bad" not in u for u in urls) and bool(urls),
                    "sources": [
                        {
                            "name": "cam0",
                            "url": u,
                            "ok": "bad" not in u,
                            "frames": 0 if "bad" in u else 120,
                            "decode_fps": 25.0,
                            "backend": "cv2gst",
                        }
                        for u in urls
                    ],
                }
            ) + "\n", 0
        if cmd == "info":
            return json.dumps(
                {
                    "runtime_version": self.runtime_version,
                    "numpy": "1.26",
                    "tensorrt": "10.3.0",
                    "cv2": "4.8",
                    "PIL": "missing (X)",
                    "cudart_version": 12050,
                }
            ) + "\n", 0
        if cmd == "build":
            if self.fail == "prepare":
                return json.dumps(
                    {"error": {"type": "RuntimeError", "message": "trtexec failed"}}
                ) + "\n", 1
            dep = args[args.index("--engine") + 1].split("/deployments/")[1].split("/")[0]
            self.deployments[dep].add("model.engine")
            return json.dumps(
                {
                    "engine_path": f"{HOME}/camthink-ai/deployments/{dep}/model.engine",
                    "engine_sha256": "e" * 64,
                    "engine_size_bytes": 12345,
                    "build_params": {"precision": args[args.index("--precision") + 1]},
                    "environment": {"tensorrt": "10.3.0"},
                    "duration_s": 95.0,
                    "builder_peak_rss_mb": 1800,
                }
            ) + "\n", 0
        if cmd == "smoke":
            if self.fail == "smoke_test":
                return json.dumps(
                    {"passed": False, "details": {"reason": "no detections", "total_detections": 0}}
                ) + "\n", 0
            return json.dumps(
                {"passed": True, "details": {"images": 3, "total_detections": 5}}
            ) + "\n", 0
        if cmd == "benchmark":
            return json.dumps(
                {
                    "metrics": self.bench,
                    "iterations": int(args[args.index("--iterations") + 1]),
                    "details": {"tegrastats": {"available": True}},
                }
            ) + "\n", 0
        if cmd == "health":
            ok = self.healthy and self.current is not None and self.service_state == "active"
            if self.fail == "health_check" and self.current.endswith("dep-002"):
                ok = False
            return json.dumps(
                {
                    "healthy": ok,
                    "current": self.current,
                    "details": {"service_state": self.service_state},
                }
            ) + "\n", 0
        if cmd == "infer":
            return json.dumps(
                {
                    "images": [
                        {
                            "image": "x.jpg",
                            "detections": [
                                {
                                    "class_name": "person",
                                    "confidence": 0.8,
                                    "bbox_xyxy": [1, 2, 3, 4],
                                }
                            ],
                        }
                    ],
                    "total_detections": 1,
                }
            ) + "\n", 0
        raise AssertionError(f"unexpected edge command {cmd}")


def _target(fake: FakeSsh, **cfg) -> CamThinkNG4511Target:
    conf = CamThinkNG4511TargetConfig(type="camthink_ng4511", ssh_host="camthink-test", **cfg)
    return CamThinkNG4511Target(name="ng4511-lab", cfg=conf, host=fake)


@pytest.fixture
def use_fake_target(monkeypatch):
    """Route get_target('ng4511-lab') in the orchestrator to a target with a FakeSsh."""
    holder: dict = {}

    def install(fake: FakeSsh, **cfg):
        target = _target(fake, **cfg)
        holder["target"] = target
        import ai_pipeline.core.deployment as dep_mod

        real = dep_mod.get_target

        def patched(name, settings):
            return target if name == "ng4511-lab" else real(name, settings)

        monkeypatch.setattr(dep_mod, "get_target", patched)
        return target

    return install


def test_root_expansion_and_command_composition():
    fake = FakeSsh()
    t = _target(fake)
    assert t.root == f"{HOME}/camthink-ai"
    assert (
        t._edge("health", "--root", t.root)
        == f"PYTHONPATH={HOME}/camthink-ai/runtime python3 -m camthink_edge health --root {HOME}/camthink-ai"
    )
    assert "systemctl --user restart camthink-ai-inference" in t._service_cmd("restart")
    assert _target(fake, service_name=None)._service_cmd("restart") is None
    assert _target(fake, install_root="/opt/x").root == "/opt/x"


def test_setup_runtime_installs_bundle_and_unit():
    fake = FakeSsh()
    t = _target(fake)
    info = t.setup_runtime()
    assert info["bundle"] == {
        "runtime_version": bundle_version(),
        "updated": True,
        "previous": None,
    }
    assert fake.unit_installed and info["service"]["name"] == "camthink-ai-inference"
    assert any("enable-linger" in w for w in info["warnings"])
    assert any("daemon-reload" in c for c in fake.commands) and any(
        "enable camthink-ai-inference" in c for c in fake.commands
    )
    # idempotent
    assert t.setup_runtime()["bundle"]["updated"] is False


def test_verify_reports_environment():
    fake = FakeSsh()
    t = _target(fake)
    info = t.verify()
    assert info["reachable"] and info["current"] is None and info["staged"] == []
    env = info["environment"]
    assert env["machine"] == "aarch64" and env["trtexec"] == "/usr/src/tensorrt/bin/trtexec"
    assert env["runtime_version"] is None and env["runtime_version_local"] == bundle_version()
    assert env["mem_total_mb"] == 7620 and info["service"] == "inactive"
    t.setup_runtime()
    assert t.verify()["environment"]["tensorrt"] == "10.3.0"


def test_full_deployment_through_ng4511_target(exported_project, use_fake_target):
    p = exported_project
    fake = FakeSsh()
    use_fake_target(fake)
    rec = start_deployment(p, target_name="ng4511-lab")
    assert rec["state"] == "ACTIVE", rec.get("error")
    assert rec["target_type"] == "camthink_ng4511"
    # bundle was installed implicitly during stage; files staged; engine built; service restarted
    assert fake.runtime_version == bundle_version()
    assert fake.deployments["demo-dep-001"] >= {
        "model.onnx",
        "runtime.json",
        "deployment.json",
        "model.engine",
    }
    staged_runtime = [
        f for files, remote in fake.uploads for f in files if f.endswith("runtime.json")
    ]
    assert staged_runtime
    assert (
        rec["runtime_config"]["classes"] == ["person"]
        and rec["runtime_config"]["input_size"] == 640
    )
    assert rec["runtime_config"]["postprocess"]["confidence"] == 0.25
    assert len(rec["samples"]) == 1  # 12-image fixture dataset -> 1 test image
    assert (
        rec["runtime_artifact"]["engine_sha256"] == "e" * 64
        and rec["runtime_artifact"]["duration_s"] == 95.0
    )
    assert rec["benchmark"]["metrics"]["fps"] == 80.0 and rec["acceptance"]["passed"]
    assert fake.current == "demo-dep-001" and fake.service_state == "active"
    assert rec["activation"]["service_restarted"] is True
    assert rec["health"]["healthy"] and rec["health"]["details"]["attempts"] == 1
    assert any("--iterations 100 --warmup 20" in c for c in fake.commands)
    assert any("--min-detections 1" in c for c in fake.commands)


def test_ng4511_prepare_failure_leaves_active_untouched(exported_project, use_fake_target):
    p = exported_project
    fake = FakeSsh()
    use_fake_target(fake)
    assert start_deployment(p, target_name="ng4511-lab")["state"] == "ACTIVE"
    fake.fail = "prepare"
    bad = start_deployment(p, target_name="ng4511-lab")
    assert bad["state"] == "FAILED" and bad["error"]["phase"] == "prepare"
    assert (
        bad["error"]["code"] == "REMOTE_OPERATION_FAILED"
        and "trtexec failed" in bad["error"]["message"]
    )
    assert fake.current == "demo-dep-001" and p.pointers()["active_deployment"] == "dep-001"


def test_ng4511_health_failure_rolls_back(exported_project, use_fake_target):
    p = exported_project
    fake = FakeSsh()
    use_fake_target(fake)
    start_deployment(p, target_name="ng4511-lab")
    fake.fail = "health_check"  # dep-002 unhealthy after activation
    bad = start_deployment(p, target_name="ng4511-lab")
    assert bad["state"] == "ROLLED_BACK" and bad["rollback"]["to"] == "dep-001"
    assert bad["health"]["details"]["attempts"] >= 2  # polled until the short timeout
    assert fake.current == "demo-dep-001" and fake.service_state == "active"
    assert p.pointers()["active_deployment"] == "dep-001"


def test_ng4511_policy_rejection(exported_project, use_fake_target):
    p = exported_project
    fake = FakeSsh(
        bench={
            "latency_ms_p50": 90.0,
            "latency_ms_p95": 120.0,
            "fps": 11.0,
            "memory_mb": 400.0,
            "gpu_util_pct": 99,
        }
    )
    use_fake_target(fake)
    rec = start_deployment(p, target_name="ng4511-lab")
    assert rec["state"] == "REJECTED"
    assert sorted(c["metric"] for c in rec["acceptance"]["checks"] if not c["passed"]) == [
        "fps",
        "latency_ms_p50",
    ]
    assert fake.current is None and fake.service_state == "inactive"


def test_ng4511_infer_and_setup_via_orchestrator(exported_project, use_fake_target, tmp_path):
    p = exported_project
    fake = FakeSsh()
    use_fake_target(fake)
    info = setup_target(p, "ng4511-lab")
    assert info["target"] == "ng4511-lab" and info["bundle"]["updated"] is True
    start_deployment(p, target_name="ng4511-lab")
    img = tmp_path / "x.jpg"
    img.write_bytes(b"jpg")
    res = infer_on_target(p, [img], target_name="ng4511-lab")
    assert res["total_detections"] == 1 and res["deployment_id"] == "dep-001"
    assert any(remote.startswith(f"{HOME}/camthink-ai/tmp/infer-") for _, remote in fake.uploads)
    assert any(c.startswith("rm -rf") and "/tmp/infer-" in c for c in fake.commands)  # cleanup
    with pytest.raises(OperationError) as ei:
        infer_on_target(p, [tmp_path / "missing.jpg"], target_name="ng4511-lab")
    assert ei.value.code == "IMAGE_NOT_FOUND"


def test_stage_renames_only_when_needed(tmp_path):
    fake = FakeSsh()
    t = _target(fake)
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"x")
    t.stage("dep-001", onnx, {"runtime_config": {"classes": ["p"]}})
    assert not any(c.startswith("mv ") for c in fake.commands)  # already model.onnx
    other = tmp_path / "best.onnx"
    other.write_bytes(b"y")
    t.stage("dep-002", other, {})
    assert any(c.startswith("mv ") and "best.onnx" in c for c in fake.commands)
    assert "model.onnx" in fake.deployments["dep-002"]


def test_stage_refuses_existing_deployment_dir():
    fake = FakeSsh()
    fake.deployments["dep-001"] = {"model.onnx"}
    t = _target(fake)
    with pytest.raises(OperationError) as ei:
        t.stage("dep-001", Path("/nonexistent.onnx"), {})
    assert ei.value.code == "TARGET_DEPLOYMENT_EXISTS"


# -- M7: service control (stop marker, no engine rebuild) ---------------------------------------


def test_service_stop_start_restart_commands_and_marker():
    fake = FakeSsh()
    fake.deployments["demo-dep-001"] = {"model.onnx", "model.engine"}
    fake.current, fake.service_state = "demo-dep-001", "active"
    t = _target(fake)

    st = t.service_status()
    assert st == {
        "service": "camthink-ai-inference",
        "systemd_state": "active",
        "current": "demo-dep-001",
        "stopped": None,
        "heartbeat": {"deployment": "demo-dep-001", "fps": 5.0, "frames": 42, "errors": 0},
    }

    res = t.service_stop()
    assert res["previous_state"] == "active" and res["state"] == "inactive"
    assert res["stopped"]["deployment"] == "demo-dep-001" and fake.stop_marker
    assert json.loads(fake.stop_marker)["by"] == "ai-pipeline service stop"
    assert any("systemctl --user stop camthink-ai-inference" in c for c in fake.commands)
    # health reports the deliberate stop; nothing was staged/built/re-pointed
    hc = t.health_check()
    assert hc["healthy"] is False and hc["details"]["stopped_intentionally"]["deployment"]
    assert fake.current == "demo-dep-001" and "model.engine" in fake.deployments["demo-dep-001"]
    assert not any("camthink_edge build" in c for c in fake.commands)
    assert t.service_status()["systemd_state"] == "inactive"
    assert t.service_status()["stopped"]["deployment"] == "demo-dep-001"

    res = t.service_start()
    assert res == {"previous_state": "inactive", "state": "active"}
    assert fake.stop_marker is None and fake.service_state == "active"
    assert t.health_check()["healthy"] is True

    n = len(fake.commands)
    res = t.service_restart()
    assert res["state"] == "active" and fake.stop_marker is None
    assert any("systemctl --user restart" in c for c in fake.commands[n:])
    assert not any("ln -sfn" in c or "camthink_edge build" in c for c in fake.commands[n:])


def test_activate_and_rollback_clear_stop_marker():
    fake = FakeSsh()
    fake.deployments["demo-dep-001"] = {"model.onnx", "model.engine"}
    fake.deployments["demo-dep-002"] = {"model.onnx", "model.engine"}
    fake.current, fake.service_state = "demo-dep-001", "active"
    t = _target(fake)
    t.service_stop()
    assert fake.stop_marker
    t.activate("demo-dep-002")
    assert fake.stop_marker is None and fake.current == "demo-dep-002"
    t.service_stop()
    t.rollback("demo-dep-001")
    assert fake.stop_marker is None and fake.current == "demo-dep-001"
    assert fake.service_state == "active"


def test_service_control_requires_service():
    t = _target(FakeSsh(), service_name=None)
    assert t.service_status()["service"] is None
    with pytest.raises(OperationError) as ei:
        t.service_stop()
    assert ei.value.code == "TARGET_NO_SERVICE"


def test_setup_installs_liveview_unit_without_enabling_it():
    fake = FakeSsh()
    t = _target(fake)
    info = t.setup_runtime()
    assert info["liveview"] == {
        "name": "camthink-ai-liveview",
        "unit": "~/.config/systemd/user/camthink-ai-liveview.service",
        "port": 8080,
        "enabled": False,
    }
    assert set(fake.units) == {"camthink-ai-inference.service", "camthink-ai-liveview.service"}
    assert (
        f"camthink_edge liveview --root {HOME}/camthink-ai"
        in fake.units["camthink-ai-liveview.service"]
    )
    assert (
        f"camthink_edge serve --root {HOME}/camthink-ai --fps-limit 5.0"
        in fake.units["camthink-ai-inference.service"]
    )
    assert not any("enable camthink-ai-liveview" in c for c in fake.commands)
    assert fake.liveview_state == "inactive"
    # no live view configured: unit not installed
    fake2 = FakeSsh()
    info = _target(fake2, liveview_service_name=None).setup_runtime()
    assert "liveview" not in info and set(fake2.units) == {"camthink-ai-inference.service"}


def test_app_config_round_trip_never_puts_the_url_on_a_command_line():
    fake = FakeSsh()
    t = _target(fake)
    assert t.app_config_read() is None
    cfg = {"schema_version": 1, "sources": [{"name": "cam0", "url": "rtsp://u:pw@cam/x"}]}
    t.app_config_write(cfg)
    assert json.loads(fake.app_json) == cfg
    assert not any("pw@" in c for c in fake.commands)
    assert t.app_config_path == f"{HOME}/camthink-ai/service/app.json"
    assert any(c.startswith("chmod 600 ") for c in fake.commands)
    assert t.app_config_read() == cfg
    fake.app_json = "{not json"
    with pytest.raises(OperationError) as ei:
        t.app_config_read()
    assert ei.value.code == "APP_CONFIG_INVALID"


def test_source_probe_uses_stdin_for_candidate_urls():
    fake = FakeSsh()
    t = _target(fake)
    res = t.source_probe(seconds=2, url="rtsp://u:pw@cam/x")
    assert res["ok"] and res["sources"][0]["url"] == "rtsp://u:pw@cam/x"
    assert fake.stdin_seen == ["rtsp://u:pw@cam/x\n"]
    assert not any("pw@" in c for c in fake.commands)
    assert any("probe --root" in c and "--source-stdin" in c for c in fake.commands)
    fake.app_json = json.dumps({"sources": [{"name": "cam0", "url": "rtsp://bad"}]})
    res = t.source_probe(seconds=2)
    assert res["ok"] is False


def test_liveview_control_and_url():
    fake = FakeSsh()
    t = _target(fake)
    assert t.liveview_url() == "http://192.168.5.222:8080/"
    assert _target(fake, liveview_host="jetson.lan", liveview_port=9000).liveview_url() == (
        "http://jetson.lan:9000/"
    )
    st = t.liveview_status()
    assert st["systemd_state"] == "inactive" and st["enabled"] == "disabled"
    res = t.liveview_start()
    assert res == {"previous_state": "inactive", "state": "active"}
    assert fake.liveview_enabled == "enabled"
    # the inference service was never touched
    assert not any(
        "camthink-ai-inference" in c and ("restart" in c or "stop" in c) for c in fake.commands
    )
    assert fake.service_state == "inactive"
    res = t.liveview_stop()
    assert res["state"] == "inactive" and fake.liveview_enabled == "disabled"
    assert t.liveview_restart()["state"] == "active"
    assert t.live_stats() is None
    fake.live_stats = json.dumps({"state": "running", "fps": 9.5})
    assert t.live_stats() == {"state": "running", "fps": 9.5}
    assert t.live_dir == "/dev/shm/camthink-ai-home_julian_camthink-ai"
    nolv = _target(FakeSsh(), liveview_service_name=None)
    assert nolv.liveview_status()["service"] is None
    with pytest.raises(OperationError) as ei:
        nolv.liveview_start()
    assert ei.value.code == "TARGET_NO_LIVEVIEW"
