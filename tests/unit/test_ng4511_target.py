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
        self.commands: list[str] = []
        self.uploads: list[tuple[list[str], str]] = []
        self.unit_installed = False

    # -- SshHost API -------------------------------------------------------------------

    def run(self, command: str, *, timeout: float = 120, check: bool = True) -> CommandResult:
        self.commands.append(command)
        out, rc = self._handle(command)
        if check and rc != 0:
            raise RemoteError(
                f"exit {rc}", code="REMOTE_COMMAND_FAILED", details={"command": command}
            )
        return CommandResult(rc, out, "")

    def run_json(self, command: str, *, timeout: float = 120) -> dict:
        res = self.run(command, timeout=timeout, check=False)
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

    def download(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    # -- command handling ----------------------------------------------------------------------

    def _handle(self, c: str) -> tuple[str, int]:
        if c == "echo $HOME":
            return HOME + "\n", 0
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
            return self._edge(c)
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

    def _edge(self, c: str) -> tuple[str, int]:
        args = shlex.split(c.split("-m camthink_edge ", 1)[1])
        cmd = args[0]
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
