"""CamThink NeoEdge NG4511 deployment target (Jetson Orin Nano, JetPack 6.x).

Everything happens over the operator's SSH alias with the ``camthink_edge`` runtime bundle
installed under a **user-owned** root (default ``~/camthink-ai``) — no sudo, and nothing touches
the vendor's own inference service.

Device layout::

    ~/camthink-ai/
      runtime/camthink_edge/        the runtime bundle (uploaded by `target setup`)
      deployments/<dep>/            model.onnx  runtime.json  deployment.json  samples/
                                    model.engine  model.engine.build.log  (built on device)
      current -> deployments/<dep>  atomic symlink swap
      service/status.json           heartbeat written by the inference service
      ~/.config/systemd/user/<service>.service   `python3 -m camthink_edge serve`
"""

from __future__ import annotations

import json
import shlex
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from ai_pipeline.core.config.schema import CamThinkNG4511TargetConfig
from ai_pipeline.core.errors import OperationError
from ai_pipeline.core.ids import utc_now_iso
from ai_pipeline.core.logging import get_logger
from ai_pipeline.core.remote import SshHost

log = get_logger("target.ng4511")

SERVICE_UNIT = """[Unit]
Description=CamThink AI inference runtime (ai-pipeline)
After=network.target

[Service]
Type=simple
Environment=PYTHONPATH={root}/runtime
Environment=PYTHONUNBUFFERED=1
ExecStart={python} -m camthink_edge serve --root {root} --fps-limit {fps}{source}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def bundle_dir() -> Path:
    return Path(str(resources.files("ai_pipeline.edge_bundle") / "camthink_edge"))


def bundle_version() -> str:
    text = (bundle_dir() / "__init__.py").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("VERSION"):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


class CamThinkNG4511Target:
    type = "camthink_ng4511"

    def __init__(
        self, *, name: str, cfg: CamThinkNG4511TargetConfig, host: SshHost | None = None
    ) -> None:
        self.name = name
        self.cfg = cfg
        self.host = host or SshHost(cfg.ssh_host, connect_timeout=cfg.connect_timeout_s)
        self._root: str | None = None

    # -- helpers -------------------------------------------------------------------------

    @property
    def root(self) -> str:
        if self._root is None:
            raw = self.cfg.install_root
            if raw.startswith("~"):
                home = self.host.run("echo $HOME", timeout=30).stdout.strip()
                raw = home + raw[1:]
            self._root = raw.rstrip("/")
        return self._root

    def _dep(self, dep: str) -> str:
        return f"{self.root}/deployments/{dep}"

    def _edge(self, *args: str) -> str:
        return (
            f"PYTHONPATH={shlex.quote(self.root + '/runtime')} {self.cfg.python} -m camthink_edge "
            + " ".join(shlex.quote(a) for a in args)
        )

    def _systemctl(self, *args: str) -> str:
        return 'XDG_RUNTIME_DIR="/run/user/$(id -u)" systemctl --user ' + " ".join(
            shlex.quote(a) for a in args
        )

    def _service_cmd(self, action: str) -> str | None:
        if not self.cfg.service_name:
            return None
        return self._systemctl(action, self.cfg.service_name)

    @property
    def _stop_marker(self) -> str:
        """Written by `service_stop`, removed by anything that (re)starts the service, so a
        deliberate stop is distinguishable from a crash."""
        return f"{self.root}/service/stopped.json"

    def _clear_stop_marker_cmd(self) -> str:
        return f"rm -f {shlex.quote(self._stop_marker)}"

    def _remote_runtime_version(self) -> str | None:
        res = self.host.run(
            f"cat {shlex.quote(self.root)}/runtime/camthink_edge/__init__.py 2>/dev/null | grep '^VERSION'",
            timeout=30,
            check=False,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        return res.stdout.strip().split("=", 1)[1].strip().strip('"')

    def _ensure_runtime(self) -> dict[str, Any]:
        """Upload the runtime bundle if missing or a different version (idempotent)."""
        local = bundle_version()
        remote = self._remote_runtime_version()
        if remote == local:
            return {"runtime_version": local, "updated": False}
        rt = f"{self.root}/runtime"
        self.host.run(
            f"mkdir -p {shlex.quote(rt + '.new')} {shlex.quote(self.root)}/deployments {shlex.quote(self.root)}/service {shlex.quote(self.root)}/logs",
            timeout=30,
        )
        self.host.upload(bundle_dir(), rt + ".new/", recursive=True)
        self.host.run(
            f"rm -rf {shlex.quote(rt + '.old')} && (test -e {shlex.quote(rt)} && mv {shlex.quote(rt)} {shlex.quote(rt + '.old')} || true) "
            f"&& mv {shlex.quote(rt + '.new')} {shlex.quote(rt)} && rm -rf {shlex.quote(rt + '.old')} "
            f"&& find {shlex.quote(rt)} -name __pycache__ -type d -exec rm -rf {{}} +",
            timeout=60,
        )
        log.bind(target=self.name).info("runtime bundle %s installed (was %s)", local, remote)
        return {"runtime_version": local, "updated": True, "previous": remote}

    # -- one-off setup ------------------------------------------------------------------------

    def setup_runtime(self) -> dict[str, Any]:
        """Install/update the runtime bundle and the systemd --user unit. Safe to repeat."""
        info: dict[str, Any] = {"root": self.root, "bundle": self._ensure_runtime(), "warnings": []}
        deps = self.host.run_json(self._edge("info"), timeout=120)
        info["device"] = deps
        for mod in ("numpy", "tensorrt"):
            if str(deps.get(mod, "")).startswith("missing"):
                info["warnings"].append(f"python module {mod} missing on device")
        if str(deps.get("cv2", "")).startswith("missing") and str(deps.get("PIL", "")).startswith(
            "missing"
        ):
            info["warnings"].append("neither OpenCV nor Pillow available for image IO")
        if self.cfg.service_name:
            unit_dir = "$HOME/.config/systemd/user"
            source = (
                f" --source {shlex.quote(self.cfg.inference_source)}"
                if self.cfg.inference_source
                else ""
            )
            unit = SERVICE_UNIT.format(
                root=self.root, python=self.cfg.python, fps=self.cfg.fps_limit, source=source
            )
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / f"{self.cfg.service_name}.service"
                p.write_text(unit, encoding="utf-8")
                self.host.run(f"mkdir -p {unit_dir}", timeout=30)
                self.host.upload(p, f"~/.config/systemd/user/{self.cfg.service_name}.service")
            self.host.run(self._systemctl("daemon-reload"), timeout=60)
            self.host.run(self._systemctl("enable", self.cfg.service_name), timeout=60)
            linger = self.host.run(
                'loginctl show-user "$(id -un)" -p Linger 2>/dev/null || echo Linger=unknown',
                timeout=30,
                check=False,
            ).stdout.strip()
            info["service"] = {
                "name": self.cfg.service_name,
                "unit": f"~/.config/systemd/user/{self.cfg.service_name}.service",
                "linger": linger,
            }
            if "Linger=yes" not in linger:
                info["warnings"].append(
                    "user lingering is not enabled: the inference service stops when the user's last "
                    "session ends. One-time fix on the device: `loginctl enable-linger $(id -un)`"
                )
        return info

    # -- DeploymentTarget protocol -------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        root = shlex.quote(self.root)
        trtexec = shlex.quote(self.cfg.trtexec)
        cmd = (
            f"echo machine=$(uname -m); echo python=$({self.cfg.python} -c 'import platform;print(platform.python_version())' 2>/dev/null); "
            "echo l4t=$(head -1 /etc/nv_tegra_release 2>/dev/null); "
            f"echo trtexec=$(test -x {trtexec} && echo yes || echo no); "
            f"echo root_exists=$(test -d {root} && echo yes || echo no); "
            f"echo current=$(readlink {root}/current 2>/dev/null | xargs -r basename); "
            f"echo staged=$(ls {root}/deployments 2>/dev/null | tr '\\n' ','); "
            "echo disk_free_mb=$(df -Pm \"$HOME\" | tail -1 | awk '{print $4}'); "
            "echo mem_total_mb=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo); "
            "echo mem_avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"
        )
        raw = self.host.run(cmd, timeout=60).stdout
        info = dict(line.split("=", 1) for line in raw.strip().splitlines() if "=" in line)
        remote_version = self._remote_runtime_version()
        env: dict[str, Any] = {
            "device": "CamThink NG4511",
            "machine": info.get("machine"),
            "l4t": info.get("l4t") or None,
            "python": info.get("python") or None,
            "trtexec": self.cfg.trtexec if info.get("trtexec") == "yes" else None,
            "runtime_version": remote_version,
            "runtime_version_local": bundle_version(),
            "mem_total_mb": _int(info.get("mem_total_mb")),
            "mem_available_mb": _int(info.get("mem_avail_mb")),
            "disk_free_mb": _int(info.get("disk_free_mb")),
        }
        if remote_version:
            try:
                dev = self.host.run_json(self._edge("info"), timeout=120)
                env.update(
                    {
                        k: v
                        for k, v in dev.items()
                        if k in ("tensorrt", "numpy", "cv2", "PIL", "cudart_version")
                    }
                )
            except OperationError as exc:  # runtime present but broken: report, do not hide
                env["runtime_info_error"] = exc.message
        if env["trtexec"] is None:
            raise OperationError(
                f"trtexec not found at {self.cfg.trtexec} on {self.cfg.ssh_host}",
                code="TARGET_TRTEXEC_MISSING",
                details={"trtexec": self.cfg.trtexec},
                hint="Set deployment_targets.<name>.trtexec in infrastructure.yaml.",
            )
        staged = [x for x in (info.get("staged") or "").split(",") if x]
        return {
            "reachable": True,
            "environment": env,
            "current": info.get("current") or None,
            "staged": staged,
            "root": self.root,
            "service": self._service_state(),
        }

    def _service_state(self) -> str | None:
        cmd = self._service_cmd("is-active")
        if cmd is None:
            return None
        return self.host.run(cmd, timeout=30, check=False).stdout.strip() or "unknown"

    def stage(self, deployment_id: str, artifact: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        self._ensure_runtime()
        dep = self._dep(deployment_id)
        exists = (
            self.host.run(f"test -e {shlex.quote(dep)}", timeout=30, check=False).returncode == 0
        )
        if exists:
            raise OperationError(
                f"deployment directory {dep} already exists on {self.cfg.ssh_host}",
                code="TARGET_DEPLOYMENT_EXISTS",
                details={"path": dep},
                hint="Deployment ids are never reused; remove the stale directory on the device first.",
            )
        self.host.run(f"mkdir -p {shlex.quote(dep + '/samples')}", timeout=30)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            (tdp / "runtime.json").write_text(
                json.dumps(metadata.get("runtime_config") or {}, indent=2), encoding="utf-8"
            )
            meta = {k: v for k, v in metadata.items() if k not in ("runtime_config", "samples")}
            (tdp / "deployment.json").write_text(
                json.dumps({**meta, "staged_at": utc_now_iso()}, indent=2), encoding="utf-8"
            )
            self.host.upload(
                [artifact, tdp / "runtime.json", tdp / "deployment.json"], dep + "/", timeout=900
            )
            if Path(artifact).name != "model.onnx":
                self.host.run(
                    f"mv {shlex.quote(dep + '/' + Path(artifact).name)} {shlex.quote(dep + '/model.onnx')}",
                    timeout=30,
                )
            samples = [Path(s) for s in metadata.get("samples") or [] if Path(s).is_file()]
            if samples:
                self.host.upload(samples, dep + "/samples/", timeout=300)
        out = self.host.run(
            f"sha256sum {shlex.quote(dep + '/model.onnx')} && stat -c %s {shlex.quote(dep + '/model.onnx')}",
            timeout=60,
        ).stdout.split()
        return {
            "path": dep + "/model.onnx",
            "sha256": out[0],
            "size_bytes": int(out[-1]),
            "samples": len(samples),
        }

    def prepare(
        self, deployment_id: str, *, precision: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        dep = self._dep(deployment_id)
        res = self.host.run_json(
            self._edge(
                "build",
                "--onnx",
                f"{dep}/model.onnx",
                "--engine",
                f"{dep}/model.engine",
                "--precision",
                precision,
                "--trtexec",
                self.cfg.trtexec,
                "--workspace-mb",
                str(self.cfg.workspace_mb),
                "--timeout-s",
                str(self.cfg.build_timeout_s),
                "--reuse-scan",
                f"{self.root}/deployments",
            ),
            timeout=self.cfg.build_timeout_s + 120,
        )
        return {
            "engine_path": res["engine_path"],
            "engine_sha256": res["engine_sha256"],
            "engine_size_bytes": res.get("engine_size_bytes"),
            "build_params": res["build_params"],
            "environment": res.get("environment", {}),
            "duration_s": res.get("duration_s"),
            "builder_peak_rss_mb": res.get("builder_peak_rss_mb"),
            "reused_from": res.get("reused_from"),
        }

    def smoke_test(self, deployment_id: str) -> dict[str, Any]:
        dep = self._dep(deployment_id)
        min_det = str(self._smoke_min_detections)
        return self.host.run_json(
            self._edge(
                "smoke",
                "--engine",
                f"{dep}/model.engine",
                "--config",
                f"{dep}/runtime.json",
                "--min-detections",
                min_det,
                f"{dep}/samples",
            ),
            timeout=600,
        )

    def benchmark(self, deployment_id: str) -> dict[str, Any]:
        dep = self._dep(deployment_id)
        first = self.host.run(
            f"ls {shlex.quote(dep + '/samples')} 2>/dev/null | head -1", timeout=30, check=False
        ).stdout.strip()
        args = [
            "benchmark",
            "--engine",
            f"{dep}/model.engine",
            "--config",
            f"{dep}/runtime.json",
            "--iterations",
            str(self._bench_iterations),
            "--warmup",
            str(self._bench_warmup),
        ]
        if first:
            args += ["--image", f"{dep}/samples/{first}"]
        return self.host.run_json(self._edge(*args), timeout=900)

    def activate(self, deployment_id: str) -> dict[str, Any]:
        previous = self.current()
        root = shlex.quote(self.root)
        self.host.run(
            f"test -f {shlex.quote(self._dep(deployment_id) + '/model.engine')} && "
            f"ln -sfn deployments/{shlex.quote(deployment_id)} {root}/current.tmp && mv -T {root}/current.tmp {root}/current",
            timeout=30,
        )
        restarted = False
        cmd = self._service_cmd("restart")
        if cmd:
            self.host.run(f"{self._clear_stop_marker_cmd()}; {cmd}", timeout=120)
            restarted = True
        return {
            "previous": previous,
            "current": deployment_id,
            "service_restarted": restarted,
            "at": utc_now_iso(),
        }

    def health_check(self) -> dict[str, Any]:
        args = ["health", "--root", self.root]
        if self.cfg.service_name:
            args += ["--service", self.cfg.service_name]
        result = self.host.run_json(self._edge(*args), timeout=300)
        details = result.setdefault("details", {})
        if not result.get("healthy") and details.get("service_state") not in (None, "active"):
            marker = self._read_stop_marker()
            if marker is not None:
                details["stopped_intentionally"] = marker
        return result

    # -- service control (no engine rebuild, no pointer change) -----------------------------------

    def _read_stop_marker(self) -> dict[str, Any] | None:
        res = self.host.run(
            f"cat {shlex.quote(self._stop_marker)} 2>/dev/null", timeout=30, check=False
        )
        if res.returncode != 0 or not res.stdout.strip():
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError:
            return {"raw": res.stdout.strip()[:200]}

    def service_status(self) -> dict[str, Any]:
        """Service state, device ``current``, stop marker and heartbeat in one round trip."""
        root = shlex.quote(self.root)
        is_active = self._service_cmd("is-active")
        # Files may lack a trailing newline, hence the `echo` after each `cat`.
        cmd = (
            f"echo; echo '__STATE__'; {is_active + ' 2>/dev/null' if is_active else 'echo none'}; "
            f"echo; echo '__CURRENT__'; readlink {root}/current 2>/dev/null; "
            f"echo; echo '__STOPPED__'; cat {shlex.quote(self._stop_marker)} 2>/dev/null; "
            f"echo; echo '__HEARTBEAT__'; cat {root}/service/status.json 2>/dev/null; true"
        )
        out = self.host.run(cmd, timeout=60, check=False).stdout
        sections: dict[str, str] = {}
        key = None
        for line in out.splitlines():
            if line.startswith("__") and line.endswith("__"):
                key = line.strip("_").lower()
                sections[key] = ""
            elif key:
                sections[key] += line + "\n"

        def _json(text: str) -> dict[str, Any] | None:
            text = text.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text[:200]}

        link = sections.get("current", "").strip()
        state = sections.get("state", "").strip() or "unknown"
        return {
            "service": self.cfg.service_name,
            "systemd_state": state if self.cfg.service_name else None,
            "current": link.rsplit("/", 1)[-1] if link else None,
            "stopped": _json(sections.get("stopped", "")),
            "heartbeat": _json(sections.get("heartbeat", "")),
        }

    def _require_service(self) -> str:
        if not self.cfg.service_name:
            raise OperationError(
                f"target {self.name} has no inference service configured (service_name: null)",
                code="TARGET_NO_SERVICE",
                hint="Set deployment_targets.<name>.service_name in infrastructure.yaml and run "
                "`target setup`.",
            )
        return self.cfg.service_name

    def service_stop(self) -> dict[str, Any]:
        self._require_service()
        before = self._service_state()
        marker = {
            "deployment": self.current(),
            "at": utc_now_iso(),
            "by": "ai-pipeline service stop",
            "previous_state": before,
        }
        self.host.run(
            f"mkdir -p {shlex.quote(self.root + '/service')} && "
            f"printf '%s' {shlex.quote(json.dumps(marker))} > {shlex.quote(self._stop_marker)} && "
            f"{self._service_cmd('stop')}",
            timeout=120,
        )
        return {"previous_state": before, "state": self._service_state(), "stopped": marker}

    def service_start(self) -> dict[str, Any]:
        self._require_service()
        before = self._service_state()
        self.host.run(
            f"{self._clear_stop_marker_cmd()} && {self._service_cmd('start')}", timeout=120
        )
        return {"previous_state": before, "state": self._service_state()}

    def service_restart(self) -> dict[str, Any]:
        self._require_service()
        before = self._service_state()
        self.host.run(
            f"{self._clear_stop_marker_cmd()} && {self._service_cmd('restart')}", timeout=120
        )
        return {"previous_state": before, "state": self._service_state()}

    def current(self) -> str | None:
        res = self.host.run(
            f"readlink {shlex.quote(self.root + '/current')} 2>/dev/null || true",
            timeout=30,
            check=False,
        )
        link = res.stdout.strip()
        return link.rsplit("/", 1)[-1] if link else None

    def rollback(self, to_deployment_id: str | None) -> dict[str, Any]:
        previous = self.current()
        root = shlex.quote(self.root)
        if to_deployment_id is None:
            self.host.run(f"rm -f {root}/current", timeout=30)
            cmd = self._service_cmd("stop")
        else:
            self.host.run(
                f"test -f {shlex.quote(self._dep(to_deployment_id) + '/model.engine')} && "
                f"ln -sfn deployments/{shlex.quote(to_deployment_id)} {root}/current.tmp && mv -T {root}/current.tmp {root}/current",
                timeout=30,
            )
            cmd = self._service_cmd("restart")
            if cmd:
                cmd = f"{self._clear_stop_marker_cmd()}; {cmd}"
        if cmd:
            self.host.run(cmd, timeout=120, check=to_deployment_id is not None)
        return {"previous": previous, "current": to_deployment_id, "at": utc_now_iso()}

    # -- extras ------------------------------------------------------------------------------------

    def infer(self, images: list[Path], *, deployment_id: str | None = None) -> dict[str, Any]:
        """Run the deployed detector on local images (uploaded to a temp dir on the device)."""
        dep = self._dep(deployment_id) if deployment_id else f"{self.root}/current"
        tmp = f"{self.root}/tmp/infer-{utc_now_iso().replace(':', '').replace('.', '')}"
        self.host.run(f"mkdir -p {shlex.quote(tmp)}", timeout=30)
        try:
            self.host.upload(list(images), tmp + "/", timeout=600)
            return self.host.run_json(
                self._edge(
                    "infer",
                    "--engine",
                    f"{dep}/model.engine",
                    "--config",
                    f"{dep}/runtime.json",
                    tmp,
                ),
                timeout=600,
            )
        finally:
            self.host.run(f"rm -rf {shlex.quote(tmp)}", timeout=30, check=False)

    # knobs the orchestrator sets from project config before the phases run
    _smoke_min_detections: int = 1
    _bench_iterations: int = 100
    _bench_warmup: int = 20

    def configure(
        self, *, smoke_min_detections: int, bench_iterations: int, bench_warmup: int
    ) -> None:
        self._smoke_min_detections = smoke_min_detections
        self._bench_iterations = bench_iterations
        self._bench_warmup = bench_warmup
