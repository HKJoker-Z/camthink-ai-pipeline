"""System resource sampler for Jetson (and, degraded, any Linux box).

Order of preference:

1. ``jtop`` (jetson-stats service; the user must be in the ``jtop`` group): CPU/GPU load,
   GPU clock, RAM, temperatures, **power rails** and nvpmodel — no root needed.
2. ``tegrastats`` output parsing (RAM, per-core CPU, GR3D load, temperatures; power rails only
   when it runs as root).
3. ``/proc/stat`` + ``/proc/meminfo`` (CPU and RAM only).

``SystemMonitor.snapshot()`` returns the latest sample; ``summary()`` aggregates everything
sampled since ``reset()`` (mean/max) for benchmark reports.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import Any


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _cpu_times() -> tuple[int, int] | None:
    line = _read("/proc/stat")
    if not line:
        return None
    parts = line.splitlines()[0].split()
    vals = [int(v) for v in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def _meminfo() -> dict[str, float]:
    out: dict[str, float] = {}
    txt = _read("/proc/meminfo") or ""
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            out[k] = int(v.split()[0]) / 1024.0
    if out:
        return {
            "ram_total_mb": round(out.get("MemTotal", 0), 0),
            "ram_used_mb": round(out.get("MemTotal", 0) - out.get("MemAvailable", 0), 0),
        }
    return {}


def _gpu_freq_mhz() -> float | None:
    import glob

    for p in glob.glob("/sys/devices/gpu.0/devfreq/*/cur_freq") + glob.glob(
        "/sys/class/devfreq/*gpu*/cur_freq"
    ):
        v = _read(p)
        if v and v.isdigit():
            return int(v) / 1e6
    return None


_GPU_LOAD_PATHS = (
    "/sys/devices/platform/gpu.0/load",  # JetPack 6 (Orin)
    "/sys/devices/gpu.0/load",  # JetPack 4/5
    "/sys/devices/platform/17000000.gpu/load",
)


def _gpu_load_pct() -> float | None:
    """GPU load in tenths of a percent from sysfs - readable without root on Jetson."""
    for path in _GPU_LOAD_PATHS:
        v = _read(path)
        if v and v.isdigit():
            return round(int(v) / 10.0, 1)
    return None


def nvpmodel_mode() -> str | None:
    exe = shutil.which("nvpmodel") or "/usr/sbin/nvpmodel"
    try:
        out = subprocess.run([exe, "-q"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"NV Power Mode:\s*(\S+)", out)
    return m.group(1) if m else None


class SystemMonitor:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._samples: list[dict[str, Any]] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self.backend = "proc"
        self.power_mode = nvpmodel_mode()
        self._prev_cpu = _cpu_times()
        self._tegra: subprocess.Popen | None = None

    # -- lifecycle ---------------------------------------------------------------------------

    def start(self) -> SystemMonitor:
        self._thread = threading.Thread(target=self._run, name="sysmon", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        if self._tegra and self._tegra.poll() is None:
            self._tegra.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def reset(self) -> None:
        with self._lock:
            self._samples = []

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
        out: dict[str, Any] = {
            "samples": len(samples),
            "backend": self.backend,
            "power_mode": self.power_mode,
        }
        if not samples:
            return out

        def agg(key: str, fn, nd=1):
            vals = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
            return round(fn(vals), nd) if vals else None

        for key in (
            "cpu_pct",
            "gpu_pct",
            "ram_used_mb",
            "power_total_mw",
            "power_cpu_gpu_cv_mw",
            "gpu_mhz",
        ):
            out[key + "_mean"] = agg(key, lambda v: sum(v) / len(v))
            out[key + "_max"] = agg(key, max)
        out["gpu_mhz_min"] = agg("gpu_mhz", min)
        for key in ("temp_cpu_c", "temp_gpu_c", "temp_tj_c"):
            out[key + "_max"] = agg(key, max)
        out["ram_total_mb"] = samples[-1].get("ram_total_mb")
        # Thermal/power capping shows as the GPU clock sitting below its maximum while busy.
        gpu_max = next((s["gpu_max_mhz"] for s in samples if s.get("gpu_max_mhz")), None)
        if gpu_max:
            busy = [s for s in samples if (s.get("gpu_pct") or 0) >= 80 and s.get("gpu_mhz")]
            capped = [s for s in busy if s["gpu_mhz"] < 0.9 * gpu_max]
            out["gpu_max_mhz"] = gpu_max
            out["gpu_busy_samples"] = len(busy)
            out["gpu_clock_capped_samples"] = len(capped)
        out["temp_tj_c_max"] = agg("temp_tj_c", max)
        return out

    # -- sampling ----------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            from jtop import jtop  # type: ignore

            with jtop(interval=self.interval_s) as j:
                if j.ok():
                    self.backend = "jtop"
                    while not self._stop_evt.is_set() and j.ok():
                        self._push(self._from_jtop(j))
                        time.sleep(self.interval_s)
                    return
        except Exception:  # noqa: BLE001 - jtop missing or service down: fall back
            pass
        if shutil.which("tegrastats"):
            try:
                self._run_tegrastats()
                return
            except Exception:  # noqa: BLE001
                pass
        self.backend = "proc"
        while not self._stop_evt.is_set():
            self._push(self._from_proc())
            time.sleep(self.interval_s)

    def _push(self, sample: dict[str, Any]) -> None:
        sample["ts"] = time.time()
        with self._lock:
            self._latest = sample
            self._samples.append(sample)
            if len(self._samples) > 100_000:
                self._samples = self._samples[-50_000:]

    def _from_proc(self) -> dict[str, Any]:
        s: dict[str, Any] = {}
        cur = _cpu_times()
        if cur and self._prev_cpu:
            dt, didle = cur[0] - self._prev_cpu[0], cur[1] - self._prev_cpu[1]
            if dt > 0:
                s["cpu_pct"] = round(100.0 * (1 - didle / dt), 1)
        self._prev_cpu = cur
        s.update(_meminfo())
        f = _gpu_freq_mhz()
        if f:
            s["gpu_mhz"] = f
        g = _gpu_load_pct()
        if g is not None:
            s["gpu_pct"] = g
        return s

    def _from_jtop(self, j) -> dict[str, Any]:
        s = self._from_proc()  # cpu % from /proc is cheap and consistent across backends
        try:
            cpus = j.cpu.get("cpu", []) if isinstance(j.cpu, dict) else []
            loads = [c.get("user", 0) + c.get("system", 0) for c in cpus if c.get("online", True)]
            if loads:
                s["cpu_pct_jtop"] = round(sum(loads) / len(loads), 1)
            gpu = next(iter(j.gpu.values())) if j.gpu else {}
            s["gpu_pct"] = round(float(gpu.get("status", {}).get("load", 0.0)), 1)
            freq = gpu.get("freq", {})
            if freq.get("cur"):
                s["gpu_mhz"] = round(freq["cur"] / 1000.0, 0)
            mem = j.memory.get("RAM", {}) if isinstance(j.memory, dict) else {}
            if mem.get("tot"):
                s["ram_total_mb"] = round(mem["tot"] / 1024.0, 0)
                s["ram_used_mb"] = round(mem.get("used", 0) / 1024.0, 0)
            temps = j.temperature or {}
            for key, name in (("temp_cpu_c", "CPU"), ("temp_gpu_c", "GPU"), ("temp_tj_c", "tj")):
                t = temps.get(name, {})
                if isinstance(t, dict) and t.get("online", True):
                    s[key] = round(float(t.get("temp", 0)), 1)
            power = j.power or {}
            tot = power.get("tot", {})
            if tot.get("power") is not None:
                s["power_total_mw"] = int(tot["power"])
            rails = power.get("rail", {})
            for rail, key in (
                ("VDD_CPU_GPU_CV", "power_cpu_gpu_cv_mw"),
                ("VDD_SOC", "power_soc_mw"),
            ):
                if rail in rails and rails[rail].get("power") is not None:
                    s[key] = int(rails[rail]["power"])
            if freq.get("max"):
                s["gpu_max_mhz"] = round(freq["max"] / 1000.0, 0)
            if self.power_mode is None and getattr(j, "nvpmodel", None):
                self.power_mode = str(j.nvpmodel)
        except Exception as exc:  # noqa: BLE001 - jtop field layouts differ between versions
            s["sysmon_error"] = f"{type(exc).__name__}: {exc}"
        return s

    _TS_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    _TS_CPU = re.compile(r"CPU \[([^\]]+)\]")
    _TS_GPU = re.compile(r"GR3D_FREQ (\d+)%(?:@(\d+))?")
    _TS_TEMP = re.compile(r"\b(CPU|GPU|tj)@([\d.]+)C", re.IGNORECASE)  # JetPack 6 lowercases
    _TS_POWER = re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC) (\d+)mW")

    def _run_tegrastats(self) -> None:
        self._tegra = subprocess.Popen(
            ["tegrastats", "--interval", str(int(self.interval_s * 1000))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.backend = "tegrastats"
        assert self._tegra.stdout is not None
        for line in self._tegra.stdout:
            if self._stop_evt.is_set():
                break
            s = self._from_proc()
            m = self._TS_RAM.search(line)
            if m:
                s["ram_used_mb"], s["ram_total_mb"] = int(m.group(1)), int(m.group(2))
            m = self._TS_CPU.search(line)
            if m:
                cores = [c for c in m.group(1).split(",") if "%" in c]
                vals = [int(c.split("%")[0]) for c in cores if c.split("%")[0].isdigit()]
                if vals:
                    s["cpu_pct_tegrastats"] = round(sum(vals) / len(vals), 1)
            m = self._TS_GPU.search(line)
            if m:
                s["gpu_pct"] = int(m.group(1))
            for name, val in self._TS_TEMP.findall(line):
                key = {"cpu": "temp_cpu_c", "gpu": "temp_gpu_c", "tj": "temp_tj_c"}[name.lower()]
                s[key] = float(val)
            for rail, mw in self._TS_POWER.findall(line):
                key = {
                    "VDD_IN": "power_total_mw",
                    "VDD_CPU_GPU_CV": "power_cpu_gpu_cv_mw",
                    "VDD_SOC": "power_soc_mw",
                }[rail]
                s[key] = int(mw)
            self._push(s)
