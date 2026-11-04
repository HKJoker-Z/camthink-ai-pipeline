#!/usr/bin/env bash
# Read-only inspection of the NG4511 for the M5 edge runtime. Run FROM THE AI SERVER:
#
#   bash scripts/inspect_ng4511.sh camthink-test > ng4511-inspection.txt 2>&1
#
# Connects with BatchMode + StrictHostKeyChecking=yes (nothing is added to known_hosts) and
# only reads state on the device. The single write probe creates and removes a temp file in
# $HOME to confirm the user-owned install root is writable. Nothing is installed or changed.
set -u
HOST="${1:?usage: inspect_ng4511.sh <ssh-alias>}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes "$HOST" --)

"${SSH[@]}" 'bash -s' <<'REMOTE'
section() { printf '\n===== %s =====\n' "$1"; }
run() { printf -- '--- %s\n' "$*"; "$@" 2>&1 || printf '(exit %s)\n' "$?"; }

section "IDENTITY"
date -u +"%Y-%m-%dT%H:%M:%SZ"; echo "user=$(id -un) uid=$(id -u) groups=$(id -Gn)"; hostname; uname -a
cat /etc/nv_tegra_release 2>/dev/null | head -1
grep -E "Product_Name|Software_Version|Image_Name" /etc/nv_tegra_release 2>/dev/null
echo "HOME=$HOME  shell=$SHELL  umask=$(umask)  XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-unset}"
echo "linger: $(loginctl show-user "$(id -un)" -p Linger 2>/dev/null || echo unknown)"
echo "sudo -n: $(sudo -n true 2>/dev/null && echo passwordless || echo 'password required')"

section "RESOURCES"
run free -h
run df -h / /home "$HOME"
run nproc
cat /sys/devices/soc0/family 2>/dev/null; cat /sys/devices/soc0/machine 2>/dev/null
run nvpmodel -q
echo "--- tegrastats (one sample, 1s):"; (timeout 3 tegrastats --interval 1000 2>/dev/null | head -1) || echo "(tegrastats unavailable)"
echo "--- jetson_clocks: $(command -v jetson_clocks || echo missing)"

section "WRITABLE INSTALL ROOT"
for d in "$HOME" "$HOME/camthink-ai" /opt/camthink-ai; do
  if [ -d "$d" ]; then
    if touch "$d/.aip_probe.$$" 2>/dev/null; then rm -f "$d/.aip_probe.$$"; echo "  $d  writable"; else echo "  $d  NOT writable"; fi
  else echo "  $d  (missing)"; fi
done
echo "--- ~/.config/systemd/user: $(ls -d "$HOME/.config/systemd/user" 2>/dev/null || echo missing)"

section "TENSORRT / CUDA"
dpkg-query -W -f='${Package} ${Version}\n' nvidia-jetpack tensorrt libnvinfer10 libnvinfer-bin python3-libnvinfer python3-libnvinfer-dev cuda-runtime-12-6 cuda-cudart-12-6 cuda-cudart-12-5 2>/dev/null
ls -la /usr/src/tensorrt/bin/trtexec 2>/dev/null || echo "trtexec missing"
/usr/src/tensorrt/bin/trtexec --version 2>&1 | tail -2
echo "--- libcudart:"; ls /usr/local/cuda*/lib64/libcudart.so* /usr/lib/aarch64-linux-gnu/libcudart.so* 2>/dev/null || echo "(no libcudart found)"
echo "--- libnvinfer:"; ls /usr/lib/aarch64-linux-gnu/libnvinfer.so* 2>/dev/null | head -3
ls -d /usr/local/cuda* 2>/dev/null
nvcc --version 2>/dev/null | tail -1 || echo "nvcc: missing (fine)"

section "PYTHON (system python3 is what the runtime would use)"
command -v python3; python3 --version
python3 - <<'PYEOF'
import importlib, sys
print("executable:", sys.executable)
for name in ("tensorrt", "numpy", "cv2", "PIL", "cuda", "pycuda", "torch", "onnx", "onnxruntime", "yaml", "psutil"):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "present")
        extra = ""
        if name == "cv2":
            info = m.getBuildInformation()
            extra = " | GStreamer: " + ("YES" if "GStreamer:                   YES" in info else "no") + " | CUDA: " + ("YES" if "NVIDIA CUDA:                   YES" in info else "no")
        print(f"  {name:12s} {v}{extra}")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:12s} missing ({type(e).__name__})")
try:
    import ctypes, ctypes.util, glob
    cands = [ctypes.util.find_library("cudart")] + sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
    print("  cudart via ctypes candidates:", [c for c in cands if c])
    for c in cands:
        if c:
            lib = ctypes.CDLL(c); v = ctypes.c_int(); lib.cudaRuntimeGetVersion(ctypes.byref(v)); print("  cudaRuntimeGetVersion:", v.value, "via", c); break
except Exception as e:  # noqa: BLE001
    print("  ctypes cudart probe failed:", type(e).__name__, e)
try:
    import tensorrt as trt
    print("  tensorrt python:", trt.__version__, "| execute_async_v3:", hasattr(trt.IExecutionContext, "execute_async_v3"))
except Exception as e:  # noqa: BLE001
    print("  tensorrt python import failed:", type(e).__name__, e)
PYEOF
echo "--- pip / venv:"; python3 -m pip --version 2>&1 | head -1; python3 -c "import venv, ensurepip; print('venv+ensurepip: yes')" 2>&1
echo "--- user site: $(python3 -m site --user-site 2>/dev/null)"

section "SERVICES / PORTS (to avoid conflicts; vendor service is left untouched)"
systemctl --no-pager --no-legend list-units --type=service --state=running 2>/dev/null | grep -i -E "camthink|detect|deepstream|triton|inference" || echo "(none matching)"
systemctl cat camthink-detect-web.service 2>/dev/null | grep -E "ExecStart|User|WorkingDirectory|Environment" | head -6
echo "--- listening TCP ports:"; (ss -ltnp 2>/dev/null || netstat -ltn 2>/dev/null) | head -15
echo "--- systemd --user reachable: $(XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user is-system-running 2>&1 | head -1)"
echo "--- /run/user/$(id -u): $(ls -d /run/user/$(id -u) 2>/dev/null || echo missing)"

section "GPU PROCESSES (nvidia-smi is not available on Jetson; use tegrastats above)"
ps -eo pid,user,rss,pcpu,comm,args --sort=-rss 2>/dev/null | head -8

section "NETWORK"
ip -4 -o addr show scope global 2>/dev/null | awk '{print "  "$2, $4}'
echo "--- internet (HEAD):"; curl -sS -o /dev/null -I -m 8 -w '  pypi.org -> %{http_code}\n' https://pypi.org/simple/numpy/ 2>&1 || echo "  (curl failed)"

section "DONE (read-only)"
REMOTE
echo "ssh exit: $?"
