#!/usr/bin/env bash
# Read-only inspection of a prospective ai-pipeline AI server (M4).
#
# Collects the facts needed to decide the production Python environment and to validate GPU
# execution. It installs, removes and modifies NOTHING: every command only reads state.
# Network checks are HEAD/GET requests that download nothing persistent.
#
# Usage (run as the user that will operate ai-pipeline):
#   bash scripts/inspect_ai_server.sh [NG4511_SSH_HOST] > ai-server-inspection.txt 2>&1
#
#   NG4511_SSH_HOST  optional ssh alias/host for the edge device; connection is attempted with
#                    BatchMode and StrictHostKeyChecking=yes so nothing is written to
#                    known_hosts and no password prompt can appear.
#
# Environment knobs (optional): AIP_CANDIDATE_PYTHONS="python3 python3.10 /opt/x/bin/python"
set -u
NG_HOST="${1:-${NG4511_HOST:-}}"

section() { printf '\n===== %s =====\n' "$1"; }
run() {  # run <label> <cmd...> : print the command, then its output or "not available"
  printf -- '--- %s: ' "$1"; shift
  if command -v "$1" >/dev/null 2>&1 || [ -x "$1" ]; then
    printf '%s\n' "$*"; "$@" 2>&1 || printf '(exit %s)\n' "$?"
  else
    printf '%s (not available)\n' "$1"
  fi
}
readfile() { printf -- '--- %s\n' "$1"; if [ -r "$1" ]; then cat "$1"; else echo "(unreadable or missing)"; fi; }

section "WHEN / WHO / WHERE"
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "user=$(id -un) uid=$(id -u) groups=$(id -Gn)"
echo "hostname=$(hostname -f 2>/dev/null || hostname)"
echo "cwd=$PWD"
echo "shell=$SHELL"
echo "HOME=$HOME"
echo "umask=$(umask)"

section "OS"
readfile /etc/os-release
run "kernel" uname -a
run "arch" uname -m
run "lsb_release" lsb_release -a
run "uptime" uptime

section "CPU / MEMORY"
run "nproc" nproc
run "lscpu (head)" sh -c 'lscpu | head -20'
run "free" free -h

section "DISK"
run "df (common roots)" df -h / /home /data /opt /mnt /srv /var /tmp
run "home usage" sh -c 'du -sh "$HOME" 2>/dev/null'
echo "--- writable candidate roots (test = touch+rm in dir; nothing persists):"
for d in "$HOME" /data /opt/camthink-ai /srv /mnt /var/lib/camthink-ai; do
  if [ -d "$d" ]; then
    if touch "$d/.aip_write_probe.$$" 2>/dev/null; then rm -f "$d/.aip_write_probe.$$"; echo "  $d  writable"; else echo "  $d  NOT writable"; fi
  else
    echo "  $d  (missing)"
  fi
done
run "block devices" lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT

section "NVIDIA GPU / DRIVER"
run "nvidia-smi" nvidia-smi
run "nvidia-smi query" nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,compute_cap,pstate --format=csv
run "nvidia-smi -L" nvidia-smi -L
readfile /proc/driver/nvidia/version
run "lspci nvidia" sh -c 'lspci 2>/dev/null | grep -i -E "nvidia|3d controller" || echo "(lspci unavailable or no NVIDIA device)"'
run "nvcc" nvcc --version
echo "--- CUDA toolkits on disk:"; ls -d /usr/local/cuda* 2>/dev/null || echo "(none under /usr/local)"
echo "--- cudnn:"; ls /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/libcudnn* 2>/dev/null | head -5 || true
echo "--- nvidia-container-toolkit:"; command -v nvidia-ctk >/dev/null 2>&1 && nvidia-ctk --version 2>&1 | head -2 || echo "(not installed)"

section "PYTHON INTERPRETERS"
CANDIDATES="${AIP_CANDIDATE_PYTHONS:-python3 python3.10 python3.11 python3.12 python3.13 python /usr/bin/python3 /usr/local/bin/python3}"
for py in $CANDIDATES; do
  if command -v "$py" >/dev/null 2>&1; then
    printf -- '--- %s -> %s : ' "$py" "$(command -v "$py")"; "$py" --version 2>&1
    "$py" - <<'PYEOF' 2>&1 | sed 's/^/      /'
import sys, importlib.util, sysconfig
print("executable:", sys.executable)
print("version:", sys.version.split()[0], "| prefix:", sys.prefix, "| venv:", sys.prefix != sys.base_prefix)
for mod in ("venv", "ensurepip", "pip"):
    print(f"{mod}:", "yes" if importlib.util.find_spec(mod) else "NO")
print("site-packages:", sysconfig.get_paths().get("purelib"))
PYEOF
  fi
done
run "pip3" pip3 --version
run "uv" uv --version
run "pipx" pipx --version
run "conda" conda --version
echo "--- venv creation test (temp dir, removed immediately):"
T="$(mktemp -d 2>/dev/null || echo /tmp/aip_venv_probe.$$)"
if python3 -m venv "$T/v" >/dev/null 2>&1; then echo "  python3 -m venv: OK ($("$T/v/bin/python" --version 2>&1))"; else echo "  python3 -m venv: FAILED (python3-venv package missing?)"; fi
rm -rf "$T"

section "ML LIBRARIES VISIBLE TO EACH INTERPRETER (import only, nothing installed)"
for py in $CANDIDATES; do
  command -v "$py" >/dev/null 2>&1 || continue
  printf -- '--- %s\n' "$py"
  "$py" - <<'PYEOF' 2>&1 | sed 's/^/      /'
import importlib
def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        return f"missing ({type(e).__name__})"
for name in ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "tensorrt", "cv2", "numpy", "PIL", "pydantic", "typer", "yaml"):
    print(f"{name:12s} {ver(name)}")
try:
    import torch
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count():", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i} {torch.cuda.get_device_name(i)} cc={p.major}.{p.minor} mem={p.total_memory // 2**20} MiB")
        print("cudnn:", torch.backends.cudnn.version())
except Exception as e:  # noqa: BLE001
    print("torch probe:", type(e).__name__, e)
PYEOF
done

section "DOCKER (informational only)"
run "docker" docker --version
run "docker info (gpu runtimes)" sh -c 'docker info 2>/dev/null | grep -i -E "runtimes|nvidia|server version" || echo "(docker daemon not reachable for this user)"'
echo "--- docker group membership: $(id -Gn | tr ' ' '\n' | grep -x docker || echo no)"

section "TOOLS"
run "git" git --version
run "ssh" ssh -V
run "scp" sh -c 'command -v scp'
run "rsync" rsync --version
run "curl" curl --version
run "wget" wget --version
run "tmux" tmux -V
run "systemctl --user" systemctl --user --version
run "systemd-run" sh -c 'command -v systemd-run'
run "nohup" sh -c 'command -v nohup'

section "NETWORK / INTERNET (HEAD requests only)"
for url in https://pypi.org/simple/ultralytics/ https://download.pytorch.org/whl/ https://github.com/ultralytics/assets/releases https://github.com/; do
  if command -v curl >/dev/null 2>&1; then
    code=$(curl -sS -o /dev/null -I -m 10 -w '%{http_code}' "$url" 2>&1); echo "  $url -> HTTP $code"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --spider -T 10 "$url" && echo "  $url -> reachable" || echo "  $url -> NOT reachable"
  else
    echo "  (neither curl nor wget available)"; break
  fi
done
echo "--- proxy env:"; env | grep -i -E '^(http|https|no)_proxy=' || echo "(none)"
echo "--- DNS:"; readfile /etc/resolv.conf | head -5

section "REPOSITORY"
if git -C "$PWD" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "cwd is inside a git repo: $(git -C "$PWD" rev-parse --show-toplevel)"
  run "remote" git -C "$PWD" remote -v
  run "branch/commit" git -C "$PWD" log --oneline -1
  run "status (short)" git -C "$PWD" status --short --branch
  echo "--- can we reach origin? (read-only ls-remote):"; git -C "$PWD" ls-remote --heads origin 2>&1 | head -5 || true
else
  echo "cwd is not a git checkout (clone target still to be decided)"
fi
echo "--- GitHub SSH auth (read-only):"; ssh -T -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes git@github.com 2>&1 | head -2 || true

section "SSH CLIENT CONFIG"
readfile "$HOME/.ssh/config"
echo "--- keys present (names only):"; ls -la "$HOME/.ssh" 2>/dev/null | awk '{print "  "$1, $9}' || echo "(no ~/.ssh)"
echo "--- known_hosts entries: $(wc -l < "$HOME/.ssh/known_hosts" 2>/dev/null || echo 0)"
echo "--- ssh-agent: ${SSH_AUTH_SOCK:+present} $(ssh-add -l 2>&1 | head -3)"

section "AI SERVER -> NG4511"
if [ -n "$NG_HOST" ]; then
  echo "target: $NG_HOST"
  echo "--- resolved config (ssh -G):"; ssh -G "$NG_HOST" 2>/dev/null | grep -E '^(hostname|user|port|identityfile|stricthostkeychecking|userknownhostsfile) ' || true
  echo "--- known_hosts match: "; ssh-keygen -F "$(ssh -G "$NG_HOST" 2>/dev/null | awk '/^hostname /{print $2}')" 2>/dev/null | head -2 || echo "  (no pinned host key)"
  echo "--- connection test (BatchMode, no host-key additions, read-only remote commands):"
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes "$NG_HOST" '
    echo "remote user=$(id -un) uid=$(id -u) groups=$(id -Gn)"; hostname; uname -a
    cat /etc/nv_tegra_release 2>/dev/null || echo "(no /etc/nv_tegra_release)"
    (dpkg-query -W nvidia-jetpack 2>/dev/null || apt-cache policy nvidia-jetpack 2>/dev/null | head -3 || echo "(jetpack version unknown)")
    dpkg-query -W -f="tensorrt \${Version}\n" tensorrt 2>/dev/null || dpkg -l 2>/dev/null | grep -i -E "^ii +(nvidia-tensorrt|tensorrt|libnvinfer[0-9]*) " | head -3 || echo "(tensorrt version unknown)"
    ls /usr/src/tensorrt/bin/trtexec 2>/dev/null || command -v trtexec 2>/dev/null || echo "(trtexec not found)"
    python3 --version 2>&1; command -v rsync >/dev/null && echo "rsync: yes" || echo "rsync: no"
    df -h / 2>/dev/null | tail -1; free -h 2>/dev/null | head -2
    sudo -n true 2>/dev/null && echo "passwordless sudo: yes" || echo "passwordless sudo: no"
    ls -ld /opt/camthink-ai 2>/dev/null || echo "/opt/camthink-ai: missing"
    systemctl --version 2>/dev/null | head -1; systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null | grep -i -E "camthink|neo|deepstream|triton|inference" || echo "(no obvious inference services running)"
    command -v docker >/dev/null && docker --version || echo "docker: no"
    ip -4 -o addr show scope global 2>/dev/null | awk "{print \$2, \$4}"
    nvpmodel -q 2>/dev/null | head -3 || echo "(nvpmodel unavailable)"
  ' 2>&1 | sed 's/^/  /'
  echo "  (ssh exit: ${PIPESTATUS[0]})"
else
  echo "no NG4511 host given — rerun with: bash scripts/inspect_ai_server.sh <ng4511-ssh-alias>"
fi

section "DONE"
echo "Nothing was installed or modified. Share this output to continue M4."
