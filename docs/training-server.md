# AI server setup (M4)

Based on the read-only inspection of the real machine (`scripts/inspect_ai_server.sh`,
2026-11-04). Facts below are measured, not assumed.

## Topology

```
Grok Bot / Human ──SSH──▶ ai-server ── ai-pipeline ── local ComputeBackend ── RTX 3080 (GPU 0)
                                   └── passwordless SSH (key, pinned host key) ──▶ NG4511 (M5)
```

The AI server is control host **and** compute host; the `local` backend is the production
backend. `ssh` stays a schema slot and is not implemented.

## The machine

| Item | Measured |
|---|---|
| Host / account | `ai-server`, user `redahk` (uid 1000, in `sudo`; `umask 0002`) |
| OS / kernel | Ubuntu 24.04.4 LTS, Linux 6.8.0-139-generic, x86_64 |
| CPU / RAM | Intel i7-14700K (28 threads), 125 GiB RAM, 8 GiB swap |
| GPUs | 3 × NVIDIA GeForce RTX 3080 **20 GB**, compute capability 8.6 |
| Driver / CUDA | driver 595.84 (open kernel module), supports CUDA ≤ 13.2; **no CUDA toolkit / nvcc on disk** (not needed: pip torch wheels bundle the CUDA runtime) |
| GPU usage | GPU 1 is busy (another process, `~/venvs/yolo`, ~10 GB, 100 %); GPUs 0 and 2 idle → the pipeline uses **GPU 0** |
| Disk | `/` 226 GB NVMe (135 GB free); **`/home` 7.3 TB (6.4 TB free)** → repo and artifacts live under `/home/redahk` |
| Python | system Python **3.12.3** with `venv` + `ensurepip`; no system `pip`, no `uv`/conda/pipx |
| Docker | not available to this user → not used |
| Tools | git 2.43, OpenSSH 9.6, scp, rsync 3.2.7, curl, wget, tmux, systemd-run, nohup |
| Internet | PyPI, download.pytorch.org, github.com reachable, no proxy |
| GitHub | no GitHub host key / credentials configured yet (clone needs auth, see Step 1) |
| NG4511 link | `~/.ssh/config` alias `camthink-test` → `julian@192.168.5.222`, key `~/.ssh/camthink_edge`, host key pinned in `known_hosts`; BatchMode login **works** |

Two gotchas observed:

- `import ultralytics` "succeeds" in the *system* Python when run from `/home/redahk`, although
  torch/numpy are missing there. That is almost certainly a directory named `ultralytics` in the
  home folder shadowing the package. The pipeline never runs Python from `~` (jobs run inside
  their job directory), but do not use `python3 -c "import ultralytics"` from `~` as a check.
- An existing `~/venvs/yolo` environment is in active use by another process. The pipeline gets
  its **own** venv and never touches that one.

## Environment decision

**System Python 3.12 + `python3 -m venv` + pip, one venv for both the CLI and the jobs.**
Torch comes from PyPI (the default Linux x86_64 wheels are CUDA 12.x builds, compatible with
driver 595) as a dependency of ultralytics; `onnx`/`onnxslim`/`onnxruntime` are pinned in the
`provider-ultralytics` extra so ultralytics never pip-installs anything at export time.
No `uv`, no conda, no Docker, no `sudo`, no CUDA toolkit install.

## Setup commands (run on ai-server as `redahk`)

```bash
# 1. clone (private repo: GitHub will prompt for a token over HTTPS)
cd ~
git clone https://github.com/HKJoker-Z/camthink-ai-pipeline.git
cd ~/camthink-ai-pipeline
git checkout cursor/m4-ai-server-a101        # until the M4 branch is merged to main

# 2. one venv for CLI + jobs (ControlHost == ComputeHost)
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,provider-ultralytics]"   # pulls torch (CUDA wheels), ultralytics, onnx*

# 3. prove CUDA is usable from this interpreter
python -c "import torch; print(torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
#   expected: 2.x.x+cu12x  cuda 12.x  True  3  ['NVIDIA GeForce RTX 3080', ...]
#   if it prints "+cpu" / None / False, stop and report — explicit-GPU training would fail
#   with GPU_DEVICE_UNAVAILABLE by design.

# 4. control plane sanity + unit tests (no GPU used)
ai-pipeline --version
ai-pipeline project validate demo_person_detection
pytest -q -m "not slow"

# 5. GPU integration test through the real job runner (downloads yolo11n.pt once)
AIP_TEST_DEVICE=0 AIP_ML_PYTHON=$PWD/.venv/bin/python pytest -q -m slow tests/integration
```

`infrastructure.yaml` needs **no change**: `compute_backends.local` without `python` uses the
CLI's own interpreter (the venv), `artifact_root: ./artifacts` resolves to
`/home/redahk/camthink-ai-pipeline/artifacts` on the 7.3 TB disk. Step 5 asserts from run
metadata that training ran on CUDA (`environment.device.policy == "explicit-gpu"`).

If `git clone` over HTTPS is inconvenient, the alternative is an SSH deploy key plus
`ssh-keyscan github.com >> ~/.ssh/known_hosts` (the inspection showed no GitHub host key).

## GPU policy (strict)

`training.device` in `project.yaml`:

| value | behaviour |
|---|---|
| `"0"`, `"0,2"`, `"cuda"`, `"cuda:2"` | explicit GPU: every index must exist **or the job fails** with `GPU_DEVICE_UNAVAILABLE` (details: requested, `cuda_available`, `device_count`, `device_names`, torch/CUDA versions). Never a silent CPU fallback. |
| `"cpu"` | CPU, always |
| `"auto"` | first GPU if available, else CPU (explicit opt-in to fallback) |

The demo project uses `"0"` (idle GPU; GPU 1 is occupied by another workload). Every run
records `environment.device = {requested, used, policy, cuda_available, device_count,
device_names}` plus `gpu`, `cuda`, `torch`, `ultralytics`; `run status -o json` exposes it.

## M4 acceptance workflow

```bash
cd ~/camthink-ai-pipeline && . .venv/bin/activate
bash scripts/m4_acceptance.sh demo_person_detection
```

CLI only, edits no configuration. Downloads the COCO128 sample (~7 MB, CC BY 4.0) into
`projects/demo_person_detection/data/coco128/`, then: `data import` → `label auto` (pretrained
`yolo11n.pt`, `person` only) → `dataset validate/build` → `train start --wait` on GPU 0
(20 epochs, 640 px; the detection head is re-initialised for the single-class list) → GPU proof
from run metadata → `evaluate run --wait` (`mAP50 ≥ 0.3`) → `model select` → `export start
--wait` (ONNX, opset 17) → `status`. Expected wall time on an RTX 3080: a few minutes, dominated
by weight downloads and the 20 epochs. Re-running is safe (imports dedupe, dataset build is
idempotent, runs get new ids). Envelopes of every step are kept in a temp dir printed at the
start; `ai-pipeline -o json run status demo_person_detection <run_id>` shows the full metadata.

A CPU dry run of the same script (device `cpu`, 10 epochs, 320 px) completed in the development
VM: eval mAP50 0.995 → ACCEPTED → `exp-001` (10.5 MB ONNX).

## Operating notes

- Jobs are detached subprocesses (`start_new_session`); the SSH session may close.
  `ai-pipeline run status <p> <run_id>` or `run list <p>` picks runs up later.
- Logs: `artifacts/<p>/runs/<run_id>/runner.log`; `ai-pipeline run logs <p> <run_id> --tail 50`.
- ultralytics downloads `yolo11n.pt` into the job's working directory on first use (5 MB per
  job that references a bare filename); point `weights` at an absolute path to share one copy.
- Nothing requires `sudo`; the user's existing `~/venvs/yolo` is left untouched.

## Facts gathered for M5 (NG4511) — do not act on them yet

| Item | Measured on `camthink-test` (192.168.5.222) |
|---|---|
| Device | CamThink NeoEdge **NG4511**, image `aibox-NG4511-36.4.3-super-jetpack-V1.1`, hostname `julian-camthink`, aarch64 |
| Software | L4T **R36.4.3**, **JetPack 6.2** (+b77), kernel 5.15.148-tegra, **TensorRT 10.3.0.30 + CUDA 12.5**, `trtexec` at `/usr/src/tensorrt/bin/trtexec`, Python 3.10.12, Docker 29.1.3, rsync present, systemd 249 |
| Resources | 233 GB NVMe (172 GB free), 7.4 GiB RAM (5.3 GiB available), power mode **25W** |
| Account | `julian` (uid 1000; groups incl. `sudo`, `video`, `render`, `gpio`), **passwordless sudo: no**; `/opt/camthink-ai` does not exist |
| Existing software | **`camthink-detect-web.service` is running ("CamThink Detect Web (YOLO11n on RTSP)")** — a vendor/demo inference service that must be inspected before deciding the M5 runtime and service-control strategy |
| Link | AI server → device SSH works non-interactively with key `camthink_edge`; host key pinned |

Implications to settle in M5 (not now): install root without sudo (`/home/julian/…` or a one-time
sudo for `/opt/camthink-ai`), whether to integrate with or replace `camthink-detect-web`,
`systemd --user` vs system service, `trtexec` FP16 build parameters for TensorRT 10.3, and
transfer via `scp` (baseline) or `rsync` (available on both ends).
