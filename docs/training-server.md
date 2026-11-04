# AI server setup (M4)

Status: **environment inspection pending.** The steps below are the baseline that the
pipeline needs; the environment-specific choices marked *decide after inspection* are filled
in once `scripts/inspect_ai_server.sh` has been run on the real machine.

## Topology

```
Grok Bot / Human ──SSH──▶ AI Server ── ai-pipeline ── local ComputeBackend ── GPU
                                   └── passwordless SSH ──▶ NG4511 (M5)
```

The AI server is control host **and** compute host. The `local` backend is the production
backend; `ssh` stays a schema slot for a possible future split and is not implemented.

## Step 0 — inspect (read-only)

Run as the account that will operate the pipeline, from any directory:

```bash
bash scripts/inspect_ai_server.sh <ng4511-ssh-alias> > ai-server-inspection.txt 2>&1
```

It installs/modifies nothing (write probes create and delete a temp file; the venv probe uses a
temp dir; SSH uses `BatchMode` + `StrictHostKeyChecking=yes`, so no keys are added). It
reports: OS/arch, CPU/RAM/disk and writable roots, `nvidia-smi`/driver/CUDA toolkit, every
Python interpreter with `venv`/`pip` availability and visible `torch`/`ultralytics`/`onnx`,
Docker (informational), git/ssh/scp/rsync/curl/systemd, internet reachability (PyPI, PyTorch
wheels, GitHub), git remote reachability, SSH config/keys, and the AI server → NG4511 link
(remote user, L4T/JetPack, TensorRT, `trtexec`, Python, rsync, disk, sudo, services, IPs).

## Step 1 — repository

```bash
git clone <repo-url> ~/camthink-ai-pipeline        # location: decide after inspection
cd ~/camthink-ai-pipeline
```

Everything the pipeline writes goes under `artifact_root` (`./artifacts` by default) and
`projects/<p>/data/`; both are gitignored. Put `artifact_root` on the largest writable disk
(`infrastructure.yaml` → `artifact_root: /data/camthink-ai/artifacts`, *decide after inspection*).

## Step 2 — Python environment (baseline: system Python + venv + pip)

No `uv`, conda or Docker is required. `uv` was only a convenience in the development VM.

```bash
python3 -m venv .venv                      # needs python3-venv on Debian/Ubuntu
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"                    # control plane + tests (no ML libraries)
```

One interpreter can serve both the CLI and the jobs (ControlHost == ComputeHost). Install the
ML extra into the **same** venv or a second one named in `compute_backends.local.python`:

```bash
# torch must match the driver's CUDA capability. Pick the wheel index after inspection:
#   driver >= 550 : --index-url https://download.pytorch.org/whl/cu124   (decide after inspection)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
pip install -e ".[provider-ultralytics]"   # ultralytics (+ onnx for export)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print a CUDA version and `True`; a `+cpu` wheel or `False` means explicit
GPU training will fail with `GPU_DEVICE_UNAVAILABLE` (by design — see below).

If the server already provides a working torch (system package, existing venv, module), point
`compute_backends.local.python` at that interpreter instead and only install
`-e ".[provider-ultralytics]"` into it. Docker is an option only if inspection shows no
usable host CUDA stack.

## Step 3 — infrastructure.yaml

```yaml
artifact_root: /data/camthink-ai/artifacts     # decide after inspection
compute_backends:
  local:
    type: local
    python: /home/<user>/camthink-ai-pipeline/.venv/bin/python   # interpreter with torch+ultralytics
deployment_targets:
  fake: {type: fake, root: ./.fake-target}
  ng4511-lab: {type: camthink_ng4511, ssh_host: ng4511, ...}     # M5
```

## Step 4 — GPU policy (strict)

`training.device` in `project.yaml`:

| value | behaviour |
|---|---|
| `"0"`, `"0,1"`, `"cuda"`, `"cuda:1"` | explicit GPU: every index must exist **or the job fails** with `GPU_DEVICE_UNAVAILABLE` (details: requested, `cuda_available`, `device_count`, `device_names`, torch/CUDA versions; hint: check `nvidia-smi` and the torch wheel). Never a silent CPU fallback. |
| `"cpu"` | CPU, always |
| `"auto"` | first GPU if available, else CPU (explicit opt-in to fallback) |

Production projects use an explicit GPU index. Every run records
`environment.device = {requested, used, policy, cuda_available, device_count, device_names}`
plus `environment.gpu`, `environment.cuda`, `environment.torch`, `environment.ultralytics`;
`run status -o json` exposes it, which is how the acceptance test proves GPU execution.

## Step 5 — verify

```bash
pytest -m "not slow"                                        # 183 unit tests, no GPU needed
AIP_TEST_DEVICE=0 AIP_ML_PYTHON=$PWD/.venv/bin/python pytest -m slow tests/integration
```

The slow test trains, evaluates, auto-labels and exports through the real provider and, with
`AIP_TEST_DEVICE=0`, asserts the run metadata proves CUDA execution.

## Step 6 — M4 acceptance workflow

```bash
bash scripts/m4_acceptance.sh demo_person_detection
```

CLI only; downloads the COCO128 sample (~7 MB, CC BY 4.0) into
`projects/demo_person_detection/data/coco128/`, then: `data import` → `label auto` (pretrained
`yolo11n.pt`, person only) → `dataset validate/build` → `train start --wait` (GPU) → GPU proof
from run metadata → `evaluate run --wait` → `model select` → `export start --wait` → `status`.
It edits no configuration; the demo project's `project.yaml` governs device (`"0"`), epochs
(20 — the detection head is re-initialised for the single-class list), thresholds
(`mAP50 ≥ 0.3`, `max_empty_label_ratio 0.7` because ~40 % of COCO128 images contain a person).
Re-running is safe: imports dedupe, the dataset build is idempotent, runs get new ids.

A CPU dry run of the same script (device `cpu`, 10 epochs, 320 px) completed in the development
VM: eval mAP50 0.995 → ACCEPTED → `exp-001` (10.5 MB ONNX).

## Operating notes

- Jobs are detached subprocesses (`start_new_session`); the SSH session that started
  `ai-pipeline train start` may close, `run status <p> <run_id>` picks the run up later.
- Logs: `artifacts/<p>/runs/<run_id>/runner.log`; `ai-pipeline run logs <p> <run_id> --tail 50`.
- Pretrained weights are downloaded by ultralytics on first use into its cache
  (`~/.config/Ultralytics`, `yolo11n.pt` in the CWD of the job); internet or a pre-seeded
  cache is required — *decide after inspection*.
- Nothing requires `sudo`.
