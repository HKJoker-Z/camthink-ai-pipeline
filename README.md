# camthink-ai-pipeline

Reusable, automation-friendly AI model lifecycle pipeline for CamThink edge AI devices.
First target: CamThink NeoEdge NG4511 (NVIDIA Jetson Orin Nano 8GB). Training runs on an
NVIDIA GPU AI server, which also hosts the pipeline itself.

```
Human / automation agent
        ↓
    AI Server   — ai-pipeline CLI, training jobs, state store, model artifacts
        ↓  passwordless SSH
    NG4511      — model runtime, inference, benchmark, health check, rollback
```

The pipeline is generic: projects are configuration (`projects/<name>/project.yaml`), the
core knows about datasets, runs, models, deployments — never about specific object classes.

**Status:** M3 — project creation and validation (M0); immutable content-addressed assets,
canonical annotations, deterministic dataset versions (M1); jobs executed by a compute backend
through a model provider, run tracking, auto-labeling, evaluation with project-defined
acceptance, model registry (M2); ONNX export as a job, first-class deployments with benchmark
policy, activation, automatic rollback and a fake target (M3); validated on the real AI server
with GPU training, evaluation and ONNX export (M4, see
[docs/training-server.md](docs/training-server.md)); own TensorRT inference runtime deployed,
benchmarked (7.7 ms p50 / 121 fps FP16 on the Orin Nano) and rolled back on the real NG4511
(M5, see [docs/ng4511-deployment.md](docs/ng4511-deployment.md)). The full V1 lifecycle is
proven end to end. See
[`docs/design/v1-architecture-proposal.md`](docs/design/v1-architecture-proposal.md).

## Install

Requires Python 3.10+.

```bash
python -m venv .venv && . .venv/bin/activate     # or: uv venv && uv pip install ...
pip install -e ".[dev]"
cp .env.example .env                            # optional; only for ${VAR} references
ai-pipeline --help
```

## Quick start

```bash
ai-pipeline project create my_detector          # scaffolds projects/my_detector/project.yaml
$EDITOR projects/my_detector/project.yaml       # classes, candidates, thresholds, target
ai-pipeline project validate my_detector
ai-pipeline data import my_detector ./images    # or omit the path to use dataset.sources
ai-pipeline label import my_detector ./labels   # YOLO txt, matched by filename stem
ai-pipeline dataset validate my_detector        # task-owned report, writes nothing
ai-pipeline dataset build my_detector           # immutable ds-v001; idempotent on content
ai-pipeline dataset show my_detector ds-v001
ai-pipeline train start my_detector --wait      # job on the compute backend -> mv-001 (TRAINED)
ai-pipeline evaluate run my_detector mv-001 --wait   # metrics + acceptance -> ACCEPTED/REJECTED
ai-pipeline model select my_detector            # best accepted by evaluation.selection
ai-pipeline label auto my_detector --wait       # pretrained model -> new label set (kind: model)
ai-pipeline export start my_detector --wait     # export job -> exp-001 (ONNX)
ai-pipeline target setup my_detector            # install the edge runtime + user service on the device
ai-pipeline deploy start my_detector            # stage/TensorRT build/smoke/benchmark/policy/activate/health
ai-pipeline deploy infer my_detector photo.jpg  # real inference on the device
ai-pipeline health my_detector                  # target health vs active deployment
ai-pipeline rollback my_detector                # restore previous known-good deployment
ai-pipeline status my_detector                  # pointers, counts, next valid operations
ai-pipeline -o json status my_detector          # machine-readable envelope
```

`infrastructure.yaml` (repo root) names the compute backends and deployment targets that
projects refer to. Generated state lives under `artifacts/` (gitignored).

## Documentation

- [docs/architecture.md](docs/architecture.md) — layers, seams, where code runs
- [docs/configuration.md](docs/configuration.md) — `project.yaml` and `infrastructure.yaml`
- [docs/cli.md](docs/cli.md) — commands, JSON envelope, exit codes
- [docs/datasets.md](docs/datasets.md) — asset store, annotation schema, validation rules, split and hash
- [docs/runs-and-models.md](docs/runs-and-models.md) — jobs, backends, providers, runs, registry, evaluation policy
- [docs/deployment.md](docs/deployment.md) — exports, deployment targets, deployment state machine, rollback
- [docs/training-server.md](docs/training-server.md) — AI server inspection, setup, strict GPU policy, M4 acceptance
- [docs/ng4511-deployment.md](docs/ng4511-deployment.md) — edge runtime (`camthink_edge`), NG4511 target, device layout, service
- [docs/operator-guide.md](docs/operator-guide.md) — commands, workflow, JSON/exit codes, monitoring, failures, new projects, approvals
- [docs/design/v1-architecture-proposal.md](docs/design/v1-architecture-proposal.md) — full design, milestones, open items

## Development

```bash
ruff check src tests && ruff format --check src tests
pytest                      # unit tests; no GPU, server or device needed (fake provider)
# optional: real ultralytics provider through the job runner, on CPU
python -m venv /tmp/venv-ml && /tmp/venv-ml/bin/pip install -e ".[provider-ultralytics]"
AIP_ML_PYTHON=/tmp/venv-ml/bin/python pytest -m slow tests/integration
```

The CLI never imports ML libraries. Jobs run in the interpreter named by
`compute_backends.<name>.python` in `infrastructure.yaml`; install `.[provider-ultralytics]`
there.
