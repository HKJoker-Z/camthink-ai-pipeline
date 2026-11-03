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

**Status:** M0 (skeleton) — project creation, configuration validation, state store, CLI and
JSON output. Data, training, export and deployment stages land in later milestones. See
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
ai-pipeline status my_detector                  # pointers, counts, next valid operations
ai-pipeline -o json status my_detector          # machine-readable envelope
```

`infrastructure.yaml` (repo root) names the compute backends and deployment targets that
projects refer to. Generated state lives under `artifacts/` (gitignored).

## Documentation

- [docs/architecture.md](docs/architecture.md) — layers, seams, where code runs
- [docs/configuration.md](docs/configuration.md) — `project.yaml` and `infrastructure.yaml`
- [docs/cli.md](docs/cli.md) — commands, JSON envelope, exit codes
- [docs/design/v1-architecture-proposal.md](docs/design/v1-architecture-proposal.md) — full design, milestones, open items

## Development

```bash
ruff check src tests && ruff format --check src tests
pytest                      # unit tests; no GPU, server or device needed
pytest -m "not slow"        # (slow/hardware markers are reserved for later milestones)
```
