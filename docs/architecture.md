# Architecture

This document describes what exists. The full design, rationale and roadmap are in
[design/v1-architecture-proposal.md](design/v1-architecture-proposal.md).

## Topology (V1)

```
Human / automation agent
        ↓
    AI Server   — ai-pipeline CLI, compute backend (local), store, artifacts
        ↓  passwordless SSH
    NG4511      — model runtime, inference, benchmark, health check, rollback
```

The AI server is both control host and compute host. The roles stay separate in code
(`ComputeBackend` seam) so they can be split later without touching orchestration.

## Layers

| Layer | Package | Contains |
|---|---|---|
| C — Projects | `projects/<name>/project.yaml` | configuration only: classes, candidates, thresholds, backend/target names |
| B — Task modules | `ai_pipeline.tasks.*` | task vocabulary and rules (label formats, metric names; dataset validation from M1) |
| A — Generic core | `ai_pipeline.core.*` | config, project, store, state, ids, logging, errors, status |
| seams | `ai_pipeline.providers`, `.backends`, `.targets` | model-specific, compute-location-specific, device-specific code |

Only `providers` names exist in M0; `backends` and `targets` are added in M2/M3.

## Core modules (M0)

| Module | Responsibility |
|---|---|
| `core.config.schema` | pydantic models for `project.yaml` and `infrastructure.yaml`; unknown keys rejected |
| `core.config.loader` | YAML load, `${VAR}` substitution, pydantic → `ConfigError`, semantic cross-checks |
| `core.settings` | resolve infrastructure file, projects root, artifact root (CLI > env > infra > default) |
| `core.project` | `Project`: paths, config, pointers; `create` from template; `load` with validation |
| `core.store` | one JSON file per entity under `artifacts/<project>/<kind>/`, atomic writes |
| `core.state` | per-entity state enums and allowed-transition tables |
| `core.status` | derived project summary and `next_operations` (never persisted) |
| `core.ids` | time-based ids for events (runs, deployments), sequential ids for versions |
| `core.logging` | context-bound logging to stderr (+ optional per-operation file) |
| `core.errors` | `PipelineError` hierarchy with `code`, `details`, `hint`, exit code |
| `cli.*` | typer commands; `run_command` turns exceptions into envelopes + exit codes |

## Filesystem layout at runtime

```
projects/<name>/project.yaml        committed
projects/<name>/data/               gitignored: raw images, labels
artifacts/<name>/project.json       pointers (created lazily if missing)
artifacts/<name>/{imports,labels,datasets,runs,models,deployments}/<id>.json
artifacts/<name>/logs/
```

The artifact root is configurable (`artifact_root` in `infrastructure.yaml`,
`AIP_ARTIFACT_ROOT`, or `--artifact-root`). Every reference stored in metadata is relative to
it, so the directory can be moved or synced as a unit.

## State model

There is no single project state. Each entity kind has its own machine
(`core.state`); illegal transitions raise `StateError` (exit 3). Every mutating command writes
an in-progress record first and a terminal record (`…_COMPLETE`/`FAILED` with error details)
last, so a crash never leaves an ambiguous record.

```
Run         CREATED → SUBMITTED → RUNNING → COMPLETED | FAILED | CANCELLED
Dataset     BUILDING → READY | FAILED
Model       TRAINED → EVALUATED → ACCEPTED | REJECTED → EXPORTED → BENCHMARKED → DEPLOYED → SUPERSEDED
Deployment  PENDING → UPLOADED → PREPARED → SMOKE_PASSED → BENCHMARKED → ACTIVE | FAILED | ROLLED_BACK
```
