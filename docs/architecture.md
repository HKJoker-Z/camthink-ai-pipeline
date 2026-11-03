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

## Data modules (M1) — details in [datasets.md](datasets.md)

| Module | Responsibility |
|---|---|
| `core.data.assets` | immutable content-addressed `AssetStore` (SHA-256 ids, read-only files, sidecar metadata) |
| `core.data.importer` | `data import`: walk sources, probe images, add to the store, record the import event |
| `core.labeling.annotations` | canonical `Annotation` model (normalised bbox, class name, provenance), JSONL IO |
| `core.labeling.labelsets` | `label import`: match label files to assets by stem, convert via the task module, store a label set |
| `core.dataset.split` | deterministic hash-ranked split with largest-remainder sizes |
| `core.dataset.hashing` | `content_hash` over canonical dataset content (`dataset-hash-v1`) |
| `core.dataset.validation` | task-agnostic `ValidationReport` / `Issue` / `Severity` |
| `core.dataset.build` | plan (validate + hash + lookup) and immutable materialisation of a version |
| `tasks.object_detection.yolo` | YOLO txt ↔ canonical conversion |
| `tasks.object_detection.validation` | detection-specific validation rules (ERROR vs WARNING) |

The task module is the only place that knows YOLO; core never reads or writes label files
directly.

## Execution modules (M2) — details in [runs-and-models.md](runs-and-models.md)

| Module | Responsibility |
|---|---|
| `core.jobs.spec` | `JobSpec` / `JobStatus` / `JobResult` file schemas (the only contract across the compute boundary) |
| `core.jobs.runner` | `python -m ai_pipeline.core.jobs.runner <job_dir>`: loads the provider, executes, writes status/result |
| `core.jobs.context` | what a provider sees: job dir, resolve/relpath helpers, `report_progress` |
| `core.runs` | run records; `start_training/evaluation/autolabel`, `submit`, `sync`, `wait`, `cancel`; post-completion hooks |
| `core.registry` | model versions: `register_from_run`, `apply_evaluation`, `select_model`, state transitions |
| `core.evaluation.policy` | `apply_rules(metrics, rules)` — project-configured acceptance, no built-in policy |
| `backends.local` | detached subprocess backend; dead-runner detection; cancel |
| `providers.fake` | deterministic provider for tests |
| `providers.ultralytics` | YOLO train/val/predict → task metric names; imported only inside the runner |

## Export and deployment modules (M3) — details in [deployment.md](deployment.md)

| Module | Responsibility |
|---|---|
| `core.exports` | `start_export` (export job, request-identity reuse), `export_from_run` hook → immutable `exp-NNN` |
| `core.deployment` | deployment entity + phase machine, benchmark policy, activation, automatic/manual rollback, recovery, health |
| `targets.base` | `DeploymentTarget` protocol (verify, stage, prepare, smoke_test, benchmark, activate, health_check, rollback) |
| `targets.fake` | directory-based target with atomic `current` symlink and `inject.json` failure injection |

Seam status: `ComputeBackend` has `local` (production for V1, where control host == compute
host) and a schema slot for `ssh` (M4). `DeploymentTarget` has `fake`; `camthink_ng4511` is a
schema slot until M5.

## Filesystem layout at runtime

```
projects/<name>/project.yaml        committed
projects/<name>/data/               gitignored: raw images, labels
artifacts/<name>/project.json       pointers (created lazily if missing)
artifacts/<name>/assets/            content-addressed immutable images + sidecars
artifacts/<name>/{imports,labels,datasets,runs,models,deployments}/<id>.json
artifacts/<name>/labels/<id>/annotations.jsonl
artifacts/<name>/datasets/<ds_version>/     immutable materialised dataset
artifacts/<name>/runs/<run_id>/             job dir: job.json, status.json, runner.log, out/
artifacts/<name>/logs/
.fake-target/                               fake device: deployments/<dep>/, current -> …
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
Model       TRAINED → EVALUATED → ACCEPTED | REJECTED          (quality only; re-evaluable)
Export      READY | FAILED                                     (immutable artifact)
Deployment  CREATED → VERIFYING → STAGING → PREPARING → SMOKE_TESTING → BENCHMARKING
            → READY_TO_ACTIVATE → ACTIVATING → HEALTH_CHECKING → ACTIVE
            pre-activation error → FAILED; policy → REJECTED;
            post-activation error → ROLLING_BACK → ROLLED_BACK | ROLLBACK_FAILED;
            ACTIVE ↔ SUPERSEDED
```
