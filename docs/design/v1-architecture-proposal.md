# camthink-ai-pipeline — V1 Architecture Proposal

Status: **ACCEPTED for M0 (2026-11-03) with the topology decision below.**

## Review decisions (2026-11-03)

- **Topology:** `ControlHost == ComputeHost == AI Server` for V1. The AI server
  runs the `ai-pipeline` CLI, training jobs, the state/metadata store, the model
  artifacts, and orchestrates deployment to the NG4511 over passwordless SSH
  (confirmed working). The NG4511 is only a deployment/inference target.
- **No separate control-host machine** is introduced for V1.
- The `ComputeBackend` and `DeploymentTarget` seams stay intact; the production
  `infrastructure.yaml` simply uses the `local` compute backend. The `ssh`
  backend remains a supported type for a future split.
- Additional M4/M5 verification items were added (§25, items 20–25).
- **M3 review (2026-11-03):** export is *not* a ModelVersion state. The model lifecycle is
  `TRAINED → EVALUATED → ACCEPTED | REJECTED` only; exports are immutable `ExportArtifact`
  entities produced by compute jobs, deployments are first-class entities with their own
  phase machine, and the TensorRT engine is target-owned (recorded on the Deployment). This
  supersedes the `EXPORTED/BENCHMARKED/DEPLOYED/SUPERSEDED` model states in §11.1/§12 and
  §16 of the brief. See `docs/deployment.md`.
- **M4 redefined (2026-11-04):** M4 = *real AI-server validation* with the `local` compute
  backend (ControlHost == ComputeHost). The `ssh` backend stays schema/design-only and is not
  implemented unless a separate compute machine appears. Explicit GPU requests must fail
  (`GPU_DEVICE_UNAVAILABLE`) instead of falling back to CPU. Deployment stays synchronous for
  V1; `scp` is the M5 transfer baseline; deployment pruning is deferred to an explicit
  `deploy prune` with protections. See `docs/training-server.md`.

```
Human / future Grok Bot
        ↓
    AI Server   — ai-pipeline CLI, compute backend (local), store, artifacts
        ↓  passwordless SSH
    NG4511      — model/runtime, inference, benchmark, health check, rollback
```

This document proposes the V1 design of a reusable, automation-friendly AI model
lifecycle pipeline for CamThink edge devices (first target: NeoEdge NG4511 /
Jetson Orin Nano 8GB), with training on a separate NVIDIA GPU server.

Every section that makes a design choice is structured as
*why needed → alternatives → V1 choice*. Open questions are collected in
[§25 Unknowns](#25-unknowns-that-must-be-verified-before-implementation).
Anything marked **ASSUMPTION** is a working assumption that review should
confirm or correct.

---

## 0. Current state (inspection results)

**Repository.** Empty apart from a one-line `README.md`. No code, config, CI, or
`.gitignore`. Nothing constrains the design.

**Development environment (this Cursor VM).**

| Item | Finding |
|---|---|
| OS / arch | Linux, x86_64 |
| Python | 3.12.3, pip 24.0, `numpy`, `PyYAML` only |
| GPU / CUDA | none (`nvidia-smi` absent) |
| Docker | not installed |
| SSH client / rsync | `ssh` present; `rsync` **not** installed |
| Credentials | no env vars, no `~/.ssh/config` entries for an AI server or NG4511 |

Implications:

- The Cursor VM can run unit tests and fake-backend integration tests only.
  Real training, export verification, and device work must happen on the AI
  server / NG4511 via a *control host* that can reach both (see §19, §25).
- Cursor is a development tool only; nothing in the design depends on it.

**AI server and NG4511.** Nothing is known from inside this repository. All
hardware/software facts are listed as verification items in §25. No decision
below is irreversible with respect to those unknowns.

---

## 1. Design goals and non-goals

Goals (V1):

1. Prove one object-detection lifecycle end to end, reproducibly, through a CLI.
2. Generic core: no application classes (dog, helmet, …) in source.
3. Every operation callable non-interactively, with `--output json`, meaningful
   exit codes, and persisted state an automation agent can read.
4. Local testability without server, device, or camera.
5. Versioned, rollback-able deployments on the device.

Non-goals (V1): see §27 "Explicitly postponed".

---

## 2. Proposed architecture

### 2.1 Three layers (as requested) plus three "plugin seams"

```
Layer C  projects/<name>/project.yaml        configuration only
            │ references by name
Layer B  ai_pipeline.tasks.object_detection  task semantics (labels, metrics, validation rules)
            │
Layer A  ai_pipeline.core                    generic lifecycle, state, CLI, store, remote exec
            │ uses via small protocols
         ai_pipeline.providers.ultralytics   model-specific train/eval/export/predict
         ai_pipeline.backends.{local,ssh}    where compute jobs run
         ai_pipeline.targets.{fake,ng4511}   where models are deployed
```

The three seams — **ModelProvider**, **ComputeBackend**, **DeploymentTarget** —
are the only abstractions introduced. Each solves a concrete separation problem
the brief calls out (YOLO spread, remote training, NG4511 spread). Nothing else
is abstracted up front.

### 2.2 Modular monolith, single Python package

- *Why:* one process, one install, easy to debug and run by hand.
- *Alternatives:* services + queue (premature); separate packages for
  core/providers/targets (packaging overhead for no benefit yet).
- *V1:* one package `ai_pipeline` with optional dependency extras so the control
  host does not need `torch` (see §22).

### 2.3 Where code runs

Three roles, one codebase:

| Role | Runs | Needs |
|---|---|---|
| **Control host** | the CLI, state store, orchestration | `ai_pipeline[core]`, `ssh`, `rsync`/`scp` |
| **Compute host** | job runner: autolabel / train / evaluate / export | `ai_pipeline[provider-ultralytics]` + GPU stack |
| **Edge device** (NG4511) | runtime bundle: TRT engine build, inference service, health/benchmark scripts | `edge/` bundle + JetPack stack |

**V1 decision:** control host and compute host are the **same machine, the AI
server** (`local` compute backend). The roles stay distinct in code so that the
`ssh` backend can split them later without touching orchestration. The control
host ships job specs and artifacts to the device, never Python edits.

---

## 3. Repository directory structure

```
camthink-ai-pipeline/
├── pyproject.toml                  # package metadata, extras, pytest/ruff config
├── README.md
├── .env.example
├── .gitignore
├── infrastructure.yaml             # named backends/targets (no secrets)   §6.2
├── src/ai_pipeline/
│   ├── cli/                        # typer app; thin adapters only          §13
│   │   ├── main.py
│   │   ├── project.py  data.py  label.py  dataset.py  train.py
│   │   ├── evaluate.py model.py  export.py benchmark.py deploy.py status.py
│   │   └── output.py               # JSON envelope / human rendering        §14
│   ├── core/                       # Layer A
│   │   ├── config/                 # pydantic schemas + loader + env subst  §6
│   │   ├── project.py              # Project: paths, config, store handle   §7
│   │   ├── store.py                # JSON-file metadata store (atomic)     §12
│   │   ├── ids.py                  # run/version id generation
│   │   ├── state.py                # entity state enums + transition rules  §11
│   │   ├── errors.py               # PipelineError hierarchy → exit codes  §16
│   │   ├── logging.py              # context-aware logger                  §15
│   │   ├── remote.py               # Host protocol: LocalHost, SshHost     §19
│   │   ├── artifacts.py            # artifact root, hashing, manifests     §20
│   │   ├── data/                   # ingest (local dir, rtsp capture)
│   │   ├── labeling/               # label store, importers, auto-label job
│   │   ├── dataset/                # validate, split, build version        §10
│   │   ├── jobs/                   # job spec + runner (runs on compute)   §9
│   │   ├── training/               # submit/poll/fetch orchestration       §9
│   │   ├── evaluation/             # metrics merge + acceptance policy     §11.3
│   │   ├── registry/               # model versions                        §12
│   │   ├── export/                 # export stage orchestration
│   │   └── deployment/             # deploy/rollback/health orchestration  §18
│   ├── tasks/                      # Layer B
│   │   ├── base.py                 # TaskModule protocol
│   │   └── object_detection/       # YOLO-txt label format, validation, metric names
│   ├── providers/                  # model seam
│   │   ├── base.py                 # ModelProvider protocol
│   │   ├── fake.py                 # for tests
│   │   └── ultralytics/
│   ├── backends/                   # compute seam
│   │   ├── base.py                 # ComputeBackend protocol
│   │   ├── local.py
│   │   └── ssh.py
│   ├── targets/                    # deployment seam
│   │   ├── base.py                 # DeploymentTarget protocol
│   │   ├── fake.py
│   │   └── camthink_ng4511/
│   └── templates/                  # project scaffold (project.yaml, README)
├── edge/                           # runtime bundle shipped to the device    §18.3
│   ├── build_engine.sh             # ONNX → TensorRT (runs on device)
│   ├── inference_service.py        # minimal service + /health
│   ├── smoke_test.py  benchmark.py  healthcheck.py
│   └── camthink-ai-inference.service
├── projects/
│   └── demo_person_detection/
│       ├── project.yaml
│       └── data/                   # gitignored: raw/, labels/
├── artifacts/                      # gitignored (configurable root)       §20
├── tests/
│   ├── unit/  integration/  hardware/
│   └── fixtures/                   # tiny synthetic images only
└── docs/
    ├── design/v1-architecture-proposal.md   (this file)
    ├── architecture.md  configuration.md  cli.md  training-server.md
    ├── ng4511-deployment.md  model-lifecycle.md  troubleshooting.md
    └── (written as features land, not before)
```

Notes:

- `src/` layout prevents accidentally importing the uninstalled tree in tests.
- `edge/` is deliberately outside the Python package: it is copied to the
  device as a bundle and must have minimal dependencies.
- Per-project *data* lives next to the config (`projects/<p>/data/`) because
  operators think of it as the project's; *generated* artifacts (dataset
  versions, runs, models) live under one `artifacts/` root so they can be moved
  or synced as a unit.

---

## 4. Major Python modules and their responsibilities

| Module | Responsibility | Must not |
|---|---|---|
| `core.config` | Load/validate `project.yaml` + `infrastructure.yaml`; `${ENV}` substitution; schema versioning | know about providers/targets beyond their names |
| `core.project` | Resolve paths; hold config; open store | contain lifecycle logic |
| `core.store` | Read/write entity JSON atomically; list/query | hold business rules |
| `core.state` | Entity state enums; `transition(entity, to)` with allowed-transition table | perform IO |
| `core.remote` | `Host` protocol (`run`, `upload`, `download`, `exists`) via `ssh`/`rsync` subprocesses | contain backend/target logic |
| `core.jobs` | `JobSpec` (kind, inputs, provider, params) + `run_job()` executed on compute host writing `status.json` | know about SSH |
| `core.training` | Submit job via backend; poll; fetch artifacts; create run record | call provider directly |
| `core.evaluation` | Merge metrics (eval + benchmark) and apply acceptance policy from config | know metric semantics (task module supplies names) |
| `core.registry` | Model version records + lifecycle | contain file-copy logic beyond artifact refs |
| `core.deployment` | Orchestrate deploy/rollback/health using a `DeploymentTarget` | contain SSH or device paths |
| `tasks.object_detection` | Label format (YOLO txt), dataset validation rules, class handling, metric names, provider data-config generation | reference any model library |
| `providers.ultralytics` | `train/evaluate/export/predict/autolabel` using ultralytics; translate params | be imported on the control host |
| `backends.*` | Execute a `JobSpec` somewhere; report status; move files | interpret job results |
| `targets.*` | Device-specific verify/upload/prepare/smoke/benchmark/activate/health/rollback | decide acceptance |

Plugin lookup is a plain dict registry (`name → class`) in each seam package.
Python entry points can be added later if third-party plugins ever matter.

---

## 5. Project abstraction (Layer C)

A **project** is a directory `projects/<name>/` containing `project.yaml` and
(gitignored) `data/`. Its generated state lives under `artifacts/<name>/`.

```
artifacts/<project>/
├── project.json            # created_at, schema_version, current pointers
├── imports/<import_id>.json
├── labels/                 # label sets (manual / imported / auto) + manifests
├── datasets/<ds_version>/  # built, immutable dataset versions
├── runs/<run_id>/          # training/eval/export jobs: spec, status, logs, outputs
├── models/<model_version>/ # registry records + artifact refs
├── deployments/<deploy_id>.json
└── logs/
```

`ai-pipeline project create <name>` renders `templates/` into `projects/<name>/`
and writes `artifacts/<name>/project.json`. Creating a project never touches
Python files.

- *Alternatives:* multiple YAML files per project (`dataset.yaml`,
  `evaluation.yaml`, …) as suggested in the brief.
- *V1 choice:* **one `project.yaml` with sections**. One file is simpler to
  validate as a whole, diff, and pass around. The loader will accept an optional
  split (`project.yaml` + `<section>.yaml` overrides) later if files get long;
  the schema does not change either way. Infrastructure endpoints are *not* in
  the project file (§6.2), which is the split that actually matters.

---

## 6. Configuration structure

### 6.1 `projects/<name>/project.yaml` (draft schema, not final)

```yaml
schema_version: 1

project:
  name: demo_person_detection
  task_type: object_detection
  description: Temporary demo project to validate the pipeline.

dataset:
  classes: [person]                 # project class list; order = class index
  label_format: yolo                # task module decides what is allowed
  sources:
    - type: local_images            # local_images | rtsp_capture (V1)
      path: ./data/raw
  split: { train: 0.8, val: 0.1, test: 0.1, seed: 42 }
  validation:
    min_images: 20
    min_instances_per_class: 10
    max_empty_label_ratio: 0.5

labeling:
  auto:
    provider: ultralytics
    weights: yolo11n.pt             # pretrained; provider resolves/downloads
    class_map: { person: person }   # source class -> project class (others dropped)
    confidence: 0.4
    backend: ai-server              # where the auto-label job runs

model:
  provider: ultralytics
  candidates:
    - { name: yolo11n, weights: yolo11n.pt }
  # more candidates later -> one run per candidate, selection picks

training:
  backend: ai-server                # key into infrastructure.yaml
  epochs: 5
  image_size: 640
  batch_size: 16
  device: "0"
  provider_args: {}                 # passthrough, validated loosely

evaluation:
  split: test
  metrics: [precision, recall, mAP50, mAP50-95]
  acceptance:                       # all rules must pass
    mAP50: { min: 0.30 }
  selection:
    primary_metric: mAP50-95        # highest wins among accepted

export:
  format: onnx
  opset: 17
  image_size: 640
  dynamic: false

deployment:
  target: ng4511-lab                # key into infrastructure.yaml
  runtime: tensorrt
  precision: fp16
  acceptance:                       # from device benchmark
    latency_ms_p50: { max: 60 }
    fps: { min: 15 }
    memory_mb: { max: 3000 }
  health:
    timeout_s: 120
    interval_s: 5
  keep_versions: 3
```

### 6.2 `infrastructure.yaml` (repo root; committed; no secrets)

```yaml
schema_version: 1
artifact_root: ./artifacts          # overridable by AIP_ARTIFACT_ROOT

compute_backends:
  local:
    type: local
    python: python                  # interpreter with provider extra installed
  ai-server:
    type: ssh
    ssh_host: ai-server             # alias in ~/.ssh/config, or user@host
    workdir: /data/camthink-ai-pipeline
    python: /data/venvs/aip/bin/python
    # docker_image: ...             # alternative executor, decided after §25

deployment_targets:
  fake:
    type: fake
    root: ./.fake-target
  ng4511-lab:
    type: camthink_ng4511
    ssh_host: ng4511
    install_root: /opt/camthink-ai
    service_name: camthink-ai-inference
    inference_source: "${NG4511_RTSP_URL}"   # optional; env-substituted
```

- *Why split project vs infrastructure:* the same project must run against a
  lab device today and a different device tomorrow without editing project
  config; endpoints are environment facts, classes/thresholds are project facts.
- *Alternatives:* everything in `project.yaml` (couples project to one lab);
  per-host `.env` only (unvalidated, invisible).
- *V1:* two validated files; secrets only via env/SSH (§17).

### 6.3 Validation

- pydantic v2 models, `extra="forbid"` on every section → typos fail early with
  path-qualified messages (`training.epoch: extra field not permitted`).
- Cross-field checks: `task_type` must have a registered task module;
  `provider`, `backend`, `target` names must resolve; metric names in
  `evaluation.metrics/acceptance` must be known to the task module; `classes`
  non-empty and unique; split sums to 1.
- `ai-pipeline project validate <p>` runs config validation alone (exit 2 on
  failure, JSON error list with `--output json`).

---

## 7. Task-module abstraction (Layer B)

```python
class TaskModule(Protocol):
    name: str                                   # "object_detection"
    label_formats: tuple[str, ...]              # ("yolo",)
    metric_names: tuple[str, ...]               # ("precision","recall","mAP50","mAP50-95", per-class)
    def validate_dataset(self, images, labels, cfg) -> ValidationReport: ...
    def build_provider_data_config(self, ds_version_dir, classes) -> Path: ...  # e.g. data.yaml
    def normalize_labels(self, labels, class_map, classes) -> Labels: ...       # for import/auto-label
```

- *Why:* dataset validation rules, label formats, and metric vocabularies are
  genuinely task-specific; keeping them out of `core` is what makes
  classification/segmentation addable later.
- *Alternatives:* bake object detection into core and refactor later (would
  spread YOLO-txt assumptions into validation and registry).
- *V1:* one implementation, `object_detection`, YOLO-txt label format only.
  Adding a task later = new subpackage + registry entry, no core change.

---

## 8. Model-provider abstraction

```python
class ModelProvider(Protocol):
    name: str
    def autolabel(self, images, weights, class_map, conf, out_dir) -> LabelSet: ...
    def train(self, data_cfg, candidate, params, out_dir) -> TrainResult: ...     # weights + metrics
    def evaluate(self, weights, data_cfg, split, out_dir) -> Metrics: ...
    def export(self, weights, fmt, params, out_dir) -> ExportResult: ...           # e.g. ONNX
```

- *Why:* the brief's explicit requirement that YOLO specifics not leak.
- *Alternatives:* `Trainer`/`Evaluator`/`Exporter` as separate classes — more
  types for the same single implementation; a provider that also owns edge
  inference — rejected, edge inference belongs to the target/runtime bundle.
- *V1:* `ultralytics` provider only, plus `fake` for tests. Provider code runs
  **only inside jobs on the compute host**; the control host never imports it.

---

## 9. Compute-backend abstraction and remote training strategy

### 9.1 Jobs

All provider work (autolabel, train, evaluate, export) is a **job**:

```
runs/<run_id>/
├── job.json       # kind, project, dataset_version, provider, candidate, params, created_at
├── status.json    # {state, started_at, finished_at, progress, error}  written by runner
├── runner.log
└── out/           # weights/, metrics.json, labels/, model.onnx ...
```

The runner is `python -m ai_pipeline.jobs.run <job_dir>`; it loads the provider
named in `job.json`, executes, and writes `status.json` on every transition —
including on exception (`FAILED` with traceback). A backend is just *how* that
command is launched and how files move.

### 9.2 Backend protocol

```python
class ComputeBackend(Protocol):
    def submit(self, job_dir: Path, inputs: list[Path]) -> None    # upload + launch detached
    def status(self, job_dir: Path) -> JobStatus                   # read remote status.json
    def logs(self, job_dir: Path, tail: int) -> str
    def fetch(self, job_dir: Path) -> None                         # download out/ + status
    def cancel(self, job_dir: Path) -> None
```

- **`local`**: `subprocess.Popen` detached; used by tests and for CPU smoke runs.
- **`ssh`**: `rsync` job dir + dataset version to `workdir/jobs/<run_id>/`;
  `ssh host 'nohup python -m ai_pipeline.jobs.run … > runner.log 2>&1 &'`;
  poll by `ssh host cat status.json`; `rsync` results back.

- *Why detached + status file rather than a live SSH session:* training runs
  for hours; the control host may disconnect; polling a file is restart-safe
  and trivially agent-friendly.
- *Alternatives:* paramiko/fabric (extra dependency, reimplements ssh config,
  agent forwarding, ControlMaster); a scheduler (Slurm/Ray/Prefect) —
  premature; `systemd-run --user` instead of `nohup` — nicer (cgroup, logs) but
  depends on server setup; keep as an option after §25.
- *V1:* shell out to system `ssh`/`rsync` with `BatchMode=yes`,
  `ConnectTimeout`, and a single `Host` abstraction shared with targets.

### 9.3 Training run tracking

`runs/<run_id>/run.json` (control-host record, distinct from the job files):

```json
{
  "run_id": "run-20261103-153300-7f3a",
  "kind": "train",
  "project": "demo_person_detection",
  "dataset_version": "ds-v003",
  "candidate": {"name": "yolo11n", "weights": "yolo11n.pt"},
  "params": {"epochs": 5, "image_size": 640, "batch_size": 16, "device": "0"},
  "backend": "ai-server",
  "environment": {"python": "...", "provider_version": "...", "torch": "...", "gpu": "..."},
  "state": "TRAINING_COMPLETE",
  "timestamps": {"created": "...", "started": "...", "finished": "..."},
  "outputs": {"weights": "runs/…/out/weights/best.pt", "metrics": "runs/…/out/metrics.json"},
  "error": null
}
```

Retries: `train` always creates a new `run_id`; `train status`/`train fetch`
operate on an existing one. No silent resume in V1 (resume is an explicit
`--resume <run_id>` later if needed).

Evaluation runs on the compute backend as an `evaluate` job (GPU, same
environment as training) — simpler than maintaining a CPU eval path, and metrics
come from one place. The ultralytics train job already produces val metrics;
the separate evaluate job runs on the configured `evaluation.split` (test).

---

## 10. Dataset versioning

### 10.1 Flow

```
data/raw/ (immutable imports)  ──import──▶ artifacts/<p>/imports/<id>.json (manifest + sha256)
labels (manual | imported | auto) ──────▶ artifacts/<p>/labels/<label_set_id>/
                       dataset validate ─▶ ValidationReport (task module rules)
                       dataset build    ─▶ artifacts/<p>/datasets/<ds_version>/
```

### 10.2 Dataset version layout

```
datasets/ds-v003/
├── dataset.json      # id, created_at, project, classes, sources (import ids, label set ids),
│                     # split config+seed, counts, class distribution, validation report,
│                     # content_hash, label_format, size_bytes
├── manifest.jsonl    # one line per sample: relpath, sha256, split, n_instances
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml         # generated by task module for the provider
```

- *Why:* training must be traceable to exact files; raw `data/` must never be
  overwritten.
- *Alternatives:* DVC (good tool, but adds a second CLI and remote config for a
  single-host V1); manifest-only versions referencing raw files (fragile when
  raw changes; awkward to rsync); SQLite catalogue (no benefit at this size).
- *V1:* **filesystem copies + JSON metadata**, content-hash deterministic
  versions. Files are copied (hardlinked when same filesystem) — datasets are
  small in V1 and a self-contained directory is what rsync and the provider
  want. DVC/object storage can later replace the copy step without changing
  `dataset.json`.

Idempotency: `dataset build` computes `content_hash` over (sorted manifest of
image+label hashes, classes, split config). If a version with the same hash
exists, it returns that version and exits 0 with `"created": false`. Existing
versions are never modified.

---

## 11. Pipeline state model

### 11.1 Entity states, not one global state

- *Why:* a project holds several dataset versions, runs, and model versions at
  once; a single project-level state (`TRAINING` → `EVALUATING` …) becomes wrong
  as soon as two runs exist or a deploy fails while a new train starts.
- *Alternatives:* the single linear state list from the brief (simple, but
  ambiguous); a workflow engine (premature).
- *V1:* each entity has its own small state machine; `ai-pipeline status`
  derives a project summary and `next_operations` from them.

```
ImportRecord     : IMPORTED
LabelSet         : PENDING → READY | FAILED
DatasetVersion   : BUILDING → READY | FAILED            (validate is a report, not a state)
Run (any job)    : CREATED → SUBMITTED → RUNNING → COMPLETED | FAILED | CANCELLED
ModelVersion     : TRAINED → EVALUATED → ACCEPTED | REJECTED
                   ACCEPTED → EXPORTED → BENCHMARKED → DEPLOYED → SUPERSEDED
                   (any) → ARCHIVED
Deployment       : PENDING → UPLOADED → PREPARED → SMOKE_PASSED → BENCHMARKED
                   → ACTIVE | FAILED | ROLLED_BACK
Project pointers : current_dataset_version, selected_model_version,
                   active_deployment, last_known_good_deployment
```

Transitions are enforced by a table in `core.state`; illegal transitions raise
`StateError` (exit 3).

### 11.2 Failure handling

Every mutating command: write entity with an in-progress state → do work → write
terminal state. Any exception → terminal `FAILED` with `error{code,message,
details}` persisted. There is no state in which "something is happening" cannot
be disambiguated by reading the record plus `status.json` from the backend.
Commands may be rerun; they either create a new entity (`train`) or detect the
existing one (`dataset build`, `deploy --deployment <id> --resume` later).

### 11.3 Evaluation policy

`evaluation.acceptance` and `deployment.acceptance` are lists of
`metric → {min|max}` rules from config. `core.evaluation.apply_policy(metrics,
rules)` returns `{passed, checks:[{metric, value, rule, pass}]}`. Candidate
selection = highest `primary_metric` among `ACCEPTED`. No policy is hard-coded;
the task module only validates metric *names*.

---

## 12. Model registry and metadata store

```
models/mv-004/model.json
{
  "model_version": "mv-004",
  "project": "...", "task_type": "object_detection",
  "run_id": "run-…", "dataset_version": "ds-v003",
  "candidate": {...}, "training_params": {...},
  "metrics": {"eval": {...}, "benchmark": {...}},
  "acceptance": {"eval": {...}, "benchmark": {...}},
  "artifacts": {
    "weights":  {"path": "runs/run-…/out/weights/best.pt", "sha256": "…"},
    "onnx":     {"path": "models/mv-004/model.onnx", "sha256": "…"},
    "engine":   {"built_on": "ng4511-lab", "path_on_target": "/opt/camthink-ai/models/v007/model.engine",
                 "tensorrt_version": "…", "note": "not portable"}
  },
  "state": "DEPLOYED",
  "deployments": ["dep-…"],
  "created_at": "...", "updated_at": "..."
}
```

- *Why a store at all:* traceability requirements (§16 of brief).
- *Alternatives:* MLflow (adds a server/DB, UI we don't need yet; could be a
  later exporter); SQLite (compact, queryable, but opaque to `git diff`/`cat`
  and to rsync merges).
- *V1:* **one JSON file per entity**, written atomically (tmp + `os.replace`),
  behind a tiny `Store` class so SQLite/MLflow can replace it without touching
  callers. Listing = directory scan; volumes are tens of records.

---

## 13. CLI structure

Framework: **typer** (click-based, typed, auto `--help`). Noun-verb layout,
`<project>` always an explicit argument (agent-friendly, no "current project"
magic). Global options: `--output {human,json}`, `--infra <file>`,
`--artifact-root <dir>`, `-v/--verbose`, `--quiet`.

```
ai-pipeline project   create <p> [--task object_detection] [--template default]
                      validate <p>
                      list
ai-pipeline data      import   <p> <path> [--recursive]            # local images
                      collect  <p> --source <name> --frames N      # rtsp capture (optional in V1)
                      list     <p>
ai-pipeline label     auto     <p> [--import <id>] [--backend <b>]
                      import   <p> <path> --format yolo
                      list     <p>
ai-pipeline dataset   validate <p> [--labels <set>]
                      build    <p> [--labels <set>]
                      list     <p>
                      show     <p> <ds_version>
ai-pipeline train     start    <p> [--dataset <ds>] [--candidate <name>] [--backend <b>]
                      status   <p> <run_id> [--wait] [--tail]
                      fetch    <p> <run_id>
                      cancel   <p> <run_id>
                      list     <p>
ai-pipeline evaluate  run      <p> <run_id | model_version>
ai-pipeline model     register <p> <run_id>                     # TRAINED (auto after fetch)
                      list     <p>
                      show     <p> <mv>
                      select   <p> [<mv>]                        # explicit or policy-based
ai-pipeline export    run      <p> [<mv>] [--format onnx]
ai-pipeline target    verify   <p> | setup <p>                   # install/update edge bundle
ai-pipeline benchmark run      <p> [<mv>]                        # on device, via a deployment candidate
ai-pipeline deploy    run      <p> [<mv>] [--skip-benchmark]
                      status   <p>
                      history  <p>
ai-pipeline rollback  <p> [--to <deployment_id>]
ai-pipeline health    <p>
ai-pipeline status    <p>                                        # summary + next_operations
```

Rules: no interactive prompts (a `--yes` flag where confirmation would be
natural); every command idempotent or creates a distinct id; `train start
--wait` is the only long-blocking option and it polls with a timeout.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | operation failed (training failed, health check failed, …) |
| 2 | usage / configuration error |
| 3 | precondition / state error (e.g. deploy with no ACCEPTED model) |
| 4 | remote / connectivity error (SSH, rsync, device unreachable) |
| 5 | policy rejection (metrics below acceptance; not a bug, not a crash) |

---

## 14. Structured JSON output

With `--output json`, stdout contains exactly one JSON document; all logs go to
stderr. Envelope:

```json
{
  "schema_version": 1,
  "project": "demo_person_detection",
  "operation": "train.start",
  "status": "ok" | "failed" | "rejected",
  "result": { "run_id": "run-…", "dataset_version": "ds-v003", "state": "SUBMITTED" },
  "error": null | { "code": "REMOTE_UNREACHABLE", "message": "...", "details": {...}, "hint": "..." },
  "next_operations": ["train status demo_person_detection run-…"],
  "timestamp": "2026-11-03T07:33:00Z"
}
```

Human mode renders the same object as text (rich tables when a TTY). Exit code
is derived from `status`/`error.code`, never from parsing text. The envelope is
versioned so the future agent can rely on it.

---

## 15. Logging

- stdlib `logging`; one `LoggerAdapter` carrying context
  `{project, operation, run_id, dataset_version, model_version, target}` that
  is merged into every record.
- Two sinks: stderr (human format; JSON lines with `--log-json`) and a per-run
  file `artifacts/<p>/logs/<operation>-<timestamp>.log`.
- Subprocess output (`ssh`, `rsync`, trtexec, runner) is captured and logged
  verbatim at DEBUG, and the tail is attached to the error on failure.
- *Alternatives:* structlog/loguru — nicer, one more dependency; not needed.

---

## 16. Error handling

```
PipelineError(code, message, details, hint, exit_code)
├── ConfigError          (2)
├── StateError           (3)
├── RemoteError          (4)   host, command, exit_code, stderr_tail
├── OperationError       (1)   e.g. TrainingFailed, HealthCheckFailed, ArtifactInvalid
└── PolicyRejected       (5)   checks[]
```

Rules: never `except Exception: pass`; subprocess calls use a wrapper that
raises `RemoteError` with command, exit code, and stderr tail; every error has a
machine `code` and a human `hint` ("check `ssh ai-server` works
non-interactively"). The CLI's top-level handler converts exceptions into the
JSON envelope and exit code; unexpected exceptions become `INTERNAL_ERROR` with
traceback in the log file.

---

## 17. Secrets management

- *V1:* **SSH keys only.** `infrastructure.yaml` names hosts; authentication is
  the operator's `~/.ssh/config` + key agent. No passwords anywhere.
- Anything else sensitive (e.g. RTSP URL with credentials) is referenced as
  `${VAR}` in config and read from the environment; `.env` (gitignored) is
  loaded via `python-dotenv` if present; `.env.example` lists variables with
  descriptions and no values.
- `.gitignore` excludes `.env`, `*.pem`, `id_*`, `artifacts/`, `projects/*/data/`.
- Remote accounts: dedicated `aip` user on the AI server; on the device, a
  non-root user with `sudo` limited to `systemctl {restart,status}
  camthink-ai-inference` via sudoers — to be confirmed (§25).
- *Alternatives:* Vault/SOPS — far beyond V1 needs.

---

## 18. Deployment-target abstraction and NG4511 strategy

### 18.1 Protocol

```python
class DeploymentTarget(Protocol):
    name: str
    def verify(self) -> TargetInfo                       # reachable, arch, JetPack/TRT version, disk, service state
    def setup_runtime(self, bundle: Path) -> None         # install/update edge/ bundle (one-off)
    def upload(self, candidate: Candidate) -> str         # -> target version id (vNNN), staged, not active
    def validate_artifact(self, version) -> None          # checksum + model.json sanity
    def prepare(self, version, precision) -> PrepareResult   # ONNX -> TensorRT engine ON DEVICE
    def smoke_test(self, version) -> SmokeResult          # inference on bundled sample images
    def benchmark(self, version, n) -> BenchmarkMetrics   # latency p50/p95, fps, mem, gpu util
    def activate(self, version) -> None                   # atomic symlink swap + service restart
    def health(self) -> HealthResult                      # service active + /health JSON + recent inference
    def current(self) -> str | None; def list_versions(self) -> list[str]
    def rollback(self, to_version) -> None                # activate(previous) + health
    def prune(self, keep: int) -> None
```

Implementations: `fake` (local temp dir, "engine build" = copy, health always
OK unless a test toggles failure) and `camthink_ng4511` (SSH + `edge/` scripts).

### 18.2 Device layout and deploy sequence

```
/opt/camthink-ai/
├── runtime/                 # edge bundle (versioned by git sha), venv or system python
├── models/
│   ├── v006/  model.onnx  model.engine  model.json  labels.txt  smoke/  benchmark.json
│   └── v007/ ...
├── current -> models/v007   # symlink, swapped atomically (ln -sfn tmp && mv -T)
├── deployments.json         # history incl. last_known_good
└── logs/
```

`deploy run` = `verify → upload → validate_artifact → prepare(fp16) →
smoke_test → benchmark (+ policy) → activate → restart → health`. Any failure
after `activate` triggers automatic `rollback(last_known_good)` and the
deployment ends `FAILED` (exit 1) with both results in the record. Failures
before `activate` leave the staged version on disk (for inspection) and the
active model untouched. Retrying never overwrites an existing `vNNN`.

### 18.3 Edge runtime (what actually runs inference)

- *Why needed:* "deployed model can run inference" and health checks need a
  process on the device.
- *Alternatives:* DeepStream (powerful, steep; config-heavy for V1); Triton
  (heavy on 8 GB); whatever CamThink preinstalls (**unknown**, §25).
- *V1:* a **minimal Python inference service** in `edge/`: loads
  `current/model.engine` via ultralytics' TensorRT runtime (or raw
  `tensorrt`+`cuda` bindings if ultralytics proves heavy on the device — decided
  after §25), reads frames from the configured source (RTSP URL or looping
  sample images), exposes `http://127.0.0.1:8080/health` and `/stats` (JSON:
  model version, last inference age, fps, errors). Managed by systemd. Small
  enough to replace with DeepStream later behind the same target protocol.
- Engine generation happens **on the device** (`trtexec` or ultralytics export)
  because TRT engines are tied to TRT version + GPU; FP16 only in V1.

### 18.4 Benchmark

`edge/benchmark.py` runs N warm + M timed inferences on sample images, samples
`tegrastats` for memory/GPU, writes `benchmark.json`. Results are merged into
the model version and checked against `deployment.acceptance`.

---

## 19. Remote execution (`core.remote`)

```python
class Host(Protocol):
    def run(self, argv: list[str], *, timeout: float | None, check=True) -> CommandResult
    def upload(self, local: Path, remote: str, *, delete=False) -> None     # rsync -a --checksum
    def download(self, remote: str, local: Path) -> None
    def exists(self, remote: str) -> bool
LocalHost()                 # subprocess, shutil
SshHost(alias)              # ssh -o BatchMode=yes -o ConnectTimeout=10 ; rsync -e ssh
```

Used by both the `ssh` compute backend and the NG4511 target. `rsync` is
required on control host, server, and device (verify on device, §25); fallback
`scp -r` for hosts lacking rsync is a one-line swap behind `upload`.

The **control host** (where the CLI runs) is the AI server in V1 (review
decision). It reaches the device over passwordless SSH; the `local` compute
backend needs no SSH at all. The transfer method to the device (`rsync` vs
`scp`) is decided in M5 behind `Host.upload` (§25 item 25).

---

## 20. Model artifact storage

- `artifact_root` (default `./artifacts`, gitignored) holds everything
  generated; every reference in metadata is `relative path + sha256`, so the
  root can be moved, mounted from NAS, or rsynced.
- Large files kept: dataset versions, `best.pt`, `model.onnx`, runner logs.
  Not kept on control host: TensorRT engines (device-specific; recorded as
  path-on-target + TRT version only), training caches.
- Pretrained weights are fetched by the provider on the compute host and cached
  there (`workdir/cache/`), never committed.
- *Postponed:* remote object storage (S3/MinIO) sync — `artifacts.py` is the
  one place to add it.

---

## 21. Git strategy

- `main` always importable and tests green; feature branches
  (`feat/…`, `fix/…`, `docs/…`) merged by PR; small focused commits
  (Conventional Commits style for scan-ability, not enforced by tooling).
- `.gitignore`: `artifacts/`, `projects/*/data/`, `*.pt`, `*.onnx`, `*.engine`,
  `*.plan`, `.env`, `*.log`, `__pycache__/`, `.venv/`, `.fake-target/`,
  `runs/`, `wandb/`.
- Test fixtures: only tiny synthetic images generated at test time (no binary
  blobs in git). A handful of real sample JPEGs (< 1 MB total) for the demo
  project's smoke test are acceptable and documented.
- Tags `v0.1.0` … per milestone (§23).
- CI (GitHub Actions): `ruff` + `pytest -m "not slow and not hardware"` on
  push. Hardware/slow suites are run manually from the control host.

---

## 22. Dependencies

| Extra | Packages | Installed on |
|---|---|---|
| core (default) | `pydantic>=2`, `typer`, `pyyaml`, `python-dotenv`, `rich` (optional pretty tables) | control host, also everywhere else |
| `provider-ultralytics` | `ultralytics`, `torch`, `torchvision`, `onnx`, `onnxslim`/`onnxruntime` (ultralytics export deps) | AI server; optionally dev machine for CPU smoke |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` (lenient) | developers, CI |
| edge bundle (not pip extra) | JetPack-provided `tensorrt`, `numpy`, `opencv` (system), ultralytics Jetson wheels **or** `pycuda` — decided after §25 | NG4511 |

External tools: `ssh`, `rsync` (control/server/device), `trtexec` (device,
ships with TensorRT).

Python: `>=3.10` to stay compatible with JetPack 6 (Ubuntu 22.04 → 3.10) and
typical server distros; 3.12 on dev is fine. Packaging: `pyproject.toml` +
`pip install -e .[dev]`; `uv` optional, not required.

- *Why typer over argparse:* subcommand trees, typed options, `--help`
  generation for the agent; click dependency is acceptable.
- *Why pydantic over hand-rolled validation:* precise error paths for free.
- *Why not paramiko:* see §9.2.

---

## 23. Milestones

Each milestone ends with tests green, docs for what exists, and a tagged commit.
Hardware-dependent milestones are split so that software work is never blocked
on access.

**M0 — Skeleton (no hardware).** Package, `pyproject`, `.gitignore`,
`.env.example`, config schemas + loader + `${ENV}` substitution, `Store`,
`state` tables, errors/exit codes, logging, JSON envelope, `project create |
validate | list`, `status`. Unit tests for all of it. Docs: `architecture.md`,
`configuration.md`, `cli.md` (skeleton).

**M1 — Data → dataset version (no hardware).** `data import`, label store,
`label import` (YOLO txt), `object_detection` task module (validation rules,
`data.yaml`), `dataset validate | build | list | show`, deterministic
versioning. `fake` provider `autolabel` to test the auto-label path. Tests with
synthetic images.

**M2 — Jobs, backends, training, evaluation, registry (local backend +
fake provider).** `JobSpec`, runner, `local` backend, `train start | status |
fetch`, `evaluate run`, registry + `model list | show | select`, acceptance
policy. Full lifecycle integration test with fakes. Then the `ultralytics`
provider, verified with a `slow` CPU test (1 epoch, tiny dataset) on the dev
machine.

**M3 — Export + deployment loop with `fake` target.** `export run`, `deploy
run | status | history`, `rollback`, `health`, `benchmark run`, deployment
records, automatic rollback on post-activation failure. Integration test:
dataset → train (fake) → export → deploy → health → rollback, all local.

**M4 — Real AI server.** Resolve §25 server items; `ssh` backend; `aip` user
and `workdir` setup script; `training-server.md`. Demo project trains
`yolo11n` for a few epochs on the server through the CLI; artifacts fetched;
evaluated; registered.

**M5 — Real NG4511.** Resolve §25 device items; `camthink_ng4511` target;
`edge/` bundle (engine build, service, health, smoke, benchmark);
`target setup`; `ng4511-deployment.md`. Deploy the M4 model, run inference on
sample/RTSP input, benchmark, health check, rollback to previous version.

**M6 — V1 acceptance.** Run the §27 (brief) 16-step workflow end to end from a
clean checkout using only documented CLI commands; capture JSON outputs;
`model-lifecycle.md`, `troubleshooting.md`; tag `v1.0.0`.

Dependency order: M0 → M1 → M2 → M3 are purely local; M4 and M5 need access
and can start as soon as their §25 items are answered (M4 can run in parallel
with M3).

---

## 24. Major technical risks

| Risk | Impact | Mitigation |
|---|---|---|
| JetPack / TensorRT / ultralytics version mismatch on NG4511 | engine build or runtime fails | verify versions first (§25); build engines on device; keep raw-TRT fallback path for the service |
| Orin Nano 8 GB memory during engine build + running service | OOM, slow builds | build with the service stopped or on the staged version while old service runs (measure); FP16 only; `keep_versions` pruning |
| Network topology: control host cannot reach both server and device | pipeline cannot orchestrate | decide the control host early; SSH jump hosts are supported via `~/.ssh/config` without code changes |
| Training environment drift on AI server (CUDA/torch) | irreproducible runs | pin provider versions in `pyproject`; record environment in `run.json`; consider Docker executor if server already uses Docker |
| Ultralytics API/CLI changes | provider breaks | provider is the only module touching it; pin version |
| Large datasets later make copy-based versioning slow | disk/time | hardlinks now; manifest is storage-agnostic so DVC/object store can replace copies |
| Auto-label quality for the demo | poor demo metrics | irrelevant to V1 goal; acceptance thresholds in config are low for the demo |
| Device service model: CamThink may ship its own inference stack | duplicated/conflicting services | §25; target protocol isolates the choice |

---

## 25. Unknowns that must be verified before implementation

Items marked **(blocks M4)** / **(blocks M5)** gate those milestones only;
M0–M3 can proceed regardless.

**Control host**
1. ~~Where will the CLI run in production?~~ **Resolved: the AI server.**
2. OS/Python on the AI server; is `rsync` available? (merged into item 3)

**AI server (blocks M4)**
3. OS, Python version, GPU model(s), driver/CUDA version, free disk.
4. SSH access: user, key-based auth, sudo? Can a dedicated `aip` user be
   created? Is there a shared data path (e.g. `/data`)?
5. Existing ML tooling: venv/conda/Docker? Preferred isolation method?
6. Internet access (to download pretrained weights / pip packages)?
7. Is `nohup`-style detached execution acceptable, or is a job system already
   in use (Slurm, etc.)?

**CamThink NG4511 (blocks M5)**
8. JetPack / L4T version, TensorRT and CUDA versions, Python version.
9. SSH access: user, key auth, sudo policy, whether root is permitted.
10. Free disk, storage device (eMMC/NVMe), and power mode (affects benchmarks).
11. What software CamThink preinstalls: any inference service, DeepStream,
    Docker, camera management? Must we integrate with it or can we own
    `/opt/camthink-ai` + a systemd service?
12. Is `systemd` available and may our user control one unit via sudoers?
13. Camera input: RTSP URL(s), resolution, codec; or are only sample images
    needed for V1 acceptance?
14. Is `rsync` and `trtexec` present on the device?
15. Network: can the device reach the control host/server (for pulling), or is
    only inbound SSH from the control host possible (push model, assumed)?

**Project / data**
16. Source of demo images: existing folder, a public sample subset, or frames
    captured from the device camera? Approximate count?
17. Is any manual label review tool wanted in V1 (e.g. Label Studio import
    adapter), or is auto-label + spot check sufficient for the demo?

**Process**
18. Confirm model family baseline (`yolo11n` via ultralytics) and its license
    (AGPL-3.0 / enterprise) is acceptable for this project.
19. Confirm that tiny real sample JPEGs may be committed for smoke tests.

**AI Server → NG4511 link (added at review; verify during M4/M5)**
20. Which Linux user on the AI server performs the SSH to the NG4511, and which
    remote user it logs in as.
21. Whether that remote user has write permission to the deployment root
    (`/opt/camthink-ai`) and may control the inference service.
22. Whether passwordless `sudo` is available on the NG4511, and whether it is
    needed (it may be avoidable with a user-owned install root and a
    `systemd --user` service).
23. NG4511 hostname/IP strategy (static IP, DHCP reservation, mDNS
    `*.local`, or an `~/.ssh/config` alias) — the alias is what
    `infrastructure.yaml` will reference.
24. SSH host-key verification strategy: pinned entry in `known_hosts`
    (preferred), `StrictHostKeyChecking=accept-new`, or disabled (rejected).
25. Transfer method to the device: `rsync` (if installed on the NG4511) or
    `scp`/`sftp`; `Host.upload` hides the choice.

---

## 26. Explicitly stated assumptions

- **A1.** Push-based topology: the AI server (control + compute host)
  initiates all SSH connections to the device; the device never needs to reach
  the server. (Confirmed at review.)
- **A2.** One AI server and one device in V1; names in `infrastructure.yaml`
  already allow more.
- **A3.** YOLO-family via ultralytics is acceptable as the single V1 provider.
- **A4.** FP16 TensorRT, static input shape, single model per device at a time.
- **A5.** Datasets in V1 are small (hundreds to low thousands of images), making
  copy-based versioning and rsync transfers practical.
- **A6.** The device runs our own minimal inference service; no CamThink service
  integration is required for V1 (to be confirmed by §25 item 11).

---

## 27. Explicitly postponed

- Any application project (unleashed dog, helmet, faces, fire, vehicles).
- Additional task modules (classification, segmentation, tracking, pose, face).
- Additional providers (non-ultralytics detectors), INT8 calibration, DLA.
- Grok Bot / agent integration code; HTTP API server (CLI + JSON is the V1 API).
- Active learning, auto-retraining, scheduling, multi-candidate hyper-parameter
  sweeps beyond "one run per listed candidate".
- Remote artifact storage (S3/MinIO/NAS sync), DVC, MLflow.
- Fleet / multi-device management, OTA of the edge bundle beyond `target setup`.
- DeepStream / Triton runtimes; web dashboards; Prometheus exporters (the
  `/stats` endpoint is shaped so one can be added).
- Resume-in-place for interrupted training runs.
- Docker packaging of the pipeline itself (optional Docker executor on the
  server is kept as an option, not a requirement).

---

## 28. Review checklist

Please confirm or correct:

1. Single `project.yaml` + separate `infrastructure.yaml` (§5, §6.2).
2. Entity-level states instead of one project state machine (§11).
3. Evaluation/export run as jobs on the compute backend, not on the control
   host (§9.3) — meaning the control host never needs torch.
4. Shelling out to `ssh`/`rsync` instead of paramiko (§9.2).
5. Minimal own inference service on the device for V1 (§18.3).
6. Answers (or "unknown, verify at Mx") for the items in §25.

Implementation starts with M0 only after this review.
