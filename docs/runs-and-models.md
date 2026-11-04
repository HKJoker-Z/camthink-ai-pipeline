# Jobs, runs, providers, backends and the model registry

This document describes the M2 execution model: how a training/evaluation/auto-label request
becomes a **job** executed by a **compute backend** through a **model provider**, how the
control plane tracks it as a **run**, and how a completed training run becomes a registered
**model version**.

## The dependency boundary

```
control plane  (ai-pipeline CLI, ai_pipeline.core.*, ai_pipeline.backends.*)
    │   knows provider *names* only; never imports torch / ultralytics
    │   writes  runs/<run_id>/job.json
    │   reads   runs/<run_id>/status.json, out/result.json
    ▼
compute host   python -m ai_pipeline.core.jobs.runner <job_dir>
    │   ai_pipeline.providers.load_provider(name)  ← heavy imports happen here
    ▼
ModelProvider.train / evaluate / autolabel
```

In the V1 topology both sides are the AI server, but nothing relies on that: the boundary is a
directory of files, so the `ssh` backend (M4) only has to move the directory and launch the same
command remotely. A unit test asserts that importing the CLI, core, backends and the provider
registry loads no `torch`/`ultralytics`/`cv2` module.

## Job directory

```
artifacts/<p>/runs/<run_id>/
  job.json       JobSpec   — written once by the control plane
  status.json    JobStatus — PENDING (backend) → RUNNING → COMPLETED | FAILED | CANCELLED (runner)
  runner.log     runner stdout/stderr + its own log lines
  inputs/        optional staged inputs (e.g. images.txt for autolabel)
  out/           provider outputs; out/result.json is the JobResult
```

`JobSpec`: `run_id, kind (autolabel|train|evaluate), project, task_type, provider, inputs{},
params{}, created_at`. `JobStatus`: `state, handle{}, submitted_at, started_at, finished_at,
progress{}, error{type,message,traceback}, result_file`. `JobResult`: `artifacts{name: path
relative to job dir}, metrics{}, environment{}, extra{}`.

`handle` is an opaque dict only the backend that wrote it interprets (local: `pid`, `python`).
Every path inside these files is relative to the job directory or absolute on the compute host.

### Runner behaviour

1. write `status.json` `RUNNING` with `pid`/`hostname`;
2. `load_provider(spec.provider)` and dispatch on `spec.kind`;
3. on success: write `out/result.json`, then `status.json` `COMPLETED` (`result_file` set);
4. on any exception: `status.json` `FAILED` with `error.type/message/traceback`, exit 1; for
   `PipelineError`s the structured `code`, `details` and `hint` are kept too (e.g.
   `GPU_DEVICE_UNAVAILABLE`) and surface in `run status` as `error.details.cause`.

Providers may call `ctx.report_progress(epoch=3, total_epochs=5)`; it merges into
`status.progress` while RUNNING.

## Compute backends

`ComputeBackend` protocol: `submit(job_dir)`, `status(job_dir)`, `logs(job_dir, tail)`,
`fetch(job_dir)`, `cancel(job_dir)`. Backends never interpret results.

**`local`** (`ai_pipeline.backends.local`):

- `submit`: writes `PENDING`, launches `<python> -m ai_pipeline.core.jobs.runner <job_dir>` with
  `Popen(start_new_session=True)`, stdout+stderr appended to `runner.log`. The CLI can exit; the
  job keeps running. `<python>` is `compute_backends.<name>.python` (default: the CLI's own
  interpreter).
- `status`: reads `status.json`; if non-terminal and the recorded `pid` is dead, writes `FAILED`
  (`RunnerDied`, with a log tail) — a crashed runner never leaves an ambiguous state.
- `cancel`: SIGTERM (then SIGKILL) the process group, writes `CANCELLED`.
- `fetch`: no-op (shared filesystem).

`ssh` is accepted by the schema but returns `BACKEND_NOT_IMPLEMENTED` until M4.

## Runs (`artifacts/<p>/runs/<run_id>.json`)

```json
{
  "run_id": "run-20261103-090006-e8bd", "kind": "train",
  "project": "demo_person_detection", "task_type": "object_detection",
  "state": "COMPLETED", "backend": "local", "provider": "ultralytics",
  "candidate": {"name": "yolo11n", "weights": "yolo11n.pt"},
  "inputs": {"dataset_version": "ds-v001", "dataset_content_hash": "6bf3…",
             "dataset_dir": "/…/datasets/ds-v001", "data_config": "/…/ds-v001/data.yaml"},
  "params": {"epochs": 1, "image_size": 64, "batch_size": 8, "device": "cpu",
             "provider_args": {}, "classes": ["person"], "candidate": {...}},
  "job_dir": "demo_person_detection/runs/run-…",
  "timestamps": {"created": "…", "submitted": "…", "started": "…", "finished": "…"},
  "progress": {"epoch": 1, "total_epochs": 1},
  "outputs": {"weights": "demo_person_detection/runs/run-…/out/train/weights/best.pt",
              "metrics": "…/out/metrics.json", "result": "…/out/result.json"},
  "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mAP50": 0.0, "mAP50-95": 0.0,
              "per_class": {...}},
  "environment": {"python": "3.12.3", "platform": "…", "hostname": "…", "ai_pipeline": "…",
                  "provider": "ultralytics", "ultralytics": "8.4.199", "torch": "2.15.0+cpu",
                  "cuda": null, "cuda_available": false, "gpu": null,
                  "device": {"requested": "cpu", "used": "cpu", "policy": "explicit-cpu",
                             "cuda_available": false, "device_count": 0, "device_names": []}},
  "handle": {"pid": 1234, "python": "/…/bin/python", "backend": "local"},
  "error": null,
  "model_version": "mv-002", "label_set_id": null,
  "post_processed": true, "post_error": null
}
```

Kind-specific `inputs`: train — `dataset_version, dataset_content_hash, dataset_dir,
data_config`; evaluate — `model_version, dataset_version, split, weights, data_config`;
autolabel — `import_ids, asset_count` (the image list is staged in `inputs/images.txt`).
`outputs` paths are relative to the artifact root.

### Lifecycle

```
CREATED ──submit──▶ SUBMITTED ──runner starts──▶ RUNNING ──▶ COMPLETED | FAILED | CANCELLED
```

- `train start` / `evaluate run` / `label auto` create a run (new id every time, never resume)
  and submit it. A failed submit marks the run FAILED.
- `sync_run` (behind `run status`, `--wait`, `run list`, `model register`) reads the backend
  status, advances the run (walking through RUNNING if a fast job finished before it was
  observed), copies `outputs`, `metrics`, `environment` from `result.json`, and then runs the
  **post hook** for the kind:

| kind | hook | result |
|---|---|---|
| `train` | `core.registry.register_from_run` | `ModelVersion` in state `TRAINED`, `run.model_version` set |
| `evaluate` | `core.registry.apply_evaluation` | model `metrics.eval`, `evaluation.acceptance`, state `EVALUATED → ACCEPTED \| REJECTED` |
| `autolabel` | `runs._autolabel_to_label_set` | new label set with `source.kind = model`, `run.label_set_id` set |
| `export` | `core.exports.export_from_run` | immutable `ExportArtifact` (`exp-NNN`), `run.export_id` set — see [deployment.md](deployment.md) |

Hooks are idempotent (`post_processed`); a hook failure leaves the run COMPLETED with
`post_error` and exit code 1 (`RUN_POST_PROCESS_FAILED`); the next sync retries it.

## Model versions (`artifacts/<p>/models/<mv>.json`)

```json
{
  "model_version": "mv-001", "state": "ACCEPTED",
  "project": "…", "task_type": "object_detection", "provider": "ultralytics",
  "candidate": {"name": "yolo11n", "weights": "yolo11n.pt"},
  "training_run_id": "run-…", "dataset_version": "ds-v001",
  "training_params": {...}, "backend": "local", "environment": {...},
  "artifacts": {"weights": {"path": "…/out/train/weights/best.pt", "sha256": "…", "size_bytes": 5447386}},
  "metrics": {"train": {...}, "eval": {"precision": …, "recall": …, "mAP50": …, "mAP50-95": …, "split": "test"}},
  "evaluation": {"run_id": "run-…", "dataset_version": "ds-v001", "split": "test", "at": "…",
                 "acceptance": {"passed": true, "checks": [{"metric": "mAP50", "value": 0.75, "rule": {"min": 0.3}, "passed": true}]}},
  "exports": ["exp-001"], "deployments": ["dep-001", "dep-002"],
  "history": [{"at": "…", "state": "TRAINED", "note": "registered from run run-…"},
              {"at": "…", "state": "EVALUATED", "note": "evaluated by run run-…"},
              {"at": "…", "state": "ACCEPTED", "note": "all acceptance rules passed"}],
  "created_at": "…", "updated_at": "…"
}
```

Artifacts are referenced (path + sha256), never copied: the training run directory is the
storage. Lifecycle (`core.state.MODEL_TRANSITIONS`) concerns model *quality* only:

```
TRAINED → EVALUATED → ACCEPTED | REJECTED      (re-evaluation always allowed)
any → ARCHIVED
```

Exports and deployments are separate entities; the model record only lists their ids
(`exports`, `deployments`). See [deployment.md](deployment.md).

`model select [<mv>]`: explicit selection requires an `ACCEPTED` model; implicit selection picks
the accepted model with the highest `evaluation.selection.primary_metric`. Sets the project
pointer `selected_model_version`, which `export start` and `deploy start` use by default.

## Evaluation policy

`evaluation.acceptance` in `project.yaml` is a map `metric → {min, max}`;
`core.evaluation.apply_rules` returns `{passed, checks[]}`. A missing or non-numeric metric
fails its rule. No policy is built in; metric names are the task module's (`precision`,
`recall`, `f1`, `mAP50`, `mAP50-95`), and the provider is responsible for mapping library
names onto them (ultralytics: `metrics/mAP50(B)` → `mAP50`, etc.).

## Auto-labeling

`label auto <project> [--import id]… [--weights w] [--confidence c] [--wait]`

- inputs: assets in scope (file paths staged in `inputs/images.txt`); provider/weights/
  confidence/class_map from `labeling.auto` (overridable);
- job output: `out/detections.jsonl` rows `{image, class_name (source vocabulary), bbox
  normalised xyxy, confidence}`;
- hook: maps `class_map[source] → project class`, drops unmapped source classes (counted as
  `dropped_unmapped`), writes a label set with `source = {kind: model, provider, weights,
  confidence, class_map, run_id}` and every annotation carrying `source.kind=model`,
  `source.ref=<weights>`, `source.run_id`, `source.provider`, `metadata.source_class`.

Nothing is built or merged: the new set is simply the latest READY label set, so a subsequent
`dataset build` uses it, and a reviewed/corrected set can later point at it via
`parent_label_set_id`.

## Providers

| name | module | notes |
|---|---|---|
| `fake` | `ai_pipeline.providers.fake` | deterministic; `provider_args.fake_metrics / fake_fail / fake_sleep_s` for tests; `export` writes a fake `model.onnx` |
| `ultralytics` | `ai_pipeline.providers.ultralytics` | YOLO; writes a job-local `data.resolved.yaml` with an absolute dataset path because ultralytics resolves `path:` against the CWD; **strict device policy** via `providers.device.resolve_device` (explicit GPU → `GPU_DEVICE_UNAVAILABLE` rather than CPU fallback; `auto` opts into fallback; export always runs on CPU); `export` works on a job-local copy of the weights |

Install the ultralytics extra only in the job interpreter:
`pip install -e '.[provider-ultralytics]'` and point `compute_backends.local.python` at it.
The opt-in integration test `pytest -m slow tests/integration` (with `AIP_ML_PYTHON=<that
python>`) runs train → evaluate → autolabel → ONNX export through the real provider on CPU.
