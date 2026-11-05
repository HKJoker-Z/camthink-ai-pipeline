# CLI

`ai-pipeline` is the only operator interface. Every command is non-interactive, takes the
project name explicitly, and can emit a machine-readable envelope with `--output json`.

## Global options

| Option | Purpose |
|---|---|
| `-o, --output human\|json` | output format (default `human`) |
| `--infra <file>` | `infrastructure.yaml` to use (default: search upward from CWD; env `AIP_INFRA_FILE`) |
| `--projects-root <dir>` | override `projects_root` (env `AIP_PROJECTS_ROOT`) |
| `--artifact-root <dir>` | override `artifact_root` (env `AIP_ARTIFACT_ROOT`) |
| `-v` / `-vv` | info / debug logging on stderr |
| `-q` | errors only |
| `--log-json` | JSON-lines logs on stderr |
| `--version` | print version |

Logs always go to **stderr**; stdout carries only the command's output.

## Commands implemented

### Project (M0)

| Command | Effect | Typical exit codes |
|---|---|---|
| `project create <name> [--task object_detection] [--template default] [config options…]` | scaffold `projects/<name>/project.yaml`, `README.md`, `data/raw/`; initialise `artifacts/<name>/project.json`. Configuration options (below) are applied to the template and validated before anything is written | 0, 2 (bad name/task, `CONFIG_INVALID`, `CONFIG_INFRA_MISMATCH`, `SET_PATH_PROTECTED`), 3 (exists) |
| `project configure <name> <config options…>` | validated edit of `project.yaml`: the new document must pass schema, task and infrastructure validation, then it is written atomically; `result.changes[] = {path, old, new}`, `result.changed` | 0, 2 (`NO_CHANGES`, `SET_SYNTAX`, `SET_PATH_PROTECTED`, `CONFIG_INVALID`, `CONFIG_INFRA_MISMATCH` — file untouched), 3 (not found) |
| `project show <name>` | dump the validated configuration (`result.config`) | 0, 2, 3 |
| `project validate <name>` | load and fully validate the project config, including infrastructure references | 0, 2, 3 (not found) |
| `project list` | list projects under the projects root with validity | 0 |

Configuration options (M7), identical for `create` and `configure`: `--description`,
`--classes a,b` (sets `dataset.classes` and resets `labeling.auto.class_map` to the identity map
unless `--auto-label-map` is given), `--source <dir>` (single `local_images` source),
`--weights <file>` (single model candidate), `--auto-label-weights`, `--auto-label-map
'coco:project,…'`, `--auto-label-confidence`, `--epochs`, `--image-size` (training **and**
export), `--batch-size`, `--device`, `--min-images`, `--min-instances`, `--min-map50`,
`--target`, `--precision`, `--confidence`, and `--set key.path=value` (repeatable; YAML value;
`project.name`, `project.task_type`, `schema_version` are protected). Comments in
`project.yaml` are not preserved.
| `status <name>` | pointers, entity counts (incl. assets), latest/active runs, `next_operations` | 0, 3 |

### Data, labels, datasets (M1) — see [datasets.md](datasets.md)

| Command | Effect | Typical exit codes |
|---|---|---|
| `data import <name> [<source>] [--no-recursive]` | import images into the content-addressed asset store; default source = `dataset.sources` of type `local_images` | 0 (also when nothing new), 1 (`NO_IMAGES_FOUND`, `IMPORT_SOURCE_MISSING`), 2 (`NO_LOCAL_IMAGE_SOURCE`) |
| `data list <name>` | import events and total asset count | 0 |
| `label import <name> <path> [--format yolo] [--classes-file f] [--import id]…` | convert label files to a canonical label set (`kind: imported`) | 0, 1 (`NO_ASSETS`, `AMBIGUOUS_STEMS`, `LABEL_SOURCE_MISSING`) |
| `label list <name>` | label sets with counts and parse-issue totals | 0 |
| `dataset validate <name> [--labels id] [--import id]… [--verify-assets]` | task-owned validation report; writes nothing | 0 valid, 1 `DATASET_INVALID` (report still in `result`) |
| `dataset build <name> [--labels id] [--import id]…` | validate, then build an immutable version; idempotent on content (`created: false`) | 0, 1 (`DATASET_INVALID`, `DATASET_BUILD_FAILED`) |
| `dataset list <name>` | versions with state, counts, short hash; `*` marks the current pointer | 0 |
| `dataset show <name> <version> [--manifest]` | full `dataset.json` (provenance, config snapshot, validation) | 0, 3 (`DATASET_NOT_FOUND`) |

### Runs, training, evaluation, registry (M2) — see [runs-and-models.md](runs-and-models.md)

| Command | Effect | Typical exit codes |
|---|---|---|
| `label auto <name> [--import id]… [--weights w] [--confidence c] [--backend b] [--wait] [--timeout s]` | submit an auto-label job; on completion a `kind: model` label set is created (nothing built/merged) | 0, 1 (`RUN_FAILED`), 3 (`NO_ASSETS`, `NO_AUTOLABEL_CONFIG`) |
| `train start <name> [--dataset ds] [--candidate c] [--backend b] [--wait] [--timeout s]` | create + submit a training run (new run id each time) | 0, 1 (`RUN_FAILED`, `BACKEND_NOT_IMPLEMENTED`), 3 (`NO_DATASET`, `DATASET_NOT_READY`, `CANDIDATE_UNKNOWN`) |
| `train status <name> <run_id> [--wait] [--tail N]` / `train list <name>` | aliases of `run status` / `run list --kind train` | |
| `run status <name> <run_id> [--wait] [--timeout s] [--tail N]` | sync the run with its job, run post-processing on completion | 0, 1 (`RUN_FAILED`, `RUN_CANCELLED`, `RUN_POST_PROCESS_FAILED`, `RUN_WAIT_TIMEOUT`), 3 (`RUN_NOT_FOUND`) |
| `run list <name> [--kind k] [--no-sync]` | list runs, refreshing active ones | 0 |
| `run logs <name> <run_id> [--tail N]` | runner log | 0 |
| `run cancel <name> <run_id>` | cancel a non-terminal run | 0 |
| `evaluate run <name> <mv> [--dataset ds] [--split s] [--backend b] [--wait]` | submit an evaluation; on completion the model gets `metrics.eval` and an acceptance verdict (`ACCEPTED`/`REJECTED`) | 0, 1, 3 (`MODEL_NOT_FOUND`) |
| `evaluate status <name> <run_id>` | alias of `run status` | |
| `model list <name>` / `model show <name> <mv>` | registry listing / full record | 0, 3 |
| `model register <name> <run_id>` | register a model version from a COMPLETED train run (normally automatic) | 0, 3 (`RUN_NOT_COMPLETED`, `RUN_NOT_TRAIN`) |
| `model select <name> [<mv>]` | set `selected_model_version` (explicit must be accepted; implicit = best accepted by `selection.primary_metric`) | 0, 3 (`MODEL_NOT_ACCEPTED`, `NO_ACCEPTED_MODEL`) |

Run commands return `status: failed` with the run record in `result` when the run itself
FAILED/CANCELLED, so the agent gets the error, timestamps and log pointer in one call. The
`result.environment` block (torch/CUDA/GPU, resolved `device`) is included so GPU execution can
be verified from the envelope; structured job errors appear as `error.details.cause`
(e.g. `GPU_DEVICE_UNAVAILABLE`).

### Export and deployment (M3) — see [deployment.md](deployment.md)

| Command | Effect | Typical exit codes |
|---|---|---|
| `export start <name> [<mv>] [--backend b] [--rerun] [--wait]` | export job → immutable `exp-NNN`; reuses an identical READY export unless `--rerun` | 0, 1 (`RUN_FAILED`), 3 (`NO_SELECTED_MODEL`, `MODEL_NOT_FOUND`) |
| `export status <name> <run_id>` / `export list <name>` / `export show <name> <exp>` | run alias / listing / full record | 0, 3 |
| `target verify <name> [--target t]` | reachability + environment of the target | 0, 1 (`TARGET_NOT_IMPLEMENTED`) |
| `deploy start <name> [<mv>] [--export exp] [--target t]` | verify → stage → prepare → smoke → benchmark → policy → activate → health; synchronous, every phase persisted | 0 ACTIVE, 5 `POLICY_REJECTED`, 1 `DEPLOYMENT_FAILED` / `DEPLOYMENT_ROLLED_BACK` / `DEPLOYMENT_ROLLBACK_FAILED`, 3 preconditions (`NO_EXPORT`, `MODEL_REJECTED`, `MODEL_NOT_ACCEPTED`) |
| `deploy status <name> [<dep>] [--full]` / `deploy list <name>` | inspect deployments (default: active) | 0 |
| `deploy infer <name> <image>… [--deployment dep] [--target t]` | real inference on the target with the active (or given) deployment; detections + timings | 0, 1 (`IMAGE_NOT_FOUND`, remote errors) |
| `target setup <name> [--target t]` | install/update the edge runtime bundle and the `systemd --user` service on the device (idempotent) | 0, 4 (`REMOTE_*`) |
| `rollback <name> [--to dep]` | restore the previous known-good deployment; `--to` is also the recovery path after `ROLLBACK_FAILED` | 0, 3 (`NO_ACTIVE_DEPLOYMENT`, `NO_PREVIOUS_DEPLOYMENT`), 1 |
| `health <name> [--target t]` | target health + consistency with `active_deployment` | 0, 1 (`SERVICE_STOPPED` when the service was stopped with `service stop`, `UNHEALTHY`, `POINTER_MISMATCH`) |

Deployment commands return the deployment summary in `result` for every outcome, so the
verdicts (`acceptance`, `rollback`, `error.phase`) are readable without logs.

### Edge application control (M7) — see [deployment.md](deployment.md#service-control)

| Command | Effect | Typical exit codes |
|---|---|---|
| `target status [<name>] [--target t]` | what each device actually runs: `current` ref, owning `project`, `deployment_id`, `model_version`, `service.state` (`running` / `stopped` / `failed` / `inactive` / `unknown` / `none` / `unreachable`), heartbeat, `healthy`, `consistent`; with `<name>` also that project's pointers and `this_project_is_current`. Read-only; device state wins over project pointers | 0 always (unreachable devices are reported inline) |
| `service status <name>` | alias of `target status <name>` | 0 |
| `service stop <name> [--target t]` | write a stop marker on the device and stop the inference service of the project's current deployment; nothing is removed or re-pointed | 0 (idempotent, `changed: false` if already stopped), 3 (`NO_ACTIVE_DEPLOYMENT`, `DEVICE_OCCUPIED`, `DEPLOYMENT_NOT_INSTALLED`), 1 (`SERVICE_STOP_FAILED`, `TARGET_NO_SERVICE*`), 4 |
| `service start <name> [--target t] [--switch]` | clear the marker and start the service of the installed deployment (no staging, no engine build), wait for health, verify `current`. Refuses to replace another project's deployment unless `--switch` re-points the device to this project's installed deployment | 0 (idempotent), 3 (`DEVICE_OCCUPIED`, …), 1 (`SERVICE_START_FAILED`), 4 |
| `service restart <name> [--target t]` | restart the service and wait for health; deployment id and model version unchanged | 0, 3, 1 (`SERVICE_RESTART_FAILED`), 4 |

Service actions are appended to the deployment record as `service_events` (visible with
`deploy status <name> <dep> --full`).

## JSON envelope

```json
{
  "schema_version": 1,
  "project": "demo_person_detection",
  "operation": "project.validate",
  "status": "ok",
  "result": { "...": "command-specific" },
  "error": null,
  "next_operations": ["data import demo_person_detection <path>"],
  "timestamp": "2026-11-03T07:33:00Z"
}
```

- `status` is `ok`, `failed`, or `rejected` (policy rejection; not a crash).
- On failure `result` is `{}` and `error` is
  `{ "code", "message", "details", "hint" }`. `code` is stable and machine-matchable
  (e.g. `PROJECT_NOT_FOUND`, `CONFIG_INVALID`, `CONFIG_INFRA_MISMATCH`, `PROJECT_EXISTS`,
  `ENV_VAR_MISSING`, `INFRA_FILE_MISSING`, `ILLEGAL_TRANSITION`, `DATASET_INVALID`,
  `AMBIGUOUS_STEMS`, `INTERNAL_ERROR`).
- `dataset validate` is the one command that returns `status: failed` *with* a populated
  `result` (the full report), so an agent can read the issues without a second call.
- `next_operations` are suggestions, expressed as `ai-pipeline` argument strings.
- Exactly one JSON document is written to stdout, also on failure.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | operation failed (including unexpected internal errors) |
| 2 | usage or configuration error |
| 3 | precondition / state error (not found, already exists, illegal transition) |
| 4 | remote / connectivity error (`REMOTE_UNREACHABLE`, `REMOTE_COMMAND_FAILED`, `REMOTE_OPERATION_FAILED`, `REMOTE_TRANSFER_FAILED`, `REMOTE_TIMEOUT`) |
| 5 | policy rejection (metrics below configured acceptance) |

Codes 4 and 5 are defined now and used from M2/M3 onward.
