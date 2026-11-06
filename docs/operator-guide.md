# Operator guide

How to run `camthink-ai-pipeline` day to day — by a human or by an automation agent. Every
operation is a CLI command with `--output json`, stable ids, meaningful exit codes and
`next_operations`; nothing requires editing source code or logging into the device.

Where: the AI server (`ai-server`, user `redahk`), repository `~/camthink-ai-pipeline`,
`. .venv/bin/activate` first. All commands take the project name explicitly. An agent that
operates the pipeline remotely (Grok Bot) uses the SSH entry point and rules in
[grok-bot-operator.md](grok-bot-operator.md).

## 1. Commands

| Group | Commands |
|---|---|
| project | `project create <p> [--task object_detection] [config options…]`, `project configure <p> <config options…>`, `project show <p>`, `project validate <p>`, `project list` |
| data | `data import <p> [<dir-or-file>]`, `data list <p>` |
| label | `label import <p> <dir> [--format yolo] [--classes-file f] [--import id] [--parent lbl]`, `label auto <p> [--wait]`, `label list <p>`, `label export <p> <lbl> [--include-images]`, `label exports <p>` |
| dataset | `dataset validate <p>`, `dataset build <p>`, `dataset list <p>`, `dataset show <p> <ds>` |
| train / runs | `train start <p> [--dataset ds] [--candidate c] [--wait]`, `run status <p> <run> [--wait] [--tail N]`, `run list <p>`, `run logs <p> <run>`, `run cancel <p> <run>` |
| evaluate / model | `evaluate run <p> <mv> [--wait]`, `model list <p>`, `model show <p> <mv>`, `model register <p> <run>`, `model select <p> [<mv>]` |
| export | `export start <p> [<mv>] [--rerun] [--wait]`, `export list <p>`, `export show <p> <exp>` |
| target / deploy | `target setup <p>`, `target verify <p>`, `target status [<p>] [--target t]`, `deploy start <p> [<mv>] [--export exp] [--target t]`, `deploy status <p> [<dep>] [--full]`, `deploy list <p>`, `deploy infer <p> <img>…`, `rollback <p> [--to dep]`, `health <p>` |
| edge service | `service status <p>`, `service stop <p>`, `service start <p> [--switch]`, `service restart <p>` — control the installed application's systemd service without redeploying or rebuilding |
| overview | `status <p>` |

Global: `-o json`, `-v`/`-vv`, `--infra`, `--artifact-root`. Reference: [cli.md](cli.md).

## 2. Normal workflow

```
project create P            # once; then edit projects/P/project.yaml (classes, thresholds, target)
project validate P
data import P [<dir>]       # default: dataset.sources in project.yaml; re-import is a no-op
label import P <dir> --classes-file names.txt     # or: label auto P --wait  (pretrained model)
label export P <lbl>        # auto-labels -> review ZIP; reviewed labels come back via label import --parent <lbl>
dataset validate P          # fix reported ERRORs; WARNINGs do not block unless configured
dataset build P             # -> ds-vNNN (idempotent for identical content)
train start P --wait        # -> run-…, model mv-NNN (TRAINED)
evaluate run P mv-NNN --wait   # -> ACCEPTED | REJECTED by evaluation.acceptance
model select P              # best ACCEPTED by evaluation.selection.primary_metric
export start P --wait       # -> exp-NNN (ONNX)
target setup P              # once per device/runtime version
deploy start P              # -> dep-NNN ACTIVE | REJECTED (policy) | FAILED | ROLLED_BACK
health P                    # service active, heartbeat fresh, pointer consistent
deploy infer P img.jpg      # real detections from the device
rollback P                  # restore previous known-good deployment
```

`status P` prints the pointers (`current_dataset_version`, `selected_model_version`,
`active_deployment`, `last_known_good_deployment`), counts, model and deployment states and
`next_operations` — follow those when unsure what comes next.

## 3. JSON envelope

Every command prints exactly one JSON document on stdout with `-o json` (logs go to stderr):

```json
{"schema_version": 1, "project": "P", "operation": "deploy.start",
 "status": "ok" | "failed" | "rejected",
 "result": {...}, "error": null | {"code", "message", "details", "hint"},
 "next_operations": ["health P", "deploy status P"], "timestamp": "…"}
```

Exit codes: `0` ok · `1` operation failed (job/deployment failed, rolled back) · `2`
configuration/usage · `3` state/precondition (not found, not ready, not accepted) · `4` remote/SSH
· `5` policy rejected (metrics below acceptance, nothing activated). Decide from `status`,
`error.code` and the exit code — never from log text. Failed runs/deployments still carry their
record in `result` (state, phases, benchmark, verdicts, `error.phase`).

## 4. Monitoring

- Training / evaluation / export / auto-label are **runs**: `train start` returns immediately
  unless `--wait`. Poll `run status P <run_id>` (states `CREATED → SUBMITTED → RUNNING →
  COMPLETED | FAILED | CANCELLED`; `result.progress.epoch`, `result.environment.device`,
  `result.metrics`). `run list P` refreshes all active runs. Logs: `run logs P <run_id> --tail 50`.
  Jobs are detached; the SSH session may close.
- Deployments are **synchronous**: `deploy start` returns when the deployment is terminal
  (`ACTIVE`, `REJECTED`, `FAILED`, `ROLLED_BACK`, `ROLLBACK_FAILED`). Every phase is persisted, so
  another process can read `deploy status P <dep> --full` while it runs (TensorRT build takes
  minutes). `health P` checks the device service and heartbeat (`details.heartbeat.fps`,
  `frames`, `errors`, `last_detections`).
- Run metadata proves where training ran: `result.environment.device.policy == "explicit-gpu"`,
  `cuda_available`, `gpu`.

## 5. Failures

| Symptom | Meaning | Action |
|---|---|---|
| `dataset validate` exit 1, `DATASET_INVALID` | label/image problems or policy (`TOO_FEW_IMAGES`, `UNKNOWN_CLASS`, `BBOX_OUT_OF_BOUNDS`, …) listed in `result.report.issues` | fix labels/data or `dataset.validation` thresholds, rerun |
| run `FAILED`, `error.details.cause = GPU_DEVICE_UNAVAILABLE` | explicit GPU not available (CUDA/torch mismatch) | check `nvidia-smi`, torch wheel; never silently falls back to CPU |
| run `FAILED` otherwise | provider error; `run logs P <run>` has the traceback | fix input/config; a new `train start` creates a new run (no resume) |
| `RUN_POST_PROCESS_FAILED` | job finished but registration/label-set creation failed | fix the cause, `run status` again (hook retried) |
| model `REJECTED` | evaluation below `evaluation.acceptance` | improve data/epochs or thresholds; REJECTED models cannot be deployed |
| deploy `REJECTED` (exit 5) | benchmark below `deployment.acceptance`; nothing activated | tune model/precision/thresholds |
| deploy `FAILED` | error before activation (`error.phase`: verify/stage/prepare/smoke_test/benchmark); active deployment untouched | read `error`, fix, `deploy start` again (new id) |
| deploy `ROLLED_BACK` | activation or health failed, previous deployment restored automatically | inspect `deploy status P <dep> --full` (`health.details`, device log) |
| deploy `ROLLBACK_FAILED` | restore failed; `active_deployment` cleared, `rollback.target_current` shows device state | `health P`, then `rollback P --to <last-known-good>`; human attention |
| `health` exit 1 `UNHEALTHY` / `POINTER_MISMATCH` | service down/stale heartbeat, or device `current` differs from the project's active deployment (e.g. another project deployed to the same device) | `deploy status`, `rollback P` or redeploy |
| exit 4 `REMOTE_*` | SSH/scp problem | `ssh -o BatchMode=yes <alias> true`; `details.stderr_tail` |

Rules of thumb: commands are safe to retry; ids are never reused; nothing on the device is
overwritten; the last known-good deployment is kept for rollback.

## 6. Creating a new object-detection project

1. `project create my_detector` — scaffolds `projects/my_detector/project.yaml` from the
   template (an example overlay-driven creation is `scripts/v1_acceptance.sh`).
2. Edit `project.yaml` only:
   - `dataset.classes` (order = class index), `dataset.sources` (image folder), `dataset.validation`
     thresholds;
   - labels: `label import` of existing YOLO labels (`--classes-file` lists the label files' class
     order) or `labeling.auto.class_map` to map a pretrained model's classes onto yours;
   - `model.candidates` (e.g. `yolo11n.pt`), `training.epochs/image_size/batch_size/device`;
   - `evaluation.acceptance` / `selection`; `export.image_size`;
   - `deployment.target` (`ng4511-lab`), `deployment.acceptance` (fps/latency/memory),
     `deployment.inference` (confidence/iou), `smoke`, `health`.
3. `project validate my_detector`, then the normal workflow. No Python changes, no new target or
   runtime code: the device runtime reads classes/thresholds from the generated `runtime.json`.

Infrastructure (hosts, paths, interpreters) lives in `infrastructure.yaml`, not in projects.

## 7. Operations that need human approval

The CLI does not enforce approvals; an automation agent must ask before:

- `deploy start` to a production target (changes what runs on the device; automatic rollback
  exists but the device is briefly disrupted during activation) and `rollback`;
- `rollback --to <dep>` after `ROLLBACK_FAILED` (device state is unknown);
- `service stop` (ends inference on the device), `service restart` (interrupts it), and
  `service start --switch` (replaces the application another project runs on the device);
  plain `service start` of an application that was stopped with `service stop` is routine;
- `target setup` (installs/updates the runtime bundle and the user service on the device);
- `project configure` on a project with an active deployment or a run in progress (and any
  class change on a project that already has label sets), or editing `infrastructure.yaml`;
- deleting anything (deployment directories on the device, artifacts, datasets) — the pipeline
  never deletes; no `deploy prune` exists yet;
- stopping the vendor service or any system service (requires sudo; out of scope);
- large downloads/trainings on shared GPUs (choose `training.device` to avoid busy GPUs);
- using `--rerun` on exports or re-evaluating an accepted model (changes verdicts).

Routine, safe without approval: `status`, `*/list`, `*/show`, `health`, `target status`,
`run status/logs`, `project create`/`configure` on a fresh project, `data import`,
`label import/auto`, `dataset validate/build`, `train start`, `evaluate run`, `export start`,
`model select`, `deploy infer`, `service start` of a deliberately stopped application.
