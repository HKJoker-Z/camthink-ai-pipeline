# Grok Bot operator guide

How Grok Bot operates `camthink-ai-pipeline` V1 from natural-language requests. This file is
written to be used as a reusable Skill: sections 2–8 are the rules the bot follows; section 1
is what a human configures once. Everything below maps onto the existing CLI
(`ai-pipeline 0.1.0.dev0`) — no new service, API or database. Background and failure tables:
[operator-guide.md](operator-guide.md), [cli.md](cli.md), [deployment.md](deployment.md),
[ng4511-deployment.md](ng4511-deployment.md).

**Skill card**

| | |
|---|---|
| Name | `camthink-ai-pipeline-operator` |
| Use when | the user asks to create or configure a detection project, upload/import data/labels, build a dataset, train, evaluate, select, export, deploy to an edge device, check what runs on a device, stop/start/restart the edge application, roll back, or report pipeline state |
| Tool needed | one shell/SSH tool that runs `ssh ai-server <args>` and returns stdout, stderr and the exit code; for dataset upload the same tool must be able to feed a local file to the command's stdin (section 7.0) |
| Scope | V1: YOLO object detection (`task_type: object_detection`), targets `ng4511-lab` (NG4511) and other configured targets; no face recognition, no new task types |
| Hard rules | always JSON; decide from `status`/`error.code`/exit code, never from log text; one active deployment per device; the device's `target status` beats any project pointer; ask a human before anything in section 6 |

---

## 1. Connecting Grok Bot to the AI server (one-time setup)

Grok Bot runs in its own environment and reaches the pipeline **only over SSH** to the AI
server. Nothing else is exposed.

| Item | Value |
|---|---|
| AI server | `ai-server`, LAN IP `192.168.2.202`, SSH port 22, user `redahk` |
| Repository | `/home/redahk/camthink-ai-pipeline`, Python env `.venv` |
| Host key (ED25519) | `SHA256:mgw2j3CyVAxXsazhPqfvNtG1qu4kX6Lez2Ua7Z3uO0k` (verify out of band before pinning) |
| Entry point | `scripts/grok_bot/aip` — `cd`s into the repo, uses `.venv`, forces `--output json` |
| Devices | reached from the AI server over its existing SSH aliases (`camthink-test` = NG4511, `JestonNX8` = Orin); Grok Bot never talks to a device directly |

Assumption: Grok Bot's execution environment has a network path to `192.168.2.202:22` (same
LAN, VPN/Tailscale, or a jump host). Cursor "My Machines" access does **not** carry over; the
steps below create a dedicated key for Grok Bot.

### 1.1 On the Grok Bot side

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/grok_bot_ai_server -C grok-bot
cat ~/.ssh/grok_bot_ai_server.pub          # give this line to the AI-server admin (step 1.2)

cat >> ~/.ssh/config <<'EOF'
Host ai-server
    HostName 192.168.2.202
    User redahk
    IdentityFile ~/.ssh/grok_bot_ai_server
    IdentitiesOnly yes
    BatchMode yes
    ConnectTimeout 10
    ServerAliveInterval 30
    StrictHostKeyChecking yes
EOF
ssh-keyscan -t ed25519 192.168.2.202 >> ~/.ssh/known_hosts   # then compare the fingerprint above
```

### 1.2 On the AI server (admin, as `redahk`)

Append one line to `~/.ssh/authorized_keys` so that this key can run **nothing but the
pipeline entry point** (`restrict` also disables port/agent forwarding and PTYs):

```
restrict,command="/home/redahk/camthink-ai-pipeline/scripts/grok_bot/aip" ssh-ed25519 AAAA…grok-bot
```

With this forced command the bot's SSH arguments *are* the `ai-pipeline` arguments:
`ssh ai-server status v1_person` runs `ai-pipeline -o json status v1_person`. Arguments are
split on whitespace only (no quoting, no shell); anything that is not a CLI command is refused
by the CLI itself (exit 2). An unrestricted key would instead call the script explicitly:
`ssh ai-server /home/redahk/camthink-ai-pipeline/scripts/grok_bot/aip status v1_person`.

### 1.3 Grok Bot tool definition

One tool is enough. Suggested contract:

```
aip(args: string, timeout_s: int = 120) -> {stdout: str, stderr: str, exit_code: int}
   runs: ssh ai-server <args>
```

- `stdout` is exactly one JSON document (the envelope, section 3); `stderr` carries logs.
- Default timeout 120 s. Use a longer tool timeout only for `--wait` on short runs; for anything
  that may exceed the tool timeout use section 4 (poll runs; `detach` for deployments).
- Never run two state-changing commands for the same project concurrently.

### 1.4 Read-only connection test

```
ssh ai-server connection-test
```

Prints one JSON document (`status: ok` and exit 0 when the CLI works) with the server identity,
git branch/commit, CLI version, the project list with their pointers and `next_operations`,
GPUs (`nvidia-smi`), and whether each configured device is reachable *from the AI server*
(`checks.targets.<name>.reachable`; unreachable devices are listed in `warnings`, they do not
fail the test). It changes nothing. Run it first in every session and whenever a command returns
exit 4.

---

## 2. Commands (verified against the CLI; `<p>` = project name)

Every command is `ssh ai-server <command>` (forced key) and returns the JSON envelope.

| Stage | Command | Notes |
|---|---|---|
| overview | `status <p>` | pointers, counts, model/deployment states, `next_operations` |
| project | `project list` · `project create <p> [--task object_detection] [--template default] [config options…]` · `project configure <p> <config options…>` · `project show <p>` · `project validate <p>` | name `^[a-z][a-z0-9_]*$`; create scaffolds `projects/<p>/project.yaml` and applies the options (section 7.1); configure edits an existing file with full validation; show dumps the validated config |
| upload (wrapper only) | `upload <p> <name>.zip  < local-archive` | the only write path of the forced key: unpacks into `projects/<p>/data/uploads/<name>/` and returns the folders to import (section 7.0) |
| data | `data import <p> [<dir-or-file>] [--no-recursive]` · `data list <p>` | default source: `dataset.sources` in `project.yaml`; paths are on the AI server |
| labels | `label import <p> <dir> [--format yolo] [--classes-file f] [--import id]…` · `label auto <p> [--import id]… [--weights w] [--confidence c] [--backend b] [--wait] [--timeout s]` · `label list <p>` | `label auto` is a run (section 4) |
| dataset | `dataset validate <p> [--labels id] [--import id]… [--verify-assets]` · `dataset build <p> [--labels id] [--import id]…` · `dataset list <p>` · `dataset show <p> <ds-vNNN> [--manifest]` | validate writes nothing; build is idempotent (`created: false`) |
| train | `train start <p> [--dataset ds] [--candidate c] [--backend b] [--wait] [--timeout s]` · `train status <p> <run> [--wait] [--timeout s] [--tail N]` · `train list <p>` | new run id every time; completion auto-registers `mv-NNN` |
| runs | `run status <p> <run> [--wait] [--timeout s] [--tail N]` · `run list <p> [--kind train\|evaluate\|autolabel\|export] [--no-sync]` · `run logs <p> <run> [--tail N]` · `run cancel <p> <run>` | works for every run kind |
| evaluate | `evaluate run <p> <mv> [--dataset ds] [--split val\|test] [--backend b] [--wait] [--timeout s]` · `evaluate status <p> <run>` | sets `ACCEPTED`/`REJECTED` by `evaluation.acceptance` |
| models | `model list <p>` · `model show <p> <mv>` · `model select <p> [<mv>]` · `model register <p> <run>` | select without `<mv>` = best ACCEPTED by `selection.primary_metric` |
| export | `export start <p> [<mv>] [--backend b] [--rerun] [--wait] [--timeout s]` · `export status <p> <run>` · `export list <p>` · `export show <p> <exp-NNN>` | ONNX; identical request reuses the export (`created: false`) |
| target | `target status [<p>] [--target t]` · `target verify <p> [--target t]` · `target setup <p> [--target t]` | status = what every device (or `<p>`'s device) actually runs, read-only, exit 0 even when stopped/unreachable (section 5); verify is read-only; setup installs the runtime + user service (approval) |
| deploy | `deploy start <p> [<mv>] [--export exp] [--target t]` · `deploy status <p> [<dep>] [--full]` · `deploy list <p>` · `deploy infer <p> <image>… [--deployment dep] [--target t]` | start is synchronous and long (approval; section 4.2) |
| edge service | `service status <p>` (alias of `target status <p>`) · `service stop <p> [--target t]` · `service start <p> [--target t] [--switch]` · `service restart <p> [--target t]` | control the systemd service of the *installed* deployment: no staging, no TensorRT build, pointers untouched (section 5.2); stop/restart/`--switch` = approval |
| recovery | `rollback <p> [--to dep]` · `health <p> [--target t]` | rollback = approval; health is read-only |
| helpers (wrapper only) | `connection-test` · `detach <command…>` · `detach-status <handle>` · `upload <p> <archive>` | sections 1.4, 4.2 and 7.0 |

Global flags accepted before the command: `-v`/`-vv` (more logs on stderr), `--infra`,
`--projects-root`, `--artifact-root`. There is no `--yes`, `--force`, `--skip-benchmark`,
`deploy prune` or delete command of any kind.

### 2.1 Project configuration options (`project create` and `project configure`)

Both commands take the same options; every option maps to one `project.yaml` path and the
resulting file is validated (schema, task vocabulary, `infrastructure.yaml` references) before
anything is written. `project configure` with an invalid value exits 2 (`CONFIG_INVALID` /
`CONFIG_INFRA_MISMATCH`) and leaves the file untouched; `project create` with an invalid option
creates nothing.

| Option | `project.yaml` path | Notes |
|---|---|---|
| `--description <text>` | `project.description` | |
| `--classes cat,dog` | `dataset.classes` | order = class index; also resets `labeling.auto.class_map` to the identity map (`cat: cat, dog: dog`) unless `--auto-label-map` is given in the same call |
| `--source <dir>` | `dataset.sources` (single `local_images` entry) | relative to `projects/<p>/`, e.g. `./data/uploads/cats_dogs/images` |
| `--weights yolo11n.pt` | `model.candidates` (single candidate named after the file stem) | |
| `--auto-label-weights w` · `--auto-label-map 'cat:cat,dog:dog'` · `--auto-label-confidence c` | `labeling.auto.weights` / `.class_map` / `.confidence` | map = `pretrained:project` pairs; a bare name `cat` means `cat:cat` |
| `--epochs n` · `--image-size n` · `--batch-size n` · `--device "0"\|"1"\|cpu` | `training.epochs` / `training.image_size` **and** `export.image_size` / `training.batch_size` / `training.device` | |
| `--min-images n` · `--min-instances n` | `dataset.validation.min_images` / `.min_instances_per_class` | |
| `--min-map50 x` | `evaluation.acceptance.mAP50.min` | |
| `--target t` · `--precision fp16\|fp32` · `--confidence x` | `deployment.target` / `.precision` / `.inference.confidence` | `t` must exist in `infrastructure.yaml` |
| `--set key.path=value` (repeatable) | any other key, e.g. `--set dataset.split.train=0.7 --set deployment.keep_versions=2` | value is parsed as YAML; `project.name`, `project.task_type`, `schema_version` are protected (`SET_PATH_PROTECTED`, exit 2); unknown keys are rejected |

`project configure` returns `result.changes[] = {path, old, new}` and `result.changed`
(`false` when the requested values were already in place — safe to repeat). Comments in
`project.yaml` are not preserved; a header line points to `docs/configuration.md`. Nothing in
the table changes a project's name or task type, and no option deletes data, runs, models or
deployments.

---

## 3. Reading results

### 3.1 Envelope

```json
{"schema_version": 1, "project": "v1_person", "operation": "train.start",
 "status": "ok" | "failed" | "rejected",
 "result": {…}, "error": null | {"code": "…", "message": "…", "details": {…}, "hint": "…"},
 "next_operations": ["run status v1_person run-20261105-042301-9fcd"],
 "timestamp": "2026-11-05T04:23:01Z"}
```

Decision procedure, in this order:

1. **Exit code**: `0` ok · `1` operation failed (run/deployment failed or rolled back, internal
   error) · `2` usage/config (wrong flag, bad project.yaml, unknown command) · `3` state/
   precondition (not found, not ready, not accepted, already exists) · `4` remote/SSH to the
   device (`REMOTE_UNREACHABLE`, `REMOTE_COMMAND_FAILED`, `REMOTE_TIMEOUT`, …) · `5` policy
   rejected (metrics below acceptance; nothing was activated).
2. **`status`** and **`error.code`** (stable, machine-matchable). Report `error.message` and
   `error.hint` to the user verbatim; `error.details` holds specifics (`issues_by_code`,
   `stderr_tail`, `cause`, `phase`).
3. **`result`** is populated even on failure for runs, deployments and `dataset validate`, so one
   call is enough to explain what happened (run `state`/`error`, deployment `state`/`phases`/
   `acceptance`/`rollback`/`error.phase`, validation `report.issues`).
4. **`next_operations`** are CLI argument strings the pipeline considers sensible now. When the
   user's request is vague ("continue", "what's next"), run `status <p>` and follow them; they
   are advisory and never bypass the approval rules of section 6.

### 3.2 Identifiers (never guess them, always take them from a previous envelope)

| Id | Example | Produced by | Used by |
|---|---|---|---|
| run | `run-20261105-042301-9fcd` | `train start`, `evaluate run`, `label auto`, `export start` (`result.run_id`) | `run status/logs/cancel` |
| dataset version | `ds-v001` | `dataset build` | `train start --dataset` |
| model version | `mv-002` | train run completion (`result.model_version`) | `evaluate run`, `model select/show`, `export start`, `deploy start` |
| export | `exp-001` | `export start` (`result.export_id` or `result.export.export_id`) | `deploy start --export` |
| deployment | `dep-003` | `deploy start` (`result.deployment_id`) | `deploy status`, `rollback --to`, `deploy infer --deployment` |
| device ref | `v1_person-dep-001` | `result.target_ref` of a deployment | what the device calls the deployment (`health.target_current`) |
| detach handle | `20261105T042120Z-deploy-start-v1-person` | `detach` | `detach-status` |
| upload folder | `/home/redahk/camthink-ai-pipeline/projects/<p>/data/uploads/<name>/images` | `upload` (`result.image_root`, `result.label_root`, `result.class_name_files`) | `data import`, `label import` |

Ids are never reused; retrying a command creates a new id. Pointers (`status <p>` →
`result.pointers`): `current_dataset_version`, `selected_model_version`, `active_deployment`,
`last_known_good_deployment`.

---

## 4. Long-running operations

### 4.1 Runs (training, evaluation, export, auto-label) — asynchronous jobs

`train start`, `evaluate run`, `export start`, `label auto` return immediately with
`result.state = SUBMITTED` and a `run_id`; the job runs detached on the AI server, so the SSH
session may close.

- Poll `run status <p> <run> --tail 5` every 30–60 s (training) or 10 s (evaluation/export).
  States: `CREATED → SUBMITTED → RUNNING → COMPLETED | FAILED | CANCELLED`. Exit 0 while running
  or completed, 1 when FAILED/CANCELLED (`error.code = RUN_FAILED`, `error.details.cause`,
  `error.details.log_tail`), 3 if the id is unknown.
- Progress: with the real (ultralytics) provider `result.progress` stays `{}`; the epoch counter
  is only visible in the log (`result.log_tail` with `--tail`, or `run logs <p> <run> --tail 20`).
  Final metrics appear in `result.metrics` on completion; `result.model_version` is set when the
  model was registered; `result.environment.device` proves GPU use
  (`policy: explicit-gpu`, `used: "0"`, `cuda_available: true`).
- `--wait --timeout <s>` blocks inside one call; use it only when the tool timeout allows
  (evaluation/export of a small model: tens of seconds; training: minutes to hours — poll
  instead). A timeout returns exit 1 `RUN_WAIT_TIMEOUT`; the job keeps running.
- `run list <p>` refreshes all active runs; `run cancel <p> <run>` stops one (approval not
  required for a run the bot started itself in the same conversation; otherwise ask).
- Shared GPUs: `connection-test` lists GPU load. `training.device` in `project.yaml` is fixed per
  project (e.g. `"0"`); if that GPU is busy, tell the user instead of changing the config.

### 4.2 Synchronous operations (deploy, rollback, target setup) — use `detach`

`deploy start`, `rollback` and `target setup` block until finished (TensorRT build on the device:
~9 min on the NG4511 for a new engine, seconds when an identical engine is reused; Orin
~13 min) and are **not resumable** if the CLI process dies with the SSH session: the deployment
record would stay in a non-terminal phase. Therefore:

```
ssh ai-server detach deploy start v1_person
→ {"handle": "20261105T0430…-deploy-start-v1-person", "pid": …, "poll": "aip detach-status <handle>"}

ssh ai-server detach-status 20261105T0430…-deploy-start-v1-person
→ {"state": "running" | "finished" | "lost", "exit_code": null | 0 | 1 | 5 | …,
   "envelope": <the deploy start envelope once finished>, "stderr_tail": [...]}
```

- Poll `detach-status` every 30–60 s. While it runs, `deploy list <p>` shows the new `dep-NNN`
  with its live phase (`VERIFYING … BENCHMARKING … HEALTH_CHECKING`) and
  `deploy status <p> <dep> --full` the persisted phase results.
- `detach` refuses to start while another detached command is running
  (`DETACHED_COMMAND_RUNNING`, exit 3). Never start a second deployment to the same device
  while one is in flight.
- `state: lost` (process gone, no exit code) means the AI server killed/rebooted mid-command:
  run `health <p>` and `deploy list <p>`, report to a human, do not retry blindly.
- Final verdict comes from the `envelope` exactly as in section 3: exit 0 `ACTIVE`; 5
  `POLICY_REJECTED` (benchmark below `deployment.acceptance`, nothing changed on the device); 1
  `DEPLOYMENT_FAILED` (`error.phase` before activation, previous deployment untouched),
  `DEPLOYMENT_ROLLED_BACK` (activation/health failed, previous restored automatically),
  `DEPLOYMENT_ROLLBACK_FAILED` (device state unknown — human attention); 3 preconditions
  (`NO_EXPORT`, `MODEL_NOT_ACCEPTED`, `MODEL_REJECTED`, `NO_SELECTED_MODEL`); 4 device
  unreachable.

---

## 5. Verifying what actually runs on a device

V1 rule: **a device has exactly one `current` deployment**, named by its `target_ref`
(`<project>-dep-NNN`). Several projects may deploy to the same device; the latest activation
wins and silently supersedes the other project's deployment on the device (the superseded
project's own pointers still say `active_deployment: dep-NNN`). Pointers on the AI server are
therefore not proof; the device is.

### 5.1 `target status` — device-level truth (read-only, no approval)

```
ssh ai-server target status                 # every target in infrastructure.yaml
ssh ai-server target status v1_person       # the device v1_person deploys to, plus v1_person's pointers
ssh ai-server target status --target ng4511-lab
```

One entry per device (`result.targets[]`, or `result` itself for a single device):

```json
{"target": "ng4511-lab", "type": "camthink_ng4511", "reachable": true,
 "current": "v1_person-dep-003",            // what the device runs — authoritative
 "project": "v1_person", "deployment_id": "dep-003", "model_version": "mv-003",
 "export_id": "exp-002", "deployment_state": "ACTIVE", "project_active_deployment": "dep-003",
 "service": {"name": "camthink-ai-inference", "state": "running",
             "systemd_state": "active", "stopped": null},
 "heartbeat": {"deployment": "v1_person-dep-003", "ts": "…", "fps": 5.0, "frames": 19447,
               "errors": 0, "state": "running"},
 "healthy": true, "health_reason": null, "consistent": true, "error": null,
 "project_pointers": {…}, "this_project_is_current": true}   // only with <p>
```

- `service.state` is one of `running` · `stopped` (deliberate `service stop`; `service.stopped`
  holds who/when/which deployment) · `failed` (systemd failed) · `inactive` (not running, not
  stopped by the pipeline — unexpected) · `unknown` · `none` (no service configured) ·
  `unreachable` (`reachable: false`, `error` set). Exit code is 0 in every case: this command
  reports, it does not judge.
- `project`/`deployment_id`/`model_version` are resolved from the device's `current` ref, not from
  any project pointer. If a project's pointers disagree with the device
  (`this_project_is_current: false`, `consistent: false`), **the device is right** and the
  project's model is not running (`project_active_deployment` tells you what it believes).
- `healthy` is the device's own verdict (service active, heartbeat ≤ 15 s old and for `current`);
  `health_reason` explains a `false`. `consistent` compares the owning project's
  `active_deployment` with `current`.
- Read it with `--target` when no project claims the device; it also works for devices whose
  project folder does not exist on this server (then `project` etc. are derived from the ref
  only and `model_version` is `null`).

### 5.2 `service stop` / `start` / `restart` — control the installed application

These act on the **installed** deployment's `systemd --user` service only. Nothing is staged,
exported, re-exported or rebuilt (the TensorRT engine stays on disk, verified in section 8), no
pointer or rollback history changes, and the deployment record only gains a `service_events`
entry (`deploy status <p> <dep> --full`). Each returns the same shape:

```json
{"target": "ng4511-lab", "action": "stop|start|restart", "deployment_id": "dep-003",
 "target_ref": "v1_person-dep-003", "model_version": "mv-003", "export_id": "exp-002",
 "service": {"name": "camthink-ai-inference", "state_before": "running", "state": "stopped",
             "systemd_state": "inactive"},
 "changed": true, "switched_from": null, "current_after": "v1_person-dep-003",
 "health": {…}}                              // start/restart: the health result after waiting
```

| Command | Effect | Idempotent | Refused when |
|---|---|---|---|
| `service stop <p>` | writes a stop marker on the device, stops the service, waits until inactive. The application stays installed and `current`; `health <p>` then exits 1 with `SERVICE_STOPPED` (not `UNHEALTHY`) and `target status` shows `state: stopped` | yes (`changed: false` if already stopped) | the device runs another project (`DEVICE_OCCUPIED`, exit 3) |
| `service start <p>` | clears the marker, starts the service, waits for a healthy heartbeat, verifies `current` is still this deployment | yes | `DEVICE_OCCUPIED` when another project's deployment is current — never silently replaces it. With **approval**, `service start <p> --switch` re-points the device to `<p>`'s installed deployment (same engine, no rebuild; `switched_from` names what was running) |
| `service restart <p>` | restarts the service and waits for health; `deployment_id`/`model_version` are unchanged | — (always restarts) | `DEVICE_OCCUPIED` |

Other errors: `NO_ACTIVE_DEPLOYMENT` (3, project never deployed or rolled back to nothing),
`DEPLOYMENT_NOT_INSTALLED` (3, the directory is gone from the device → `deploy start`),
`TARGET_NO_SERVICE` / `TARGET_NO_SERVICE_CONTROL` (1, target type or config has no service →
`target setup`), `SERVICE_STOP_FAILED` / `SERVICE_START_FAILED` / `SERVICE_RESTART_FAILED` (1,
`details` carries the last status; run `target status` and report), exit 4 when the device is
unreachable. A `start`/`restart` whose health wait fails does **not** roll anything back; the
device is left for a human decision (`rollback <p>` needs approval).

The device's `service.stopped` marker is what distinguishes "an operator stopped it" from "it
crashed": `stopped` ⇒ `health` says `SERVICE_STOPPED`; `inactive`/`failed` without marker ⇒
`UNHEALTHY` with `details.reason`. `deploy start`, `rollback` and `service start` all clear the
marker.

### 5.3 Procedure

To answer "which model is running on the NG4511?" / "is my deployment live?": run
`target status <p>` (section 5.1) first; it is the one-call answer. `health <p>` is the older,
project-centric view with a verdict exit code:

1. `health <p>` — read-only, contacts the device:

```json
{"status": "ok", "result": {
  "target": "ng4511-lab", "healthy": true, "consistent": true,
  "target_current": "v1_person-dep-001",          // what the device runs now
  "active_deployment": "dep-001",                  // what this project believes
  "active_target_ref": "v1_person-dep-001",        // the same, in device naming
  "last_known_good_deployment": "dep-001",
  "details": {"service_state": "active", "engine_present": true, "config_present": true,
              "runtime_version": "0.1.1",
              "heartbeat": {"deployment": "v1_person-dep-001", "fps": 5.0, "frames": 12345,
                            "errors": 0, "last_detections": [...]}}}}
```

   The deployment is live **only if the exit code is 0, `result.healthy` is true AND
   `result.consistent` is true**. `healthy` is computed on the device (service active, fresh
   heartbeat, and the heartbeat's `deployment` equals `current`); `consistent` compares the
   device's `current` with this project's `active_target_ref`. `details.reason` is present
   whenever `healthy` is false.
   - exit 1 `POINTER_MISMATCH` (`healthy: true, consistent: false`): the device runs
     `target_current` — usually another project's deployment. Report *that* project and
     deployment (`<project>` and `dep-NNN` are readable from the ref); this project's model is
     **not** running even though its pointers say ACTIVE. Redeploying requires approval.
   - exit 1 `SERVICE_STOPPED` (`healthy: false`, `details.stopped_intentionally` present): an
     operator stopped the application with `service stop`; the right deployment is installed.
     `service start <p>` resumes it without approval (section 6).
   - exit 1 `UNHEALTHY` (`healthy: false`): `details.reason` says why (service inactive without
     a stop marker, stale heartbeat, missing engine). `consistent: true` with `healthy: false`
     means the right deployment is installed but not serving — e.g. the Orin on 2026-11-05
     reported `reason: "service camthink-ai-app is inactive"` while `target_current` matched.
     Report it; `service restart <p>` / `service start <p>` on an unexpectedly dead service is
     an approval item because the cause is unknown (`rollback`/`deploy start` likewise).
   - exit 4 `REMOTE_UNREACHABLE`: the device is off or unreachable from the AI server; nothing
     can be concluded about it. Say so.
2. Map the device ref to the model: `deploy status <project> <dep-NNN>` →
   `result.model_version`, `result.export_id`, `result.benchmark`, `result.acceptance`,
   `result.timestamps`; `model show <project> <mv>` for training/eval metrics.
3. `target verify <p>` (read-only) lists `result.current` and `result.staged` (every deployment
   directory still on the device) plus the device environment — useful to see which other
   projects' deployments are still installed (candidates for `service start --switch`).
4. Proof by inference: `deploy infer <p> <image-on-ai-server>…` runs the current deployment on
   the device and returns `result.images[].detections` with class names and confidences, plus
   `result.runtime_version` and engine info.

`health` exit 0 for project A implies every other project that targets the same device is
*not* live, whatever its pointers say.

---

## 6. Approval policy (the CLI does not enforce it — the bot must)

Ask the human, state the exact command and the expected effect, and wait for an explicit
"yes" before any of these:

| Operation | Why |
|---|---|
| `deploy start …` (any target) | replaces the device's active deployment — including another project's — and briefly interrupts inference; TensorRT build loads the device for minutes |
| `service stop <p>` | stops a live application: inference on that device ends until someone starts it again |
| `service restart <p>` | interrupts inference for a few seconds; also used on a crashed service whose cause is unknown |
| `service start <p> --switch` | replaces the application another project is running on the device (no rebuild, but the other project stops serving) |
| `service start <p>` when the service is `failed`/`inactive` **without** a stop marker | the service died unexpectedly; starting it hides the cause — report first |
| `rollback <p>` / `rollback <p> --to <dep>` | changes what runs on the device; `--to` after `ROLLBACK_FAILED` acts on an unknown device state |
| `target setup <p>` | installs/updates the runtime bundle and the user service on the device |
| `project configure <p> …` on a project that has an **active deployment** or a run in progress, and any change of `dataset.classes` / `--auto-label-map` on a project that already has label sets or datasets | changes the ground rules of existing artifacts (the pipeline keeps old versions, but the next build/train will differ) |
| `export start … --rerun`, re-running `evaluate run` on an already ACCEPTED/REJECTED model | changes verdicts/artifacts that later stages depend on |
| `run cancel` of a run the bot did not start | someone else's work |
| `upload <p> <name>` when `projects/<p>/data/uploads/<name>/` already exists | the previous extraction with that name is replaced |
| editing `infrastructure.yaml`, creating a project when one with the intended name exists, anything with `sudo`, deleting files/deployment directories/datasets/artifacts | the pipeline never deletes; there is no CLI for these — they are human actions |
| training when the project's GPU is already busy (`connection-test` → `checks.gpu`) | shared hardware |

Routine, no approval needed: `connection-test`, `status`, every `list`/`show`/`logs`/`status`
command, `target status` / `service status`, `health`, `target verify`, `project create` (new
name, with any configuration options), `project configure` on a project that has no deployment
and no run in progress, `project show`/`validate`, `upload` to a new name, `data import`,
`label import`, `label auto`, `dataset validate`/`build`, `train start`, `evaluate run` on a
TRAINED model, `model select`, `export start` (without `--rerun`), `deploy infer`, `run status`,
`detach-status`, and **`service start <p>` of an application that `target status` shows as
`stopped` (stop marker present) and whose device still has this project's deployment as
`current`** — that is resuming the installed application, no project switch, no rebuild.

When approval is given, quote the approved command back and run exactly that (no changed
arguments). Never chain an approved `deploy start` into a `rollback` on your own initiative:
automatic rollback is built into `deploy start`; a manual rollback is a new approval. Never add
`--switch` to a `service start` that was not approved with `--switch`.

---

## 7. End-to-end workflows

### 7.0 Getting a dataset from the Windows machine onto the AI server

The forced key cannot `scp`/`sftp`; the single write path is `upload`, which reads an archive
from **stdin** and unpacks it under `projects/<p>/data/uploads/<name>/` — nowhere else
(absolute paths and `..` inside the archive are rejected, `ARCHIVE_INVALID`; size cap 4 GB,
`UPLOAD_TOO_LARGE`). The project must already exist (`PROJECT_NOT_FOUND`, exit 3).

1. Build one archive locally (`.zip`, `.tar`, `.tar.gz`/`.tgz`). Any layout works; the usual one is
   `images/…` and, if labelled, `labels/…` (YOLO `.txt` per image) plus a `classes.txt`/`names.txt`.
   PowerShell: `Compress-Archive -Path .\cats_dogs -DestinationPath .\cats_dogs.zip`.
2. Send it. PowerShell has no `<` redirection, so go through `cmd`:

```
cmd /c "ssh ai-server upload cat_dog cats_dogs.zip < C:\data\cats_dogs.zip"
```

   (from WSL/Git Bash/Linux: `ssh ai-server upload cat_dog cats_dogs.zip < cats_dogs.zip`). The
   tool call must keep stdin attached and allow a timeout proportional to the size (30 MB took
   <1 s on the LAN).
3. Read the envelope: `result.images`, `result.label_files`, `result.image_root`,
   `result.label_root`, `result.class_name_files`, and `next_operations` already spelled out:

```
data import cat_dog /home/redahk/camthink-ai-pipeline/projects/cat_dog/data/uploads/cats_dogs/images
label import cat_dog /home/redahk/camthink-ai-pipeline/projects/cat_dog/data/uploads/cats_dogs/labels --format yolo --classes-file …/classes.txt
```

   or `label auto cat_dog` when the archive has no labels. Uploading the same `<name>` again
   replaces the previous extraction (approval, section 6); choose a new name for a new batch —
   `data import` deduplicates images by content anyway, so re-importing overlapping batches is
   harmless (`images_existing`).

Validation and class mapping are unchanged: `label import` maps class ids through
`--classes-file` (or the index order of `dataset.classes`), `dataset validate` reports
`UNKNOWN_CLASS`/`EMPTY_IMAGES`/… exactly as for locally copied data, and each `dataset build`
produces a new immutable `ds-vNNN` only when the content changed.

### 7.1 New object-detection project

```
project create <p> --classes cat,dog --weights yolo11n.pt --epochs 20 --device 0 \
       --min-images 100 --min-instances 50 --min-map50 0.5 --target ng4511-lab --precision fp16 \
       --source ./data/uploads/<name>/images
                                         # exit 3 PROJECT_EXISTS if taken → ask, do not pick another name silently
                                         # exit 2 CONFIG_INVALID / CONFIG_INFRA_MISMATCH → nothing created; fix the option
                                         # result.config echoes classes/candidates/training/acceptance/deployment
upload <p> <name>.zip < archive          # section 7.0 (or: HUMAN copies files to the AI server)
project configure <p> --auto-label-confidence 0.35 …   # any later adjustment; result.changes lists old→new
project show <p>                         # full validated configuration when the user asks "how is it set up?"
data import <p> <image_root>             # result.counts.images_new, result.total_assets
label import <p> <label_root> --format yolo --classes-file <names.txt>   # or: label auto <p>  → poll run status
dataset validate <p>                     # exit 1 DATASET_INVALID: report result.report.issues (by code) and stop
dataset build <p>                        # result.dataset_version (ds-vNNN)
train start <p>                          # result.run_id → poll run status <p> <run> --tail 5 until COMPLETED
                                         #   → result.model_version (mv-NNN, state TRAINED)
evaluate run <p> <mv> --wait --timeout 900   # result.model_state ACCEPTED|REJECTED, result.acceptance.checks
model select <p>                         # result.selected_model_version (best ACCEPTED by primary_metric)
export start <p> --wait --timeout 900    # result.export.export_id or result.export_id (reused)
target verify <p>                        # read-only: device reachable, result.current
[approval] target setup <p>              # first time per device / runtime version; use detach
[approval] detach deploy start <p>       # poll detach-status → envelope: ACTIVE (0) / REJECTED (5) / FAILED (1)
health <p>                               # healthy && consistent (section 5)
deploy infer <p> <image-on-server>…      # real detections from the device
```

### 7.2 Existing project, new training round

```
status <p>                               # pointers + next_operations
train start <p>   → poll run status      # mv-NNN TRAINED
evaluate run <p> <mv> --wait             # ACCEPTED?
model list <p>                           # compare eval metrics; model select <p> [<mv>] if the user wants it
export start <p> --wait                  # new export for the newly selected model
[approval] detach deploy start <p>  → poll → health <p> → deploy infer <p> <img>
```

### 7.3 "Something is wrong on the device"

```
target status <p>                        # device truth: current, service.state, heartbeat, healthy, consistent
health <p>        → SERVICE_STOPPED / UNHEALTHY / POINTER_MISMATCH / REMOTE_UNREACHABLE (section 5)
deploy status <p> [<dep>] --full         # phases, acceptance, rollback, service_events, error.phase
deploy list <p>                          # which deployments exist and their states
service start <p>                        # only if SERVICE_STOPPED (stop marker) and current == this project — no approval
[approval] service restart <p>           # service failed/inactive without marker, or hung heartbeat
[approval] rollback <p>                  # previous known-good; after ROLLBACK_FAILED: rollback <p> --to <last_known_good>
health <p>
```

### 7.4 Stop / start / restart the edge application (no retrain, no export, no rebuild)

```
target status <p>                        # confirm: current == <p>'s deployment, service.state running
[approval] service stop <p>              # result.service.state stopped, result.stopped = marker; current unchanged
target status <p>                        # service.state stopped, stopped.by = "ai-pipeline service stop"; healthy false
health <p>                               # exit 1 SERVICE_STOPPED (expected while stopped)
service start <p>                        # no approval: resumes the installed app; result.health.healthy true
health <p>                               # exit 0, healthy && consistent
[approval] service restart <p>           # deployment_id / model_version unchanged, new heartbeat pid
```

Report `deployment_id`, `model_version`, `service.state_before → state`, `changed`, and
`current_after` from each envelope. If the user's project is **not** what the device runs
(`DEVICE_OCCUPIED`), say which project/deployment is live (`error.details.current_project`,
`current_deployment`) and ask whether to `service start <p> --switch` (approval) — never do it
unasked, never pick another project's model to satisfy the request.

---

## 8. Tested on 2026-11-05 (AI server, project `v1_person`, through `scripts/grok_bot/aip`)

| Step | Result |
|---|---|
| SSH forced command (`restrict,command="…/aip"`, temporary key, from the server to itself) | `connection-test` and `status v1_person` returned envelopes; `ls /; id` was refused by the CLI (exit 2, "No such command 'ls'"); a plain login printed the wrapper usage and exited 2 |
| `connection-test` | `status: ok`; 3× RTX 3080 (GPU 1 busy at 100 %); `orin-lab` reachable, `ng4511-lab` unreachable ("No route to host") — the NG4511 was offline during the test |
| read-only chain: `status`, `project validate`, `data list`, `label list`, `dataset validate`, `dataset list`, `model list`, `export list`, `deploy list`, `deploy status` | all exit 0, `status: ok` |
| `train start v1_person` | exit 0, `SUBMITTED`, `run-20261105-042301-9fcd`, `next_operations: ["run status v1_person run-…"]` |
| poll `run status … --tail 3` every 10 s | `RUNNING` (progress `{}`), then `COMPLETED` after ~20 s: `mv-002`, mAP50 0.942, device `used: "0"`, `policy: explicit-gpu` |
| `evaluate run v1_person mv-002 --wait --timeout 600` | exit 0, `COMPLETED`, `model_state: ACCEPTED`, mAP50 0.923 ≥ 0.3 |
| `export start v1_person` | exit 0, `exp-001` reused (`created: false`, selected model still `mv-001`) |
| `run status v1_person run-nope` | exit 3, `RUN_NOT_FOUND`, hint present |
| `health v1_person` | exit 4, `REMOTE_UNREACHABLE` (NG4511 offline) — correctly inconclusive |
| `health ppe_safety` (Orin, read-only) | exit 1 `UNHEALTHY`: `target_current: ppe_safety-dep-001`, `consistent: true`, `details.reason: "service camthink-ai-app is inactive"`, heartbeat `state: stopped` — the "installed but not serving" case of section 5 |
| `target verify ppe_safety` | exit 0, `current: ppe_safety-dep-001`, `staged: ["ppe_safety-dep-001"]` |
| `detach status v1_person` → `detach-status <handle>` | `state: finished`, `exit_code: 0`, envelope embedded |

Not exercised: `deploy start`/`rollback` on the NG4511 (device offline; both require approval
anyway). `mv-002` was left `ACCEPTED` and unselected; no pointer of `v1_person` changed.

### 8.1 M7 acceptance (2026-11-05, same server, all through `scripts/grok_bot/aip`)

**Test A — new project `cat_dog` entirely through the new interface** (800 images: 400 cats,
400 dogs sampled from the Microsoft "Cats and Dogs" set, labelled by `label auto` with the
COCO `cat`/`dog` classes; GPU 0 while GPU 1 was busy with an unrelated job):

| Step | Result |
|---|---|
| `project create cat_dog --description … --classes cat,dog --weights yolo11n.pt --epochs 20 --image-size 640 --batch-size 16 --device 0 --min-images 100 --min-instances 50 --min-map50 0.5 --target ng4511-lab --precision fp16 --confidence 0.3 --source ./data/uploads/cat_dog_images` | exit 0; `result.config` showed the two classes, identity `class_map`, candidate `yolo11n`, acceptance `mAP50 ≥ 0.5`, target `ng4511-lab` |
| `upload cat_dog cat_dog_images.zip < …zip` via the forced-command path (`SSH_ORIGINAL_COMMAND`) | 30,051,272 bytes, `files_extracted: 800`, `images: 800`, `image_root: …/projects/cat_dog/data/uploads/cat_dog_images`, `next_operations: ["data import cat_dog …", "label auto cat_dog"]` |
| `project configure cat_dog --auto-label-confidence 0.35` | `changes: [{labeling.auto.confidence: 0.4 → 0.35}]` |
| `data import cat_dog <image_root>` | `images_new: 800`, `rejected: 0` |
| `label auto cat_dog` → `run status` | `COMPLETED` in 5 s, 1179 detections, label set `lbl-…86db` |
| `dataset validate cat_dog` / `dataset build cat_dog` | valid (1 warning `EMPTY_IMAGES`: 58 backgrounds), classes cat 443 / dog 396; `ds-v001` (640/80/80) |
| `train start cat_dog` → poll `run status` | `COMPLETED` after 56 s on GPU 0 (`policy: explicit-gpu`, `used: "0"`), `mv-001`, mAP50 0.930 |
| `evaluate run cat_dog mv-001` → `model select cat_dog` | test split mAP50 0.914, mAP50-95 0.733 → `ACCEPTED`; `selected_model_version: mv-001` |
| `export start cat_dog` → `export show cat_dog exp-001` | ONNX `exp-001`, 10.66 MB, sha256 recorded |
| not run | `deploy start cat_dog` — it would replace `v1_person` on `ng4511-lab`; left for approval (`status cat_dog` → `next_operations: ["deploy start cat_dog"]`) |

**Test B — service control on the live NG4511** (`ng4511-lab`, `v1_person-dep-003`, the lab
device; nothing else runs on it):

| Step | Result |
|---|---|
| `target status` (fleet) | `ng4511-lab`: `current v1_person-dep-003`, `mv-003`, `service running`, heartbeat 5 fps / 19,447 frames, `healthy: true`, `consistent: true` |
| `service stop v1_person` (approved test device) | exit 0, `running → stopped`, `changed: true`, marker `{"deployment": "v1_person-dep-003", "by": "ai-pipeline service stop", …}`; `current_after` unchanged; device: `systemctl is-active` → `inactive`, deployment directory intact |
| `health v1_person` while stopped | exit 1 `SERVICE_STOPPED`, hint "`service start v1_person` resumes it" |
| `target status v1_person` while stopped | `service.state: stopped` with the marker, heartbeat frozen (`state: stopped`), `healthy: false`, `consistent: true` |
| `service stop v1_person` again | exit 0, `changed: false` (idempotent) |
| `service start v1_person` | exit 0 in ~5 s, `stopped → running`, `health.healthy: true`, `current_after: v1_person-dep-003`, `switched_from: null`; `health v1_person` → exit 0 healthy+consistent |
| `service start v1_person` again | `changed: false` |
| `service restart v1_person` | exit 0, `deployment_id dep-003` / `mv-003` / `exp-002` unchanged, new service PID (7139 → 9890 → 10373), healthy |
| no rebuild / no retrain | `model.engine` mtime unchanged (15:03:09 before and after), `run list v1_person` 9 → 9, `export list` 2 → 2; `deploy status v1_person dep-003 --full` → `service_events: stop, stop, start, start, restart` |
| `service start v1_wildlife` (its pointer says `dep-001 ACTIVE`) | exit 3 `DEVICE_OCCUPIED`: "device ng4511-lab currently runs v1_person-dep-003, not v1_wildlife-dep-001", hint offers `--switch` with approval; device untouched; `target status v1_wildlife` → `this_project_is_current: false` |

**Test C — regression**: `connection-test` ok; forced command refused `ls /; cat /etc/passwd`
and `bash` (CLI exit 2 "No such command"); `deploy infer v1_person <image>` on the NG4511
returned 2 detections from `dep-003`; `deploy list v1_person` and `rollback --help` intact;
`ruff check`, `ruff format --check` and `pytest -m "not slow and not hardware"` (215 tests,
including the new `tests/unit/test_cli_m7.py`, NG4511 service-control simulation and
`tests/unit/test_grok_bot_upload.py`) pass.

### 8.2 Troubleshooting

| Symptom | Meaning / action |
|---|---|
| `project create`/`configure` exit 2 `CONFIG_INFRA_MISMATCH` | `--target`/backend name not in `infrastructure.yaml`; `connection-test` lists the valid names |
| `project configure` exit 2 `CONFIG_INVALID` | value out of range or unknown key under `--set`; `error.details` has the pydantic location; the file was not changed |
| `project configure` exit 2 `NO_CHANGES` | no option given |
| `upload` exit 3 `PROJECT_NOT_FOUND` | run `project create` first |
| `upload` `ARCHIVE_INVALID` / `EMPTY_UPLOAD` / `NO_STDIN` | the archive was not piped into stdin (PowerShell: use `cmd /c "… < file"`), or it contains `..`/absolute members |
| `service *` exit 3 `DEVICE_OCCUPIED` | another project is live on the device; report `error.details.current_project` and ask about `--switch` |
| `service start` exit 1 `SERVICE_START_FAILED` | the service came up but no healthy heartbeat / `current` changed underneath; run `target status --target <t>`, report, do not retry in a loop |
| `health` exit 1 `SERVICE_STOPPED` | expected after `service stop`; `service start <p>` resumes |
| `target status` shows `service.state: inactive` with `stopped: null` | the service died without a pipeline stop → approval item (`service restart`), include `health_reason` |
| `target status` `reachable: false` | device off or unreachable from the AI server; nothing can be concluded, nothing changed |

---

## 9. Known limitations (V1)

- One active deployment per device; cross-project supersede is visible through `target status`
  (`this_project_is_current: false`) and `health` (`POINTER_MISMATCH`) — the superseded
  project's record is not marked `SUPERSEDED`.
- `deploy start`, `rollback`, `target setup` are synchronous, not resumable, and have no
  concurrency guard in the CLI; `detach` serialises them for the bot only. `service stop/start/
  restart` are quick (seconds) and run inline.
- Real training runs report no `progress` fields; epoch progress is log-only.
- The forced-command key's only write path is `upload` (archives into
  `projects/<p>/data/uploads/`); deletions and `infrastructure.yaml` edits remain human steps.
  Paths given to `data import`, `label import`, `deploy infer` are paths on the AI server.
- `project configure` covers the ordinary settings (classes, sources, weights, auto-label,
  training, validation thresholds, acceptance, target/precision/confidence) and any other
  single key via `--set`; it cannot rename a project or change its task type, and it drops
  comments from `project.yaml`. Lists (several candidates/sources) go through `--set` with a
  space-free JSON value, because the forced command splits on whitespace:
  `--set model.candidates=[{"name":"yolo11n","weights":"yolo11n.pt"},{"name":"yolo11s","weights":"yolo11s.pt"}]`.
- Service control exists for `camthink_ng4511` targets (and the test `fake` target); other
  target types answer `TARGET_NO_SERVICE_CONTROL`.
- No deployment pruning; device directories accumulate (`target verify` → `staged`).
- Device access is only from the AI server; if the AI server cannot reach a device (exit 4),
  the bot cannot either.
- Object detection only. Face recognition and other task types are out of scope for this
  integration.
