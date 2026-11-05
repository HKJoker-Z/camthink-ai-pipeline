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
| Use when | the user asks to create a detection project, import data/labels, build a dataset, train, evaluate, select, export, deploy to an edge device, check what runs on a device, roll back, or report pipeline state |
| Tool needed | one shell/SSH tool that runs `ssh ai-server <args>` and returns stdout, stderr and the exit code |
| Scope | V1: YOLO object detection (`task_type: object_detection`), targets `ng4511-lab` (NG4511) and other configured targets; no face recognition, no new task types |
| Hard rules | always JSON; decide from `status`/`error.code`/exit code, never from log text; one active deployment per device; ask a human before anything in section 6 |

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
| project | `project list` · `project create <p> [--task object_detection] [--template default]` · `project validate <p>` | name `^[a-z][a-z0-9_]*$`; create scaffolds `projects/<p>/project.yaml` |
| data | `data import <p> [<dir-or-file>] [--no-recursive]` · `data list <p>` | default source: `dataset.sources` in `project.yaml`; paths are on the AI server |
| labels | `label import <p> <dir> [--format yolo] [--classes-file f] [--import id]…` · `label auto <p> [--import id]… [--weights w] [--confidence c] [--backend b] [--wait] [--timeout s]` · `label list <p>` | `label auto` is a run (section 4) |
| dataset | `dataset validate <p> [--labels id] [--import id]… [--verify-assets]` · `dataset build <p> [--labels id] [--import id]…` · `dataset list <p>` · `dataset show <p> <ds-vNNN> [--manifest]` | validate writes nothing; build is idempotent (`created: false`) |
| train | `train start <p> [--dataset ds] [--candidate c] [--backend b] [--wait] [--timeout s]` · `train status <p> <run> [--wait] [--timeout s] [--tail N]` · `train list <p>` | new run id every time; completion auto-registers `mv-NNN` |
| runs | `run status <p> <run> [--wait] [--timeout s] [--tail N]` · `run list <p> [--kind train\|evaluate\|autolabel\|export] [--no-sync]` · `run logs <p> <run> [--tail N]` · `run cancel <p> <run>` | works for every run kind |
| evaluate | `evaluate run <p> <mv> [--dataset ds] [--split val\|test] [--backend b] [--wait] [--timeout s]` · `evaluate status <p> <run>` | sets `ACCEPTED`/`REJECTED` by `evaluation.acceptance` |
| models | `model list <p>` · `model show <p> <mv>` · `model select <p> [<mv>]` · `model register <p> <run>` | select without `<mv>` = best ACCEPTED by `selection.primary_metric` |
| export | `export start <p> [<mv>] [--backend b] [--rerun] [--wait] [--timeout s]` · `export status <p> <run>` · `export list <p>` · `export show <p> <exp-NNN>` | ONNX; identical request reuses the export (`created: false`) |
| target | `target verify <p> [--target t]` · `target setup <p> [--target t]` | verify is read-only; setup installs the runtime + user service (approval) |
| deploy | `deploy start <p> [<mv>] [--export exp] [--target t]` · `deploy status <p> [<dep>] [--full]` · `deploy list <p>` · `deploy infer <p> <image>… [--deployment dep] [--target t]` | start is synchronous and long (approval; section 4.2) |
| recovery | `rollback <p> [--to dep]` · `health <p> [--target t]` | rollback = approval; health is read-only |
| helpers (wrapper only) | `connection-test` · `detach <command…>` · `detach-status <handle>` | section 1.4 and 4.2 |

Global flags accepted before the command: `-v`/`-vv` (more logs on stderr), `--infra`,
`--projects-root`, `--artifact-root`. There is no `--yes`, `--force`, `--skip-benchmark`,
`deploy prune` or delete command of any kind.

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

Procedure to answer "which model is running on the NG4511?" / "is my deployment live?":

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
   - exit 1 `UNHEALTHY` (`healthy: false`): `details.reason` says why (service inactive, stale
     heartbeat, missing engine). `consistent: true` with `healthy: false` means the right
     deployment is installed but not serving — e.g. the Orin on 2026-11-05 reported
     `reason: "service camthink-ai-app is inactive"` while `target_current` matched. Report; do
     not restart anything yourself (`rollback`/`deploy start` need approval).
   - exit 4 `REMOTE_UNREACHABLE`: the device is off or unreachable from the AI server; nothing
     can be concluded about it. Say so.
2. Map the device ref to the model: `deploy status <project> <dep-NNN>` →
   `result.model_version`, `result.export_id`, `result.benchmark`, `result.acceptance`,
   `result.timestamps`; `model show <project> <mv>` for training/eval metrics.
3. `target verify <p>` (read-only) lists `result.current` and `result.staged` (every deployment
   directory still on the device) plus the device environment — useful when no project claims
   the device.
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
| `rollback <p>` / `rollback <p> --to <dep>` | changes what runs on the device; `--to` after `ROLLBACK_FAILED` acts on an unknown device state |
| `target setup <p>` | installs/updates the runtime bundle and the user service on the device |
| `export start … --rerun`, re-running `evaluate run` on an already ACCEPTED/REJECTED model | changes verdicts/artifacts that later stages depend on |
| `run cancel` of a run the bot did not start | someone else's work |
| editing `project.yaml` / `infrastructure.yaml`, creating a project when one with the intended name exists, anything with `sudo`, stopping services, deleting files/deployment directories/datasets/artifacts | the pipeline never deletes; there is no CLI for these — they are human actions |
| training when the project's GPU is already busy (`connection-test` → `checks.gpu`) | shared hardware |

Routine, no approval needed: `connection-test`, `status`, every `list`/`show`/`logs`/`status`
command, `health`, `target verify`, `project create` (new name) and `project validate`,
`data import`, `label import`, `label auto`, `dataset validate`/`build`, `train start`,
`evaluate run` on a TRAINED model, `model select`, `export start` (without `--rerun`),
`deploy infer`, `run status`, `detach-status`.

When approval is given, quote the approved command back and run exactly that (no changed
arguments). Never chain an approved `deploy start` into a `rollback` on your own initiative:
automatic rollback is built into `deploy start`; a manual rollback is a new approval.

---

## 7. End-to-end workflows

### 7.1 New object-detection project

```
project create <p>                       # exit 3 PROJECT_EXISTS if taken → ask, do not pick another name silently
   ↳ HUMAN: copy images (and YOLO labels) to the AI server, e.g. projects/<p>/data/raw,
            and edit projects/<p>/project.yaml (classes, dataset.sources, training.epochs/device,
            evaluation.acceptance, deployment.target). Grok Bot has no file access via the forced key.
project validate <p>                     # exit 2 CONFIG_* → report error.message/hint to the human
data import <p> [<dir>]                  # result.counts.images_new, result.total_assets
label import <p> <labels-dir> --classes-file <names.txt>     # or: label auto <p>  → poll run status
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
health <p>        → UNHEALTHY / POINTER_MISMATCH / REMOTE_UNREACHABLE (section 5)
deploy status <p> [<dep>] --full         # phases, acceptance, rollback, error.phase, device log pointers
deploy list <p>                          # which deployments exist and their states
[approval] rollback <p>                  # previous known-good; after ROLLBACK_FAILED: rollback <p> --to <last_known_good>
health <p>
```

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

---

## 9. Known limitations (V1)

- One active deployment per device; cross-project supersede is only visible through `health`
  (`POINTER_MISMATCH`) — the superseded project's record is not marked `SUPERSEDED`.
- `deploy start`, `rollback`, `target setup` are synchronous, not resumable, and have no
  concurrency guard in the CLI; `detach` serialises them for the bot only.
- Real training runs report no `progress` fields; epoch progress is log-only.
- The forced-command key gives the bot no file access: data upload, `project.yaml` edits and
  deletions are human steps. Paths given to `data import`, `label import`, `deploy infer` are
  paths on the AI server.
- `project.yaml` is the only place to change classes, thresholds, GPU and target; there are no
  CLI flags for them.
- No deployment pruning; device directories accumulate (`target verify` → `staged`).
- Device access is only from the AI server; if the AI server cannot reach a device (exit 4),
  the bot cannot either.
- Object detection only. Face recognition and other task types are out of scope for this
  integration.
