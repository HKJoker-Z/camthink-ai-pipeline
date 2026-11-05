# Export, deployment targets, deployments and rollback

M3 adds the last stages of the lifecycle with local/fake components. The real NG4511 target
(M5) implements the same `DeploymentTarget` protocol; the orchestration, records and state
machine described here do not change.

```
ModelVersion (ACCEPTED)
    │  export start            compute job: ModelProvider.export → ONNX
    ▼
ExportArtifact  exp-NNN        immutable, portable (ONNX)
    │  deploy start
    ▼
Deployment      dep-NNN        one attempt on one target; target builds the runtime artifact
    │  verify → stage → prepare → smoke → benchmark → policy → activate → health
    ▼
ACTIVE  (or FAILED / REJECTED / ROLLED_BACK / ROLLBACK_FAILED)
```

## Two separate acceptance concepts

| Concept | Produced by | Metrics | Recorded on | Gate |
|---|---|---|---|---|
| **Model acceptance** | `evaluate run` (`evaluation.acceptance`) | precision, recall, f1, mAP50, mAP50-95 | `ModelVersion.state` = `ACCEPTED` / `REJECTED` | may this model be deployed at all |
| **Deployment acceptance** | benchmark on the target (`deployment.acceptance`) | latency_ms_p50/p95, fps, memory_mb, gpu_util_pct | `Deployment.acceptance`, state `REJECTED` | may this deployment be activated on this target |

A model can be quality-ACCEPTED and deployment-REJECTED on one target. Neither verdict changes
the other entity's state.

## ModelVersion lifecycle (unchanged by export/deploy)

```
TRAINED → EVALUATED → ACCEPTED | REJECTED     (re-evaluation always allowed; any → ARCHIVED)
```

Export and deployment never transition a model. The model record keeps `exports: [exp-…]` and
`deployments: [dep-…]` for traceability only.

## Export (`artifacts/<p>/exports/exp-NNN.json`)

```json
{
  "export_id": "exp-001", "state": "READY", "project": "demo_person_detection",
  "model_version": "mv-001", "format": "onnx", "provider": "ultralytics",
  "export_run_id": "run-20261103-…",
  "source_weights": {"path": "…/runs/run-…/out/train/weights/best.pt", "sha256": "…"},
  "export_params": {"opset": 17, "image_size": 640, "dynamic": false},
  "environment": {"python": "…", "ultralytics": "8.4.199", "torch": "2.15.0+cpu", …},
  "artifact": {"path": "…/runs/run-…/out/model.onnx", "sha256": "…", "size_bytes": 10567382},
  "details": {"exporter": "ultralytics", "opset_imports": [...], "inputs": ["images"], "outputs": ["output0"]},
  "created_at": "…"
}
```

- `export start <p> [<mv>] [--rerun] [--wait]` — default model: `selected_model_version`;
  exporting a non-selected model explicitly is allowed. Runs as a `kind: export` job on the
  compute backend (`ModelProvider.export`), so the CLI never imports the ML runtime.
- **Idempotency**: a READY export with an identical *request* (model version, source weights
  sha256, format, `export_params`) is reused (`created: false`); `--rerun` forces a new
  `exp-NNN`. Records are never overwritten; a changed parameter is a different request.
- The ultralytics provider exports from a job-local copy of the weights so the training run
  directory is never written to. Only ONNX is implemented; the schema (`format`) does not
  preclude others.
- `export.provider_args` in `project.yaml` is passed through to the provider.

## DeploymentTarget protocol (`ai_pipeline.targets.base`)

| Phase | Returns | Fake implementation |
|---|---|---|
| `verify()` | reachable, environment, current, staged | creates `<root>/target.json` |
| `stage(dep, artifact, meta)` | path, sha256, size | copies to `<root>/deployments/<dep>/model.onnx` (+ `deployment.json`) |
| `prepare(dep, precision, params)` | engine_path, engine_sha256, build_params, environment | writes a fake `model.engine` |
| `smoke_test(dep)` | passed, details | checks the engine exists |
| `benchmark(dep)` | metrics, iterations, details | deterministic synthetic metrics |
| `activate(dep)` | previous, current | atomic symlink swap `current -> deployments/<dep>` |
| `health_check()` | healthy, current, details | engine present under `current` |
| `rollback(to)` | previous, current | re-points `current` (or removes it for `None`) |
| `service_status()` (optional) | service, systemd_state, current, stopped, heartbeat | reads `<root>/service.json` |
| `service_stop()` / `service_start()` / `service_restart()` (optional) | previous_state, state | flips `<root>/service.json`; stop records a marker |

The TensorRT engine is **target-owned**: it is built by `prepare` on the device and recorded on
the Deployment (`runtime_artifact`), never on the ModelVersion or ExportArtifact.

### Fake target failure injection (tests only)

`<root>/inject.json`:

```json
{"fail": "verify|stage|prepare|smoke_test|benchmark|activate|health_check|rollback|service_stop|service_start|service_restart",
 "health_fail_for": ["dep-002"],
 "benchmark_metrics": {"fps": 9.0}}
```

`<root>` is `deployment_targets.<name>.root` (relative to `infrastructure.yaml`).

## Service control

`service stop|start|restart <p>` (`ai_pipeline.core.edge_service`) act on the systemd service of
the project's *current* deployment on the device, through the optional `service_*` methods of
the target. They never stage, build or re-point anything:

- `stop` writes `<root>/service/stopped.json` (`deployment`, `at`, `by`, `previous_state`) and
  stops the unit. The marker is how a deliberate stop is told apart from a crash: `health`
  reports `SERVICE_STOPPED` instead of `UNHEALTHY` while it exists, and `target status` shows
  `service.state: stopped` with the marker's content. `activate`, `rollback` and `service start`
  remove it.
- `start` is refused with `DEVICE_OCCUPIED` when the device's `current` belongs to another
  project — the request is never satisfied by silently serving a different model. With
  `--switch` the device is re-pointed to this project's *installed* deployment via the
  target's `rollback(to)` (symlink + restart, reusing the engine already on disk).
- `restart` requires `current` to be this project's deployment and verifies it still is after
  the health wait.
- Each action appends `{action, at, changed, switched_from, service_state}` to the deployment
  record's `service_events`; deployment state, pointers and rollback history are untouched.

`target status [<p>] [--target t]` is the device-centric read: for every target it resolves the
owning project from the `current` ref (`<project>-dep-NNN`), looks up that deployment's model
version, and reports service state, heartbeat, health and whether the owning project's
`active_deployment` agrees. It is the authority when per-project pointers are stale.

## Deployment (`artifacts/<p>/deployments/dep-NNN.json`)

```json
{
  "deployment_id": "dep-002", "state": "ROLLED_BACK", "project": "…",
  "model_version": "mv-001", "export_id": "exp-001",
  "export_artifact": {"path": "…", "sha256": "…", "size_bytes": 174},
  "target": "fake", "target_type": "fake", "runtime": "tensorrt", "precision": "fp16",
  "acceptance_rules": {"fps": {"min": 15.0}, "latency_ms_p50": {"max": 60.0}},
  "previous_deployment": "dep-001",
  "timestamps": {"created": "…", "updated": "…", "finished": "…"},
  "phases": [{"state": "CREATED", "at": "…"}, …, {"state": "ROLLED_BACK", "at": "…"}],
  "target_info": {"reachable": true, "environment": {...}, "current": "dep-001", "staged": [...]},
  "staged": {"path": "…", "sha256": "…", "size_bytes": 174},
  "runtime_artifact": {"engine_path": "…/model.engine", "engine_sha256": "…",
                       "build_params": {"precision": "fp16", "runtime": "tensorrt"}, "environment": {...}},
  "smoke": {"passed": true, "details": {...}},
  "benchmark": {"metrics": {"latency_ms_p50": 25.0, "latency_ms_p95": 41.0, "fps": 32.0,
                            "memory_mb": 1850.0, "gpu_util_pct": 55.0}, "iterations": 50, "details": {...}},
  "acceptance": {"passed": true, "checks": [{"metric": "fps", "value": 32.0, "rule": {"min": 15.0}, "passed": true}, …]},
  "activation": {"previous": "dep-001", "current": "dep-002", "at": "…"},
  "health": {"healthy": false, "current": "dep-002", "details": {"reason": "…"}},
  "rollback": {"trigger": "automatic", "reason": "HEALTH_CHECK_FAILED", "to": "dep-001", "at": "…",
               "result": {"previous": "dep-002", "current": "dep-001"}, "health": {"healthy": true, …}},
  "error": {"code": "HEALTH_CHECK_FAILED", "message": "…", "phase": "health_check", "details": {...}, "hint": null}
}
```

Every phase result is persisted as it happens, so a deployment can be inspected from another
process while it runs or after a crash. Project pointers: `active_deployment` (what is live),
`last_known_good_deployment` (rollback destination).

## Deployment state machine

```
CREATED → VERIFYING → STAGING → PREPARING → SMOKE_TESTING → BENCHMARKING
        → READY_TO_ACTIVATE → ACTIVATING → HEALTH_CHECKING → ACTIVE

pre-activation error  (CREATED … READY_TO_ACTIVATE) → FAILED      active deployment untouched
BENCHMARKING, policy not satisfied                  → REJECTED    not activated; exit code 5
ACTIVATING | HEALTH_CHECKING error                  → ROLLING_BACK → ROLLED_BACK | ROLLBACK_FAILED
ACTIVE → SUPERSEDED   (a newer deployment became ACTIVE)      SUPERSEDED → ACTIVE  (restored)
ACTIVE → ROLLING_BACK → ROLLED_BACK                 (`rollback`, manual)
```

- **Benchmark execution vs policy**: the benchmark always runs and is persisted (also with no
  rules configured, in which case `acceptance = {passed: true, checks: []}`). Rules come from
  `deployment.acceptance`; there is no `--skip-benchmark`.
- **Automatic rollback** restores `last_known_good_deployment` (the previous ACTIVE one; `None`
  for a first deployment, leaving nothing active), re-checks health, and flips the restored
  record back to `ACTIVE`.
- **`ROLLBACK_FAILED`** is explicit: the target's actual `current` is recorded
  (`rollback.target_current`), `active_deployment` is cleared because what is live is unknown,
  `last_known_good_deployment` is preserved, and `status`/`next_operations` steer to
  `health` / `deploy status` / `rollback --to <dep>`. `rollback <p> --to <dep>` is the recovery
  path when nothing is marked active.

## Commands and exit codes

| Command | Outcome → exit |
|---|---|
| `deploy start <p> [<mv>] [--export exp] [--target t]` | ACTIVE → 0; REJECTED → 5 (`POLICY_REJECTED`); FAILED / ROLLED_BACK / ROLLBACK_FAILED → 1 (`DEPLOYMENT_FAILED` / `DEPLOYMENT_ROLLED_BACK` / `DEPLOYMENT_ROLLBACK_FAILED`); preconditions → 3 (`NO_EXPORT`, `MODEL_REJECTED`, `MODEL_NOT_ACCEPTED`, `EXPORT_MODEL_MISMATCH`, `NO_SELECTED_MODEL`) |
| `deploy status <p> [<dep>] [--full]`, `deploy list <p>` | 0 |
| `rollback <p> [--to dep]` | ROLLED_BACK (or recovered) → 0; `NO_ACTIVE_DEPLOYMENT` / `NO_PREVIOUS_DEPLOYMENT` → 3; `ROLLBACK_HEALTH_FAILED` → 1 |
| `health <p> [--target t]` | healthy and consistent with `active_deployment` → 0; `UNHEALTHY` / `POINTER_MISMATCH` → 1 |
| `target verify <p> [--target t]` | 0; `TARGET_NOT_IMPLEMENTED` → 1 |

The JSON envelope of every deployment command carries the deployment summary (state, phases,
benchmark, acceptance, rollback, error, pointers) in `result` even when the status is `failed`
or `rejected`, so an agent never needs the logs to decide what happened.

## Preconditions for `deploy start`

- model: `selected_model_version` or explicit; must be `ACCEPTED` (`REJECTED` and merely
  `TRAINED`/`EVALUATED` models are refused; there is no override flag);
- export: `--export` or the latest READY export of that model in `export.format`;
- target: `deployment.target` or `--target`, must be implemented (`fake` in M3).

Deployment ids are sequential and never reused; the fake target refuses to stage into an
existing `deployments/<dep>` directory.
