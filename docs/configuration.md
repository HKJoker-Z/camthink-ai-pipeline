# Configuration

Two files, both validated strictly (unknown keys are errors, error messages include the
path of the offending key). Neither file may contain secrets; use `${VAR}` or
`${VAR:-default}` to pull values from the environment (a `.env` file in the working
directory is loaded automatically; see `.env.example`). A reference to an unset variable
without a default is a configuration error.

## `infrastructure.yaml`

Environment facts: which compute backends and deployment targets exist. Lives at the
repository root and is found by walking up from the current directory (override with
`--infra` or `AIP_INFRA_FILE`). Relative paths in this file are resolved against the file's
own directory.

```yaml
schema_version: 1
projects_root: ./projects      # override: AIP_PROJECTS_ROOT / --projects-root
artifact_root: ./artifacts     # override: AIP_ARTIFACT_ROOT / --artifact-root

compute_backends:              # at least one
  local:                       # name referenced by project.yaml
    type: local                # local | ssh (ssh: schema only until M4)
    # python: /path/to/venv/bin/python   # interpreter that runs jobs; must have the
                                         # provider extra installed. Default: the CLI's own.
  # remote:
  #   type: ssh
  #   ssh_host: ai-server      # ~/.ssh/config alias or user@host
  #   workdir: /data/aip
  #   python: /data/venvs/aip/bin/python

deployment_targets:            # at least one
  fake:
    type: fake                 # fake | camthink_ng4511
    root: ./.fake-target       # simulated device directory (relative to this file)
  ng4511-lab:                  # schema only until M5 (TARGET_NOT_IMPLEMENTED)
    type: camthink_ng4511
    ssh_host: ng4511           # alias in ~/.ssh/config (host, user, key, host-key pinning)
    install_root: /opt/camthink-ai
    service_name: camthink-ai-inference
    inference_source: "${NG4511_RTSP_URL:-}"
```

## `projects/<name>/project.yaml`

Project facts: what to detect, with which model, how good it must be, where to deploy.
Created by `ai-pipeline project create <name>` from `ai_pipeline/templates/default/`.

| Section | Keys | Notes |
|---|---|---|
| `schema_version` | `1` | |
| `project` | `name`, `task_type`, `description` | `name` must equal the directory name and match `^[a-z][a-z0-9_]{0,63}$`; `task_type` must be a registered task module (`object_detection`) |
| `dataset` | `classes`, `label_format`, `sources[]`, `split{train,val,test,seed}`, `validation{min_images,min_instances_per_class,max_empty_label_ratio,fail_on_warnings}` | classes unique and ordered (index = position); `label_format` must be supported by the task module; split sums to 1 with train, val > 0; `fail_on_warnings` (default false) makes validation warnings block `dataset build` |
| `dataset.sources[]` | `type: local_images` + `path` (relative to the project dir; default source for `data import`), or `type: rtsp_capture` + `url`, `frames`, `interval_s` (capture not implemented yet) | |
| `labeling.auto` | `provider`, `weights`, `class_map`, `confidence`, `backend` | optional; used by `label auto`. `class_map` maps the pretrained model's class names → project classes; detections of unmapped source classes are dropped |
| `model` | `provider`, `candidates[{name, weights}]` | provider ∈ `ultralytics`, `fake`; candidate names unique; `train start --candidate` picks one (default: first) |
| `training` | `backend`, `epochs`, `image_size`, `batch_size`, `device`, `provider_args{}` | `backend` must exist in `infrastructure.yaml`; `provider_args` are passed through to the provider unvalidated (fake provider: `fake_metrics`, `fake_fail`, `fake_sleep_s`) |
| `evaluation` | `split` (`val`/`test`), `metrics[]`, `acceptance{metric: {min,max}}`, `selection.primary_metric` | metrics must be known to the task module (`precision`, `recall`, `f1`, `mAP50`, `mAP50-95`); acceptance keys and `primary_metric` must be in `metrics`. All acceptance rules must pass for a model to become `ACCEPTED`; `primary_metric` ranks accepted models for `model select` |
| `export` | `format` (`onnx`), `opset`, `image_size`, `dynamic`, `provider_args{}` | these form the export *request identity*: `export start` reuses a READY export with the same model, weights hash and params |
| `deployment` | `target`, `runtime` (`tensorrt`), `precision` (`fp16`/`fp32`), `acceptance{}`, `health{timeout_s, interval_s}`, `keep_versions` | `target` must exist in `infrastructure.yaml`; acceptance keys ∈ `latency_ms_p50`, `latency_ms_p95`, `fps`, `memory_mb`, `gpu_util_pct` and gate *activation* (benchmark still runs and is persisted when empty). `health.*` and `keep_versions` are reserved for the real target (M5) |

An acceptance rule needs at least one of `min`/`max`. All rules must pass for a candidate to
be accepted; `selection.primary_metric` picks among accepted candidates.

Validate with `ai-pipeline project validate <name>` (exit 2 and a list of
`{path, message}` errors in `--output json` on failure).

## Project record

`artifacts/<name>/project.json` holds pointers (`current_dataset_version`,
`selected_model_version`, `active_deployment`, `last_known_good_deployment`) and timestamps.
It is created by `project create` and re-created automatically on first use if the artifact
root is empty (e.g. after a fresh git checkout). It is never edited by hand.
