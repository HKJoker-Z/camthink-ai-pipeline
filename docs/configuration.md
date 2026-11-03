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
    type: local                # local | ssh
    python: python             # interpreter with the provider extra installed
  # remote:
  #   type: ssh
  #   ssh_host: ai-server      # ~/.ssh/config alias or user@host
  #   workdir: /data/aip
  #   python: /data/venvs/aip/bin/python

deployment_targets:            # at least one
  fake:
    type: fake                 # fake | camthink_ng4511
    root: ./.fake-target
  ng4511-lab:
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
| `dataset` | `classes`, `label_format`, `sources[]`, `split{train,val,test,seed}`, `validation{min_images,min_instances_per_class,max_empty_label_ratio}` | classes unique and ordered (index = position); `label_format` must be supported by the task module; split sums to 1 with train, val > 0 |
| `dataset.sources[]` | `type: local_images` + `path`, or `type: rtsp_capture` + `url`, `frames`, `interval_s` | |
| `labeling.auto` | `provider`, `weights`, `class_map`, `confidence`, `backend` | optional; `class_map` maps pretrained class → project class, unmapped classes are dropped |
| `model` | `provider`, `candidates[{name, weights}]` | provider ∈ `ultralytics`, `fake`; candidate names unique |
| `training` | `backend`, `epochs`, `image_size`, `batch_size`, `device`, `provider_args{}` | `backend` must exist in `infrastructure.yaml`; `provider_args` are passed through unvalidated |
| `evaluation` | `split` (`val`/`test`), `metrics[]`, `acceptance{metric: {min,max}}`, `selection.primary_metric` | metrics must be known to the task module (`precision`, `recall`, `f1`, `mAP50`, `mAP50-95`); acceptance keys and `primary_metric` must be in `metrics` |
| `export` | `format` (`onnx`), `opset`, `image_size`, `dynamic` | |
| `deployment` | `target`, `runtime` (`tensorrt`), `precision` (`fp16`/`fp32`), `acceptance{}`, `health{timeout_s, interval_s}`, `keep_versions` | `target` must exist in `infrastructure.yaml`; acceptance keys ∈ `latency_ms_p50`, `latency_ms_p95`, `fps`, `memory_mb`, `gpu_util_pct` |

An acceptance rule needs at least one of `min`/`max`. All rules must pass for a candidate to
be accepted; `selection.primary_metric` picks among accepted candidates.

Validate with `ai-pipeline project validate <name>` (exit 2 and a list of
`{path, message}` errors in `--output json` on failure).

## Project record

`artifacts/<name>/project.json` holds pointers (`current_dataset_version`,
`selected_model_version`, `active_deployment`, `last_known_good_deployment`) and timestamps.
It is created by `project create` and re-created automatically on first use if the artifact
root is empty (e.g. after a fresh git checkout). It is never edited by hand.
