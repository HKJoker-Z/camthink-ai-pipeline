# NG4511 deployment (M5)

Our own, independent inference runtime on the CamThink NeoEdge NG4511, driven through the
generic `DeploymentTarget` interface and the M3 lifecycle. The vendor's
`camthink-detect-web.service` is **not** touched, replaced or depended upon.

```
AI server (ai-pipeline)                         NG4511 (julian@192.168.5.222, alias camthink-test)
  deploy start                                   ~/camthink-ai/                      (user-owned, no sudo)
    verify      ── ssh probes ───────────────▶     runtime/camthink_edge/            runtime bundle
    stage       ── scp onnx+runtime.json+samples ▶ deployments/dep-NNN/{model.onnx, runtime.json, samples/}
    prepare     ── camthink_edge build ────────▶   trtexec --fp16 → model.engine     (built on the device)
    smoke       ── camthink_edge smoke ────────▶   inference on the shipped samples
    benchmark   ── camthink_edge benchmark ────▶   latency/fps/RSS/tegrastats
    policy      (deployment.acceptance, on the AI server)
    activate    ── ln -sfn + systemctl --user restart ▶ current -> deployments/dep-NNN ; service reload
    health      ── camthink_edge health (polled) ▶ service active + fresh heartbeat for `current`
  rollback      ── same primitives in reverse
```

## Device facts used (inspection 2026-11-04)

L4T R36.4.3 / JetPack 6.2, TensorRT 10.3.0 + CUDA 12.5, `trtexec` at
`/usr/src/tensorrt/bin/trtexec`, Python 3.10.12, 7.4 GiB RAM (5.3 GiB available with the vendor
service running), 172 GB free NVMe, 25 W power mode, user `julian` **without passwordless
sudo**, rsync present, systemd 249. Dependencies of the runtime (numpy, TensorRT Python
bindings, libcudart, OpenCV or Pillow) are checked by `scripts/inspect_ng4511.sh` and
`ai-pipeline target setup`.

## The runtime: `camthink_edge`

Source: `src/ai_pipeline/edge_bundle/camthink_edge/` (shipped as data, never imported by the
control plane). Pure Python 3.10: stdlib + numpy + the device's `tensorrt` bindings; CUDA
memory via `ctypes` on `libcudart` (no pycuda/cuda-python); image IO via OpenCV if present,
otherwise Pillow.

| Command | Purpose |
|---|---|
| `build --onnx --engine --precision fp16 --trtexec … --workspace-mb N --timeout-s N` | ONNX → TensorRT engine with the device's `trtexec` (`--fp16 --skipInference`); atomic write; returns sha256, size, duration, builder peak RSS, TRT version |
| `infer --engine --config IMG…` | detections per image with pre/inference/post timings |
| `smoke --engine --config --min-detections N SAMPLES…` | runs on the shipped sample images; `passed` iff outputs finite and detections ≥ N |
| `benchmark --engine --config --iterations N --warmup W [--image]` | p50/p95/mean latency (H2D+execute+D2H), fps, process RSS, end-to-end ms, `tegrastats` RAM/GPU samples |
| `health --root ROOT [--service NAME]` | `current` valid, engine + runtime.json present, service `active`, heartbeat fresh (≤ 15 s) and for the current deployment |
| `serve --root ROOT [--source RTSP] --fps-limit F` | long-running loop (RTSP via OpenCV, or the deployment's samples when no source), heartbeat `service/status.json` every 2 s |
| `info` | versions of python/numpy/tensorrt/cv2/PIL/cudart |

Everything model-specific comes from **`runtime.json`**, generated per deployment from
`project.yaml` (classes, `export.image_size`, `deployment.inference`):

```json
{"schema_version": 1, "task": "object_detection", "classes": ["person"], "input_size": 640,
 "precision": "fp16",
 "preprocess": {"letterbox": true, "pad_value": 114, "scale": 0.00392156862745098, "channels": "rgb"},
 "postprocess": {"type": "yolo", "confidence": 0.25, "iou": 0.45, "max_detections": 300}}
```

Post-processing decodes the ultralytics head output `(1, 4+nc, N)` (or transposed), applies the
confidence threshold and class-aware NMS, and maps boxes back through the letterbox. Another YOLO
detection model with different classes/input size reuses the runtime by configuration only.
Parity check against ultralytics' own predictions (ONNX + onnxruntime, 128 COCO images): 100 vs
98 detections, 86 boxes identical at IoU > 0.9, confidences within ~0.02.

## Device layout

```
~/camthink-ai/
  runtime/camthink_edge/            bundle; VERSION compared on every stage/setup, re-uploaded when different
  deployments/dep-NNN/              model.onnx runtime.json deployment.json samples/ model.engine model.engine.build.log
  current -> deployments/dep-NNN    atomic (ln -sfn tmp && mv -T)
  service/status.json               heartbeat: deployment, fps, frames, errors, last detections, rss
  tmp/infer-*/                      scratch for `deploy infer` (removed afterwards)
~/.config/systemd/user/camthink-ai-inference.service   ExecStart python3 -m camthink_edge serve …
```

The TensorRT engine is built on the device and recorded on the **Deployment**
(`runtime_artifact`), never on the model version or export — engines are TRT/GPU specific.

## Target configuration (`infrastructure.yaml`)

```yaml
ng4511-lab:
  type: camthink_ng4511
  ssh_host: camthink-test                   # ~/.ssh/config alias (user, key, pinned host key)
  install_root: ~/camthink-ai               # user-owned
  service_name: camthink-ai-inference       # systemd --user unit; null = no service (health = one-shot inference)
  inference_source: "${NG4511_RTSP_URL:-}"  # RTSP URL for the service; empty = loop over samples
  trtexec: /usr/src/tensorrt/bin/trtexec
  workspace_mb: 1024                        # TensorRT builder workspace (device has ~5 GiB free)
  build_timeout_s: 1800
  fps_limit: 5                              # service frame rate cap (keeps GPU/RAM headroom for the vendor service)
```

Project-side knobs (`project.yaml` → `deployment`): `precision`, `acceptance` (benchmark
policy), `inference` (confidence/iou/max_detections), `smoke` (`samples`, `min_detections`),
`benchmark` (`iterations`, `warmup`), `health` (`timeout_s`, `interval_s` — polled after
activation because the service needs a few seconds to load the engine).

## Commands

```bash
ai-pipeline target setup  demo_person_detection      # upload bundle, install+enable the user unit (idempotent)
ai-pipeline target verify demo_person_detection      # reachability, L4T, TensorRT, trtexec, memory, current, staged
ai-pipeline deploy start  demo_person_detection      # full lifecycle; exit 0 ACTIVE / 5 policy / 1 failed-rolled back
ai-pipeline deploy status demo_person_detection --full
ai-pipeline deploy infer  demo_person_detection a.jpg b.jpg   # real inference with the active deployment
ai-pipeline health        demo_person_detection
ai-pipeline rollback      demo_person_detection      # previous known-good; --to <dep> after ROLLBACK_FAILED
bash scripts/m5_acceptance.sh demo_person_detection  # the whole sequence incl. supersede + rollback
```

## Service and sessions

The inference service is a `systemd --user` unit so no sudo is needed. For it to keep running
when no SSH session is open, user lingering must be enabled once on the device:
`loginctl enable-linger julian` (asks for the user's own password over SSH; `target setup`
warns while `Linger=no`). Without lingering the service still works while a session exists.
Setting `service_name: null` disables service management entirely; health then performs a
one-shot inference on a sample.

## Memory and GPU

The device shares its 7.4 GiB and the GPU with the vendor service (`camthink-detect-web`).
Every deployment records: `runtime_artifact.builder_peak_rss_mb` (trtexec peak RSS during the
engine build), `benchmark.metrics.memory_mb` (runtime process RSS), `benchmark.details`
(`system_mem_before/after_mb`, `tegrastats` RAM/GPU-utilisation samples). The service is
frame-rate capped (`fps_limit`). Use `deploy status --full` to inspect; adjust `workspace_mb`
downward if the build fails for memory.

## Not done / deferred

Deployment pruning (`deploy prune`, with protections) — storage is ample (172 GB); INT8; RTSP
has been implemented in `serve` (OpenCV) but is exercised only when `inference_source` is set;
the vendor service remains untouched and uninspected beyond its name.
