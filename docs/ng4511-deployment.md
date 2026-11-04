# NG4511 deployment (M5)

Our own, independent inference runtime on the CamThink NeoEdge NG4511, driven through the
generic `DeploymentTarget` interface and the M3 lifecycle. The vendor's
`camthink-detect-web.service` is **not** touched, replaced or depended upon (it is no longer
needed; stopping it is an operator decision requiring sudo, see "Vendor service" below).
**M5 status: accepted** — measured results at the end of this document.

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

Deployments are named on the device by their **`target_ref` = `<project>-<deployment_id>`**
(e.g. `demo_person_detection-dep-004`), so several projects can share one device without
colliding; a device has one `current` deployment at a time, and deploying project B supersedes
project A on the device (A's `health` then reports `POINTER_MISMATCH` until A is redeployed).

```
~/camthink-ai/
  runtime/camthink_edge/            bundle; VERSION compared on every stage/setup, re-uploaded when different
  deployments/<project>-dep-NNN/    model.onnx runtime.json deployment.json samples/
                                    model.engine model.engine.json (provenance) model.engine.build.log
  current -> deployments/<ref>      atomic (ln -sfn tmp && mv -T)
  service/status.json               heartbeat: deployment, fps, frames, errors, last detections, rss
  tmp/infer-*/                      scratch for `deploy infer` (removed afterwards)
~/.config/systemd/user/camthink-ai-inference.service   ExecStart python3 -m camthink_edge serve …
```

The TensorRT engine is built on the device and recorded on the **Deployment**
(`runtime_artifact`), never on the model version or export — engines are TRT/GPU specific.

**Engine reuse**: `build` writes `model.engine.json` (ONNX sha256, precision, TensorRT version,
engine sha256). A later deployment of the *identical* ONNX with the same precision on the same
TensorRT version copies that engine instead of rebuilding (`runtime_artifact.reused_from`), which
turns a ~8-minute prepare phase into seconds; a sidecar whose engine is missing or tampered is
ignored. The smoke test and benchmark still run on the copy.

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

## Vendor service

`camthink-detect-web.service` (system unit, user `julian`, port 8766, its own venv under
`~/camthink-detect-web`) ran unchanged throughout M5; both services coexisted (RAM peak
2.9 GB of 7.6 GB with our build + benchmark running). It is not required by our runtime. To
retire it, an operator with sudo can run on the device
`sudo systemctl disable --now camthink-detect-web.service`; the pipeline deliberately never
manages system services.

## M5 acceptance result (ai-server → camthink-test, 2026-11-04)

| Step | Result |
|---|---|
| `target setup` / `verify` | runtime 0.1.0 installed into `~/camthink-ai`, user unit enabled; L4T R36.4.3, Python 3.10.12, TensorRT 10.3.0, numpy 1.21.5, OpenCV 4.8.0, cudart 12.6; 5.4 GiB RAM available, 175 GB disk free |
| `deploy start` → `dep-002` | **ACTIVE** through all 10 phases |
| TensorRT FP16 build (on device) | `model.engine` 8.66 MB, sha `4ee46a7e…`, **508 s** build, trtexec peak RSS 1.68 GB; engine IO `images [1,3,640,640] f32` → `output0 [1,5,8400] f32` |
| smoke (3 test-split samples) | 2 person detections (0.49, 0.40) → passed |
| benchmark (100 iters, warmup 20) | **p50 7.67 ms, p95 10.8 ms, 121 fps** (H2D+execute+D2H); end-to-end incl. pre/post 13.3 ms; process RSS 361 MB; system available RAM 4.81 → 4.62 GB; tegrastats RAM max 2.92 GB / GPU util mean 83 %, max 92 % |
| deployment acceptance | fps 121 ≥ 15, p50 7.67 ≤ 60 → activated |
| health | service `active`, heartbeat age 1 s for `dep-002`, 0 errors |
| `deploy infer` (6 COCO images) | 1 person (0.62) on `000000000036.jpg`; per image pre ≈ 5 ms (OpenCV), inference ≈ 14 ms cold-start process incl. first-call 108 ms, post ≈ 0.3 ms |
| second `deploy start` → `dep-003` | ACTIVE, p50 6.82 ms, 131 fps; `dep-002` SUPERSEDED |
| `rollback` | `dep-003` → ROLLED_BACK, `current -> dep-002`, service restarted, healthy on the 2nd poll (~6 s), `dep-002` ACTIVE again |
| vendor service | untouched, still running |

Observations: the FP16 engine build dominates deployment time (~8.5 min at 25 W with the vendor
service resident); redeploying the *same* export rebuilds the engine — an engine cache keyed by
ONNX sha + precision + TensorRT version is a candidate M6 improvement. The failed `dep-001`
(pre-fix stage error) left `~/camthink-ai/deployments/dep-001/` (ONNX + samples, no engine) on
the device; it is harmless and can be removed by hand — there is no automatic pruning in V1.

## M6 V1 acceptance result (two new projects created by CLI, 2026-11-04)

| | `v1_person` (COCO128, `label auto`) | `v1_wildlife` (african-wildlife, `label import`) |
|---|---|---|
| classes | person | buffalo, elephant, rhino, zebra |
| assets / annotations | 128 / 151 (72 background) | 1463 unique of 1501 files (content dedup) / 2685 |
| dataset | `ds-v001` hash `ed40ff6d…` (identical to the M5 demo — deterministic identity), split 102/13/13 | `ds-v001` hash `7b48fb90…`, split 1171/146/146 |
| training (GPU 0, explicit-gpu, torch 2.15+cu132) | 20 epochs, mAP50 0.942 / mAP50-95 0.643 | 15 epochs, mAP50 0.950 / mAP50-95 0.785 |
| evaluation (test) | mAP50 0.923 → ACCEPTED | mAP50 0.944 → ACCEPTED |
| export | `exp-001` ONNX 10.66 MB | `exp-001` ONNX 10.67 MB |
| deploy A | engine 8.76 MB built in 528 s (peak RSS 1.68 GB); p50 6.97 ms / p95 10.67 ms / 131 fps; RSS 356 MB; GPU 79 % mean; ACTIVE, healthy, service 5.0 fps | engine 8.61 MB built in 526 s; p50 7.12 ms / 125 fps; RSS 353 MB; ACTIVE, healthy |
| real inference | 1 person (0.62) on 8 images, 13.8 ms/img after warm-up | 10 detections on 8 images (buffalo 0.84–0.94, one rhino), 13.9 ms/img |
| deploy B | **engine reused from `v1_person-dep-001`, prepare 0 s**, 120 fps, ACTIVE | **reused from `v1_wildlife-dep-001`, prepare 0 s**, 128 fps, ACTIVE |
| rollback | dep-002 → ROLLED_BACK, dep-001 ACTIVE, healthy, pointers consistent | same |

No pipeline source was modified between the two runs; the second project differed only in
its `project.yaml` (overlay). Deploying `v1_wildlife` superseded `v1_person` on the device as
documented.

## RTSP / camera input

`serve --source <rtsp-url>` reads frames with OpenCV (GStreamer backend present on the device)
and reconnects on stream errors; set `inference_source` (`NG4511_RTSP_URL` in `.env`) and run
`target setup` to rewrite the user unit. **Status: not yet verified with a real camera** — the
acceptance runs used the sample-image loop. A real-camera test is a separate pending item.

## Not done / deferred

Deployment pruning (`deploy prune`, with protections) — storage is ample; INT8; real-camera
RTSP verification; marking another project's deployment SUPERSEDED when a different project
takes over the device.
