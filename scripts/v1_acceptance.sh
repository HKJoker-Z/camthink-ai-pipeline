#!/usr/bin/env bash
# V1 end-to-end acceptance (M6): the complete lifecycle for a *new* project, through the CLI
# only, on the real AI server + NG4511.
#
#   create project -> (edit config = overlay) -> import data -> label (auto | import)
#   -> validate/build dataset -> GPU training -> evaluation -> select -> ONNX export
#   -> target setup/verify -> deploy (TensorRT FP16 on device, smoke, benchmark, policy,
#   activate, health) -> real inference -> second deploy (engine reuse) -> rollback -> status
#
# Run ON THE AI SERVER from the repository root inside the ai-pipeline venv:
#   bash scripts/v1_acceptance.sh v1_person   --dataset coco128
#   bash scripts/v1_acceptance.sh v1_wildlife --dataset african-wildlife
#
# Options: --overlay FILE (default scripts/acceptance/<dataset>.yaml), --reuse-project
# (continue with an existing project instead of creating it), --data-dir DIR.
# Pipeline source code is never modified; the only "edit" is the project's own project.yaml.
set -euo pipefail

PROJECT="${1:?usage: v1_acceptance.sh <project> --dataset coco128|african-wildlife [--overlay FILE]}"; shift
DATASET=""; OVERLAY=""; REUSE=0; DATA_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --overlay) OVERLAY="$2"; shift 2 ;;
    --reuse-project) REUSE=1; shift ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$DATASET" ] || { echo "--dataset is required (coco128 | african-wildlife)" >&2; exit 2; }
case "$DATASET" in
  coco128) OVERLAY="${OVERLAY:-scripts/acceptance/person_coco128.yaml}"; URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"; LABELING=auto ;;
  african-wildlife) OVERLAY="${OVERLAY:-scripts/acceptance/wildlife.yaml}"; URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"; LABELING=import ;;
  *) echo "unknown dataset $DATASET" >&2; exit 2 ;;
esac
DATA_DIR="${DATA_DIR:-projects/$PROJECT/data/$DATASET}"
AIP="${AIP_CLI:-ai-pipeline}"; PY="${AIP_PY:-python3}"

say() { printf '\n==> %s\n' "$*"; }
jget() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get('result') or {}; print($2)" "$1"; }
step() {  # step <name> <expected-exit> <cli args...>
  local name="$1" want="$2"; shift 2
  local out="$WORK/$name.json"; set +e
  "$AIP" -o json "$@" > "$out"; local code=$?; set -e
  if [ "$code" = "$want" ]; then
    echo "   ok   $name (exit $code, $(jget "$out" "d['operation']"))"
  else
    echo "   FAIL $name: exit $code (expected $want) $(jget "$out" "(d.get('error') or {}).get('code')") $(jget "$out" "(d.get('error') or {}).get('message')")" >&2
    echo "   hint: $(jget "$out" "(d.get('error') or {}).get('hint')")" >&2
    exit 1
  fi
}
WORK="$(mktemp -d)"
say "V1 acceptance: project=$PROJECT dataset=$DATASET overlay=$OVERLAY (envelopes in $WORK)"
"$AIP" --version

say "1. create project from the template and apply the configuration overlay"
if [ "$REUSE" = 1 ] && [ -f "projects/$PROJECT/project.yaml" ]; then
  echo "   reusing existing project"
else
  step create 0 project create "$PROJECT" --task object_detection
fi
"$PY" - "projects/$PROJECT/project.yaml" "$OVERLAY" <<'PYEOF'
import sys, yaml
from pathlib import Path
cfg_path, overlay_path = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load(cfg_path.read_text()); overlay = yaml.safe_load(overlay_path.read_text()) or {}
def merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v
merge(cfg, overlay)
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"   classes={cfg['dataset']['classes']} epochs={cfg['training']['epochs']} device={cfg['training']['device']} target={cfg['deployment']['target']}")
PYEOF
step validate 0 project validate "$PROJECT"

say "2. obtain dataset ($DATASET)"
if [ -d "$DATA_DIR/images" ]; then
  echo "   using existing $DATA_DIR"
else
  mkdir -p "$DATA_DIR"; curl -fsSL -o "$WORK/ds.zip" "$URL"
  "$PY" - "$WORK/ds.zip" "$DATA_DIR" "$DATASET" <<'PYEOF'
import sys, zipfile, shutil
from pathlib import Path
zip_path, dest, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ext = dest.parent / f".{kind}-extract"
with zipfile.ZipFile(zip_path) as z:
    z.extractall(ext)
if kind == "coco128":
    shutil.copytree(ext / "coco128" / "images" / "train2017", dest / "images", dirs_exist_ok=True)
else:  # african-wildlife: images/{train,val,test}, labels/{train,val,test}
    shutil.copytree(ext / "images", dest / "images", dirs_exist_ok=True)
    shutil.copytree(ext / "labels", dest / "labels", dirs_exist_ok=True)
    (dest / "names.txt").write_text("buffalo\nelephant\nrhino\nzebra\n")
shutil.rmtree(ext)
print(f"   {sum(1 for _ in (dest / 'images').rglob('*.jpg'))} images under {dest / 'images'}")
PYEOF
fi

say "3. import images"
step import 0 data import "$PROJECT" "$DATA_DIR/images"
echo "   assets: $(jget "$WORK/import.json" "r['total_assets']") (new: $(jget "$WORK/import.json" "r['counts']['images_new']"), rejected: $(jget "$WORK/import.json" "r['counts']['rejected']"))"

say "4. labels ($LABELING)"
if [ "$LABELING" = auto ]; then
  step label 0 label auto "$PROJECT" --wait
  echo "   run $(jget "$WORK/label.json" "r['run_id']") -> $(jget "$WORK/label.json" "r['label_set_id']"): $(jget "$WORK/label.json" "r['label_set']['counts']")"
else
  step label 0 label import "$PROJECT" "$DATA_DIR/labels" --format yolo --classes-file "$DATA_DIR/names.txt"
  echo "   $(jget "$WORK/label.json" "r['label_set_id']"): $(jget "$WORK/label.json" "r['counts']")"
fi

say "5. validate + build dataset version"
step dsvalidate 0 dataset validate "$PROJECT"
echo "   $(jget "$WORK/dsvalidate.json" "r['report']['stats']")  warnings=$(jget "$WORK/dsvalidate.json" "r['report']['warning_count']")"
step dsbuild 0 dataset build "$PROJECT"
echo "   $(jget "$WORK/dsbuild.json" "r['dataset_version']") created=$(jget "$WORK/dsbuild.json" "r['created']") hash=$(jget "$WORK/dsbuild.json" "r['content_hash'][:16]") splits=$(jget "$WORK/dsbuild.json" "r['counts']['splits']")"

say "6. GPU training"
step train 0 train start "$PROJECT" --wait --timeout 14400
RUN=$(jget "$WORK/train.json" "r['run_id']"); MV=$(jget "$WORK/train.json" "r['model_version']")
echo "   $RUN -> $MV metrics=$(jget "$WORK/train.json" "r['metrics']")"
echo "   device: $(jget "$WORK/train.json" "r['environment'].get('device')") gpu=$(jget "$WORK/train.json" "r['environment'].get('gpu')") torch=$(jget "$WORK/train.json" "r['environment'].get('torch')") ultralytics=$(jget "$WORK/train.json" "r['environment'].get('ultralytics')")"
"$PY" - "$WORK/train.json" <<'PYEOF'
import json, sys
env = json.load(open(sys.argv[1]))["result"]["environment"]; dev = env.get("device") or {}
if str(dev.get("requested", "")).lower() in ("cpu", "auto"):
    print("   GPU PROOF: n/a (device is cpu/auto)"); sys.exit(0)
ok = dev.get("policy") == "explicit-gpu" and dev.get("used") not in (None, "cpu") and env.get("cuda_available") is True
print("   GPU PROOF:", "PASS - training ran on CUDA" if ok else "FAIL - metadata does not prove GPU execution"); sys.exit(0 if ok else 1)
PYEOF

say "7. evaluation + model acceptance"
step evaluate 0 evaluate run "$PROJECT" "$MV" --wait --timeout 3600
echo "   $MV -> $(jget "$WORK/evaluate.json" "r['model_state']") checks=$(jget "$WORK/evaluate.json" "[(c['metric'], c['value'], c['passed']) for c in r['acceptance']['checks']]")"

say "8. select + ONNX export"
step select 0 model select "$PROJECT"
step export 0 export start "$PROJECT" --wait --timeout 3600
EXP=$(jget "$WORK/export.json" "r.get('export_id') or r.get('export', {}).get('export_id')")
"$AIP" -o json export show "$PROJECT" "$EXP" > "$WORK/export_show.json"
echo "   $EXP size=$(jget "$WORK/export_show.json" "r['artifact']['size_bytes']") sha=$(jget "$WORK/export_show.json" "r['artifact']['sha256'][:16]") details=$(jget "$WORK/export_show.json" "r['details']")"

say "9. target setup + verify"
step setup 0 target setup "$PROJECT"
echo "   runtime=$(jget "$WORK/setup.json" "r['bundle']") warnings=$(jget "$WORK/setup.json" "r.get('warnings')")"
step verify 0 target verify "$PROJECT"
echo "   current=$(jget "$WORK/verify.json" "r.get('current')") staged=$(jget "$WORK/verify.json" "r.get('staged')") mem_available_mb=$(jget "$WORK/verify.json" "r['environment'].get('mem_available_mb')")"

say "10. deploy A (stage -> TensorRT FP16 -> smoke -> benchmark -> policy -> activate -> health)"
step deploy_a 0 deploy start "$PROJECT"
DEP_A=$(jget "$WORK/deploy_a.json" "r['deployment_id']")
echo "   $DEP_A ($(jget "$WORK/deploy_a.json" "r['target_ref']")) state=$(jget "$WORK/deploy_a.json" "r['state']")"
echo "   benchmark=$(jget "$WORK/deploy_a.json" "r['benchmark']") acceptance=$(jget "$WORK/deploy_a.json" "r['acceptance']['passed']") health=$(jget "$WORK/deploy_a.json" "r['health']")"
"$AIP" -o json deploy status "$PROJECT" "$DEP_A" --full > "$WORK/deploy_a_full.json"
echo "   engine: $(jget "$WORK/deploy_a_full.json" "{k: r['record']['runtime_artifact'].get(k) for k in ('engine_sha256','engine_size_bytes','duration_s','builder_peak_rss_mb','reused_from')}")"
echo "   smoke: $(jget "$WORK/deploy_a_full.json" "r['record']['smoke']['details'].get('per_image')")"
echo "   memory/gpu: $(jget "$WORK/deploy_a_full.json" "{k: v for k, v in r['record']['benchmark']['details'].items() if k in ('end_to_end_ms_mean','process_rss_mb','system_mem_before_mb','system_mem_after_mb','tegrastats')}")"

say "11. health + real inference"
step health_a 0 health "$PROJECT"
echo "   $(jget "$WORK/health_a.json" "{k: r[k] for k in ('healthy','target_current','active_deployment','consistent')}")  heartbeat=$(jget "$WORK/health_a.json" "{k: (r['details'].get('heartbeat') or {}).get(k) for k in ('fps','frames','errors','last_detection_count')}")"
mapfile -t IMGS < <(find "$DATA_DIR/images" -type f -name '*.jpg' | sort | head -8)
step infer 0 deploy infer "$PROJECT" "${IMGS[@]}"
"$PY" - "$WORK/infer.json" <<'PYEOF'
import json, sys
from pathlib import Path
r = json.load(open(sys.argv[1]))["result"]
print(f"   runtime={r.get('runtime_version')} tensorrt={r.get('engine', {}).get('tensorrt')} classes={r.get('classes')}")
for img in r["images"]:
    t = img.get("timing_ms", {})
    dets = ", ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in img["detections"][:4])
    print(f"   {Path(img['image']).name}: {len(img['detections'])} det [{dets}]  inf={t.get('inference_ms')}ms")
print(f"   total detections: {r['total_detections']}")
PYEOF

say "12. deploy B (identical export -> engine reuse expected) then rollback to $DEP_A"
step deploy_b 0 deploy start "$PROJECT"
DEP_B=$(jget "$WORK/deploy_b.json" "r['deployment_id']")
"$AIP" -o json deploy status "$PROJECT" "$DEP_B" --full > "$WORK/deploy_b_full.json"
echo "   $DEP_B state=$(jget "$WORK/deploy_b.json" "r['state']") engine_reused_from=$(jget "$WORK/deploy_b_full.json" "r['record']['runtime_artifact'].get('reused_from')") prepare_s=$(jget "$WORK/deploy_b_full.json" "r['record']['runtime_artifact'].get('duration_s')") benchmark=$(jget "$WORK/deploy_b.json" "r['benchmark']")"
step rollback 0 rollback "$PROJECT"
echo "   $(jget "$WORK/rollback.json" "r['deployment_id']") -> $(jget "$WORK/rollback.json" "r['state']"), restored $(jget "$WORK/rollback.json" "r['rollback']['to']"), healthy=$(jget "$WORK/rollback.json" "r['rollback']['health']['healthy']")"
step health_b 0 health "$PROJECT"
echo "   $(jget "$WORK/health_b.json" "{k: r[k] for k in ('healthy','target_current','active_deployment','consistent')}")"

say "13. final state"
"$AIP" deploy list "$PROJECT"
"$AIP" -o json status "$PROJECT" | "$PY" -c "import json,sys; r=json.load(sys.stdin)['result']; print('   counts:', r['counts']); print('   pointers:', r['pointers']); print('   models:', r['model_states']); print('   deployments:', r['deployment_states']); print('   next:', r['next_operations'])"
echo
echo "V1 acceptance for $PROJECT completed. Envelopes in $WORK"
