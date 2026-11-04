#!/usr/bin/env bash
# M4 acceptance workflow: prove the pipeline runs end to end on the real AI server, through the
# CLI only, with real GPU training. Uses the temporary validation project
# (demo_person_detection) and the small public COCO128 sample (CC BY 4.0, ~7 MB).
#
#   dataset -> real GPU training -> run tracking -> ModelVersion -> real evaluation -> ACCEPTED
#   -> model select -> real ONNX export -> ExportArtifact
#
# Usage (from the repository root, inside the ai-pipeline environment):
#   bash scripts/m4_acceptance.sh [project] [--data-dir DIR] [--skip-download]
#
# Nothing in this script edits source code or project configuration; the project's own
# project.yaml (device, epochs, thresholds) governs the run. Every step is a normal CLI call;
# re-running is safe (imports dedupe, dataset build is idempotent, runs get new ids).
set -euo pipefail

PROJECT="${1:-demo_person_detection}"; shift || true
DATA_DIR=""
SKIP_DOWNLOAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

AIP="${AIP_CLI:-ai-pipeline}"
PY="${AIP_PY:-python3}"
COCO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

say() { printf '\n==> %s\n' "$*"; }
# jget <json-file> <python-expr using d (envelope) and r (result)>
jget() { "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get('result') or {}; print($2)" "$1"; }
step() {  # step <name> <cli args...> : run with JSON output, keep the envelope, fail on non-zero
  local name="$1"; shift
  local out="$WORK/$name.json"
  if "$AIP" -o json "$@" > "$out"; then
    echo "   ok   $name  ($(jget "$out" "d['operation']"))"
  else
    local code=$?
    echo "   FAIL $name (exit $code): $(jget "$out" "(d.get('error') or {}).get('code')") $(jget "$out" "(d.get('error') or {}).get('message')")" >&2
    echo "   hint: $(jget "$out" "(d.get('error') or {}).get('hint')")" >&2
    exit "$code"
  fi
}

WORK="$(mktemp -d)"
say "M4 acceptance for project '$PROJECT' (envelopes in $WORK)"
"$AIP" --version

say "1. validate project configuration"
step validate project validate "$PROJECT"
DEVICE=$("$PY" - "$PROJECT" <<'PYEOF'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(f"projects/{sys.argv[1]}/project.yaml").read_text())
print(cfg["training"]["device"])
PYEOF
)
echo "   training.device = $DEVICE  (explicit GPU requests never fall back to CPU)"

say "2. obtain sample images (COCO128)"
DATA_DIR="${DATA_DIR:-projects/$PROJECT/data/coco128}"
if [ "$SKIP_DOWNLOAD" = 1 ] || [ -d "$DATA_DIR/images" ]; then
  echo "   using existing $DATA_DIR"
else
  mkdir -p "$DATA_DIR"
  curl -fsSL -o "$WORK/coco128.zip" "$COCO_URL"
  "$PY" - "$WORK/coco128.zip" "$DATA_DIR" <<'PYEOF'
import sys, zipfile, shutil
from pathlib import Path
zip_path, dest = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as z:
    z.extractall(dest.parent / ".coco128-extract")
src = dest.parent / ".coco128-extract" / "coco128"
shutil.copytree(src / "images" / "train2017", dest / "images", dirs_exist_ok=True)
shutil.rmtree(dest.parent / ".coco128-extract")
print(f"   extracted {len(list((dest / 'images').glob('*.jpg')))} images to {dest / 'images'}")
PYEOF
fi

say "3. import images"
step import data import "$PROJECT" "$DATA_DIR/images"
echo "   assets: $(jget "$WORK/import.json" "r['total_assets']") (new: $(jget "$WORK/import.json" "r['counts']['images_new']"))"

say "4. auto-label with the pretrained model (labeling.auto; only class_map classes are kept)"
step autolabel label auto "$PROJECT" --wait
echo "   run $(jget "$WORK/autolabel.json" "r['run_id']") -> label set $(jget "$WORK/autolabel.json" "r['label_set_id']"), $(jget "$WORK/autolabel.json" "r['label_set']['counts']['annotations']") annotations on $(jget "$WORK/autolabel.json" "r['label_set']['counts']['assets_with_annotations']") images"
echo "   device used by the auto-label job: $(jget "$WORK/autolabel.json" "r.get('environment', {}).get('device', {}).get('used', '(see run)')")"

say "5. validate + build dataset version"
step dsvalidate dataset validate "$PROJECT"
step dsbuild dataset build "$PROJECT"
DS=$(jget "$WORK/dsbuild.json" "r['dataset_version']")
echo "   $DS created=$(jget "$WORK/dsbuild.json" "r['created']") hash=$(jget "$WORK/dsbuild.json" "r['content_hash'][:16]") counts=$(jget "$WORK/dsbuild.json" "r['counts']")"

say "6. real training on the local compute backend (device=$DEVICE)"
step train train start "$PROJECT" --wait --timeout 14400
RUN=$(jget "$WORK/train.json" "r['run_id']")
MV=$(jget "$WORK/train.json" "r['model_version']")
echo "   $RUN -> $MV  metrics=$(jget "$WORK/train.json" "r['metrics']")"

say "7. GPU execution proof from run metadata"
"$AIP" -o json run status "$PROJECT" "$RUN" > "$WORK/train_status.json"
"$PY" - "$WORK/train_status.json" "$DEVICE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
requested = sys.argv[2]
env = d["result"].get("environment") or {}
dev = env.get("device") or {}
print(f"   environment: python={env.get('python')} torch={env.get('torch')} cuda={env.get('cuda')} "
      f"ultralytics={env.get('ultralytics')} hostname={env.get('hostname')}")
print(f"   gpu={env.get('gpu')} cuda_available={env.get('cuda_available')}")
print(f"   device: requested={dev.get('requested')} used={dev.get('used')} policy={dev.get('policy')} "
      f"names={dev.get('device_names')}")
if requested.lower() not in ("cpu", "auto"):
    ok = (dev.get("policy") == "explicit-gpu" and dev.get("used") not in (None, "cpu")
          and env.get("cuda_available") is True)
    print("   GPU PROOF:", "PASS - training ran on CUDA" if ok else "FAIL - metadata does not prove GPU execution")
    if not ok:
        sys.exit(1)
else:
    print("   (device is cpu/auto; GPU proof not applicable)")
PYEOF

say "8. real evaluation + model acceptance policy"
step evaluate evaluate run "$PROJECT" "$MV" --wait --timeout 3600
echo "   model state: $(jget "$WORK/evaluate.json" "r['model_state']")  checks: $(jget "$WORK/evaluate.json" "[(c['metric'], c['value'], c['passed']) for c in r['acceptance']['checks']]")"

say "9. select model"
step select model select "$PROJECT"
echo "   selected $(jget "$WORK/select.json" "r['selected_model_version']") ($(jget "$WORK/select.json" "r['primary_metric']")=$(jget "$WORK/select.json" "r['value']"))"

say "10. real ONNX export"
step export export start "$PROJECT" --wait --timeout 3600
EXP=$(jget "$WORK/export.json" "r.get('export_id') or r.get('export', {}).get('export_id')")
"$AIP" -o json export show "$PROJECT" "$EXP" > "$WORK/export_show.json"
echo "   $EXP format=$(jget "$WORK/export_show.json" "r['format']") size=$(jget "$WORK/export_show.json" "r['artifact']['size_bytes']") sha256=$(jget "$WORK/export_show.json" "r['artifact']['sha256'][:16]")"
echo "   artifact: $(jget "$WORK/export_show.json" "r['artifact']['path']")"
echo "   exporter: $(jget "$WORK/export_show.json" "r['details']")"

say "11. summary"
"$AIP" status "$PROJECT"
echo
echo "M4 acceptance completed. Envelopes kept in $WORK"
