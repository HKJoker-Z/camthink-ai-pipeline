#!/usr/bin/env bash
# M5 acceptance workflow: deploy the selected, exported model to the real NG4511 through the
# generic deployment lifecycle, run real inference, exercise rollback. Run ON THE AI SERVER
# from the repository root inside the ai-pipeline venv:
#
#   bash scripts/m5_acceptance.sh [project] [--target NAME] [--images DIR]
#
# Steps (all normal CLI calls; nothing here edits configuration):
#   target setup -> target verify -> deploy start (dep A: stage, TensorRT FP16 build, smoke,
#   benchmark, policy, activate, health) -> health -> deploy infer on real images
#   -> deploy start (dep B, supersedes A) -> rollback (restores A) -> health -> deploy list
set -euo pipefail

PROJECT="${1:-demo_person_detection}"; shift || true
TARGET=""
IMAGES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --images) IMAGES="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
AIP="${AIP_CLI:-ai-pipeline}"
PY="${AIP_PY:-python3}"
TOPT=(); [ -n "$TARGET" ] && TOPT=(--target "$TARGET")

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
show_dep() {
  local f="$1"
  echo "   $(jget "$f" "r['deployment_id']") state=$(jget "$f" "r['state']") model=$(jget "$f" "r['model_version']") export=$(jget "$f" "r['export_id']")"
  echo "   phases: $(jget "$f" "' -> '.join(r['phases'])")"
  echo "   benchmark: $(jget "$f" "r.get('benchmark')")"
  echo "   acceptance: $(jget "$f" "[(c['metric'], c['value'], c['passed']) for c in (r.get('acceptance') or {}).get('checks', [])]")"
  echo "   health: $(jget "$f" "r.get('health')")  rollback: $(jget "$f" "r.get('rollback')")"
}

WORK="$(mktemp -d)"
say "M5 acceptance for '$PROJECT' (envelopes in $WORK)"
"$AIP" --version

say "1. install/update the edge runtime + user service on the target"
step setup 0 target setup "$PROJECT" "${TOPT[@]}"
echo "   $(jget "$WORK/setup.json" "r['target']") runtime=$(jget "$WORK/setup.json" "r['bundle']") warnings=$(jget "$WORK/setup.json" "r.get('warnings')")"

say "2. verify target"
step verify 0 target verify "$PROJECT" "${TOPT[@]}"
echo "   $(jget "$WORK/verify.json" "r['environment']")"
echo "   current=$(jget "$WORK/verify.json" "r.get('current')") staged=$(jget "$WORK/verify.json" "r.get('staged')") service=$(jget "$WORK/verify.json" "r.get('service')")"

say "3. deploy (stage -> TensorRT FP16 build -> smoke -> benchmark -> policy -> activate -> health)"
step deploy_a 0 deploy start "$PROJECT" "${TOPT[@]}"
show_dep "$WORK/deploy_a.json"
DEP_A=$(jget "$WORK/deploy_a.json" "r['deployment_id']")
"$AIP" -o json deploy status "$PROJECT" "$DEP_A" --full > "$WORK/deploy_a_full.json"
echo "   engine: $(jget "$WORK/deploy_a_full.json" "{k: r['record']['runtime_artifact'].get(k) for k in ('engine_sha256','engine_size_bytes','duration_s','builder_peak_rss_mb','environment')}")"
echo "   smoke: $(jget "$WORK/deploy_a_full.json" "r['record']['smoke']")"
echo "   benchmark details: $(jget "$WORK/deploy_a_full.json" "{k: v for k, v in r['record']['benchmark']['details'].items() if k in ('latency_ms_mean','end_to_end_ms_mean','process_rss_mb','system_mem_before_mb','system_mem_after_mb','tegrastats')}")"

say "4. health"
step health_a 0 health "$PROJECT"
echo "   $(jget "$WORK/health_a.json" "r")"

say "5. real inference on images"
if [ -z "$IMAGES" ]; then
  IMAGES="projects/$PROJECT/data/coco128/images"
fi
mapfile -t IMGS < <(ls "$IMAGES"/*.jpg 2>/dev/null | head -6)
if [ "${#IMGS[@]}" -eq 0 ]; then echo "   no images under $IMAGES (use --images DIR)"; exit 1; fi
step infer 0 deploy infer "$PROJECT" "${IMGS[@]}"
"$PY" - "$WORK/infer.json" <<'PYEOF'
import json, sys
from pathlib import Path
r = json.load(open(sys.argv[1]))["result"]
print(f"   deployment={r.get('deployment_id')} runtime={r.get('runtime_version')} backend={r.get('image_backend')} engine={r.get('engine', {}).get('tensorrt')}")
for img in r["images"]:
    t = img.get("timing_ms", {})
    print(f"   {Path(img['image']).name}: {len(img['detections'])} det  pre={t.get('preprocess_ms')} inf={t.get('inference_ms')} post={t.get('postprocess_ms')} ms")
    for d in img["detections"][:3]:
        print(f"       {d['class_name']} {d['confidence']:.2f} {d['bbox_xyxy']}")
print(f"   total detections: {r['total_detections']}")
PYEOF

say "6. second deployment supersedes the first"
step deploy_b 0 deploy start "$PROJECT" "${TOPT[@]}"
show_dep "$WORK/deploy_b.json"
DEP_B=$(jget "$WORK/deploy_b.json" "r['deployment_id']")

say "7. rollback to $DEP_A"
step rollback 0 rollback "$PROJECT"
echo "   $(jget "$WORK/rollback.json" "r['deployment_id']") -> $(jget "$WORK/rollback.json" "r['state']") rollback=$(jget "$WORK/rollback.json" "r['rollback']")"
step health_b 0 health "$PROJECT"
echo "   $(jget "$WORK/health_b.json" "r")"

say "8. deployments"
"$AIP" deploy list "$PROJECT"
"$AIP" -o json status "$PROJECT" | "$PY" -c "import json,sys; r=json.load(sys.stdin)['result']; print('   pointers:', r['pointers']); print('   deployment_states:', r['deployment_states'])"
echo
echo "M5 acceptance completed. Envelopes in $WORK (deploy_a_full.json has the complete deployment record)."
