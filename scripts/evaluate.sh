#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ASSETS_ROOT="${DEFAULT_ASSETS_ROOT:-$ROOT/assets}"
DEFAULT_ORIGIN_DIR="${DEFAULT_ORIGIN_DIR:-}"
DEFAULT_MASK_DIR="${DEFAULT_MASK_DIR:-}"
DEFAULT_REF_DIR="${DEFAULT_REF_DIR:-}"
ONBOARDING_ROOT="${ONBOARDING_ROOT:-/mnt/cpfs/users/lyw/idvtrain}"
ONBOARDING_RUNTIME="$ONBOARDING_ROOT/.cpfs_runtime/humanvid"
ONBOARDING_FACEBENCH="$ONBOARDING_ROOT/hifivfs_facebench_package_20260521"
DEFAULT_ID_MODELS_DIR="${DEFAULT_ID_MODELS_DIR:-$ONBOARDING_FACEBENCH/eval_tools/third_party/insightface_func/checkpoints/antelope}"
DEFAULT_ARCFACE_MODEL="${DEFAULT_ARCFACE_MODEL:-$ONBOARDING_RUNTIME/models/insightface/buffalo_l/w600k_r50.onnx}"
DEFAULT_CURRICULAR_MODEL="${DEFAULT_CURRICULAR_MODEL:-$ONBOARDING_RUNTIME/models/metrics/curricularface_ir101.onnx}"
DEFAULT_COSFACE_MODEL="${DEFAULT_COSFACE_MODEL:-$ONBOARDING_FACEBENCH/eval_tools/third_party/CosFace_pytorch/checkpoints/ACC99.28.pth}"
DEFAULT_POSE_MODEL="${DEFAULT_POSE_MODEL:-$ONBOARDING_FACEBENCH/eval_tools/third_party/deep_head_pose/checkpoints/hopenet_robust_alpha1.pkl}"
DEFAULT_GAZE_MODEL="${DEFAULT_GAZE_MODEL:-$ONBOARDING_FACEBENCH/models/L2CSNet_gaze360.pkl}"
DEFAULT_DEEP3D_ROOT="${DEFAULT_DEEP3D_ROOT:-$ONBOARDING_FACEBENCH/eval_tools/third_party/Deep3DFaceRecon_pytorch}"
DEFAULT_MUSIQ_MODEL="${DEFAULT_MUSIQ_MODEL:-$ONBOARDING_ROOT/third_party/vbench_runtime/weights/musiq_spaq_ckpt-358bb6af.pth}"
DEFAULT_DINO_ROOT="${DEFAULT_DINO_ROOT:-$DEFAULT_ASSETS_ROOT/models/vbench/dino}"
DINO_CODE_ROOT="$ROOT/vendor/dino"

usage() {
  sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  printf '%s\n' "Metric names: id_strict,input_leak,id_arc,id_ins,id_cur,face_similarity,pose,gaze,expression,lighting,imaging_quality,subject_consistency,temporal_flickering"
  printf '%s\n' "Groups: all,identity,identity_multi,facebench,vbench,temporal"
}

# Usage:
#   bash scripts/evaluate.sh RESULTS_DIR [options]
#   --benchmark-mode MODE     short or long (default: short)
#   --output-dir DIR          default: RESULTS_DIR/benchmark_eval_MODE
#   --manifest FILE           default: non_long_200/manifest.json for short; manifest.json for long
#   --origin-dir DIR          override manifest origin videos
#   --mask-dir DIR            override face_boxes/mask files
#   --ref-dir DIR             override reference images
#   --metrics LIST            comma-separated metrics/groups (default: all)
#   --exclude-metrics LIST    remove metrics/groups after --metrics expansion
#   --resume / --no-resume    reuse successful matching stages (default: resume)
#   --limit N                 evaluate first N manifest cases
#   --gpu-list LIST           visible GPU IDs (default: 0)
#   --model-profile NAME      onboarding or assets (default: onboarding)
#   --models-root DIR         model root for the assets profile
#   --register                register this completed eval and rebuild its gallery
#   --register-id ID          stable unique ID (default: RESULTS_DIR basename)
#   --register-label LABEL    display label (default: register ID)
#   --register-group GROUP    display group (default: Other)
#   --register-tags LIST      comma-separated display tags
#   --register-model PATH     evaluated model/checkpoint path
#   --register-model-url URL  model source URL
#   --register-steps N        inference steps recorded in the gallery
#   --register-seed N         inference seed recorded in the gallery

[[ $# -gt 0 ]] || { usage; exit 2; }
[[ "$1" != "-h" && "$1" != "--help" ]] || { usage; exit 0; }
RESULTS_DIR="$1"; shift
OUTPUT_DIR=""
MANIFEST=""
BENCHMARK_MODE="short"
ORIGIN_DIR="$DEFAULT_ORIGIN_DIR"
MASK_DIR="$DEFAULT_MASK_DIR"
REF_DIR="$DEFAULT_REF_DIR"
ASSETS_ROOT="$DEFAULT_ASSETS_ROOT"
METRICS="all"
EXCLUDE_METRICS=""
RESUME=1
FAILED_STAGES=()
LIMIT=0
GPU_LIST=0
NUM_GPUS=""
MODEL_PROFILE="onboarding"
MODELS_ROOT=""
REGISTER=0
REGISTER_ID=""
REGISTER_LABEL=""
REGISTER_GROUP="Other"
REGISTER_TAGS=""
REGISTER_MODEL=""
REGISTER_MODEL_URL=""
REGISTER_CHECKPOINT_STEP=""
REGISTER_STEPS=""
REGISTER_SEED=""
REGISTER_NOTES=""
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --benchmark-mode) BENCHMARK_MODE="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --origin-dir) ORIGIN_DIR="$2"; shift 2 ;;
    --mask-dir) MASK_DIR="$2"; shift 2 ;;
    --ref-dir) REF_DIR="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --exclude-metrics) EXCLUDE_METRICS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --gpu-list) GPU_LIST="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --model-profile) MODEL_PROFILE="$2"; shift 2 ;;
    --models-root) MODELS_ROOT="$2"; shift 2 ;;
    --register) REGISTER=1; shift ;;
    --no-register) REGISTER=0; shift ;;
    --register-id) REGISTER_ID="$2"; shift 2 ;;
    --register-label) REGISTER_LABEL="$2"; shift 2 ;;
    --register-group) REGISTER_GROUP="$2"; shift 2 ;;
    --register-tags) REGISTER_TAGS="$2"; shift 2 ;;
    --register-model) REGISTER_MODEL="$2"; shift 2 ;;
    --register-model-url) REGISTER_MODEL_URL="$2"; shift 2 ;;
    --register-checkpoint-step) REGISTER_CHECKPOINT_STEP="$2"; shift 2 ;;
    --register-steps) REGISTER_STEPS="$2"; shift 2 ;;
    --register-seed) REGISTER_SEED="$2"; shift 2 ;;
    --register-notes) REGISTER_NOTES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

RESULTS_DIR="$(realpath "$RESULTS_DIR")"
case "$BENCHMARK_MODE" in
  short)
    EVAL_MAX_FRAMES=81
    DEFAULT_MANIFEST="$ASSETS_ROOT/benchmark/non_long_200/manifest.json"
    ;;
  long)
    EVAL_MAX_FRAMES=0
    DEFAULT_MANIFEST="$ASSETS_ROOT/benchmark/manifest.json"
    ;;
  *) printf 'Unknown benchmark mode: %s (expected short or long)\n' "$BENCHMARK_MODE" >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/benchmark_eval_$BENCHMARK_MODE}"
MANIFEST="${MANIFEST:-$DEFAULT_MANIFEST}"
[[ -f "$MANIFEST" ]] || { printf 'Benchmark manifest not found: %s\n' "$MANIFEST" >&2; exit 2; }
MANIFEST="$(realpath "$MANIFEST")"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/evaluation.log") 2>&1
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"$GPU_LIST")}" 
export PYTHONPATH="$ROOT:$ROOT/vendor/facebench:${PYTHONPATH:-}"

if [[ "$MODEL_PROFILE" == "assets" ]]; then
  MODELS_ROOT="${MODELS_ROOT:-$ASSETS_ROOT/models}"
  ID_MODELS_DIR="$MODELS_ROOT/identity/antelope"
  ARCFACE_MODEL="$MODELS_ROOT/identity/arcface_w600k_r50.onnx"
  CURRICULAR_MODEL="$MODELS_ROOT/identity/curricularface_ir101.onnx"
  COSFACE_MODEL="$MODELS_ROOT/facebench/cosface_ACC99.28.pth"
  POSE_MODEL="$MODELS_ROOT/facebench/hopenet_robust_alpha1.pkl"
  GAZE_MODEL="$MODELS_ROOT/facebench/L2CSNet_gaze360.pkl"
  DEEP3D_ROOT="$MODELS_ROOT/facebench/deep3d"
  MUSIQ_MODEL="$MODELS_ROOT/vbench/musiq_spaq_ckpt-358bb6af.pth"
  DINO_ROOT="$MODELS_ROOT/vbench/dino"
elif [[ "$MODEL_PROFILE" == "onboarding" ]]; then
  ID_MODELS_DIR="$DEFAULT_ID_MODELS_DIR"
  ARCFACE_MODEL="$DEFAULT_ARCFACE_MODEL"
  CURRICULAR_MODEL="$DEFAULT_CURRICULAR_MODEL"
  COSFACE_MODEL="$DEFAULT_COSFACE_MODEL"
  POSE_MODEL="$DEFAULT_POSE_MODEL"
  GAZE_MODEL="$DEFAULT_GAZE_MODEL"
  DEEP3D_ROOT="$DEFAULT_DEEP3D_ROOT"
  MUSIQ_MODEL="$DEFAULT_MUSIQ_MODEL"
  DINO_ROOT="$DEFAULT_DINO_ROOT"
else
  printf 'Unknown model profile: %s (expected onboarding or assets)\n' "$MODEL_PROFILE" >&2
  exit 2
fi

SELECTED="$($PYTHON_BIN - "$METRICS" "$EXCLUDE_METRICS" <<'PY'
import sys
atomic = {'id_strict','input_leak','id_arc','id_ins','id_cur','face_similarity','pose','gaze','expression','lighting','imaging_quality','subject_consistency','temporal_flickering'}
groups = {
 'all': atomic, 'identity': {'id_strict','input_leak','id_arc','id_ins','id_cur'},
 'identity_multi': {'id_arc','id_ins','id_cur'},
 'facebench': {'face_similarity','pose','gaze','expression','lighting'},
 'vbench': {'imaging_quality','subject_consistency','temporal_flickering'},
 'temporal': {'subject_consistency','temporal_flickering'},
}
def expand(value):
 out=set()
 for raw in value.split(','):
  name=raw.strip().lower().replace('-','_')
  if not name: continue
  if name in groups: out |= groups[name]
  elif name in atomic: out.add(name)
  else: raise SystemExit(f'Unknown metric/group: {raw}')
 return out
print(','.join(sorted(expand(sys.argv[1]) - expand(sys.argv[2]))))
PY
)"
[[ -n "$SELECTED" ]] || { printf '%s\n' 'No metrics remain after exclusions.' >&2; exit 2; }
has_metric() { [[ ",$SELECTED," == *",$1,"* ]]; }
has_any() { local name; for name in "$@"; do has_metric "$name" && return 0; done; return 1; }
require_file() { [[ -f "$1" ]] || { printf 'Required model not found: %s\n' "$1" >&2; exit 2; }; }
if has_any id_strict input_leak id_arc id_ins id_cur; then
  require_file "$ID_MODELS_DIR/scrfd_10g_bnkps.onnx"
fi
has_any id_strict input_leak id_ins && require_file "$ID_MODELS_DIR/glintr100.onnx"
has_metric id_arc && require_file "$ARCFACE_MODEL"
has_metric id_cur && require_file "$CURRICULAR_MODEL"
has_metric imaging_quality && require_file "$MUSIQ_MODEL"
if has_metric subject_consistency; then
  require_file "$DINO_CODE_ROOT/hubconf.py"
  require_file "$DINO_ROOT/dino_vitbase16_pretrain.pth"
fi
has_metric face_similarity && require_file "$COSFACE_MODEL"
has_metric pose && require_file "$POSE_MODEL"
has_metric gaze && require_file "$GAZE_MODEL"
if has_any expression lighting; then
  require_file "$DEEP3D_ROOT/checkpoints/pretrained/epoch_20.pth"
  require_file "$DEEP3D_ROOT/BFM/BFM_model_front.mat"
fi

stage_done() {
  local stage="$1" signature="$2" artifact="$3"
  [[ "$RESUME" == 1 && -s "$artifact" && -f "$OUTPUT_DIR/.${stage}.signature" ]] || return 1
  [[ "$(<"$OUTPUT_DIR/.${stage}.signature")" == "$signature" ]]
}
mark_done() { printf '%s' "$2" > "$OUTPUT_DIR/.$1.signature"; }
run_stage() {
  local stage="$1" signature="$2" artifact="$3"; shift 3
  if stage_done "$stage" "$signature" "$artifact"; then
    printf '[resume] %s -> %s\n' "$stage" "$artifact"
    return 0
  fi
  printf '[run] %s\n' "$stage"
  if "$@"; then mark_done "$stage" "$signature"; return 0; fi
  printf '[error] stage %s failed; partial artifact and log are preserved\n' "$stage" >&2
  FAILED_STAGES+=("$stage")
  return 0
}

MAPPING="$OUTPUT_DIR/mapping.json"
PREPARE_ARGS=(--manifest "$MANIFEST" --results-dir "$RESULTS_DIR" --output "$MAPPING" --limit "$LIMIT")
[[ -n "$ORIGIN_DIR" ]] && PREPARE_ARGS+=(--origin-dir "$ORIGIN_DIR")
[[ -n "$MASK_DIR" ]] && PREPARE_ARGS+=(--mask-dir "$MASK_DIR")
[[ -n "$REF_DIR" ]] && PREPARE_ARGS+=(--ref-dir "$REF_DIR")
"$PYTHON_BIN" "$ROOT/tools/prepare_results.py" "${PREPARE_ARGS[@]}"
"$PYTHON_BIN" "$ROOT/tools/validate_mapping.py" --mapping "$MAPPING" --output "$OUTPUT_DIR/input_report.json" --errors "$OUTPUT_DIR/errors.log" --max-frames "$EVAL_MAX_FRAMES"
MAPPING_HASH="$(sha256sum "$MAPPING" | awk '{print $1}')"
MANIFEST_HASH="$(sha256sum "$MANIFEST" | awk '{print $1}')"

if has_any id_strict input_leak; then
  run_stage identity_strict "$BENCHMARK_MODE|$SELECTED|$EVAL_MAX_FRAMES|$LIMIT|$MANIFEST_HASH|$MAPPING_HASH|$ID_MODELS_DIR" "$OUTPUT_DIR/identity_strict.json" \
    env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -m swapface_benchmark.metrics.identity_strict \
      --label swapface_benchmark --mapping "$MAPPING" --out "$OUTPUT_DIR/identity_strict.json" \
      --models-dir "$ID_MODELS_DIR" --device 0 \
      --sample-frames "$EVAL_MAX_FRAMES" --max-eval-frames "$EVAL_MAX_FRAMES" --crop-mode face-box --no-random-sampling
fi

if has_any id_arc id_ins id_cur; then
  MULTI_METRICS=""
  for metric in id_arc id_ins id_cur; do has_metric "$metric" && MULTI_METRICS="${MULTI_METRICS:+$MULTI_METRICS,}$metric"; done
  run_stage identity_multibackbone "$BENCHMARK_MODE|$MULTI_METRICS|$EVAL_MAX_FRAMES|$LIMIT|$MANIFEST_HASH|$MAPPING_HASH|$ID_MODELS_DIR|$ARCFACE_MODEL|$CURRICULAR_MODEL" "$OUTPUT_DIR/identity_multibackbone.json" \
    env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -m swapface_benchmark.metrics.identity_multibackbone \
      --mapping "$MAPPING" --output "$OUTPUT_DIR/identity_multibackbone.json" \
      --detector "$ID_MODELS_DIR/scrfd_10g_bnkps.onnx" \
      --id-arc "$ARCFACE_MODEL" --id-ins "$ID_MODELS_DIR/glintr100.onnx" \
      --id-cur "$CURRICULAR_MODEL" \
      --metrics "$MULTI_METRICS" --device 0 --sample-frames "$EVAL_MAX_FRAMES" --max-eval-frames "$EVAL_MAX_FRAMES" --batch-size 32
fi

if has_any imaging_quality subject_consistency temporal_flickering; then
  VBENCH_METRICS=""
  for metric in imaging_quality subject_consistency temporal_flickering; do has_metric "$metric" && VBENCH_METRICS="${VBENCH_METRICS:+$VBENCH_METRICS,}$metric"; done
  run_stage vbench_quality "$BENCHMARK_MODE|$VBENCH_METRICS|$EVAL_MAX_FRAMES|$LIMIT|$MANIFEST_HASH|$MAPPING_HASH|$MUSIQ_MODEL|$DINO_ROOT" "$OUTPUT_DIR/vbench_quality.json" \
    env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -m swapface_benchmark.metrics.vbench_quality \
      --mapping "$MAPPING" --output "$OUTPUT_DIR/vbench_quality.json" --metrics "$VBENCH_METRICS" \
      --musiq-model "$MUSIQ_MODEL" \
      --dino-repo "$DINO_CODE_ROOT" \
      --dino-model "$DINO_ROOT/dino_vitbase16_pretrain.pth" --device cuda:0 --batch-size 8 --max-frames "$EVAL_MAX_FRAMES"
fi

if has_any face_similarity pose gaze expression lighting; then
  FACEBENCH_LAYOUT="$OUTPUT_DIR/facebench_layout"
  FB_FLAGS=()
  has_metric face_similarity || FB_FLAGS+=(--no-enable-face-sim)
  has_metric pose || FB_FLAGS+=(--no-enable-pose)
  has_metric gaze || FB_FLAGS+=(--no-enable-gaze)
  has_any expression lighting || FB_FLAGS+=(--no-enable-exp-gamma)
  FB_SIGNATURE="$BENCHMARK_MODE|$SELECTED|$EVAL_MAX_FRAMES|$LIMIT|$MANIFEST_HASH|$MAPPING_HASH|$COSFACE_MODEL|$POSE_MODEL|$GAZE_MODEL|$DEEP3D_ROOT"
  if stage_done facebench "$FB_SIGNATURE" "$OUTPUT_DIR/facebench/evaluation_summary_sim.json"; then
    printf '[resume] facebench -> %s\n' "$OUTPUT_DIR/facebench/evaluation_summary_sim.json"
  else
    "$PYTHON_BIN" "$ROOT/tools/prepare_facebench_layout.py" --mapping "$MAPPING" --output-root "$FACEBENCH_LAYOUT"
    run_stage facebench "$FB_SIGNATURE" "$OUTPUT_DIR/facebench/evaluation_summary_sim.json" \
      env METHOD_NAME=swapface_benchmark SOURCE_VIDEO_DIR="$FACEBENCH_LAYOUT/source" \
        TARGET_VIDEO_DIR="$FACEBENCH_LAYOUT/target" OUTPUT_DIR="$OUTPUT_DIR/facebench" \
        CUDA_VISIBLE_DEVICES="$GPU_LIST" NUM_GPUS="$NUM_GPUS" PYTHON_BIN="$PYTHON_BIN" \
        MAX_EVAL_FRAMES="$EVAL_MAX_FRAMES" RANDOM_SAMPLING=0 \
        FACE_MODEL_PATH="$COSFACE_MODEL" \
        FACE_DETECT_MODEL_PATH="$(dirname "$ID_MODELS_DIR")" \
        POSE_MODEL_PATH="$POSE_MODEL" GAZE_MODEL_PATH="$GAZE_MODEL" \
        DEEP3D_CHECKPOINTS_DIR="$DEEP3D_ROOT/checkpoints" \
        DEEP3D_BFM_FOLDER="$DEEP3D_ROOT/BFM" \
        bash "$ROOT/vendor/facebench/scripts_lzk/cal_metric/method_eval_facebench_all.sh" --override-existing "${FB_FLAGS[@]}"
  fi
fi

SUMMARY_ARGS=(--input-report "$OUTPUT_DIR/input_report.json" --selected "$SELECTED" --output "$OUTPUT_DIR/summary.json")
has_any id_strict input_leak && [[ -s "$OUTPUT_DIR/identity_strict.json" ]] && SUMMARY_ARGS+=(--identity-strict "$OUTPUT_DIR/identity_strict.json")
has_any id_arc id_ins id_cur && [[ -s "$OUTPUT_DIR/identity_multibackbone.json" ]] && SUMMARY_ARGS+=(--identity-multibackbone "$OUTPUT_DIR/identity_multibackbone.json")
has_any imaging_quality subject_consistency temporal_flickering && [[ -s "$OUTPUT_DIR/vbench_quality.json" ]] && SUMMARY_ARGS+=(--vbench-quality "$OUTPUT_DIR/vbench_quality.json")
has_any face_similarity pose gaze expression lighting && [[ -s "$OUTPUT_DIR/facebench/evaluation_summary_sim.json" ]] && SUMMARY_ARGS+=(--facebench "$OUTPUT_DIR/facebench/evaluation_summary_sim.json")
"$PYTHON_BIN" "$ROOT/tools/summarize.py" "${SUMMARY_ARGS[@]}"
printf 'Benchmark complete: %s\n' "$OUTPUT_DIR/summary.json"
if [[ ${#FAILED_STAGES[@]} -gt 0 ]]; then
  printf 'Failed stages: %s\n' "${FAILED_STAGES[*]}" >&2
  exit 1
fi
if [[ "$REGISTER" == 1 ]]; then
  REGISTER_ARGS=(register --benchmark-mode "$BENCHMARK_MODE" --results-dir "$RESULTS_DIR" --evaluation-dir "$OUTPUT_DIR" --group "$REGISTER_GROUP")
  [[ -n "$REGISTER_ID" ]] && REGISTER_ARGS+=(--run-id "$REGISTER_ID")
  [[ -n "$REGISTER_LABEL" ]] && REGISTER_ARGS+=(--label "$REGISTER_LABEL")
  [[ -n "$REGISTER_TAGS" ]] && REGISTER_ARGS+=(--tags "$REGISTER_TAGS")
  [[ -n "$REGISTER_MODEL" ]] && REGISTER_ARGS+=(--model-path "$REGISTER_MODEL")
  [[ -n "$REGISTER_MODEL_URL" ]] && REGISTER_ARGS+=(--model-url "$REGISTER_MODEL_URL")
  [[ -n "$REGISTER_CHECKPOINT_STEP" ]] && REGISTER_ARGS+=(--checkpoint-step "$REGISTER_CHECKPOINT_STEP")
  [[ -n "$REGISTER_STEPS" ]] && REGISTER_ARGS+=(--inference-steps "$REGISTER_STEPS")
  [[ -n "$REGISTER_SEED" ]] && REGISTER_ARGS+=(--seed "$REGISTER_SEED")
  [[ -n "$REGISTER_NOTES" ]] && REGISTER_ARGS+=(--notes "$REGISTER_NOTES")
  "$PYTHON_BIN" "$ROOT/tools/eval_gallery.py" "${REGISTER_ARGS[@]}"
fi
