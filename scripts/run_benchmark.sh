#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_ROOT="${ASSETS_ROOT:-$ROOT/assets}"
RESULTS_DIR="${1:?usage: scripts/run_benchmark.sh RESULTS_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-$ROOT/outputs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-0}"
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"$GPU_LIST")}"
SAMPLE_FRAMES="${SAMPLE_FRAMES:-81}"
LIMIT="${LIMIT:-0}"

export PYTHONPATH="$ROOT:$ROOT/vendor/facebench:${PYTHONPATH:-}"
MAPPING="$OUTPUT_DIR/mapping.json"
"$PYTHON_BIN" "$ROOT/tools/prepare_results.py" \
  --manifest "$ASSETS_ROOT/benchmark/manifest.json" \
  --results-dir "$RESULTS_DIR" \
  --output "$MAPPING" \
  --limit "$LIMIT"

CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -m swapface_benchmark.metrics.identity_strict \
  --label swapface_benchmark \
  --mapping "$MAPPING" \
  --out "$OUTPUT_DIR/identity_strict.json" \
  --models-dir "$ASSETS_ROOT/models/identity/antelope" \
  --device 0 \
  --sample-frames "$SAMPLE_FRAMES" \
  --max-eval-frames "$SAMPLE_FRAMES" \
  --crop-mode face-box \
  --no-random-sampling

CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -m swapface_benchmark.metrics.identity_multibackbone \
  --mapping "$MAPPING" \
  --output "$OUTPUT_DIR/identity_multibackbone.json" \
  --detector "$ASSETS_ROOT/models/identity/antelope/scrfd_10g_bnkps.onnx" \
  --id-arc "$ASSETS_ROOT/models/identity/arcface_w600k_r50.onnx" \
  --id-ins "$ASSETS_ROOT/models/identity/antelope/glintr100.onnx" \
  --id-cur "$ASSETS_ROOT/models/identity/curricularface_ir101.onnx" \
  --device 0 \
  --sample-frames "$SAMPLE_FRAMES" \
  --max-eval-frames "$SAMPLE_FRAMES" \
  --batch-size 32

FACEBENCH_LAYOUT="$OUTPUT_DIR/facebench_layout"
"$PYTHON_BIN" "$ROOT/tools/prepare_facebench_layout.py" \
  --mapping "$MAPPING" \
  --output-root "$FACEBENCH_LAYOUT"

FACEBENCH_ROOT="$ROOT/vendor/facebench"
METHOD_NAME=swapface_benchmark \
SOURCE_VIDEO_DIR="$FACEBENCH_LAYOUT/source" \
TARGET_VIDEO_DIR="$FACEBENCH_LAYOUT/target" \
OUTPUT_DIR="$OUTPUT_DIR/facebench" \
CUDA_VISIBLE_DEVICES="$GPU_LIST" \
NUM_GPUS="$NUM_GPUS" \
PYTHON_BIN="$PYTHON_BIN" \
FACE_MODEL_PATH="$ASSETS_ROOT/models/facebench/cosface_ACC99.28.pth" \
FACE_DETECT_MODEL_PATH="$ASSETS_ROOT/models/identity" \
POSE_MODEL_PATH="$ASSETS_ROOT/models/facebench/hopenet_robust_alpha1.pkl" \
GAZE_MODEL_PATH="$ASSETS_ROOT/models/facebench/L2CSNet_gaze360.pkl" \
DEEP3D_CHECKPOINTS_DIR="$ASSETS_ROOT/models/facebench/deep3d/checkpoints" \
DEEP3D_BFM_FOLDER="$ASSETS_ROOT/models/facebench/deep3d/BFM" \
  bash "$FACEBENCH_ROOT/scripts_lzk/cal_metric/method_eval_facebench_all.sh" \
    --max-frames "$SAMPLE_FRAMES" \
    --no-random-sampling

CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -m swapface_benchmark.metrics.imaging_quality \
  --mapping "$MAPPING" \
  --output "$OUTPUT_DIR/vbench_imaging_quality.json" \
  --model-path "$ASSETS_ROOT/models/vbench/musiq_spaq_ckpt-358bb6af.pth" \
  --device cuda:0 \
  --batch-size 8

"$PYTHON_BIN" "$ROOT/tools/summarize.py" \
  --identity-strict "$OUTPUT_DIR/identity_strict.json" \
  --identity-multibackbone "$OUTPUT_DIR/identity_multibackbone.json" \
  --facebench "$OUTPUT_DIR/facebench/evaluation_summary_sim.json" \
  --imaging-quality "$OUTPUT_DIR/vbench_imaging_quality.json" \
  --output "$OUTPUT_DIR/summary.json"

echo "Benchmark complete: $OUTPUT_DIR/summary.json"
