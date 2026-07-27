#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/torch_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export METHOD_NAME="${METHOD_NAME:-alphaface}"
export SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-${PROJECT_DIR}/eval_datas/FaceBench/merge_data_1920}"
export TARGET_VIDEO_DIR="${TARGET_VIDEO_DIR:-${PROJECT_DIR}/output_lzk/other_baseline/alphaface}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output_lzk/other_baseline/alphaface}"
export NUM_GPUS="${NUM_GPUS:-4}"
export FACE_MODEL_PATH="${FACE_MODEL_PATH:-${PROJECT_DIR}/eval_tools/third_party/CosFace_pytorch/checkpoints/ACC99.28.pth}"
export FACE_DETECT_MODEL_PATH="${FACE_DETECT_MODEL_PATH:-${PROJECT_DIR}/eval_tools/third_party/insightface_func/checkpoints}"
export POSE_MODEL_PATH="${POSE_MODEL_PATH:-${PROJECT_DIR}/eval_tools/third_party/deep_head_pose/checkpoints/hopenet_robust_alpha1.pkl}"
export GAZE_MODEL_PATH="${GAZE_MODEL_PATH:-${PROJECT_DIR}/models/L2CSNet_gaze360.pkl}"
export DEEP3D_CHECKPOINTS_DIR="${DEEP3D_CHECKPOINTS_DIR:-${PROJECT_DIR}/eval_tools/third_party/Deep3DFaceRecon_pytorch/checkpoints}"
export DEEP3D_BFM_FOLDER="${DEEP3D_BFM_FOLDER:-${PROJECT_DIR}/eval_tools/third_party/Deep3DFaceRecon_pytorch/BFM}"

"${PYTHON_BIN:-python}" tools_lzk/cal_metric/method_eval_facebench.py "$@"
