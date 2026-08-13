#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/modelscope_models}"
ONBOARDING_ROOT="${ONBOARDING_ROOT:-/mnt/cpfs/users/lyw/idvtrain}"
FACEBENCH="$ONBOARDING_ROOT/hifivfs_facebench_package_20260521"
RUNTIME="$ONBOARDING_ROOT/.cpfs_runtime/humanvid"

copy_model() {
  local source="$1" relative="$2" destination
  destination="$OUTPUT/$relative"
  [[ -f "$source" ]] || { printf 'Missing source model: %s\n' "$source" >&2; exit 2; }
  mkdir -p "$(dirname "$destination")"
  if [[ "$(realpath "$source")" != "$(realpath -m "$destination")" ]]; then
    cp -p "$source" "$destination"
  fi
  printf '[copied] %s\n' "$relative"
}

mkdir -p "$OUTPUT"
copy_model "$FACEBENCH/eval_tools/third_party/insightface_func/checkpoints/antelope/scrfd_10g_bnkps.onnx" models/identity/antelope/scrfd_10g_bnkps.onnx
copy_model "$FACEBENCH/eval_tools/third_party/insightface_func/checkpoints/antelope/glintr100.onnx" models/identity/antelope/glintr100.onnx
copy_model "$RUNTIME/models/insightface/buffalo_l/w600k_r50.onnx" models/identity/arcface_w600k_r50.onnx
copy_model "$RUNTIME/models/metrics/curricularface_ir101.onnx" models/identity/curricularface_ir101.onnx
copy_model "$FACEBENCH/eval_tools/third_party/CosFace_pytorch/checkpoints/ACC99.28.pth" models/facebench/cosface_ACC99.28.pth
copy_model "$FACEBENCH/eval_tools/third_party/deep_head_pose/checkpoints/hopenet_robust_alpha1.pkl" models/facebench/hopenet_robust_alpha1.pkl
copy_model "$FACEBENCH/models/L2CSNet_gaze360.pkl" models/facebench/L2CSNet_gaze360.pkl
copy_model "$FACEBENCH/eval_tools/third_party/Deep3DFaceRecon_pytorch/checkpoints/pretrained/epoch_20.pth" models/facebench/deep3d/checkpoints/pretrained/epoch_20.pth
copy_model "$FACEBENCH/eval_tools/third_party/Deep3DFaceRecon_pytorch/checkpoints/dlib_predictor_recognition/shape_predictor_5_face_landmarks.dat" models/facebench/deep3d/checkpoints/dlib_predictor_recognition/shape_predictor_5_face_landmarks.dat
copy_model "$FACEBENCH/eval_tools/third_party/Deep3DFaceRecon_pytorch/BFM/BFM_model_front.mat" models/facebench/deep3d/BFM/BFM_model_front.mat
copy_model "$FACEBENCH/eval_tools/third_party/Deep3DFaceRecon_pytorch/BFM/similarity_Lm3D_all.mat" models/facebench/deep3d/BFM/similarity_Lm3D_all.mat
copy_model "$ONBOARDING_ROOT/third_party/vbench_runtime/weights/musiq_spaq_ckpt-358bb6af.pth" models/vbench/musiq_spaq_ckpt-358bb6af.pth

DINO_SOURCE="${DINO_MODEL_PATH:-$ROOT/assets/models/vbench/dino/dino_vitbase16_pretrain.pth}"
if [[ ! -f "$DINO_SOURCE" ]]; then
  printf 'DINO weight missing. Download it first with scripts/download_dino_assets.sh, or set DINO_MODEL_PATH.\n' >&2
  exit 2
fi
copy_model "$DINO_SOURCE" models/vbench/dino/dino_vitbase16_pretrain.pth

cp "$ROOT/configs/model_manifest.json" "$OUTPUT/model_manifest.json"
(cd "$OUTPUT" && find models -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
printf 'ModelScope package ready: %s\n' "$OUTPUT"
printf 'Verify with: (cd %q && sha256sum -c SHA256SUMS)\n' "$OUTPUT"
