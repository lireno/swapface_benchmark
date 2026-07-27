# HiFiVFS FaceBench Metric Package

This package is a portable copy for:

`scripts_lzk/cal_metric/method_eval_facebench_all.sh`

The copied script and Python entry have been adjusted to resolve model paths from this package directory instead of the original `/mnt/nas/.../HiFiVFS_wan` path.

## What Is Included

- `scripts_lzk/cal_metric/method_eval_facebench_all.sh`
- `tools_lzk/cal_metric/method_eval_facebench.py`
- `eval_tools/metrics_calculator_facebench.py`
- `warping_error.py`
- CosFace model: `eval_tools/third_party/CosFace_pytorch/checkpoints/ACC99.28.pth`
- InsightFace antelope ONNX models under `eval_tools/third_party/insightface_func/checkpoints`
- Deep head pose model: `eval_tools/third_party/deep_head_pose/checkpoints/hopenet_robust_alpha1.pkl`
- Gaze model: `models/L2CSNet_gaze360.pkl`
- Deep3DFaceRecon code, checkpoints, and BFM assets under `eval_tools/third_party/Deep3DFaceRecon_pytorch`
- Torch SqueezeNet cache used by LPIPS: `torch_cache/hub/checkpoints/squeezenet1_1-b8a52dc0.pth`

## What Is Not Included

The current repository did not contain FaceBench source videos under `eval_datas/FaceBench/merge_data_1920`, and it did not contain the target videos under `output_lzk/other_baseline/alphaface`. Put those datasets/results on the new server, or override paths with environment variables.

Expected default target layout:

```text
output_lzk/other_baseline/alphaface/
  sim/00000_swapped.mp4
  diff/00000_swapped.mp4
```

Expected source layout:

```text
eval_datas/FaceBench/merge_data_1920/
  00000.mp4
  00000_mask.mp4
  00000_ref_sim.png
  00000_ref_diff.png
```

## Run

```bash
cd hifivfs_facebench_package_20260521
bash scripts_lzk/cal_metric/method_eval_facebench_all.sh
```

Override paths or GPU count:

```bash
SOURCE_VIDEO_DIR=/data/FaceBench/merge_data_1920 \
TARGET_VIDEO_DIR=/data/alphaface \
OUTPUT_DIR=/data/alphaface_metrics \
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
bash scripts_lzk/cal_metric/method_eval_facebench_all.sh
```

Extra `tyro` CLI options can be appended after the script command.

## Python Environment

Install project dependencies on the new server. Start with:

```bash
pip install -r requirements-facebench.txt
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```

Choose the PyTorch / CUDA / onnxruntime builds that match the new server. If `--enable-warping-error` is turned on and no local RAFT weights are passed, torchvision may try to download RAFT weights.
