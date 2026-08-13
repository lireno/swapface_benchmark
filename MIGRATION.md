# SwapFace Benchmark 迁移与部署

本文描述如何在新机器上部署评测代码、准备模型包，并运行一键评测。GitHub
仓库只包含运行源码；benchmark 数据和模型建议分别托管在 ModelScope。

公开模型仓库：
<https://www.modelscope.cn/models/luozekai/swapface_benchmark_models>

## 1. 推荐环境

- Linux x86_64；
- Python 3.12；
- NVIDIA GPU 与 CUDA；
- 已验证组合：Python 3.12.3、PyTorch 2.9.0、TorchVision 0.24.0、CUDA 12.9；
- FFmpeg/OpenCV 能正常解码 MP4。

建议使用独立虚拟环境：

```bash
python3.12 -m venv /path/to/facebench
source /path/to/facebench/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

根据目标机器 CUDA 版本先安装匹配的 PyTorch 和 TorchVision，然后安装其余依赖：

```bash
pip install torch==2.9.0 torchvision==0.24.0
pip install -r requirements.txt
pip install --no-build-isolation --no-deps \
  git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae
```

`onnxruntime-gpu` 与 NumPy 必须兼容。本项目已验证 NumPy 1.26.4；不要再安装
CPU 版 `onnxruntime`，否则可能覆盖 GPU 版。如果所用 CUDA/Python 组合无法从
默认 PyPI 获取对应 wheel，应从机器环境提供的 CUDA 软件源安装等价版本。

## 2. 源码与模型的边界

运行源码均已包含在仓库中：

- `swapface_benchmark/`：身份与 VBench 指标适配；
- `vendor/facebench/`：FaceBench、Deep3D 推理、CosFace、Hopenet 和检测适配；
- `vendor/pyiqa/`：仅保留 MUSIQ 所需源码；
- `vendor/dino/`：仅保留 DINO ViT-B/16 所需源码；
- `tools/`、`scripts/`：输入整理、校验、汇总和一键评测。

模型、BFM 和 dlib predictor 不在 GitHub 源码仓库中。它们必须按下一节目录放置。

## 3. ModelScope 模型包结构

```text
models/
├── identity/
│   ├── antelope/
│   │   ├── scrfd_10g_bnkps.onnx
│   │   └── glintr100.onnx
│   ├── arcface_w600k_r50.onnx
│   └── curricularface_ir101.onnx
├── facebench/
│   ├── cosface_ACC99.28.pth
│   ├── hopenet_robust_alpha1.pkl
│   ├── L2CSNet_gaze360.pkl
│   └── deep3d/
│       ├── checkpoints/
│       │   ├── pretrained/epoch_20.pth
│       │   └── dlib_predictor_recognition/shape_predictor_5_face_landmarks.dat
│       └── BFM/
│           ├── BFM_model_front.mat
│           └── similarity_Lm3D_all.mat
└── vbench/
    ├── musiq_spaq_ckpt-358bb6af.pth
    └── dino/dino_vitbase16_pretrain.pth
```

总计 13 个文件。每个文件的大小、用途和 SHA256 见
[`configs/model_manifest.json`](configs/model_manifest.json)。模型包根目录还应包含：

- `model_manifest.json`：机器可读清单；
- `SHA256SUMS`：上传和下载后的完整性校验。

从当前已验证路径生成包：

```bash
DINO_MODEL_PATH=/path/to/dino_vitbase16_pretrain.pth \
bash scripts/package_modelscope_models.sh /path/to/modelscope_models
```

校验：

```bash
cd /path/to/modelscope_models
sha256sum -c SHA256SUMS
```

特别注意：BFM、InsightFace 等资源可能有独立或非商业许可。上传 ModelScope 前请
确认目标仓库可见性和分发授权，详见 `THIRD_PARTY.md`。

## 4. 新机器目录准备

从 ModelScope 下载公开模型包：

```bash
modelscope download --model luozekai/swapface_benchmark_models \
  --local_dir /data/swapface_benchmark_assets
cd /data/swapface_benchmark_assets
sha256sum -c SHA256SUMS
```

下载后目录例如：

```text
/data/swapface_benchmark_assets/models/...
```

benchmark 数据包应提供：

```text
assets/
├── benchmark/manifest.json
├── benchmark/non_long_200/manifest.json
├── origin_videos/
├── reference_images/
└── face_boxes/
```

模型包和 benchmark 数据也可以合并为同一个 `assets/` 根目录。

## 5. 环境检查

```bash
python - <<'PY'
import torch, onnxruntime, insightface, dlib, nvdiffrast.torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('ORT providers:', onnxruntime.get_available_providers())
PY
```

期望 ONNX Runtime providers 至少包含 `CUDAExecutionProvider`；无 GPU 的登录节点
可以只做导入检查，正式评测应在 GPU 节点执行。

## 6. 一键评测

```bash
export PYTHON_BIN=/path/to/facebench/bin/python

bash scripts/evaluate.sh /path/to/results \
  --benchmark-mode short \
  --model-profile assets \
  --models-root /data/swapface_benchmark_assets/models \
  --assets-root /data/swapface_benchmark_assets \
  --gpu-list 0
```

只算部分指标：

```bash
bash scripts/evaluate.sh /path/to/results \
  --model-profile assets \
  --models-root /data/models \
  --assets-root /data/assets \
  --metrics identity,temporal
```

排除指定指标：

```bash
bash scripts/evaluate.sh /path/to/results \
  --model-profile assets \
  --models-root /data/models \
  --assets-root /data/assets \
  --exclude-metrics expression,lighting
```

模式和帧协议：

- `--benchmark-mode short` 使用 `benchmark/non_long_200/manifest.json`，连续取输出视频最前面的最多 81 帧；
- `--benchmark-mode long` 使用 `benchmark/manifest.json`，连续评测全部输出帧。

默认启用 resume。成功且输入签名一致的阶段会跳过；模式和帧窗口包含在签名中，short 与 long 不会混用缓存。使用 `--no-resume` 强制重算。评测日志、错误记录、各阶段 JSON 和最终 `summary.json` 默认分别写入 `RESULTS_DIR/benchmark_eval_short/` 或 `RESULTS_DIR/benchmark_eval_long/`。

## 7. 指标与模型对应关系

| 指标 | 依赖模型 |
|---|---|
| `id_strict`, `input_leak` | SCRFD + GLIntr100 |
| `id_arc` | SCRFD + ArcFace W600K-R50 |
| `id_ins` | SCRFD + GLIntr100 |
| `id_cur` | SCRFD + CurricularFace IR101 |
| `face_similarity` | CosFace |
| `pose` | Hopenet |
| `gaze` | L2CS-Net |
| `expression`, `lighting` | Deep3D checkpoint + BFM + dlib predictor |
| `imaging_quality` | MUSIQ-SPAQ |
| `subject_consistency` | DINO ViT-B/16 |
| `temporal_flickering` | 无模型 |

## 8. 常见问题

- `No module named models`：应使用当前仓库版本；Deep3D 源码已 vendored，无需从模型目录导入源码。
- `No module named onnxruntime`：确认正在使用目标虚拟环境，并安装 `onnxruntime-gpu`。
- NumPy ABI 报错：使用 `numpy==1.26.4`，重新安装与之兼容的 ONNX Runtime。
- nvdiffrast 构建失败：先安装 PyTorch，再使用 `--no-build-isolation --no-deps` 安装。
- 原视频或 mask 比结果视频短：评测按设计记入 `errors.log`，不会静默截断。
