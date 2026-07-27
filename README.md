# SwapFace Benchmark

这是一个 200-case 长视频换脸 benchmark。数据和评测模型托管在 ModelScope，GitHub 仓库只保存评测代码、协议和模型 SHA256 清单。

- 数据与模型：<https://www.modelscope.cn/datasets/lireno/swapface_benchmark>
- 代码：<https://github.com/lireno/swapface_benchmark>

## Benchmark 内容

每个 case 包含：

- `ref_image`：需要注入的目标身份参考图；
- `origin_video`：待换脸视频；
- `face_boxes`：与 origin video 按顺序对应的人脸框；
- `gan_swapped_video`：GAN 逐帧换脸 baseline；
- FPS、face-box 帧范围和时长元数据。

筛选过程：从 Part005 随机抽取，按 `(face_box 最后帧号 - 第一帧号) / 25 > 10` 初筛，排除 FPS 大于等于 50 的视频，人工排除 30 个低质量 case，再以随机种子 `20260728` 从剩余 220 个中选择 200 个。按实际 FPS 换算，其中 179 个视频超过 10 秒。

## 评测指标

- `ID-Arc`：ArcFace W600K-R50；
- `ID-Ins`：InsightFace GLIntr100；
- `ID-Cur`：CurricularFace IR101；
- `ID Variance`：逐视频帧级 ID-Arc 相似度方差；
- `Input Leak`：生成身份与 origin video 身份的相似度；
- `Face Detection Rate`：SCRFD 有效人脸帧比例；
- `Pose`：Hopenet；
- `Gaze L2 / Gaze Cos`：L2CS-Net；
- `Expression / Lighting`：Deep3DFaceRecon；
- `VBench Imaging Quality`：MUSIQ-SPAQ。

## 1. 安装

建议使用 Python 3.10 或 3.11，并先安装与服务器 CUDA 对应的 PyTorch。

```bash
git clone https://github.com/lireno/swapface_benchmark.git
cd swapface_benchmark
pip install -r requirements.txt
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

## 2. 下载数据和模型

```bash
bash scripts/download_assets.sh
```

下载后目录为：

```text
assets/
  benchmark/
    manifest.json
    ref_images/
    origin_videos/
    face_boxes/
    gan_swapped_videos/
  models/
    manifest.json
    identity/
    facebench/
    vbench/
```

`download_assets.sh` 会按 `models/manifest.json` 自动校验全部模型 SHA256。

## 3. 准备生成结果

为每个 case 生成一个视频，文件名必须是 `<case_id>.mp4`：

```text
results/
  part005-long-001.mp4
  part005-long-002.mp4
  ...
  part005-long-250.mp4  # 仅实际入选的 200 个 ID 会出现在 manifest 中
```

准确 case 列表以 `assets/benchmark/manifest.json` 为准。

## 4. 运行完整评测

单卡：

```bash
bash scripts/run_benchmark.sh results outputs/my_method
```

多卡 FaceBench：

```bash
GPU_LIST=0,1,2,3 NUM_GPUS=4 \
  bash scripts/run_benchmark.sh results outputs/my_method
```

先跑一个 case 验证环境：

```bash
LIMIT=1 SAMPLE_FRAMES=8 \
  bash scripts/run_benchmark.sh results outputs/smoke
```

最终汇总在：

```text
outputs/my_method/summary.json
```

各指标的逐 case 结果也会单独保存在输出目录。

## 数据和结果约定

- manifest 中全部媒体路径均相对于 `assets/benchmark/manifest.json`；
- `face_boxes.json` 的数字键不要求从 0 开始，按数值排序后的第一项对应视频第 1 帧；
- 生成视频按自身时间轴抽取评测帧，origin video 按相对时间位置映射到相同数量的帧；
- 所有 ID 模型共享 SCRFD 五点对齐；
- 模型路径和 SHA256 记录在 [`configs/model_manifest.json`](configs/model_manifest.json)。

## 第三方代码与模型

本仓库 vendoring 了 FaceBench、Deep3DFaceRecon 和 pyIQA 的必要源码。模型权重遵循各上游项目的许可与使用限制，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
