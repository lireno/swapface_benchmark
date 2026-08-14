# SwapFace Benchmark

这是一个同时支持 short 与 long 两套 200-case 数据的换脸 benchmark。数据和评测模型托管在 ModelScope，GitHub 仓库只保存评测代码、协议和模型 SHA256 清单。

新机器环境部署、模型目录结构、ModelScope 打包与校验流程见
[`MIGRATION.md`](MIGRATION.md)。

- Short 数据：<https://www.modelscope.cn/datasets/lireno/swapface_benchmark/tree/master/benchmark/non_long_200>
- Long 数据：<https://www.modelscope.cn/datasets/lireno/swapface_benchmark/tree/master/benchmark>
- 评测模型：<https://www.modelscope.cn/models/luozekai/swapface_benchmark_models>
- 代码：<https://github.com/lireno/swapface_benchmark>

## Benchmark 内容

每个 case 包含：

- `ref_image`：需要注入的目标身份参考图；
- `origin_video`：待换脸视频；
- `face_boxes`：与 origin video 按顺序对应的人脸框；
- `gan_swapped_video`：GAN 逐帧换脸 baseline；
- FPS、face-box 帧范围和时长元数据。

数据分为两套独立协议：

- `short`：使用 `benchmark/non_long_200/manifest.json`。每个输出视频只评测最前面的 `min(总帧数, 81)` 帧，即帧下标 `0..80`；绝不在全片上均匀采样。
- `long`：使用 `benchmark/manifest.json`，评测输出视频的全部帧。

long 的筛选过程：从 Part005 随机抽取，按 `(face_box 最后帧号 - 第一帧号) / 25 > 10` 初筛，排除 FPS 大于等于 50 的视频，人工排除 30 个低质量 case，再以随机种子 `20260728` 从剩余 220 个中选择 200 个。按实际 FPS 换算，其中 179 个视频超过 10 秒。

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
- `VBench Subject Consistency`：DINO ViT-B/16 相邻帧和首帧特征一致性；
- `VBench Temporal Flickering`：连续帧像素 MAE（原始 VBench 口径面向静态视频，动态视频需谨慎解释）。

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

也可以分别下载数据与公开模型包：

```bash
modelscope download --dataset lireno/swapface_benchmark --local_dir assets
modelscope download --model luozekai/swapface_benchmark_models --local_dir assets
cd assets && sha256sum -c SHA256SUMS
```

下载后目录为：

```text
assets/
  benchmark/
    manifest.json
    non_long_200/
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

`download_assets.sh` 会按 `models/manifest.json` 校验原有模型 SHA256，并下载 DINO 源码与
ViT-B/16 权重；DINO revision 和权重 SHA256 会在下载及评测输出中记录。

## 3. 准备生成结果

为每个 case 生成一个视频，文件名必须是 `<case_id>.mp4`：

```text
results/
  part005-long-001.mp4
  part005-long-002.mp4
  ...
  part005-long-250.mp4  # 仅实际入选的 200 个 ID 会出现在 manifest 中
```

准确 case 列表以对应模式的 manifest 为准：short 使用 `assets/benchmark/non_long_200/manifest.json`，long 使用 `assets/benchmark/manifest.json`。

## 4. 一键评测

默认模式是 short，从 benchmark assets 读取原视频、参考图和 face boxes，输出到
`RESULTS_DIR/benchmark_eval_short/`：

```bash
bash scripts/evaluate.sh /path/to/results --benchmark-mode short
```

评测 long 数据（全部帧）：

```bash
bash scripts/evaluate.sh /path/to/results --benchmark-mode long
```

模型默认使用 `--model-profile onboarding`，路径遵循
`/mnt/cpfs/users/lyw/idvtrain/docs/2026-08-11_codex_project_onboarding.md`：

- InsightFace 与 CurricularFace：`.cpfs_runtime/humanvid/models/`；
- CosFace、Hopenet、L2CS、Deep3D：`hifivfs_facebench_package_20260521/`；
- MUSIQ：`third_party/vbench_runtime/weights/`；
- DINO 权重：当前 benchmark 的 `assets/models/vbench/dino/`；DINO 运行源码已精简并放在 `vendor/dino/`。

若要全部使用本 benchmark 下载的模型：

```bash
bash scripts/evaluate.sh results --benchmark-mode short --model-profile assets
```

也可通过 `ONBOARDING_ROOT`、`DEFAULT_*_MODEL`、`DEFAULT_ID_MODELS_DIR`、
`DEFAULT_DEEP3D_ROOT`、`DEFAULT_DINO_ROOT` 环境变量覆盖单项模型位置。

所有 metric 的运行源码均包含在本仓库的 `swapface_benchmark/` 与 `vendor/`
目录中；模型权重、BFM、检测器数据等运行资产不随源码仓库分发。

覆盖输入和输出路径：

```bash
bash scripts/evaluate.sh /path/to/results \
  --benchmark-mode short \
  --manifest /path/to/benchmark/manifest.json \
  --origin-dir /path/to/origin_videos \
  --mask-dir /path/to/face_boxes_or_masks \
  --ref-dir /path/to/ref_images \
  --output-dir /path/to/evaluation
```

按指标选择或排除：

```bash
# 只跑三套独立身份 backbone
bash scripts/evaluate.sh results --metrics id_arc,id_ins,id_cur

# 跑全部，但排除 FaceBench 和 Subject Consistency
bash scripts/evaluate.sh results \
  --metrics all --exclude-metrics facebench,subject_consistency

# 三项 VBench 指标，共享一次生成视频解码
bash scripts/evaluate.sh results --metrics vbench
```

可用分组：`all`、`identity`、`identity_multi`、`facebench`、`vbench`、
`temporal`。运行 `bash scripts/evaluate.sh --help` 查看全部原子指标。

默认启用 resume：成功阶段记录输入 mapping 和参数签名，相同配置再次运行会跳过；失败
阶段不会写成功签名，会在下次自动重试。`--no-resume` 可强制重算。

生成视频是时间基准，原视频和 mask 按实际 FPS 映射。short 只检查并使用生成视频前 81 帧对应的时间窗口；long 检查并使用完整生成视频时间轴。原视频或 mask 无法覆盖实际评测窗口时会写入 `errors.log` 和逐指标失败结果，不再静默截短或末帧补齐。

### 兼容入口

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
LIMIT=1 BENCHMARK_MODE=short \
  bash scripts/run_benchmark.sh results outputs/smoke
```

最终汇总在：

```text
outputs/my_method/summary.json
```

各指标的逐 case 结果也会单独保存在输出目录。

## 5. Short / Long 评测网页

网页使用显式注册表，Short 与 Long 完全分开。第一次使用时运行一次初始化脚本：

```bash
bash scripts/initialize_eval_gallery.sh
```

默认读取：

```text
/mnt/cpfs/users/lzk/dataset/swapface_benchmark/short_200/benchmark/non_long_200/manifest.json
/mnt/cpfs/users/lzk/dataset/swapface_benchmark/long_200/benchmark/manifest.json
```

在其他机器上可通过 `SWAPFACE_BENCHMARK_DATASET_ROOT`，或者 `SHORT_MANIFEST`、
`LONG_MANIFEST` 环境变量指定数据位置。初始化后页面位于：

```text
web_reports/short/index.html
web_reports/long/index.html
```

后续评测只需要增加 `--register`。注册成功后会自动更新对应网页：

```bash
bash scripts/evaluate.sh /path/to/results \
  --benchmark-mode short \
  --register \
  --register-id gtid-id5-flow10-step25-seed42 \
  --register-label "GTID id5 flow10 · 25 steps" \
  --register-group GTID \
  --register-tags maskroi,25step,id5 \
  --register-model /path/to/model \
  --register-model-url https://modelscope.cn/models/example/model \
  --register-steps 25 \
  --register-seed 42
```

`--register-id` 省略时使用结果目录名；相同 ID 再次注册会更新原条目，不会产生重复项。
注册要求完整的 200-case mapping 和 summary，因而 `--limit` smoke test 不会误注册成正式结果。
注册表保存在 `registries/short.json` 和 `registries/long.json`，网页视频使用指向原结果的
符号链接，不会复制 200 份视频。

网页结果视频应使用浏览器兼容的 H.264/avc1 + yuv420p。已有注册结果若是 OpenCV
`mp4v`/FMP4，可原子转码并保留帧数和 FPS：

```bash
python tools/transcode_registered_h264.py --benchmark-mode short --workers 8
```

用 HTTP 服务查看网页，避免浏览器对 `file://` JSON 和视频加载的限制：

```bash
python -m http.server 8000 --directory web_reports
# Short: http://127.0.0.1:8000/short/
# Long:  http://127.0.0.1:8000/long/
```

## 数据和结果约定

- manifest 中媒体路径相对于各自的 manifest；
- `face_boxes.json` 的数字键不要求从 0 开始，按数值排序后的第一项对应视频第 1 帧；
- short 固定连续取最前面的最多 81 帧，long 连续取全部帧；origin video 和 mask 按实际 FPS 时间戳映射；
- 所有 ID 模型共享 SCRFD 五点对齐；
- 模型路径和 SHA256 记录在 [`configs/model_manifest.json`](configs/model_manifest.json)。

## 第三方代码与模型

本仓库 vendoring 了 FaceBench、Deep3DFaceRecon 和 pyIQA 的必要源码。模型权重遵循各上游项目的许可与使用限制，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
