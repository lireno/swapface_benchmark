# SwapFace Benchmark

这是一个同时支持 short 与 long 两套 200-case 数据的换脸 benchmark。数据和评测模型托管在 ModelScope，GitHub 仓库只保存评测代码、协议和模型 SHA256 清单。

新机器环境部署、模型目录结构、ModelScope 打包与校验流程见
[`MIGRATION.md`](MIGRATION.md)。

- Short 数据：<https://www.modelscope.cn/datasets/lireno/swapface_benchmark/tree/master/benchmark/non_long_200>
- Long 数据：<https://www.modelscope.cn/datasets/lireno/swapface_benchmark/tree/master/benchmark>
- 评测模型：<https://www.modelscope.cn/models/luozekai/swapface_benchmark_models>
- 代码：<https://github.com/lireno/swapface_benchmark>

## 评测实现更新（2026-09-15）

当前协议为 `rgb_landmarks_gaze3d_directroi_v3`。详细修正、兼容性和验证记录见
[更新记录](docs/2026-09-15_evaluation_cleanup.md)。

- GAN baseline 只是可选展示资源；计算你生成视频的指标不要求它存在，也不解码或评测它。
- 默认直接读取 manifest 指定的逐帧 `face_boxes` JSON，不生成 mask 视频、不重新检测出一套评测框、不按视频长度插值。
- 显式提供 mask 视频时，从对应帧的 mask 取框；不自动寻找相邻目录里的其他 JSON。框用于选定人脸区域，ID / Deep3D 所需五点对齐仍会执行。
- Pose 和 Gaze 的输入颜色已修正；Deep3D 使用标准五点，直接提取系数，不做网格重建和渲染。
- 旧版属性分数与 v3 不应混用；重算只需要原来的生成视频，不需要重新训练。

## Benchmark 内容

每个 case 包含：

- `ref_image`：需要注入的目标身份参考图；
- `origin_video`：待换脸视频；
- `face_boxes`：与 origin video 按顺序对应的人脸框；
- `gan_swapped_video`：可选的 GAN baseline 展示资源，不是评测 GT 或运行依赖；
- FPS、face-box 帧范围和时长元数据。

数据分为两套独立协议：

- `short`：使用 `benchmark/non_long_200/manifest.json`。每个输出视频仅在最前面的 `min(总帧数, 81)` 帧窗口内每 5 帧取一帧，即 `0,5,...,80`（完整窗口 17 帧）；绝不扩展到第 81 帧之后。
- `long`：使用 `benchmark/manifest.json`。ID、FaceBench 属性和 Image Quality 固定取帧下标
  `0, 15, 30, ...`（`stride=15`）。

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

评测 long 数据（覆盖完整视频，默认每 15 帧取一帧）：

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
# 只测还原性：Pose、Gaze、Expression、Lighting
bash scripts/evaluate.sh results --benchmark-mode short --metrics restoration
bash scripts/evaluate.sh results --benchmark-mode long --metrics restoration

# 只跑三套独立身份 backbone
bash scripts/evaluate.sh results --metrics id_arc,id_ins,id_cur

# 跑全部，但排除 FaceBench 和 Subject Consistency
bash scripts/evaluate.sh results \
  --metrics all --exclude-metrics facebench,subject_consistency

# 三项 VBench 指标，共享一次生成视频解码
bash scripts/evaluate.sh results --metrics vbench
```

可用分组：`all`、`identity`、`identity_multi`、`facebench`、`vbench`、
`temporal`、`restoration`。运行 `bash scripts/evaluate.sh --help` 查看全部原子指标。

默认启用 resume：成功阶段记录输入、模型资产、代码和参数签名。代码内容变化，或原路径的视频 / 框 / 模型文件大小或修改时间变化都会使缓存失效。媒体和模型使用 stat 清单，不是每次全量读取大文件做 SHA；请勿保留原大小和 mtime 原位篡改内容。`--no-resume` 可强制重算。

失败阶段不会保留成功签名；重跑前已有汇总移动到 `.previous`，防止失败后误报上次成功结果。输入帧覆盖不足、请求指标缺失或 case 失败会返回非零退出码。部分帧检测不到人脸仍允许作为缺测，不填零。

`--gpu-list` 同时用于所有 GPU 指标。Identity strict、Identity multi-backbone 和
VBench 会按 manifest 顺序将 cases 均匀分片，每张物理 GPU 启动一个独立进程，并在
完成后恢复 manifest 顺序、重新计算全量汇总统计；FaceBench 使用 Ray actor 按 GPU
分发 batch。例如 `--gpu-list 4,5,6,7` 会让四类指标都使用物理 GPU 4–7，而不是只让
这些卡可见但仍全部运行在逻辑 `cuda:0`。

生成视频是时间基准，原视频和 mask 按实际 FPS 映射。short 只检查并使用生成视频前 81 帧对应的时间窗口；long 覆盖完整生成视频时间轴，并对所有已选指标每 15 帧取 1 帧。原视频或 mask 无法覆盖实际评测窗口时会写入 `errors.log` 和逐指标失败结果，不再静默截短或末帧补齐。

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
- `face_boxes.json` 的数字键不要求从 0 开始，按数值排序后的第一项对应 origin 视频下标 0（第 1 帧）；坐标为该 origin 视频像素空间的 xyxy。不得把另一段裁剪前视频的框直接用于当前视频；输入覆盖不足或无效框会报错；
- short 在前 81 帧窗口按 `stride=5` 抽帧；long 默认使用 `stride=15`；origin video 和 mask 按实际 FPS 时间戳映射；
- 所有 ID 模型共享 SCRFD 五点对齐；
- 模型路径和 SHA256 记录在 [`configs/model_manifest.json`](configs/model_manifest.json)。

CPU 测试：

```bash
PYTHONPATH=.:tools python -m pytest -q tests
```

`BENCHMARK_DECODE_THREADS` 控制每个 FaceBench 解码器的线程数（默认 2），避免每个 actor 都启动整机 CPU 数量的解码线程。
Metric 可以与模型训练 / 视频生成共享已分配的 GPU/PPU；外部实验队列不应等待 metric 完成才开始下一次训练或生成。不要据此假定 metric 不消耗显存。

## 第三方代码与模型

### FVD（2026-09-17）

short 和 long 的 `--metrics all` 现在均包含 `fvd`。旧指标集可用
`--exclude-metrics fvd`；仅算 FVD 用 `--metrics fvd`，无需参考图、ROI/mask
或 GAN 视频，只需 manifest 中的 `case_id`、`origin_video` 及生成结果。
仍使用公共 mapping 的文件名解析规则；原视频和生成视频必须时间零点对齐。

```bash
PYTHON_BIN=/mnt/cpfs/users/lyw/venvs/idvtrain/bin/python \
bash scripts/evaluate.sh /path/to/generated_videos \
  --manifest /path/to/short_manifest.json --benchmark-mode short \
  --metrics fvd --gpu-list 0,1,2,3 --output-dir /path/to/eval_fvd_short \
  --fvd-weights /path/to/rgb_imagenet.pt
```

long 使用对应 manifest，并改 `--benchmark-mode long` 和输出目录即可。
`--fvd-i3d-root` 需包含 `pytorch_i3d_model/pytorch_i3d.py`，默认使用本仓库
`vendor/` 内置源码，无需 pip 安装。assets profile 的权重默认
`MODELS_ROOT/fvd/rgb_imagenet.pt`；onboarding profile 默认
`/mnt/cpfs/users/lzk/modelscope_swapface_models/models/fvd/rgb_imagenet.pt`。
也可通过 `FVD_WEIGHTS` 环境变量或 `--fvd-weights` 指定。网络模块不会自动下载模型；
缺失会明确报错，不能将缺失指标当成成功。2026-09-17 已下载官方固定版本
RGB 权重到上述共享目录（50,883,138 字节），严格加载匹配通过。
SHA256：`2609088c2e8c868187c9921c50bc225329a9057ed75e76120e0b4a397a2c7538`。
来源/版本记录在权重旁的 `provenance.json`，模型包 manifest 和 SHA256SUMS 已追加登记。

协议 `paired_timestamp_i3d_logits400_v2`：

- 每 case 取一个连续 15 帧片段（`--fvd-video-length`，最小 15），seed 默认 42
  （`--fvd-seed`）。short 只在前 81 帧窗口选起点，long 在完整生成时间轴选起点；
  case hash 固定起点，原视频帧按 `round(generated_index / generated_fps * origin_fps)` 对齐。
  不独立按两段视频各自长度取比例位置，也不应用其他指标的 stride=5/15。
- 原视频必须覆盖整个评测窗口，不静默截断。真正不足片段长度的视频，两边对应序列
  按相同规则循环重复；解码失败不补帧。变速/pingpong 输入应预先通过 adapter 对齐。
- 全帧输入，不裁 ROI；保留 HiFiVFS 预处理：RGB -> 640x480 -> 224x224，
  `2*x/255-1`。I3D 400 类 logits 沿输出时间平均，集合级均值与无偏协方差，
  float64 协方差因子 SVD（与 Bures/PSD 公式等价）计算 Frechet 距离，避免
  小样本秩亏时 sqrtm 的数值不稳定。不是逐视频 FVD 平均。
- 至少两个 case；不同样本数、权重、特征层或片段协议的分数不可直接混比。
  long 的单短片段 FVD 不代表整段长期一致性。
- 按 `--gpu-list` 启动多个特征 worker，每卡默认 batch=4（`--fvd-batch-size`
  为每卡大小）。每个 worker 加载一次 I3D，处理分配 case 的原视频和生成视频。
  按 case ID 严格汇总全部特征，再在主进程计算一次集合级 FVD；绝不平均每卡 FVD。
  完成的分片特征支持断点续跑；worker 日志在 `fvd_workers/run_*/`。
  直接运行 `tools/eval_fvd_streaming.py` 同样支持 `--gpu-list 0,1,2,3`。
- 输出 `fvd.json` 和 `summary.json` 中的 `metrics_flat.fvd`（数值，越小越好）。
  `summary.protocol.fvd` 单独记录采样和集合级聚合规则，不沿用逐帧指标的聚合说明。
- 成功特征缓存包含 case ID、采样索引、视频 path/size/mtime_ns、代码与权重指纹；
  完整成功才原子写入，失败记录不充当缓存。`--no-resume` 同时绕过 FVD 特征缓存。
  保留大小和时间戳的原位修改不能被 stat 指纹识别，这种情况请强制重算。

实现：`tools/eval_fvd_streaming.py` 为 CLI，`tools/fvd_paired.py` 为核心逻辑。
CPU 回归测试：`PYTHONPATH=.:tools python -m pytest -q tests/test_fvd_paired.py`。
测试中的 stub I3D 仅用于验证接线/缓存，不是预训练模型的数值验证。

真实权重 smoke（2026-09-17）：PPU4、batch=1，已完成 opt2000 streaming 模型
short/long 各 2 个 case 的公开入口 FVD-only 验证，失败数均为 0。
输出在 `output/fvd_pretrained_smoke_20260917/{short,long}/summary.json`。
short 使用生成时的 pingpong eval_adapter 原视频，保持时序对应。
这些仅是功能 smoke，不是 200-case 正式指标；不可用于模型比较。
复现入口 `tools/smoke_fvd_pretrained.py --help`，不会触发其他评测指标。

本仓库 vendoring 了 FaceBench、Deep3DFaceRecon 和 pyIQA 的必要源码。模型权重遵循各上游项目的许可与使用限制，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
