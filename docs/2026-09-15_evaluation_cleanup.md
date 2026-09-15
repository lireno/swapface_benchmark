# 2026-09-15 公开评测流程修正与精简

## 基线与范围

仓库：`swapface_benchmark`。
按要求先提交当前工作区，基线 commit 为 **`05b5864`**（`chore: snapshot current benchmark before evaluation cleanup`）。
基线包含原有注册表、展示工具和此前已实施的 v2 metric 修正；不是最早有 RGB / 五点问题的版本。
随后在独立 `eval-cleanup-20260915` 工作树修改和验证，避免在开发过程中改动正在运行的生产评测模块。
本次是本地 Git 提交，不包含远程推送。

新 FaceBench 协议：**`rgb_landmarks_gaze3d_directroi_v3`**。
不改模型训练和生成视频；改变的是评测预处理、无关计算与完成校验。

## 1. 之前的错误与保留的修正

| 项目 | 问题 | 当前实现 |
| --- | --- | --- |
| Pose | 解码已输出 RGB，旧函数仍按 BGR 再转一次，交换红蓝 | ndarray / PIL 输入均按 RGB 处理 |
| Gaze | 关闭内部 detector 后调用 RGB 接口 `predict_gaze`，却送入 BGR | detector 关闭直接送 RGB；开启时给 `step` BGR |
| Expression / Lighting | dlib 五点实际是四个眼角加鼻子，误当成双眼中心、鼻子、双嘴角 | 使用 SCRFD/RetinaFace 标准五点，并保留 Deep3D 的坐标翻转和对齐模板 |
| Gaze Cos | 二维 pitch/yaw 数值向量的余弦不是空间视线夹角余弦 | 使用三维单位视线方向余弦 |

这些是计算接口 / 关键点语义问题，不是 MP4 容器或编码格式的问题。
两段视频同时送入错误模型输入，误差也不会保证抵消。
修正 RGB 不等于证明估计器在所有生成脸、裁剪和姿态上都已校准。
Gaze Cos 的变化是指标定义变化，不能当成模型性能提升。

上述四项已包含于 v2 基线；本次将它们纳入可重复执行的 CPU 回归测试，而不是重新声称发现或修正一次。

## 2. GAN baseline 不再参与常规评测

原来 `prepare_results.py` 强制要求 `gan_swapped_video` 存在，并将其写作 `ground_truth`。
Identity strict 随后读取 GAN 视频，检查时长、逐帧裁剪、检测、提取身份特征、计算 `gt_ref`，但标准总表只使用 `id_sim` 和 `input_leak`。
因此一个无关 GAN 文件缺失或过短，甚至可能导致用户生成视频的身份指标失败。

现在：

- manifest 可以没有 `gan_swapped_video`；即使保留一个不存在的历史路径，也不会读取它。
- Identity strict 仅处理生成视频、原视频和参考图，不读取 GAN，也不计算 GAN 身份特征。
- 原视频仍然需要：`Input Leak` 比较生成身份与原视频身份，Pose / Gaze / Exp / Lighting 比较生成视频与原视频属性。
- `gt_ref` 等旧诊断字段为兼容旧读取程序保留 null / 空列表；`ground_truth_evaluation=false` 明确表示未执行。不会用 GAN 替代真实 GT，也不会把缺失指标填成 0。
- Gallery 中 GAN baseline 的可选展示功能不因评测计算精简而删除；展示需要资源与计算需要资源是不同概念。

## 3. 直接使用明确提供的 ROI

旧公开入口的实际流程是：

```text
face_boxes.json
    -> 按框数与视频总帧数重新映射
    -> 绘制整段矩形 mask
    -> mp4v 编码
    -> H.264 再编码
    -> FaceBench 解码 mask
    -> 阈值 / 轮廓重建 bounding box
```

这不仅重复编码 / 解码，还会在框数不一致时改变时间对应关系，且有损压缩可能影响 mask 边界。
现在默认入口把 `mapping.json` 直接交给 FaceBench，没有生成 mask 视频和临时数据布局这一步：

```text
生成视频帧下标 i
    -> round(i / generated_fps * origin_fps)
    -> 读取该 origin 帧对应的原始 xyxy 框
    -> 按 origin / generated 尺寸比例换算一次
    -> 裁剪两边对应的人脸区域
```

框规则：

1. JSON 数字键按数值排序；第一项对应 manifest 的 origin 视频下标 0。键可以不是从 0 开始，遵循已有数据集约定。
2. 坐标必须处于该 origin 视频的像素坐标系。**已有框是否跟输入视频属于同一次裁剪，必须由数据准备保证；仅凭 JSON 无法自动证明。**
3. 保留原有扩框比例：左 / 右各 0.5 个框宽，上 0.75、下 0.25 个框高。没有为了分数临时调小或调大 ROI。
4. JSON 行格式错误、NaN、倒置 / 零面积框、框覆盖不到实际抽样时间点、缩放裁剪后为空，都明确报错。不丢行压缩时间轴，不选最近框，不末帧补齐。
5. 不重新检测生成一套评测 ROI，不自动猜邻近目录的 `face_boxes.json`。显式提供的 mask 视频优先于旧 manifest 中残留的其他框字段。
6. 若明确提供的是 mask 视频，按该 mask 的 FPS 对齐并取框，缩放用最近邻。mask 空白帧报错，不悄悄变成整帧评测。
7. ID 和 Deep3D 内部的**标准五点检测 / 对齐仍需要**：它们服务于模型输入对齐，和决定评价哪个人脸区域的外部 ROI 框不是同一件事。

`prepare_facebench_layout.py` 保留为兼容工具，只创建输入链接；JSON 框直接链接成 `_boxes.json`，不编码 mask。
历史像素级 LPIPS / SSIM / warping 路径若明确调用，需要真实 mask 视频；不能把矩形框伪装成分割标注。

## 4. 省掉不参与指标的模型工作

### Deep3D

旧流程调用完整 `FacereconModel.test()`：网络系数 -> 网格、纹理、光照合成 -> renderer -> 再取系数。
Expression / Lighting 实际只需要网络输出中的两个切片：

```python
with torch.inference_mode():
    coefficients = net_recon(aligned_rgb_tensor)
expression = coefficients[:, 80:144]   # 64 coefficients
lighting = coefficients[:, 227:254]    # 27 coefficients
```

新版直接严格加载同一 `epoch_20.pth['net_recon']`，保持原 resnet50 / use_last_fc=False 结构。
不创建 renderer、不计算 3D 网格、不加载识别支路；不需要 nvdiffrast 编译、dlib predictor 或 BFM 网格矩阵。
仍需 `similarity_Lm3D_all.mat` 对齐模板和标准五点 ONNX。
历史模型包及 SHA 清单保留兼容；并未删除用户已有模型资产。

### CosFace

- Face Similarity 和 Face Similarity source 共用每个生成帧的检测、对齐与特征，避免同一帧重复算两遍。
- CosFace 对齐入口只加载 SCRFD detector，不再遍历加载目录中无关的 GLIntr100 身份模型。
- 检测失败返回缺测，不再将整张裁剪图硬 resize 后伪装成已对齐人脸。逐帧 null 与有效帧数保留；不能将该变化解释成所有 case 的数值必然一致。

### 视频 I/O

- FaceBench 的视频元数据检查与帧读取复用同一个 Decord reader。
- 默认每个解码器 2 个线程，可用 `BENCHMARK_DECODE_THREADS` 调整，避免每个 actor 都创建整机 CPU 数量的线程。
- Identity / multi-ID 共享的 OpenCV 读取按顺序 read / grab，连续帧不反复 seek；重复帧索引复用已读帧，短解码报错。
- 无跨运行图像特征缓存；不同 case 的参考图 / 框可能有不同预处理，当前优先避免错误复用。

## 5. 输入、失败与缓存校验

- 输入校验按实际 FPS 和实际抽样末帧检查，移除旧的“允许多一帧”容忍。
- manifest 重复 case ID 直接报错。
- 输入不完整不再仅写 errors.log 然后继续宣称成功。
- FaceBench 错误 case 带 ref_type，写入 failure_count；按本次返回结果汇总，不扫描混入上次遗留 case 文件。
- 请求指标缺失、无有限均值或阶段缺失，最终 summary 明确列出错误并非零退出。允许部分 case 缺测，不要求所有指标恰好 200 个有效 case。
- 失败 / 强制重跑前移除旧成功签名，并将旧汇总保留为 `.previous`；旧结果不能冒充当前成功。
- resume 指纹包含代码内容 SHA256、媒体与模型资产的路径 / 大小 / mtime_ns，以及原有 mapping 和参数签名。
- 指纹按 stat 检查大文件，并不是每次全量读取模型和视频计算 SHA256；若人为保留大小和时间戳原位篡改数据，应使用新目录或 `--no-resume`。
- 公共入口要求 timestamp_strict。需要 prefix-pingpong / linspace 的生成应提供事先对齐的 manifest / adapter，不允许隐藏环境变量仅改变 FaceBench 时间对齐而其他指标不变。

## 6. 验证记录

### CPU

```bash
PYTHONPATH=.:tools python -m pytest -q tests
```

31 项通过：原有帧采样、分片合并、gallery 测试，以及新增的 RGB / BGR 契约、五点、三维 gaze cosine、框时间顺序 / 缩放 / 无效值、禁止猜测邻近框、无 GAN 映射与身份计算、顺序解码、缓存失效、缺失阶段失败、输入短一帧失败、CosFace 特征复用与缺测处理。
Python 编译检查、`bash -n` 和 `git diff --check` 通过。

### 真实 PPU

使用已有 L2 short200 生成视频的第一个 case，在物理 PPU 0 运行全部指标，取生成帧 0 / 40 / 80。
测试 manifest 删除 GAN 字段，提供原始 JSON 框；检查所有指标有有限均值、case_count=1、failure_count 全为 0，并确认没有生成 facebench_layout / mask 视频。
这只是功能 smoke，不是重跑完整 benchmark，也不用于报告模型排名。

同一真实输入、相同标准五点 / 对齐图像和同一 Deep3D 权重，分别执行旧完整渲染路径和新直接系数路径：

| 检查 | 结果 |
| --- | --- |
| 五点与网络输入图像 | 完全一致 |
| Expression 系数最大绝对差 | 0.0 |
| Lighting 系数最大绝对差 | 0.0 |
| 新路径导入 nvdiffrast | 否 |

该数值一致性测试仅覆盖一个真实输入；没有声称所有硬件 / batch 配置 bitwise 一致。
同次首次调用耗时约 2.86 秒（旧）与 1.27 秒（新），受冷启动及共享卡影响，**不能当成整套 benchmark 加速比**。
没有在全新服务器从零安装依赖验证；已更新 README / MIGRATION 和去掉标准路径不再需要的 dlib / trimesh / nvdiffrast 安装要求。

复用验证：同一输入 / 配置再次启动，四个阶段全部打印 `[resume]`，没有重新启动模型计算。

可随仓库查看的报告：[validation.json](2026-09-15_evaluation_cleanup_validation.json)。
本机日志、系数对照数组与 smoke summary 已归档到 `/mnt/cpfs/users/lyw/idvtrain/.cpfs_runtime/benchmark_cleanup_20260915/`；临时测试目录为 `/tmp/swapface_cleanup_validation/`。

## 7. 使用与兼容性

新任务推荐使用新输出目录：

```bash
bash scripts/evaluate.sh /path/to/generated_videos \
  --manifest /path/to/benchmark/manifest.json \
  --benchmark-mode short \
  --model-profile assets --models-root /path/to/models \
  --output-dir /path/to/eval_v3 --gpu-list 0,1,2,3
```

只算还原性：`--metrics restoration`，等价于 `--metrics pose,gaze,expression,lighting`。跳过 ID、CosFace 相似度、MUSIQ、DINO 和 flickering；不要求参考图存在，也不加载参考图。仍需原视频、生成视频及对应 ROI。

按用户最新要求，默认 short 在前 81 帧窗口抽取 `0,5,...,80`（17 帧），long 全长抽取 `0,15,30,...`。`--frame-stride` 仍可显式覆盖。先限制短视频评测窗口，再执行 stride，不会因步长变大而扩展到 81 帧之后。这个抽帧协议与历史 short stride=1 / long stride=10 不同，做横向表格必须统一抽帧。

新增还原性 smoke 使用 short 默认步长 5；上述全指标三帧 smoke 是显式 stride=40，不是新默认。
仍然支持显式 `--mask-dir` 覆盖，JSON 与 mask 视频都可用。
Metric 是独立工作，与训练 / 生成共享已分配卡；外部队列不要把 metric 完成作为下一次训练 / 生成的前置条件。

v2 -> v3：网络颜色和 Deep3D 五点语义相同，但直接 ROI 避免了旧有损 mask / 时间插值，空脸 CosFace 处理也更严格。
因此 **v3 不能假定与 v2 全表数值完全相同，论文最终横向对照应统一版本**。
已经在运行的 v2 L2 重算仍按 v2 报告；不能因为更新仓库就把旧结果标作 v3。
本次不擅自重跑所有 200-case 实验，也不覆盖历史指标。

未完成的进一步优化包括：跨 metric 共享 ID 检测、跨实验缓存原视频特征、模型批量推理与全量吞吐评估。
它们需要完整预处理指纹和数值回归，当前没有用未经验证的缓存替代正确性。

### 最新抽帧 / restoration 验证

CPU 共 31 项通过。对同一 81 帧测试视频，short 默认得到 17 个评测帧 `0,5,...,80`，long 模式默认得到 6 个评测帧 `0,15,...,75`；两次只输出 Pose、Gaze L2 / Cos、Expression、Lighting，失败计数均为 0。long 模式测试 manifest 完全移除参考图与 GAN 字段。该 long smoke 用于验证步长和接口，不是长视频全量评测。
