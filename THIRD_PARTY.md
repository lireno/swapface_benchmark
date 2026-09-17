# Third-party components

本项目只对 benchmark 的组织、路径适配和评测编排负责。以下源码与模型来自各自上游项目，使用时应遵循对应许可：

- InsightFace / SCRFD / ArcFace / GLIntr100: <https://github.com/deepinsight/insightface>
- CurricularFace: <https://github.com/HuangYG123/CurricularFace>
- CosFace: <https://github.com/MuggleWang/CosFace_pytorch>
- Deep Head Pose / Hopenet: <https://github.com/natanielruiz/deep-head-pose>
- L2CS-Net: <https://github.com/edavalosanaya/L2CS-Net>
- Deep3DFaceRecon: <https://github.com/sicxu/Deep3DFaceRecon_pytorch>
- pyIQA: <https://github.com/chaofengc/IQA-PyTorch>
- VBench: <https://github.com/Vchitect/VBench>
- DINO: <https://github.com/facebookresearch/dino>
- PyTorch I3D: <https://github.com/piergiaj/pytorch-i3d> (Apache-2.0; code and license in `vendor/pytorch_i3d_model`; RGB ImageNet+Kinetics weights stored separately)
- nvdiffrast: <https://github.com/NVlabs/nvdiffrast>

特别注意：InsightFace 提供的预训练模型可能带有非商业使用限制；Basel Face Model 相关资产也有独立许可。请在分发和使用前自行确认研究用途与授权范围。

仓库中的 `vendor/` 仅保留评测运行所需的上游源码与许可证；训练代码、示例、
模型权重和 BFM 数据不包含在内。
