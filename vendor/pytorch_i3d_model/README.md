# PyTorch I3D

Upstream: https://github.com/piergiaj/pytorch-i3d
Reference revision: 05783d11f9632b25fe3d50395a9c9bb51f848d6d

This copy is the existing HiFiVFS_wan/pytorch_i3d_model implementation,
which adapts that upstream architecture. It retains the checkpoint names,
400-class logits and temporal forward behavior used by the existing FVD.
Only torch is required; no PyPI package installation or auto-download occurs.
Use the upstream models/rgb_imagenet.pt (ImageNet + Kinetics pretraining).
Weights are kept outside Git under modelscope_swapface_models/models/fvd.
