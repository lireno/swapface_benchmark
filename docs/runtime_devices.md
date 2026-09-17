# Evaluation devices and CPU limits

GPU evaluation runs neural models on each worker's local CUDA device. Ray maps
local CUDA:0 onto its assigned physical GPU. This includes the Deep3D SCRFD
five-point detector (previously hard-coded to CPU), coefficient network, pose,
gaze, identity encoders, MUSIQ, DINO and I3D. VBench pixel flicker and tensor
resize/normalization for quality/FVD also run on the selected device.

GPU requests fail if ONNX CUDA initialization fails; automatic whole-session
CPU retry is disabled. CPUExecutionProvider remains listed for unsupported
individual operations, not as permission to run the entire model on CPU.
Torch FaceBench actors also fail instead of silently switching to CPU.

CPU remains necessary for the existing video decoder, file I/O, NumPy/InsightFace
alignment and NMS, and small statistical aggregation. No metric definitions or
sampling rules were intentionally changed. GPU interpolation/reductions can
introduce small floating-point differences; code fingerprints invalidate stale
evaluation stage caches rather than silently mixing implementations.

Defaults through scripts/evaluate.sh:

- ONNX intra-op 2, inter-op 1, sequential execution, both spin loops disabled.
- Torch intra-op 2, inter-op 1; OpenCV compute 1; BLAS 2; decoding 2.
- Ray scheduling reserves 2 CPUs per GPU actor; object store capacity 512 MiB.

Overrides: BENCHMARK_ORT_THREADS, BENCHMARK_TORCH_THREADS,
BENCHMARK_OPENCV_THREADS, BENCHMARK_BLAS_THREADS, BENCHMARK_DECODE_THREADS,
BENCHMARK_ACTOR_CPUS, BENCHMARK_RAY_OBJECT_STORE_MB.
Ray CPU reservations are scheduling resources, **not** OS CPU quotas. Limits are
per pool/process, not a promise of total CPU use. Monitor full-run CPU/RSS after
resuming. OpenCV FFmpeg thread configuration depends on the installed backend.

InsightFace's public get_model in the installed version drops sess_options;
runtime_limits.insightface_model uses ModelRouter to pass bounded options to
the actual session. Keep that helper when adding ONNX models.

Validation on PPU-ZW805: SCRFD GPU session loaded with CUDA provider and two
intra-op threads; eight warmed-up blank-image detections took about 0.124 s wall
and 0.124 s CPU. This is a device/thread smoke test, not full benchmark throughput.
GPU vs CPU preprocessing max absolute difference on two random frames was
6.2e-5 for DINO and 6.0e-8 for MUSIQ. Production evaluation remains stopped until
explicitly resumed.
