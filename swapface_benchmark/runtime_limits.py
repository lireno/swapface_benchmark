"""Bound per-process compute pools; Ray CPU resources alone are not thread caps."""
import os

_configured = False


def positive_env(name, default):
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return value


def ort_session_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = positive_env('BENCHMARK_ORT_THREADS', 2)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Default ORT idle spinning across CPU cores caused heavy contention.
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return options


def configure_cpu_runtime():
    global _configured
    if _configured:
        return
    import cv2
    import torch
    torch.set_num_threads(positive_env('BENCHMARK_TORCH_THREADS', 2))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(positive_env('BENCHMARK_OPENCV_THREADS', 1))
    _configured = True
    print(f'[cpu-limits] torch={torch.get_num_threads()} interop=1 '
          f'opencv={cv2.getNumThreads()} ort={positive_env("BENCHMARK_ORT_THREADS",2)} '
          'ort_spinning=off', flush=True)


def insightface_model(path, providers=None):
    """InsightFace 0.7 get_model drops sess_options; ModelRouter forwards it.

    Preserve the same network/router and provider selection, only change
    scheduling. Explicit files only: no implicit downloads or model discovery.
    """
    from insightface.model_zoo.model_zoo import ModelRouter, get_default_providers
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    selected = providers if providers is not None else get_default_providers()
    model = ModelRouter(str(path)).get_model(
        providers=selected,
        sess_options=ort_session_options())
    if any((p[0] if isinstance(p, tuple) else p) == 'CUDAExecutionProvider' for p in selected):
        require_ort_cuda(model.session)
    return model


def ort_providers(device_id):
    """Negative device explicitly requests CPU; GPU requests must never silently fall back."""
    import onnxruntime as ort
    if device_id < 0:
        return ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('GPU evaluation requires ONNX Runtime CUDAExecutionProvider')
    return [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']


def require_ort_cuda(session):
    if 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('ONNX CUDA initialization failed; refusing silent CPU evaluation')
    session.disable_fallback()
