import sys
from types import SimpleNamespace

import pytest

from swapface_benchmark.runtime_limits import positive_env, ort_session_options, insightface_model


def test_positive_env(monkeypatch):
    monkeypatch.delenv('TEST_THREAD_LIMIT', raising=False)
    assert positive_env('TEST_THREAD_LIMIT', 2) == 2
    monkeypatch.setenv('TEST_THREAD_LIMIT', '0')
    with pytest.raises(ValueError):
        positive_env('TEST_THREAD_LIMIT', 2)


def test_ort_options(monkeypatch):
    ort = pytest.importorskip('onnxruntime')
    monkeypatch.setenv('BENCHMARK_ORT_THREADS', '2')
    options = ort_session_options()
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    for pool in ('intra_op', 'inter_op'):
        assert options.get_session_config_entry(f'session.{pool}.allow_spinning') == '0'


def test_insightface_options_forwarded(monkeypatch, tmp_path):
    pytest.importorskip('onnxruntime')
    seen = {}

    class Router:
        def __init__(self, path):
            seen['path'] = path

        def get_model(self, **kwargs):
            seen.update(kwargs)
            return 'model'

    monkeypatch.setitem(sys.modules, 'insightface.model_zoo.model_zoo',
                        SimpleNamespace(ModelRouter=Router,
                                        get_default_providers=lambda: ['CPUExecutionProvider']))
    path = tmp_path / 'model.onnx'
    path.touch()
    assert insightface_model(path, ['CPUExecutionProvider']) == 'model'
    assert seen['providers'] == ['CPUExecutionProvider']
    assert seen['sess_options'].intra_op_num_threads == 2


def test_gpu_request_must_not_silently_use_cpu(monkeypatch):
    from swapface_benchmark.runtime_limits import ort_providers, require_ort_cuda
    ort = pytest.importorskip('onnxruntime')
    monkeypatch.setattr(ort, 'get_available_providers', lambda: ['CPUExecutionProvider'])
    assert ort_providers(-1) == ['CPUExecutionProvider']
    with pytest.raises(RuntimeError, match='CUDAExecutionProvider'):
        ort_providers(0)
    with pytest.raises(RuntimeError, match='refusing silent CPU'):
        require_ort_cuda(SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider']))
    calls = []
    require_ort_cuda(SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider'],
                                    disable_fallback=lambda: calls.append(True)))
    assert calls == [True]
