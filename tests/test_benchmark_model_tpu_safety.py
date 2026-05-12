from types import SimpleNamespace

import numpy as np
import torch

from noboom_benchmark.noboom_lib.core import model_objectives
from noboom_benchmark.noboom_lib.core.benchmark_utils.benchmark_model import BenchmarkModel


class _IdentityNetwork(torch.nn.Module):
    def forward(self, inputs):
        values, = inputs
        return values + 1

    def grouped_parameters(self):
        return ()


def _model_on_device(device: str, **kwargs) -> BenchmarkModel:
    model = BenchmarkModel(
        detector=object(),
        network=_IdentityNetwork(),
        losses=[],
        metrics=[],
        model_name="tpu_safety_test",
        **kwargs,
    )
    model._trainer = SimpleNamespace(strategy=SimpleNamespace(root_device=torch.device(device)))
    return model


def test_xla_bypasses_torch_compile_and_cudagraph(monkeypatch) -> None:
    model = _model_on_device(
        "xla",
        compile_torch_model=True,
        compile_torch_model_mode="max-autotune",
        compile_torch_model_fullgraph=True,
    )

    def fail_compile(*args, **kwargs):
        del args, kwargs
        raise AssertionError("torch.compile should not run on XLA")

    cudagraph_calls = []

    def record_cudagraph_call() -> None:
        cudagraph_calls.append(True)

    monkeypatch.setattr(torch, "compile", fail_compile)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", record_cudagraph_call)

    result = model._process_batch_with_compile((torch.tensor([1.0]),))

    assert result.item() == 2.0
    assert model._compiled_network is None
    assert model._mark_cudagraph_step_begin() is False
    assert cudagraph_calls == []


def test_cuda_compile_kwargs_and_cudagraph_are_preserved(monkeypatch) -> None:
    model = _model_on_device(
        "cuda",
        compile_torch_model=True,
        compile_torch_model_mode="max-autotune-no-cudagraphs",
        compile_torch_model_fullgraph=False,
    )
    compile_calls = []
    cudagraph_calls = []

    def fake_compile(network, **kwargs):
        compile_calls.append(kwargs)
        return network

    def record_cudagraph_call() -> None:
        cudagraph_calls.append(True)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", record_cudagraph_call)

    result = model._process_batch_with_compile((torch.tensor([1.0]),))

    assert result.item() == 2.0
    assert compile_calls == [{"mode": "max-autotune-no-cudagraphs", "fullgraph": False}]
    assert model._mark_cudagraph_step_begin() is True
    assert cudagraph_calls == [True]


def test_xla_autocast_disabled_context_does_not_call_torch_autocast(monkeypatch) -> None:
    model = _model_on_device("xla")

    def fail_autocast(*args, **kwargs):
        del args, kwargs
        raise AssertionError("torch.autocast should not be constructed for XLA")

    monkeypatch.setattr(torch, "autocast", fail_autocast)

    with model._torch_autocast_disabled():
        pass


def test_prediction_alignment_detaches_before_numpy_conversion() -> None:
    model = BenchmarkModel(
        detector=object(),
        window_size=1,
        prediction_horizon=None,
        label_index_offset=0,
        metrics=[],
    )
    scores = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 1, 0], dtype=torch.float32)

    padded_scores, aligned_labels = model.align_prediction_outputs(scores, labels, [3])

    np.testing.assert_allclose(padded_scores, np.array([0.2, 0.4, 0.6], dtype=np.float32))
    np.testing.assert_allclose(aligned_labels, labels.numpy())


def test_style_transfer_device_prefers_discovered_xla_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(model_objectives.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setattr(model_objectives, "_safe_xla_device", lambda: torch.device("xla"))

    assert model_objectives._resolve_style_transfer_device() == torch.device("xla")


def test_xla_checkpoint_load_uses_cpu_map_location(monkeypatch) -> None:
    model = _model_on_device("xla")
    model.ckpt_path = "/tmp/fake-xla-checkpoint.ckpt"
    map_locations = []

    def fake_torch_load(path, map_location=None):
        del path
        map_locations.append(map_location)
        return {}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    model._load_network_weights()

    assert map_locations == ["cpu"]
