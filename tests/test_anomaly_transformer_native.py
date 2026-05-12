from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from noboom_benchmark.noboom_lib.core.models.anomaly_transformer import (
    AnomalyTransformer,
    AnomalyTransformerAnomalyDetector,
    AnomalyTransformerLoss,
    anomaly_transformer_energy,
)


class _FixedAnomalyTransformer(torch.nn.Module):
    def __init__(self, predictions: tuple[object, ...], win_size: int) -> None:
        super().__init__()
        self.predictions = predictions
        self.win_size = win_size

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[object, ...]:
        del inputs
        return self.predictions


class _ChunkEchoAnomalyTransformer(torch.nn.Module):
    def __init__(self, win_size: int) -> None:
        super().__init__()
        self.win_size = win_size
        self.batch_sizes: list[int] = []
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[object, ...]:
        x = inputs[0]
        self.batch_sizes.append(int(x.shape[0]))
        uniform_attention = torch.full(
            (x.shape[0], 1, x.shape[1], x.shape[1]),
            1.0 / x.shape[1],
            dtype=x.dtype,
            device=x.device,
        )
        return (
            torch.zeros_like(x),
            [uniform_attention],
            [uniform_attention],
            [torch.ones_like(uniform_attention)],
        )


def test_anomaly_transformer_honors_native_e_layers() -> None:
    model = AnomalyTransformer(
        win_size=8,
        enc_in=2,
        c_out=2,
        d_model=8,
        n_heads=2,
        e_layers=2,
        d_ff=16,
    )

    assert len(model.network.encoder.attn_layers) == 2


def test_attention_energy_matches_uniform_kl_semantics() -> None:
    x = torch.tensor([[[1.0], [3.0]]])
    reconstruction = torch.tensor([[[2.0], [1.0]]])
    uniform_attention = torch.full((1, 1, 2, 2), 0.5)
    predictions = (
        reconstruction,
        [uniform_attention],
        [uniform_attention],
        [torch.ones(1, 1, 2, 2)],
    )

    energy = anomaly_transformer_energy(x, predictions, win_size=2)
    detector = AnomalyTransformerAnomalyDetector(
        model=_FixedAnomalyTransformer(predictions, win_size=2),
        temperature=50.0,
    )
    loss = AnomalyTransformerLoss(k=3)

    expected_energy = torch.tensor([[0.5, 2.0]])
    assert torch.allclose(energy, expected_energy)
    assert torch.allclose(detector.compute_online_anomaly_score((x,)), expected_energy[:, -1])
    assert torch.allclose(loss(predictions, (x,)), torch.tensor(5.0))


def test_anomaly_transformer_param_search_targets_native_components() -> None:
    params_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "params"
        / "anomaly_transformer.yaml"
    )
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    model_space = params["search_space"]["model"]
    assert "e_layers" in model_space["network"]["init_args"]
    assert "lr" in model_space["optimizer"]["init_args"]
    assert "detector" not in model_space


def test_detector_scores_predict_on_end_inputs_in_configured_chunks() -> None:
    x = torch.arange(5 * 2 * 1, dtype=torch.float32).reshape(5, 2, 1)
    model = _ChunkEchoAnomalyTransformer(win_size=2)
    detector = AnomalyTransformerAnomalyDetector(model=model, batch_size=2)

    scores = detector.compute_online_anomaly_score((x,))

    expected = x[:, -1].pow(2).mean(dim=-1) / x.shape[1]
    assert model.batch_sizes == [2, 2, 1]
    assert torch.allclose(scores, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device mismatch coverage")
def test_detector_moves_predict_on_end_inputs_to_model_device() -> None:
    model = AnomalyTransformer(
        win_size=4,
        enc_in=2,
        c_out=2,
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_ff=16,
    ).cuda()
    detector = AnomalyTransformerAnomalyDetector(model=model)
    cpu_inputs = (torch.randn(3, 4, 2),)

    scores = detector.compute_online_anomaly_score(cpu_inputs)

    assert scores.shape == (3,)
    assert scores.device.type == "cpu"
    assert torch.isfinite(scores).all()
