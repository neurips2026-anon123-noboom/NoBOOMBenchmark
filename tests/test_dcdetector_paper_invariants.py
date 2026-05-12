"""Semantic invariants for native DCdetector loss and scoring."""

from __future__ import annotations

import numpy as np
import torch

from noboom_benchmark.noboom_lib.core.models.dcdetector import (
    DCdetector,
    DCdetectorAnomalyDetector,
    DCdetectorLoss,
    dc_kl_loss,
)


class DeterministicAttentionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "series",
            torch.tensor(
                [
                    [
                        [
                            [0.70, 0.10, 0.10, 0.10],
                            [0.10, 0.70, 0.10, 0.10],
                            [0.10, 0.10, 0.70, 0.10],
                            [0.10, 0.10, 0.10, 0.70],
                        ]
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "prior",
            torch.tensor(
                [
                    [
                        [
                            [0.40, 0.20, 0.20, 0.20],
                            [0.20, 0.40, 0.20, 0.20],
                            [0.20, 0.20, 0.40, 0.20],
                            [0.20, 0.20, 0.20, 0.40],
                        ]
                    ]
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, inputs):
        input_data, = inputs
        batch_size = input_data.shape[0]
        return [self.series.repeat(batch_size, 1, 1, 1)], [
            self.prior.repeat(batch_size, 1, 1, 1)
        ]


class ChunkRecordingAttentionModel(DeterministicAttentionModel):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, inputs):
        input_data, = inputs
        self.batch_sizes.append(int(input_data.shape[0]))
        return super().forward(inputs)


def test_loss_matches_source_contrastive_objective() -> None:
    model = DeterministicAttentionModel()
    series, prior = model((torch.zeros(2, 4, 1),))
    prior_normalized = prior[0] / torch.unsqueeze(
        torch.sum(prior[0], dim=-1), dim=-1
    ).repeat(1, 1, 1, prior[0].shape[-1])
    expected = (
        torch.mean(dc_kl_loss(prior_normalized, series[0].detach()))
        + torch.mean(dc_kl_loss(series[0].detach(), prior_normalized))
        - torch.mean(dc_kl_loss(series[0], prior_normalized.detach()))
        - torch.mean(dc_kl_loss(prior_normalized.detach(), series[0]))
    )

    actual = DCdetectorLoss()((series, prior), (torch.zeros(2, 4, 1),))

    assert torch.allclose(actual, expected, atol=1e-7)
    assert np.isfinite(actual.detach().cpu().item())


def test_detector_returns_one_attention_energy_per_input_window() -> None:
    detector = DCdetectorAnomalyDetector(
        model=DeterministicAttentionModel(),
        temperature=50.0,
    )

    scores = detector.compute_online_anomaly_score((torch.zeros(3, 4, 1),))

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores >= 0)


def test_detector_scores_predict_on_end_inputs_in_configured_chunks() -> None:
    model = ChunkRecordingAttentionModel()
    detector = DCdetectorAnomalyDetector(
        model=model,
        temperature=50.0,
        batch_size=2,
    )

    scores = detector.compute_online_anomaly_score((torch.zeros(5, 4, 1),))

    assert model.batch_sizes == [2, 2, 1]
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()


def test_native_network_loss_and_detector_smoke() -> None:
    model = DCdetector(
        win_size=4,
        input_dim=2,
        patch_size=[1],
        n_heads=1,
        e_layers=1,
        d_model=8,
    )
    inputs = (torch.randn(3, 4, 2),)

    predictions = model(inputs)
    loss = DCdetectorLoss()(predictions, inputs)
    scores = DCdetectorAnomalyDetector(model=model).compute_online_anomaly_score(inputs)

    assert len(predictions[0]) == 1
    assert predictions[0][0].shape == (3, 1, 4, 4)
    assert torch.isfinite(loss)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_native_network_supports_meta_forward_for_flop_probe() -> None:
    with torch.device("meta"):
        model = DCdetector(
            win_size=8,
            input_dim=2,
            patch_size=[4],
            n_heads=1,
            e_layers=1,
            d_model=8,
        )
        series, prior = model((torch.randn(2, 8, 2),))

    assert len(series) == 1
    assert len(prior) == 1
    assert series[0].device.type == "meta"
    assert prior[0].device.type == "meta"
