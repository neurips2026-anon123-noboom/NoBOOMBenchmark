"""Paper-invariant tests for the native SPAGD scoring path.

The SPAGD paper specifies in Section 3.4 that inference uses the trained
anomaly detector output as the anomaly score. The ablation study contrasts this
with reconstruction scoring, so the native detector must expose only the
sigmoided classifier output at test time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from noboom_benchmark.noboom_lib.core.models.spagd import (
    SPAGD,
    SPAGDAnomalyDetector,
    SPAGDLoss,
)


class _ChunkEchoSPAGD(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.batch_sizes: list[int] = []

    def forward(self, inputs):
        x = inputs[0]
        self.batch_sizes.append(int(x.shape[0]))
        return {"test_logits": x[:, -1, 0] * self.weight}


class _OomOnceSPAGD(torch.nn.Module):
    def __init__(self, max_batch_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.max_batch_size = int(max_batch_size)
        self.batch_sizes: list[int] = []

    def forward(self, inputs):
        x = inputs[0]
        self.batch_sizes.append(int(x.shape[0]))
        if x.shape[0] > self.max_batch_size:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return {"test_logits": x[:, -1, 0] * self.weight}


def _make_network(input_dim: int = 4) -> SPAGD:
    torch.manual_seed(0)
    return SPAGD(
        input_dim=input_dim,
        win_size=8,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        top_k=2,
        candidate_ratio=0.5,
        num_chunks=2,
        dropout=0.0,
    )


def test_detector_scores_sigmoid_test_logits_without_reconstruction_fusion() -> None:
    """SPAGD Section 3.4: anomaly score = classifier output, not reconstruction error."""

    torch.manual_seed(0)
    network = _make_network(input_dim=4)
    detector = SPAGDAnomalyDetector(network)
    network.eval()

    batch = torch.randn(3, 8, 4)
    with torch.no_grad():
        expected = torch.sigmoid(network((batch,))["test_logits"])
        scores = detector.compute_online_anomaly_score((batch,))

    scores_np = scores.cpu().numpy()
    assert scores_np.shape == (3,)
    assert np.all(np.isfinite(scores_np)), "SPAGD scores must be finite"
    assert np.all(scores_np >= 0.0) and np.all(scores_np <= 1.0), (
        "SPAGD scores must lie in the sigmoid range [0, 1]"
    )
    torch.testing.assert_close(scores, expected)


def test_detector_scores_predict_on_end_inputs_in_configured_chunks() -> None:
    model = _ChunkEchoSPAGD()
    detector = SPAGDAnomalyDetector(model, batch_size=2)
    windows = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3, 1)

    scores = detector.compute_online_anomaly_score((windows,))

    expected = torch.sigmoid(windows[:, -1, 0])
    torch.testing.assert_close(scores, expected)
    assert model.batch_sizes == [2, 2, 1]


def test_detector_retries_with_smaller_chunks_after_cuda_oom() -> None:
    model = _OomOnceSPAGD(max_batch_size=2)
    detector = SPAGDAnomalyDetector(model, batch_size=4)
    windows = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3, 1)

    scores = detector.compute_online_anomaly_score((windows,))

    expected = torch.sigmoid(windows[:, -1, 0])
    torch.testing.assert_close(scores, expected)
    assert model.batch_sizes == [4, 2, 2, 1]


def test_loss_matches_reconstruction_plus_weighted_self_perturbation_objective() -> None:
    """Native Lightning loss preserves the source SPAGD training objective."""

    predictions = {
        "reconstruction": torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]]),
        "normal_logits": torch.tensor([-1.0, -2.0]),
        "perturbed_logits": torch.tensor([1.5, 2.5]),
    }
    targets = (torch.tensor([[[0.0], [1.0]], [[2.0], [1.0]]]),)
    beta = 0.25

    loss = SPAGDLoss(beta=beta)((predictions,), targets)

    reconstruction_loss = F.mse_loss(predictions["reconstruction"], targets[0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.cat([predictions["normal_logits"], predictions["perturbed_logits"]], dim=0)
    expected = reconstruction_loss + beta * F.binary_cross_entropy_with_logits(logits, labels)

    torch.testing.assert_close(loss, expected)
