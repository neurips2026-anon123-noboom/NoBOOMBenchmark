"""Paper-invariant tests for the OracleAD baseline port.

References (NeurIPS 2025, openreview id=V5kzCSeaXF):
- Eq. 17: A_score = P_score * D_score  (score fusion is multiplication).
- Eq. 12: SLS = (1/M) sum_k D^(k) over the M training windows in an epoch
  (lagged epoch-mean), used from epoch 2 onward.
- Appendix D.2: default batch size = 1024.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


OracleAD_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "models"
    / "_source"
    / "baselines"
    / "self_impl"
    / "OracleAD"
)


def _read(name: str) -> str:
    return (OracleAD_PATH / name).read_text(encoding="utf-8")


def test_default_score_fusion_is_multiply_per_paper_eq_17() -> None:
    """Paper Eq. 17 fuses prediction and deviation scores via multiplication."""

    from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.OracleAD import (  # noqa: E501
        DEFAULT_HYPER_PARAMS,
        OracleADConfig,
    )

    assert DEFAULT_HYPER_PARAMS["score_fusion"] == "multiply"
    assert OracleADConfig().score_fusion == "multiply"


def test_default_batch_size_matches_paper_appendix_d2() -> None:
    """Appendix D.2 fixes the default batch size to 1024."""

    from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.OracleAD import (  # noqa: E501
        DEFAULT_HYPER_PARAMS,
    )

    assert DEFAULT_HYPER_PARAMS["batch_size"] == 1024


def test_sls_epoch_mean_is_weighted_by_windows_not_batches() -> None:
    """Paper Eq. 12 averages SLS over all epoch windows, not over batch means."""

    from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.OracleAD import (  # noqa: E501
        OracleAD,
    )

    class DistanceModel(torch.nn.Module):
        def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            distances = history[:, 0, 0]
            reconstruction = torch.zeros_like(history)
            prediction = torch.zeros(history.size(0), history.size(2), device=history.device)
            refined = torch.zeros(history.size(0), 2, 1, device=history.device)
            refined[:, 1, 0] = distances
            return reconstruction, prediction, refined

    first_batch = torch.zeros(2, 3, 2)
    first_batch[:, 0, 0] = torch.tensor([2.0, 4.0])
    second_batch = torch.zeros(1, 3, 2)
    second_batch[:, 0, 0] = torch.tensor([10.0])

    detector = OracleAD(win_size=3)
    detector.device = torch.device("cpu")
    detector.model = DistanceModel()

    _, sls = detector._run_epoch(
        loader=[(first_batch,), (second_batch,)],
        optimizer=None,
        sls=None,
    )

    expected = torch.tensor([[0.0, 16.0 / 3.0], [16.0 / 3.0, 0.0]])
    equal_batch_mean = torch.tensor([[0.0, 6.5], [6.5, 0.0]])
    assert sls is not None
    assert torch.allclose(sls, expected)
    assert not torch.allclose(sls, equal_batch_mean)


def test_every_default_hyperparam_is_referenced_in_source() -> None:
    """Each DEFAULT_HYPER_PARAMS key must be used somewhere in the OracleAD port."""

    from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.OracleAD import (  # noqa: E501
        DEFAULT_HYPER_PARAMS,
    )

    haystack = _read("OracleAD.py") + "\n" + _read("model.py")
    missing = [name for name in DEFAULT_HYPER_PARAMS if name not in haystack]
    assert not missing, f"Unreferenced hyperparameters: {missing}"


@pytest.mark.parametrize("fusion", ["multiply", "add"])
def test_score_fusion_accepts_documented_modes(fusion: str) -> None:
    from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.OracleAD import (  # noqa: E501
        OracleADConfig,
    )

    assert OracleADConfig(score_fusion=fusion).score_fusion == fusion


def test_native_oraclead_loss_tracks_weighted_lagged_sls() -> None:
    from noboom_benchmark.noboom_lib.core.models.oraclead import OracleADLoss

    def make_predictions(distances: torch.Tensor):
        batch_size = distances.numel()
        reconstruction = torch.zeros(batch_size, 2, 2)
        prediction = torch.zeros(batch_size, 2)
        refined = torch.zeros(batch_size, 2, 1)
        refined[:, 1, 0] = distances
        return reconstruction, prediction, refined

    target = torch.zeros(2, 3, 2)
    single_target = torch.zeros(1, 3, 2)
    loss = OracleADLoss()

    loss(make_predictions(torch.tensor([2.0, 4.0])), (target,), epoch=0, num_epochs=2)
    loss(make_predictions(torch.tensor([10.0])), (single_target,), epoch=0, num_epochs=2)

    expected = torch.tensor([[0.0, 16.0 / 3.0], [16.0 / 3.0, 0.0]])
    equal_batch_mean = torch.tensor([[0.0, 6.5], [6.5, 0.0]])
    current_sls = loss.current_sls()
    assert current_sls is not None
    assert torch.allclose(current_sls, expected)
    assert not torch.allclose(current_sls, equal_batch_mean)

    loss(make_predictions(torch.tensor([1.0])), (single_target,), epoch=1, num_epochs=2)
    assert loss.sls is not None
    assert torch.allclose(loss.sls, expected)


def test_native_oraclead_detector_multiplies_prediction_and_deviation_scores() -> None:
    from noboom_benchmark.noboom_lib.core.models.oraclead import OracleADAnomalyDetector

    class FixedOracleAD(torch.nn.Module):
        win_size = 3

        def forward(self, inputs):
            window, = inputs
            batch_size = window.shape[0]
            reconstruction = torch.zeros(batch_size, 2, 2, device=window.device)
            prediction = torch.zeros(batch_size, 2, device=window.device)
            refined = torch.zeros(batch_size, 2, 1, device=window.device)
            refined[:, 1, 0] = 2.0
            return reconstruction, prediction, refined

    detector = OracleADAnomalyDetector(model=FixedOracleAD(), score_fusion="multiply")
    detector.sls = torch.zeros(2, 2)
    window = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 3.0]]])

    score = detector.compute_online_anomaly_score((window,))

    prediction_score = torch.tensor(2.0)
    deviation_score = torch.linalg.norm(torch.tensor([[0.0, 2.0], [2.0, 0.0]]), ord="fro")
    assert score.shape == (1,)
    assert torch.allclose(score[0], prediction_score * deviation_score)


def test_native_oraclead_decoder_passes_contiguous_lstm_state() -> None:
    from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.model import (
        OracleTemporalBranch,
    )

    branch = OracleTemporalBranch(hidden_dim=4, history_length=3, num_layers=2)
    seen: dict[str, bool] = {}

    class CheckingDecoder(torch.nn.Module):
        def forward(self, inputs, state):
            hidden, cell = state
            seen["hidden_contiguous"] = hidden.is_contiguous()
            seen["cell_contiguous"] = cell.is_contiguous()
            return torch.zeros(inputs.shape[0], inputs.shape[1], 4), state

    branch.decoder = CheckingDecoder()

    branch.decode(torch.randn(5, 4))

    assert seen == {"hidden_contiguous": True, "cell_contiguous": True}
