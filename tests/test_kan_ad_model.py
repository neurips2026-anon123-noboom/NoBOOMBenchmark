import sys
import types

import torch
import torch.nn as nn


def _install_timesead_stubs() -> None:
    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault(
        "timesead.models.common", types.ModuleType("timesead.models.common")
    )
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )

    class BaseModel(nn.Module):
        pass

    class AnomalyDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("dummy", torch.tensor([]), persistent=False)

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    class PredictionAnomalyDetector(AnomalyDetector):
        def forward(self, inputs):
            return None

    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.PredictionAnomalyDetector = PredictionAnomalyDetector
    models_module.BaseModel = BaseModel

    timesead_module.models = models_module
    models_module.common = common_module
    common_module.anomaly_detector = anomaly_detector_module


_install_timesead_stubs()

from noboom_benchmark.noboom_lib.core.models.kan_ad import (  # noqa: E402
    KANAD,
    KANADAnomalyDetector,
    KANADModel,
)


def test_kan_ad_forward_shapes_univariate() -> None:
    model = KANAD(window_size=96, input_dim=1, order=2, prediction_horizon=1)
    inputs = torch.randn(96, 4, 1)
    out = model((inputs,))
    assert out.shape == (1, 4, 1)


def test_kan_ad_forward_shapes_multivariate_channel_independent() -> None:
    model = KANAD(window_size=64, input_dim=3, order=2, prediction_horizon=1)
    inputs = torch.randn(64, 8, 3)
    out = model((inputs,))
    assert out.shape == (1, 8, 3)


def test_kan_ad_model_matches_official_one_step_network_shape() -> None:
    model = KANADModel(window=64, order=2)
    inputs = torch.randn(8, 64)
    out = model(inputs)
    assert out.shape == (8, 1)


def test_kan_ad_mapping_matches_reference_formula() -> None:
    model = KANADModel(window=8, order=3)
    inputs = torch.linspace(-1.0, 1.0, steps=40, dtype=torch.float32).view(5, 8)

    mapped = model._map_input(inputs)
    expected = torch.concat(
        [model.orders.repeat(inputs.size(0), 1, 1)]
        + [torch.cos(order * inputs.unsqueeze(1)) for order in range(1, model.order + 1)]
        + [inputs.unsqueeze(1)],
        dim=1,
    )

    assert torch.allclose(mapped, expected)


def test_kan_ad_mapping_keeps_input_gradients() -> None:
    model = KANADModel(window=8, order=2)
    inputs = torch.randn(4, 8, requires_grad=True)

    mapped = model._map_input(inputs)
    mapped.sum().backward()

    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape


def test_kan_ad_param_count_matches_paper_reference() -> None:
    # Table 3 of the paper reports 274 trainable parameters on UCR with
    # window=96, order=2, prediction_horizon=1, channel-independent univariate.
    # CTE differencing now lives in the data-pipeline transforms, so the model
    # has no diff-related parameters or branches.
    model = KANAD(window_size=96, input_dim=1, order=2, prediction_horizon=1)
    assert sum(p.numel() for p in model.parameters()) == 274


def test_kan_ad_detector_score_and_label_shapes() -> None:
    model = KANAD(window_size=32, input_dim=2, order=2, prediction_horizon=1)
    detector = KANADAnomalyDetector(model)
    inputs = torch.randn(32, 5, 2)
    target = torch.randn(1, 5, 2)
    label = torch.randint(0, 2, (1, 5))

    score = detector.compute_online_anomaly_score((inputs, target))
    formatted_label = detector.format_online_targets((label, target))

    assert score.shape == (5,)
    assert formatted_label.shape == (5,)


def test_kan_ad_init_does_not_accept_apply_cte() -> None:
    """`apply_cte` was removed; CTE is performed by data-pipeline transforms."""
    import pytest

    with pytest.raises(TypeError):
        KANAD(window_size=48, input_dim=1, order=4, prediction_horizon=1, apply_cte=False)


def test_kan_ad_rejects_multi_step_prediction_horizon() -> None:
    import pytest

    with pytest.raises(ValueError, match="prediction_horizon=1"):
        KANAD(window_size=48, input_dim=1, order=4, prediction_horizon=2)
