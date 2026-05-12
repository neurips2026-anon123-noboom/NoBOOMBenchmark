import sys
import types

import torch
import torch.nn as nn


def _install_timesead_stubs() -> None:
    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault(
        "timesead.models", types.ModuleType("timesead.models")
    )
    common_module = sys.modules.setdefault(
        "timesead.models.common", types.ModuleType("timesead.models.common")
    )
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )
    optim_module = sys.modules.setdefault(
        "timesead.optim", types.ModuleType("timesead.optim")
    )
    loss_module = sys.modules.setdefault(
        "timesead.optim.loss", types.ModuleType("timesead.optim.loss")
    )

    class BaseModel(nn.Module):
        pass

    class AnomalyDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("dummy", torch.tensor([]), persistent=False)

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    class Loss(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    loss_module.Loss = Loss
    models_module.BaseModel = BaseModel

    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    common_module.anomaly_detector = anomaly_detector_module
    optim_module.loss = loss_module


_install_timesead_stubs()

from noboom_benchmark.noboom_lib.core.models.igad import (  # noqa: E402
    IGAD,
    IGADAnomalyDetector,
    IGADLoss,
    MLPAutoEncoder,
    _fourier_resample,
)


def test_fourier_resample_preserves_shape_and_perturbs_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 32, 4)
    z = _fourier_resample(x)
    assert z.shape == x.shape
    assert not torch.allclose(z, x)


def test_igad_forward_returns_five_tuple_with_expected_shapes() -> None:
    model = IGAD(window_size=32, input_dim=4, hidden_dims=[64, 32], latent_dim=16)
    x = torch.randn(3, 32, 4)
    outputs = model((x,))
    assert len(outputs) == 5
    for tensor in outputs:
        assert tensor.shape == (3, 32, 4)


def test_igad_forward_handles_meta_device() -> None:
    model = IGAD(window_size=16, input_dim=2, hidden_dims=[16], latent_dim=8)
    with torch.device("meta") as meta:
        meta_model = model.to_empty(device=meta)
        x = torch.randn(2, 16, 2, device=meta)
        outputs = meta_model((x,))
    assert len(outputs) == 5
    assert all(tensor.shape == (2, 16, 2) for tensor in outputs)


def test_igad_loss_full_path_matches_reference_decomposition() -> None:
    torch.manual_seed(1)
    model = IGAD(window_size=16, input_dim=3, hidden_dims=[32], latent_dim=8)
    loss_fn = IGADLoss(idem_weight=0.3, tight_weight=0.4, alpha=1.2, recon_weight=1.0)

    x = torch.randn(4, 16, 3)
    outputs = model((x,))
    recon, fz, ff_z, f_fz, f_z = outputs
    target = x

    actual = loss_fn(outputs, (target,))

    per_sample_recon = (
        torch.nn.functional.mse_loss(recon, target, reduction="none")
        .reshape(4, -1)
        .mean(dim=-1)
    )
    expected_recon = per_sample_recon.mean()
    expected_idem = torch.nn.functional.l1_loss(f_fz, fz, reduction="mean")
    raw_tight = -(
        torch.nn.functional.l1_loss(ff_z, f_z, reduction="none").reshape(4, -1).mean(dim=-1)
    )
    clamp = torch.clamp(1.2 * per_sample_recon, min=1e-6)
    expected_tight = (torch.tanh(raw_tight / clamp) * clamp).mean()
    expected = 1.0 * expected_recon + 0.3 * expected_idem + 0.4 * expected_tight

    assert torch.allclose(actual, expected, atol=1e-6)


def test_igad_loss_eval_path_uses_only_reconstruction() -> None:
    loss_fn = IGADLoss(idem_weight=0.5, tight_weight=0.5, alpha=1.1, recon_weight=2.0)
    recon = torch.randn(2, 8, 3)
    target = torch.randn(2, 8, 3)
    expected = 2.0 * torch.nn.functional.mse_loss(recon, target)
    actual = loss_fn((recon,), (target,))
    assert torch.allclose(actual, expected)


def test_igad_loss_unwraps_nested_prediction_tuple_from_measure_flops() -> None:
    # Lightning's `measure_flops` calls `loss((out,), (target,))` where `out`
    # is already the network's full 5-tuple. The loss must unwrap that extra
    # layer instead of treating the inner tuple as a tensor.
    torch.manual_seed(7)
    model = IGAD(window_size=16, input_dim=3, hidden_dims=[32], latent_dim=8)
    loss_fn = IGADLoss(idem_weight=0.4, tight_weight=0.3, alpha=1.2, recon_weight=1.0)
    x = torch.randn(2, 16, 3)
    out = model((x,))

    direct = loss_fn(out, (x,))
    via_meta_flops_wrapping = loss_fn((out,), (x,))

    assert torch.allclose(direct, via_meta_flops_wrapping)


def test_igad_loss_drops_to_pure_reconstruction_when_weights_zero() -> None:
    torch.manual_seed(2)
    model = IGAD(window_size=12, input_dim=2, hidden_dims=[16], latent_dim=4)
    loss_fn = IGADLoss(idem_weight=0.0, tight_weight=0.0, alpha=1.1, recon_weight=1.0)
    x = torch.randn(3, 12, 2)
    outputs = model((x,))
    expected = torch.nn.functional.mse_loss(outputs[0], x)
    actual = loss_fn(outputs, (x,))
    assert torch.allclose(actual, expected)


def test_igad_detector_returns_per_window_score() -> None:
    model = IGAD(window_size=16, input_dim=4, hidden_dims=[16], latent_dim=8)
    detector = IGADAnomalyDetector(model)
    x = torch.randn(5, 16, 4)
    score = detector.compute_online_anomaly_score((x,))
    assert score.shape == (5,)


def test_frozen_copy_does_not_train_directly() -> None:
    model = IGAD(window_size=8, input_dim=2, hidden_dims=[8], latent_dim=4)
    for parameter in model.frozen_network.parameters():
        assert not parameter.requires_grad
    for parameter in model.network.parameters():
        assert parameter.requires_grad


def test_grouped_parameters_excludes_frozen_copy() -> None:
    # The optimizer must only see trainable parameters: BenchmarkModel calls
    # `manual_backward(..., inputs=opt_params)` which fails on tensors with
    # requires_grad=False (i.e. anything from the frozen_network deepcopy).
    model = IGAD(window_size=8, input_dim=2, hidden_dims=[8], latent_dim=4)
    groups = model.grouped_parameters()
    assert len(groups) == 1
    grouped = list(groups[0])
    live = list(model.network.parameters())
    assert len(grouped) == len(live)
    for tensor in grouped:
        assert tensor.requires_grad
    frozen_ids = {id(p) for p in model.frozen_network.parameters()}
    for tensor in grouped:
        assert id(tensor) not in frozen_ids


def test_mlp_autoencoder_round_trips_shape() -> None:
    ae = MLPAutoEncoder(window_size=10, input_dim=3, hidden_dims=[16], latent_dim=4)
    x = torch.randn(2, 10, 3)
    recon = ae(x)
    assert recon.shape == x.shape
