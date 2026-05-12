import importlib.util
import itertools
import math
from pathlib import Path
import sys
import types
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _install_timesead_stubs() -> None:
    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault("timesead.models.common", types.ModuleType("timesead.models.common"))
    optim_module = sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))
    loss_module = sys.modules.setdefault("timesead.optim.loss", types.ModuleType("timesead.optim.loss"))

    class BaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class AnomalyDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    class Loss(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    models_module.BaseModel = BaseModel
    common_module.AnomalyDetector = AnomalyDetector
    loss_module.Loss = Loss
    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    optim_module.loss = loss_module


_install_timesead_stubs()

BENCHMARK_HELPERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "benchmark_helpers.py"
)


def _load_pad_each_subsequence():
    spec = importlib.util.spec_from_file_location("physdiff_test_benchmark_helpers", BENCHMARK_HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.pad_each_subsequence


pad_each_subsequence = _load_pad_each_subsequence()

from noboom_benchmark.noboom_lib.core.physdiff import (  # noqa: E402
    PhysDiff,
    PhysDiffAnomalyDetector,
    PhysDiffLoss,
)
from noboom_benchmark.noboom_lib.core.models.physdiff.model import (  # noqa: E402
    MultiChannelAdaptiveFourierDecomposer,
    PotentialFieldNetwork,
)


class QuadraticPotential(nn.Module):
    def forward(self, x: torch.Tensor, gaussian_bandwidth: torch.Tensor) -> torch.Tensor:
        del gaussian_bandwidth
        return 0.5 * x.pow(2).flatten(start_dim=1).sum(dim=1, keepdim=True)


class ZeroDenoiser(nn.Module):
    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        aspe: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, high, low, aspe, energy
        return torch.zeros_like(x_t)


class CaptureZeroDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_x_t: Optional[torch.Tensor] = None

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        aspe: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, high, low, aspe, energy
        self.seen_x_t = x_t.detach().clone()
        return torch.zeros_like(x_t)


def _manual_paper_aspe(signal: torch.Tensor, order: int) -> torch.Tensor:
    perms = list(itertools.permutations(range(order)))
    batch_size, time_len, num_channels = signal.shape
    num_windows = time_len - order + 1
    values = torch.zeros(batch_size, num_channels, dtype=signal.dtype)
    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            pattern_counts = {perm: 0 for perm in perms}
            pattern_weight_sums = {perm: 0.0 for perm in perms}
            for start in range(num_windows):
                window = signal[batch_idx, start : start + order, channel_idx]
                pattern = tuple(torch.argsort(window, stable=True).tolist())
                mean_abs = window.abs().mean().clamp_min(1e-8)
                weight = float(window.std(unbiased=False) / mean_abs)
                pattern_counts[pattern] += 1
                pattern_weight_sums[pattern] += weight
            entropy = 0.0
            for perm in perms:
                count = pattern_counts[perm]
                if count == 0:
                    continue
                frequency = count / float(num_windows)
                omega = pattern_weight_sums[perm] / float(count)
                entropy -= omega * frequency * math.log(frequency)
            values[batch_idx, channel_idx] = entropy / math.log(math.factorial(order))
    return values.mean(dim=1)


def _manual_weighted_probability_aspe(signal: torch.Tensor, order: int) -> torch.Tensor:
    perms = list(itertools.permutations(range(order)))
    batch_size, time_len, num_channels = signal.shape
    num_windows = time_len - order + 1
    values = torch.zeros(batch_size, num_channels, dtype=signal.dtype)
    for batch_idx in range(batch_size):
        for channel_idx in range(num_channels):
            weighted_counts = {perm: 0.0 for perm in perms}
            for start in range(num_windows):
                window = signal[batch_idx, start : start + order, channel_idx]
                pattern = tuple(torch.argsort(window, stable=True).tolist())
                mean_abs = window.abs().mean().clamp_min(1e-8)
                weighted_counts[pattern] += float(window.std(unbiased=False) / mean_abs)
            total = sum(weighted_counts.values())
            entropy = 0.0
            for weighted_count in weighted_counts.values():
                if weighted_count == 0.0:
                    continue
                probability = weighted_count / total
                entropy -= probability * math.log(probability)
            values[batch_idx, channel_idx] = entropy / math.log(math.factorial(order))
    return values.mean(dim=1)


def _make_windows(sequence_length: int = 14, window_size: int = 8, num_channels: int = 3) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, sequence_length)
    columns = [
        torch.sin(2.0 * torch.pi * time),
        torch.cos(2.0 * torch.pi * time),
        time,
    ]
    values = torch.stack(columns[:num_channels], dim=1).float()
    return torch.stack(
        [values[start : start + window_size] for start in range(sequence_length - window_size + 1)],
        dim=0,
    )


def test_physdiff_forward_detector_and_loss_shapes() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=3,
        model_dim=32,
        ff_dim=64,
        attn_dim=8,
        num_heads=4,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    loss = PhysDiffLoss()
    inputs = _make_windows()[:2]

    outputs = model((inputs,))
    loss_value = loss(outputs, (inputs,), epoch=0, num_epochs=1)

    assert len(outputs) == 5
    assert outputs[0].shape == inputs.shape
    assert outputs[1].shape == inputs.shape
    assert outputs[2].shape == inputs.shape
    assert outputs[3].shape == (2,)
    assert outputs[4].shape == (2,)
    assert loss_value.ndim == 0
    assert torch.isfinite(loss_value)


def test_physdiff_denoiser_position_embedding_exists_and_gets_gradients() -> None:
    torch.manual_seed(3)
    model = PhysDiff(
        win_size=8,
        num_channels=3,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    assert isinstance(model.denoiser.position_embed, nn.Parameter)
    assert model.denoiser.position_embed.shape == (1, 8, 16)
    assert model.denoiser.position_embed.requires_grad

    x_t = torch.randn(2, 8, 3)
    timesteps = torch.tensor([0, 5], dtype=torch.long)
    high = torch.randn_like(x_t)
    low = torch.randn_like(x_t)
    aspe = torch.rand(2)
    energy = torch.rand(2)
    output = model.denoiser(x_t, timesteps, high, low, aspe, energy)
    output.pow(2).mean().backward()

    assert model.denoiser.position_embed.grad is not None
    assert torch.linalg.vector_norm(model.denoiser.position_embed.grad) > 0.0

    with pytest.raises(ValueError, match="position embedding length"):
        model.denoiser(torch.randn(2, 9, 3), timesteps, high, low, aspe, energy)


def test_potential_field_network_frequency_features_are_differentiable() -> None:
    torch.manual_seed(4)
    potential = PotentialFieldNetwork(win_size=8, num_channels=2, hidden_dim=16)
    x = torch.randn(3, 8, 2, requires_grad=True)
    phi = potential(x, torch.tensor(0.25))

    assert phi.shape == (3, 1)
    assert any("freq" in name for name, _param in potential.named_parameters())

    phi.sum().backward()
    assert x.grad is not None
    assert torch.linalg.vector_norm(x.grad) > 0.0


def test_mafd_aspe_matches_paper_formula_not_weighted_probability_entropy() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=2,
        dictionary_distance=0.05,
        max_magnitude=0.9,
        aspe_order=3,
    )
    signal = torch.tensor(
        [
            [
                [1.0],
                [2.0],
                [3.0],
                [100.0],
                [90.0],
                [80.0],
            ]
        ]
    )

    actual = decomposer._amplitude_sensitive_permutation_entropy(signal)
    expected = _manual_paper_aspe(signal, order=3)
    weighted_probability = _manual_weighted_probability_aspe(signal, order=3)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert not torch.allclose(expected, weighted_probability, atol=1e-3)


def test_physdiff_energy_returns_grad_energy_not_grad_phi() -> None:
    model = PhysDiff(
        win_size=4,
        num_channels=2,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    model.potential = QuadraticPotential()
    x = torch.randn(2, 4, 2)

    energy, grad_energy = model._energy(x, create_graph=False, compute_grad_energy=True)

    assert torch.allclose(energy, x.pow(2).sum(dim=(1, 2)), atol=1e-6)
    assert grad_energy is not None
    assert torch.allclose(grad_energy, 2.0 * x, atol=1e-6)


def test_p_sample_uses_grad_energy_for_langevin_update() -> None:
    model = PhysDiff(
        win_size=4,
        num_channels=2,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
        langevin_step=0.05,
        langevin_noise=False,
    )
    model.potential = QuadraticPotential()
    model.denoiser = ZeroDenoiser()
    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: torch.ones(
        timesteps.size(0), time_len // 2 + 1, device=device, dtype=dtype
    )
    x_t = torch.randn(2, 4, 2)
    timesteps = torch.zeros(2, dtype=torch.long)
    high = torch.zeros_like(x_t)
    low = torch.zeros_like(x_t)
    aspe = torch.zeros(2)

    sample = model._p_sample(x_t, timesteps, high, low, aspe)
    expected_mean = model.schedule.posterior_mean_from_noise(
        x_t, timesteps, torch.zeros_like(x_t)
    )
    expected = expected_mean - model.langevin_step * 2.0 * x_t

    assert torch.allclose(sample, expected, atol=1e-6)


def test_p_sample_applies_gamma_to_grad_energy_in_frequency_domain() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=1,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
        langevin_step=0.1,
        langevin_noise=False,
    )
    model.potential = QuadraticPotential()
    model.denoiser = ZeroDenoiser()
    gamma = torch.tensor([[1.0, 0.25, 0.0, 0.0, 0.0]], dtype=torch.float32)
    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: gamma.to(
        device=device, dtype=dtype
    )
    x_t = torch.randn(1, 8, 1)
    timesteps = torch.zeros(1, dtype=torch.long)
    high = torch.zeros_like(x_t)
    low = torch.zeros_like(x_t)
    aspe = torch.zeros(1)

    sample = model._p_sample(x_t, timesteps, high, low, aspe)
    expected_mean = model.schedule.posterior_mean_from_noise(
        x_t, timesteps, torch.zeros_like(x_t)
    )
    expected_guidance = torch.fft.irfft(
        torch.fft.rfft((2.0 * x_t).to(torch.float32), dim=1) * gamma.unsqueeze(-1),
        n=x_t.shape[1],
        dim=1,
    ).to(x_t.dtype)
    expected = expected_mean - model.langevin_step * expected_guidance

    assert torch.allclose(sample, expected, atol=1e-6)


def test_p_sample_adds_explicit_langevin_noise_by_default(monkeypatch) -> None:
    model = PhysDiff(
        win_size=4,
        num_channels=1,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
        langevin_step=0.05,
    )
    model.potential = QuadraticPotential()
    model.denoiser = ZeroDenoiser()
    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: torch.ones(
        timesteps.size(0), time_len // 2 + 1, device=device, dtype=dtype
    )

    random_calls = {"count": 0}

    def fake_randn_like(tensor):
        random_calls["count"] += 1
        if random_calls["count"] == 1:
            return torch.zeros_like(tensor)
        return torch.ones_like(tensor)

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)

    x_t = torch.randn(1, 4, 1)
    timesteps = torch.ones(1, dtype=torch.long)
    high = torch.zeros_like(x_t)
    low = torch.zeros_like(x_t)
    aspe = torch.zeros(1)

    sample = model._p_sample(x_t, timesteps, high, low, aspe)
    expected_mean = model.schedule.posterior_mean_from_noise(
        x_t, timesteps, torch.zeros_like(x_t)
    )
    expected = (
        expected_mean
        - model.langevin_step * 2.0 * x_t
        + math.sqrt(2.0 * model.langevin_step) * torch.ones_like(x_t)
    )

    assert random_calls["count"] == 2
    assert torch.allclose(sample, expected, atol=1e-6)


def test_frequency_mix_matches_manual_fft_blend() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=1,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    x_t = torch.randn(2, 8, 1)
    prior = torch.randn_like(x_t)
    high = 0.25 * prior
    low = 0.75 * prior
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: torch.zeros(
        timesteps.size(0), time_len // 2 + 1, device=device, dtype=dtype
    )
    assert torch.allclose(model._frequency_mix(x_t, high, low, timesteps), x_t, atol=1e-6)

    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: torch.ones(
        timesteps.size(0), time_len // 2 + 1, device=device, dtype=dtype
    )
    assert torch.allclose(model._frequency_mix(x_t, high, low, timesteps), prior, atol=1e-6)

    gamma = torch.tensor(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25, 0.0],
        ],
        dtype=torch.float32,
    )
    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: gamma.to(
        device=device, dtype=dtype
    )
    expected = torch.fft.irfft(
        (1.0 - gamma.unsqueeze(-1)) * torch.fft.rfft(x_t.to(torch.float32), dim=1)
        + gamma.unsqueeze(-1) * torch.fft.rfft(prior.to(torch.float32), dim=1),
        n=x_t.shape[1],
        dim=1,
    ).to(x_t.dtype)

    assert torch.allclose(model._frequency_mix(x_t, high, low, timesteps), expected, atol=1e-6)


def test_forward_denoiser_receives_mixed_state_and_energy_uses_raw_state() -> None:
    model = PhysDiff(
        win_size=4,
        num_channels=1,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    raw_x_t = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    high = torch.zeros_like(raw_x_t)
    low = torch.full_like(raw_x_t, 10.0)
    aspe = torch.zeros(1)
    capture_denoiser = CaptureZeroDenoiser()
    seen_energy_inputs = []
    model.denoiser = capture_denoiser
    model._training_priors = lambda x: (high, low, aspe)
    model.schedule.q_sample = lambda x_start, timesteps, noise: raw_x_t
    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: torch.ones(
        timesteps.size(0), time_len // 2 + 1, device=device, dtype=dtype
    )

    def fake_energy(x: torch.Tensor, create_graph: bool, compute_grad_energy: bool = True):
        del create_graph, compute_grad_energy
        seen_energy_inputs.append(x.detach().clone())
        return torch.zeros(x.size(0), dtype=x.dtype, device=x.device), None

    model._energy = fake_energy

    output = model((torch.zeros_like(raw_x_t),))[0]

    assert torch.allclose(output, torch.zeros_like(raw_x_t))
    assert len(seen_energy_inputs) == 1
    assert torch.allclose(seen_energy_inputs[0], raw_x_t)
    assert capture_denoiser.seen_x_t is not None
    assert torch.allclose(capture_denoiser.seen_x_t, high + low, atol=1e-6)
    assert not torch.allclose(capture_denoiser.seen_x_t, raw_x_t)


def test_p_sample_denoiser_receives_mixed_state_but_posterior_uses_raw_state() -> None:
    model = PhysDiff(
        win_size=4,
        num_channels=1,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
        langevin_step=0.0,
        langevin_noise=False,
    )
    model.potential = QuadraticPotential()
    capture_denoiser = CaptureZeroDenoiser()
    model.denoiser = capture_denoiser
    model._frequency_guidance_mask = lambda timesteps, time_len, device, dtype: torch.ones(
        timesteps.size(0), time_len // 2 + 1, device=device, dtype=dtype
    )
    x_t = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    high = torch.zeros_like(x_t)
    low = torch.full_like(x_t, 7.0)
    timesteps = torch.zeros(1, dtype=torch.long)
    aspe = torch.zeros(1)

    sample = model._p_sample(x_t, timesteps, high, low, aspe)
    expected = model.schedule.posterior_mean_from_noise(x_t, timesteps, torch.zeros_like(x_t))

    assert capture_denoiser.seen_x_t is not None
    assert torch.allclose(capture_denoiser.seen_x_t, high + low, atol=1e-6)
    assert not torch.allclose(capture_denoiser.seen_x_t, x_t)
    assert torch.allclose(sample, expected, atol=1e-6)


def test_physdiff_meta_forward_loss_supports_flop_probe_backward() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=3,
        model_dim=32,
        ff_dim=64,
        attn_dim=8,
        num_heads=4,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    ).to_empty(device=torch.device("meta"))
    loss = PhysDiffLoss()
    inputs = torch.randn(2, 8, 3, device="meta")
    target = torch.randn(2, 8, 3, device="meta")

    outputs = model((inputs,))
    loss_value = loss((outputs,), (target,), epoch=0, num_epochs=1)

    assert loss_value.requires_grad
    loss_value.backward()


def test_physdiff_tiny_train_predict_smoke_scores_align_to_rows() -> None:
    torch.manual_seed(7)
    window_size = 8
    windows = _make_windows(sequence_length=14, window_size=window_size, num_channels=3)
    loader = DataLoader(TensorDataset(windows), batch_size=2, shuffle=False)
    model = PhysDiff(
        win_size=window_size,
        num_channels=3,
        model_dim=32,
        ff_dim=64,
        attn_dim=8,
        num_heads=4,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    loss = PhysDiffLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    for batch, in loader:
        optimizer.zero_grad()
        loss_value = loss(model((batch,)), (batch,), epoch=0, num_epochs=1)
        loss_value.backward()
        optimizer.step()

    detector = PhysDiffAnomalyDetector(
        model,
        score_alpha=0.5,
        smoothing_kernel_size=5,
        spot_q=0.01,
        reconstruction_num_samples=1,
        component_standardize=False,
        score_normalize=False,
    )
    detector.fit(loader, window_size=window_size)
    model.eval()

    scores = torch.cat([detector((batch,)) for batch, in loader], dim=0)
    assert scores.shape == (windows.shape[0],)
    assert torch.isfinite(scores).all()

    aligned = pad_each_subsequence(
        scores,
        [scores.numel()],
        pad_prefix=window_size - 1,
        value=-torch.inf,
    )
    assert aligned.shape == (14,)
    assert torch.isneginf(aligned[: window_size - 1]).all()
    assert torch.isfinite(aligned[window_size - 1 :]).all()


def test_mafd_constrains_poles_inside_unit_disk() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=8,
        dictionary_distance=0.05,
        max_magnitude=0.95,
    )
    # Push raw params to large magnitudes; tanh(|raw|) -> 1 so |a| -> max_magnitude.
    with torch.no_grad():
        decomposer.pole_real.copy_(torch.tensor([10.0, -10.0, 5.0, 0.0, 1.0, -1.0, 3.0, -3.0]))
        decomposer.pole_imag.copy_(torch.tensor([0.0, 0.0, 5.0, 10.0, 1.0, 1.0, -3.0, 3.0]))
    poles = decomposer.constrained_poles()
    magnitudes = poles.abs()
    assert torch.all(magnitudes < 0.95 + 1e-6)
    # The two saturated entries (raw mag 10) should land essentially at max_magnitude.
    assert torch.allclose(magnitudes[:2], torch.tensor([0.95, 0.95]), atol=1e-5)


def test_mafd_basis_is_unit_modulus_on_unit_circle() -> None:
    """|B_n(e^{jt})| = sqrt(1 - |a_n|^2) / |1 - conj(a_n) e^{jt}|.

    The Blaschke prefix has unit modulus on |z| = 1, so the full basis modulus
    matches the first-term modulus. We verify this energy-preservation
    property numerically.
    """
    torch.manual_seed(0)
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=4,
        dictionary_distance=0.05,
        max_magnitude=0.9,
    )
    poles = decomposer.constrained_poles()
    basis = decomposer._tm_basis(time_len=64, device=torch.device("cpu"), real_dtype=torch.float32)
    # First-term modulus per pole (broadcast across time).
    t = 2.0 * torch.pi * torch.arange(64) / 64
    z = torch.complex(torch.cos(t), torch.sin(t))
    expected_first_mod = torch.sqrt(1.0 - poles.abs().pow(2)).unsqueeze(1) / (
        1.0 - poles.conj().unsqueeze(1) * z.unsqueeze(0)
    ).abs()
    assert torch.allclose(basis.abs(), expected_first_mod, atol=1e-5)


def test_mafd_decomposition_shapes_and_residual_definition() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=4,
        dictionary_distance=0.05,
        max_magnitude=0.9,
    )
    x = _make_windows(sequence_length=12, window_size=8, num_channels=3)[:3]
    high, low, aspe = decomposer(x)
    assert high.shape == x.shape
    assert low.shape == x.shape
    assert aspe.shape == (x.shape[0],)
    assert torch.allclose(high + low, x, atol=1e-5)


def test_mafd_constant_input_keeps_mean_in_low_component() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=4,
        dictionary_distance=0.05,
        max_magnitude=0.9,
    )
    x = torch.full((2, 8, 3), 5.0)

    high, low, _aspe = decomposer(x)

    assert torch.allclose(low, x, atol=1e-5)
    assert torch.allclose(high, torch.zeros_like(x), atol=1e-5)
    assert torch.allclose(high + low, x, atol=1e-5)


def test_mafd_mean_shifted_sinusoid_reconstructs_and_low_contains_removed_mean() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=4,
        dictionary_distance=0.05,
        max_magnitude=0.9,
    )
    time = torch.linspace(0.0, 2.0 * torch.pi, 16)
    x = (3.0 + torch.sin(time)).view(1, 16, 1)

    high, low, _aspe = decomposer(x)

    assert torch.allclose(high + low, x, atol=1e-5)
    assert torch.allclose(low.mean(dim=1), x.mean(dim=1), atol=0.1)


def test_mafd_adaptive_counts_stop_before_max_on_simple_signal() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=6,
        dictionary_distance=0.05,
        max_magnitude=0.9,
        aspe_tolerance=1.0,
    )
    x = _make_windows(sequence_length=16, window_size=12, num_channels=2)[:3]

    decomposer(x)

    assert decomposer.last_component_counts is not None
    assert torch.all(decomposer.last_component_counts >= 1)
    assert torch.all(decomposer.last_component_counts < decomposer.max_components)


def test_mafd_lower_aspe_tolerance_uses_at_least_as_many_components() -> None:
    high_tolerance = MultiChannelAdaptiveFourierDecomposer(
        max_components=6,
        dictionary_distance=0.05,
        max_magnitude=0.9,
        aspe_tolerance=1.0,
    )
    low_tolerance = MultiChannelAdaptiveFourierDecomposer(
        max_components=6,
        dictionary_distance=0.05,
        max_magnitude=0.9,
        aspe_tolerance=0.0,
    )
    low_tolerance.load_state_dict(high_tolerance.state_dict())
    x = _make_windows(sequence_length=16, window_size=12, num_channels=2)[:3]

    high_tolerance(x)
    low_tolerance(x)

    assert high_tolerance.last_component_counts is not None
    assert low_tolerance.last_component_counts is not None
    assert torch.all(low_tolerance.last_component_counts >= high_tolerance.last_component_counts)


def test_mafd_selected_poles_obey_dictionary_distance() -> None:
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=8,
        dictionary_distance=0.4,
        max_magnitude=0.9,
        aspe_tolerance=-1.0,
    )
    x = _make_windows(sequence_length=16, window_size=12, num_channels=2)[:2]

    decomposer(x)

    assert decomposer.last_selected_mask is not None
    poles = decomposer.constrained_poles().detach()
    for selected_mask in decomposer.last_selected_mask:
        selected = poles[selected_mask]
        assert selected.numel() > 1
        distances = torch.abs(selected.unsqueeze(0) - selected.unsqueeze(1))
        distances = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
        assert torch.all(distances > decomposer.dictionary_distance - 1e-6)


def test_mafd_residual_loss_reaches_pole_parameters() -> None:
    """Residual energy `‖high‖²` must produce non-zero gradients on poles
    (this is the core fix: the previous implementation had zero gradient
    flow into MAFD because the entire forward ran under torch.no_grad)."""
    torch.manual_seed(1)
    decomposer = MultiChannelAdaptiveFourierDecomposer(
        max_components=3,
        dictionary_distance=0.05,
        max_magnitude=0.9,
    )
    x = _make_windows(sequence_length=16, window_size=12, num_channels=2)[:2]
    high, _low, _aspe = decomposer(x)
    loss = high.pow(2).mean()
    loss.backward()

    assert decomposer.pole_real.grad is not None
    assert decomposer.pole_imag.grad is not None
    assert torch.linalg.vector_norm(decomposer.pole_real.grad) > 0.0
    assert torch.linalg.vector_norm(decomposer.pole_imag.grad) > 0.0


def test_mafd_pole_parameters_appear_in_physdiff_optimizer() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=3,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=4,
        dropout=0.0,
    )
    pole_param_ids = {id(model.decomposer.pole_real), id(model.decomposer.pole_imag)}
    all_param_ids = {id(p) for p in model.parameters()}
    assert pole_param_ids.issubset(all_param_ids)


def test_frequency_guidance_mask_shape_and_range() -> None:
    model = PhysDiff(
        win_size=16,
        num_channels=2,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=20,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    timesteps = torch.tensor([0, 5, 19], dtype=torch.long)
    gamma = model._frequency_guidance_mask(
        timesteps, time_len=16, device=torch.device("cpu"), dtype=torch.float32
    )
    assert gamma.shape == (3, 16 // 2 + 1)
    assert torch.all(gamma > 0.0) and torch.all(gamma < 1.0)


def test_frequency_guidance_mask_low_pass_shape() -> None:
    """Smaller σ → mask falls off more steeply with frequency."""
    model = PhysDiff(
        win_size=16,
        num_channels=2,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=20,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
        gaussian_bandwidth_init=0.10,  # narrower
    )
    timesteps = torch.tensor([model.schedule.time_steps - 1], dtype=torch.long)
    gamma_narrow = model._frequency_guidance_mask(
        timesteps, time_len=16, device=torch.device("cpu"), dtype=torch.float32
    )[0]
    # Widen σ in place and re-evaluate.
    with torch.no_grad():
        model.log_gaussian_bandwidth.copy_(torch.tensor(math.log(1.0)))
    gamma_wide = model._frequency_guidance_mask(
        timesteps, time_len=16, device=torch.device("cpu"), dtype=torch.float32
    )[0]
    # Narrow σ: highest-frequency bin attenuated more strongly.
    assert gamma_narrow[-1] < gamma_wide[-1]
    # Both masks should monotonically decrease across frequency at high-noise t.
    assert torch.all(gamma_narrow[:-1] >= gamma_narrow[1:] - 1e-6)
    assert torch.all(gamma_wide[:-1] >= gamma_wide[1:] - 1e-6)


def test_log_gaussian_bandwidth_is_a_learnable_parameter() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=2,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
        gaussian_bandwidth_init=0.3,
    )
    assert isinstance(model.log_gaussian_bandwidth, nn.Parameter)
    assert model.log_gaussian_bandwidth.requires_grad
    assert torch.allclose(model.gaussian_bandwidth(), torch.tensor(0.3), atol=1e-6)
    # Parameter must reach the optimizer via grouped_parameters.
    grouped = list(model.grouped_parameters()[0])
    assert any(p is model.log_gaussian_bandwidth for p in grouped)


def test_frequency_guidance_gradient_reaches_log_sigma() -> None:
    """A scalar loss derived from γ(t, ω) must produce a non-zero gradient on σ."""
    model = PhysDiff(
        win_size=16,
        num_channels=2,
        model_dim=16,
        ff_dim=32,
        attn_dim=8,
        num_heads=2,
        num_blocks=1,
        time_steps=20,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    timesteps = torch.tensor([10, 15, 19], dtype=torch.long)
    gamma = model._frequency_guidance_mask(
        timesteps, time_len=16, device=torch.device("cpu"), dtype=torch.float32
    )
    gamma.sum().backward()
    assert model.log_gaussian_bandwidth.grad is not None
    assert torch.linalg.vector_norm(model.log_gaussian_bandwidth.grad) > 0.0


def test_physdiff_detector_runs_under_inference_mode() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=3,
        model_dim=32,
        ff_dim=64,
        attn_dim=8,
        num_heads=4,
        num_blocks=1,
        time_steps=10,
        sampling_steps=4,
        mafd_components=2,
        dropout=0.0,
    )
    detector = PhysDiffAnomalyDetector(
        model,
        score_alpha=0.5,
        smoothing_kernel_size=1,
        spot_q=0.0,
        reconstruction_num_samples=1,
        component_standardize=False,
        score_normalize=False,
    )
    batch = _make_windows()[:2]

    model.eval()
    with torch.inference_mode():
        scores = detector((batch,))

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_physdiff_reconstruct_guidance_runs_under_no_grad_and_inference_mode() -> None:
    model = PhysDiff(
        win_size=8,
        num_channels=3,
        model_dim=32,
        ff_dim=64,
        attn_dim=8,
        num_heads=4,
        num_blocks=1,
        time_steps=10,
        sampling_steps=2,
        mafd_components=2,
        dropout=0.0,
    )
    batch = _make_windows()[:2]

    model.eval()
    with torch.no_grad():
        no_grad_reconstruction = model.reconstruct(batch)
    with torch.inference_mode():
        inference_reconstruction = model.reconstruct(batch)

    assert no_grad_reconstruction.shape == batch.shape
    assert inference_reconstruction.shape == batch.shape
    assert torch.isfinite(no_grad_reconstruction).all()
    assert torch.isfinite(inference_reconstruction).all()
