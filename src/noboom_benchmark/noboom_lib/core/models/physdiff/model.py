"""PhysDiff adapted to NoBoomBenchmark's Lightning/OmegaConf model contract."""

import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timesead.models import BaseModel
try:
    from timesead.models.common import AnomalyDetector
except ImportError:
    from timesead.models.common.anomaly_detector import AnomalyDetector
from timesead.optim.loss import Loss

PhysDiffForwardOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def smooth_scores(values: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply centered moving-average smoothing to a one-dimensional score array."""
    if kernel_size <= 1 or values.size == 0:
        return values.astype(np.float64, copy=False)
    width = min(int(kernel_size), int(values.size))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def transient_tfd_distribution(x: torch.Tensor, n_fft: Optional[int] = None) -> torch.Tensor:
    """Estimate the transient time-frequency distribution used by PhysDiff."""
    if x.ndim != 3:
        raise ValueError(f"Expected x to have shape [batch, time, channels], got {x.shape}.")

    batch_size, win_size, num_channels = x.shape
    if n_fft is None:
        n_fft = min(16, win_size)
    n_fft = max(2, min(int(n_fft), win_size))
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)

    reshaped = x.transpose(1, 2).reshape(batch_size * num_channels, win_size)
    stft = torch.stft(
        reshaped,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        return_complex=True,
    )
    power = stft.abs().pow(2)
    power = power.view(batch_size, num_channels, power.size(-2), power.size(-1)).mean(dim=1)
    flat = power.flatten(1)
    return flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-6)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion-step embedding with a small projection MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * (-scale)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.size(1) < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.size(1)))
        return self.proj(embedding.to(dtype=self.proj[0].weight.dtype))


class DiffusionSchedule(nn.Module):
    """DDPM schedule and posterior helpers."""

    def __init__(self, time_steps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        if time_steps < 2:
            raise ValueError("time_steps must be at least 2.")
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError("beta_start and beta_end must be positive.")

        betas = torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]],
            dim=0,
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance[0] = betas[0]

        self.time_steps = int(time_steps)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )
        self.register_buffer("sqrt_recip_alphas", torch.rsqrt(alphas), persistent=False)
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)

    def extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        out = values.to(device=timesteps.device).gather(0, timesteps)
        return out.view(timesteps.size(0), *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus = self.extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha.to(x_start.dtype) * x_start + sqrt_one_minus.to(x_start.dtype) * noise

    def posterior_mean_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        beta_t = self.extract(self.betas, timesteps, x_t.shape).to(x_t.dtype)
        sqrt_recip_alpha = self.extract(self.sqrt_recip_alphas, timesteps, x_t.shape).to(x_t.dtype)
        sqrt_one_minus = self.extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape).to(x_t.dtype)
        return sqrt_recip_alpha * (x_t - beta_t * noise / sqrt_one_minus.clamp_min(1e-6))


class MultiChannelAdaptiveFourierDecomposer(nn.Module):
    """Differentiable multi-channel adaptive Fourier decomposition (MAFD).

    Implements the multi-channel AFD from Wang et al., IEEE TSP 2022 (cited as
    reference [17] of the PhysDiff paper, Eq. 3) with learnable Blaschke pole
    candidates. ``max_components`` is the candidate budget; each sample greedily
    selects an adaptive subset and stops when the residual ASPE improvement is
    no larger than ``aspe_tolerance``.

    For ``N`` complex poles ``a_n`` in the open unit disk, the
    Takenaka–Malmquist orthonormal basis on the unit circle is

        B_n(z) = sqrt(1 - |a_n|^2) / (1 - conj(a_n) * z)
                 * prod_{k<n} (z - a_k) / (1 - conj(a_k) * z)

    where the trailing product is the Blaschke prefix that orthogonalises
    against earlier poles. Candidate poles are shared across all channels, but
    each channel is projected independently — matching the paper's
    statement that ``A_n = [A_{1,n}, ..., A_{C,n}]`` with cross-channel shared
    basis ``B_n`` (paper Eq. 3). Pole positions are stored as raw real/imag
    parameters and constrained to ``|a| < max_magnitude``.

    Real input signals are projected via their analytic-signal representation
    (one-sided spectrum), which is the standard treatment of AFD on real
    sequences and yields a real-valued ``low`` reconstruction. The residual
    ``high = x - low`` feeds diffusion conditioning, and residual losses still
    propagate gradients into the selected pole candidates.
    """

    def __init__(
        self,
        max_components: int,
        dictionary_distance: float,
        max_magnitude: float,
        aspe_order: int = 3,
        aspe_tolerance: float = 1e-3,
        init_pole_radius: float = 0.6,
    ) -> None:
        super().__init__()
        if max_components <= 0:
            raise ValueError("mafd_components must be positive.")
        if dictionary_distance < 0.0:
            raise ValueError("dictionary_distance must be non-negative.")
        if not (0.0 < max_magnitude < 1.0):
            raise ValueError("max_magnitude must be in (0, 1).")
        self.max_components = int(max_components)
        self.dictionary_distance = float(dictionary_distance)
        self.max_magnitude = float(max_magnitude)
        self.aspe_order = int(max(2, aspe_order))
        self.aspe_tolerance = float(aspe_tolerance)
        self.init_pole_radius = float(init_pole_radius)
        self.last_selected_mask: Optional[torch.Tensor] = None
        self.last_component_counts: Optional[torch.Tensor] = None

        # Learnable Blaschke poles, spread on a circle of `init_pole_radius`
        # in raw-parameter space. After the tanh constraint with default
        # `max_magnitude=0.95` and `init_pole_radius=0.6` the initial poles
        # land at |a| ≈ 0.95 * tanh(0.6) ≈ 0.51, leaving room for the
        # optimiser to move them in either direction.
        if self.max_components == 1:
            angles_init = torch.zeros(1, dtype=torch.float32)
        else:
            angles_init = torch.arange(self.max_components, dtype=torch.float32) * (
                2.0 * math.pi / self.max_components
            )
        init_re = self.init_pole_radius * torch.cos(angles_init)
        init_im = self.init_pole_radius * torch.sin(angles_init)
        self.pole_real = nn.Parameter(init_re)
        self.pole_imag = nn.Parameter(init_im)

        permutations = list(itertools.permutations(range(self.aspe_order)))
        self.register_buffer(
            "permutation_patterns",
            torch.tensor(permutations, dtype=torch.long),
            persistent=False,
        )

    def constrained_poles(self) -> torch.Tensor:
        """Return the complex pole vector with ``|a_n| < max_magnitude`` enforced."""
        raw_re = self.pole_real
        raw_im = self.pole_imag
        abs_val = torch.sqrt(raw_re.pow(2) + raw_im.pow(2)).clamp_min(1e-8)
        constrained_abs = self.max_magnitude * torch.tanh(abs_val)
        scale = constrained_abs / abs_val
        return torch.complex(raw_re * scale, raw_im * scale)

    def _tm_basis(
        self, time_len: int, device: torch.device, real_dtype: torch.dtype
    ) -> torch.Tensor:
        """Evaluate ``B_n(e^{jt_k})`` for all poles and unit-circle samples.

        Returns a complex tensor of shape ``(N, T)``.
        """
        poles = self.constrained_poles().to(device)
        n_components = poles.shape[0]

        t_grid = (
            2.0
            * math.pi
            * torch.arange(time_len, device=device, dtype=torch.float32)
            / time_len
        )
        z = torch.complex(torch.cos(t_grid), torch.sin(t_grid))  # (T,)

        # First term: sqrt(1 - |a_n|^2) / (1 - conj(a_n) * z)
        abs_sq_poles = poles.real.pow(2) + poles.imag.pow(2)
        norm_factor = torch.sqrt((1.0 - abs_sq_poles).clamp_min(1e-8))
        norm_factor_complex = torch.complex(norm_factor, torch.zeros_like(norm_factor))
        denom = 1.0 - poles.conj().unsqueeze(1) * z.unsqueeze(0)  # (N, T)
        first_term = norm_factor_complex.unsqueeze(1) / denom  # (N, T)

        # Blaschke factors B_k(z) = (z - a_k) / (1 - conj(a_k) * z); |B_k(e^{jt})| = 1
        blaschke_factors = (z.unsqueeze(0) - poles.unsqueeze(1)) / (
            1.0 - poles.conj().unsqueeze(1) * z.unsqueeze(0)
        )  # (N, T)

        # Prefix product: prefix[n] = prod_{k=0}^{n-1} blaschke_factors[k]
        if n_components == 1:
            blaschke_prefix = torch.ones_like(blaschke_factors)
        else:
            ones = torch.ones(
                1, time_len, dtype=blaschke_factors.dtype, device=device
            )
            cumprod = torch.cumprod(blaschke_factors[:-1], dim=0)  # (N-1, T)
            blaschke_prefix = torch.cat([ones, cumprod], dim=0)  # (N, T)

        return first_term * blaschke_prefix

    @staticmethod
    def _analytic_signal(x: torch.Tensor) -> torch.Tensor:
        """Real-to-analytic via one-sided spectrum (preserves Re == x)."""
        time_len = x.shape[1]
        spec = torch.fft.fft(x, dim=1)
        analytic_spec = torch.zeros_like(spec)
        analytic_spec[:, 0, :] = spec[:, 0, :]
        if time_len > 1:
            if time_len % 2 == 0:
                if time_len // 2 > 1:
                    analytic_spec[:, 1 : time_len // 2, :] = (
                        2.0 * spec[:, 1 : time_len // 2, :]
                    )
                analytic_spec[:, time_len // 2, :] = spec[:, time_len // 2, :]
            else:
                analytic_spec[:, 1 : (time_len + 1) // 2, :] = (
                    2.0 * spec[:, 1 : (time_len + 1) // 2, :]
                )
        return torch.fft.ifft(analytic_spec, dim=1)

    def _amplitude_sensitive_permutation_entropy(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3:
            raise ValueError(f"Expected signal to have shape [batch, time, channels], got {signal.shape}.")
        batch_size, time_len, _num_channels = signal.shape
        order = self.aspe_order
        if time_len < order:
            return signal.new_zeros(batch_size)

        windows = signal.transpose(1, 2).unfold(dimension=2, size=order, step=1)
        mean_abs = windows.abs().mean(dim=-1).clamp_min(1e-8)
        weights = windows.std(dim=-1, unbiased=False) / mean_abs
        patterns = torch.argsort(windows, dim=-1, stable=True)
        perms = self.permutation_patterns.view(1, 1, 1, -1, order)
        matches = (patterns.unsqueeze(-2) == perms).all(dim=-1)
        counts = matches.to(signal.dtype).sum(dim=2)
        num_windows = windows.size(2)
        pattern_frequency = counts / float(num_windows)
        weight_sums = (matches.to(signal.dtype) * weights.unsqueeze(-1)).sum(dim=2)
        omega = torch.where(counts > 0, weight_sums / counts.clamp_min(1.0), torch.zeros_like(weight_sums))
        entropy = -(
            omega
            * pattern_frequency
            * torch.log(pattern_frequency.clamp_min(1e-12))
        ).sum(dim=-1)
        entropy = entropy / math.log(math.factorial(order))
        return entropy.mean(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [batch, time, channels], got {x.shape}.")

        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        batch_size = x_fp32.shape[0]
        time_len = x_fp32.shape[1]

        analytic = self._analytic_signal(x_fp32)  # (B, T, C) complex
        mean_complex = analytic.mean(dim=1, keepdim=True)
        centered_analytic = analytic - mean_complex
        e_basis = self._tm_basis(time_len, x_fp32.device, torch.float32)  # (N, T) complex
        poles = self.constrained_poles().to(x_fp32.device)
        num_candidates = poles.shape[0]

        low_complex = torch.zeros_like(analytic)
        residual_complex = centered_analytic
        available = torch.ones(batch_size, num_candidates, dtype=torch.bool, device=x_fp32.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=x_fp32.device)
        selected_mask = torch.zeros_like(available)
        component_counts = torch.zeros(batch_size, dtype=torch.long, device=x_fp32.device)
        previous_aspe = self._amplitude_sensitive_permutation_entropy(centered_analytic.real.to(torch.float32))
        batch_indices = torch.arange(batch_size, device=x_fp32.device)

        for _component_idx in range(num_candidates):
            has_candidate = active & available.any(dim=1)
            if not bool(has_candidate.any()):
                break

            # A_{c,n} = (1/T) * sum_t residual_c(t) * conj(B_n(e^{jt}))
            coeffs = (
                torch.einsum("btc,nt->bnc", residual_complex, e_basis.conj()) / time_len
            )  # (B, N, C) complex
            projection_energy = coeffs.abs().pow(2).sum(dim=-1)
            projection_energy = projection_energy.masked_fill(~available, -torch.inf)
            chosen = projection_energy.argmax(dim=1)
            chosen_coeffs = coeffs[batch_indices, chosen]  # (B, C)
            chosen_basis = e_basis[chosen]  # (B, T)
            proposed_component = torch.einsum("bc,bt->btc", chosen_coeffs, chosen_basis)
            proposed_low = low_complex + proposed_component
            proposed_high = (centered_analytic - proposed_low).real.to(torch.float32)
            proposed_aspe = self._amplitude_sensitive_permutation_entropy(proposed_high)
            improvement = previous_aspe - proposed_aspe

            accept = has_candidate & (
                (component_counts == 0) | (improvement > self.aspe_tolerance)
            )
            active = active & ~(has_candidate & ~accept)
            if not bool(accept.any()):
                break

            low_complex = low_complex + proposed_component * accept.view(-1, 1, 1)
            residual_complex = centered_analytic - low_complex
            previous_aspe = torch.where(accept, proposed_aspe, previous_aspe)
            component_counts = component_counts + accept.to(component_counts.dtype)
            selected_mask = selected_mask | (
                F.one_hot(chosen, num_classes=num_candidates).to(torch.bool)
                & accept.unsqueeze(1)
            )

            chosen_poles = poles[chosen]
            too_close = torch.abs(poles.unsqueeze(0) - chosen_poles.unsqueeze(1)) <= self.dictionary_distance
            available = available & ~(too_close & accept.unsqueeze(1))
            active = active & available.any(dim=1)

        low = (mean_complex + low_complex).real.to(orig_dtype)
        high = x - low

        aspe = self._amplitude_sensitive_permutation_entropy(high)
        self.last_selected_mask = selected_mask.detach()
        self.last_component_counts = component_counts.detach()
        return high, low, aspe


class FrequencyRoutingAttention(nn.Module):
    """PhysDiff frequency-routing attention."""

    def __init__(self, d_model: int, attn_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if attn_dim <= 0 or num_heads <= 0:
            raise ValueError("attn_dim and num_heads must be positive.")
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)

        hidden = self.attn_dim * self.num_heads
        self.q_proj = nn.Linear(d_model, hidden)
        self.k_proj = nn.Linear(d_model, hidden)
        self.v_proj = nn.Linear(d_model, hidden)
        self.ph_proj = nn.Linear(d_model, hidden)
        self.pl_proj = nn.Linear(d_model, hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, prior_high: torch.Tensor, prior_low: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _dim = x.shape

        def _reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.num_heads, self.attn_dim).transpose(1, 2)

        q = _reshape(self.q_proj(x))
        k = _reshape(self.k_proj(x))
        v = _reshape(self.v_proj(x))
        p_h = _reshape(self.ph_proj(prior_high))
        p_l = _reshape(self.pl_proj(prior_low))

        qk = torch.matmul(q, k.transpose(-1, -2))
        qph = torch.matmul(q, p_h.transpose(-1, -2))
        qpl = torch.matmul(q, p_l.transpose(-1, -2))
        g_h = torch.sigmoid(qph)
        g_l = torch.sigmoid(qpl)
        scores = (qk + g_h * qph + g_l * qpl) / math.sqrt(self.attn_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.attn_dim)
        return self.out_proj(context)


class PhysDiffBlock(nn.Module):
    """Conditioned routing-attention block."""

    def __init__(self, d_model: int, ff_dim: int, attn_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = FrequencyRoutingAttention(d_model, attn_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        prior_high: torch.Tensor,
        prior_low: torch.Tensor,
        cond_bias: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = self.norm1(x + cond_bias.unsqueeze(1))
        x = x + self.attn(conditioned, prior_high, prior_low)
        x = x + self.ff(self.norm2(x))
        return x


class PotentialFieldNetwork(nn.Module):
    """Potential field used for energy regularization and reverse guidance."""

    def __init__(self, win_size: int, num_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.win_size = int(win_size)
        self.num_channels = int(num_channels)
        freq_bins = self.win_size // 2 + 1
        self.freq_proj = nn.Sequential(
            nn.Linear(3 * freq_bins * self.num_channels, hidden_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(win_size * num_channels + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, gaussian_bandwidth: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft(x.to(torch.float32), dim=1, norm="ortho")
        real = x_fft.real
        imag = x_fft.imag
        power = real.pow(2) + imag.pow(2)

        freq_bins = real.shape[1]
        omega = torch.arange(freq_bins, device=x.device, dtype=torch.float32) / float(x.shape[1])
        sigma = gaussian_bandwidth.to(device=x.device, dtype=torch.float32)
        a_omega = torch.exp(-(omega.pow(2)) / (sigma.pow(2) + 1e-8))
        weighted_power = power * a_omega.view(1, freq_bins, 1)

        spectral_features = torch.cat([real, imag, weighted_power], dim=1).flatten(start_dim=1)
        net_dtype = self.net[0].weight.dtype
        freq_context = self.freq_proj(spectral_features.to(dtype=net_dtype))
        time_features = x.flatten(start_dim=1).to(dtype=net_dtype)
        return self.net(torch.cat([time_features, freq_context], dim=-1))


class PhysDiffDenoiser(nn.Module):
    """Conditional denoiser over physical priors {P_h, P_l, H_ASPE, E(x_t)}."""

    def __init__(
        self,
        win_size: int,
        num_channels: int,
        d_model: int,
        ff_dim: int,
        attn_dim: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        self.high_proj = nn.Linear(num_channels, d_model)
        self.low_proj = nn.Linear(num_channels, d_model)
        self.position_embed = nn.Parameter(torch.zeros(1, int(win_size), d_model))
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.cond_proj = nn.Linear(2 * d_model + 2, d_model)
        self.blocks = nn.ModuleList(
            [
                PhysDiffBlock(
                    d_model=d_model,
                    ff_dim=ff_dim,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _idx in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_channels),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        aspe: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.size(1) > self.position_embed.size(1):
            raise ValueError(
                f"PhysDiffDenoiser received sequence length {x_t.size(1)}, "
                f"but position embedding length is {self.position_embed.size(1)}."
            )
        pos = self.position_embed[:, : x_t.size(1), :].to(device=x_t.device, dtype=self.input_proj.weight.dtype)
        tokens = self.input_proj(x_t) + pos
        prior_high = self.high_proj(high) + pos
        prior_low = self.low_proj(low) + pos
        pooled = torch.cat(
            [
                prior_high.mean(dim=1),
                prior_low.mean(dim=1),
                aspe.unsqueeze(-1),
                energy.unsqueeze(-1),
            ],
            dim=-1,
        )
        cond_bias = self.time_embed(timesteps) + self.cond_proj(pooled)

        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden, prior_high, prior_low, cond_bias)
        hidden = self.norm(hidden)
        return self.output_head(hidden)


class PhysDiff(BaseModel):
    """Trainable PhysDiff network compatible with ``BenchmarkModel``."""

    def __init__(
        self,
        win_size: int,
        num_channels: int,
        time_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        model_dim: int = 512,
        ff_dim: int = 2048,
        attn_dim: int = 64,
        num_heads: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.2,
        mafd_components: int = 16,
        dictionary_distance: float = 0.05,
        aspe_tolerance: float = 1e-3,
        max_magnitude: float = 0.95,
        mafd_loss_weight: float = 1.0,
        aspe_weight: float = 0.1,
        energy_loss_weight: float = 1e-3,
        disturbance_factor: float = 10.0,
        langevin_step: float = 1e-3,
        langevin_noise: bool = True,
        gaussian_bandwidth_init: float = 0.25,
        sampling_steps: Optional[int] = 64,
        legacy_disturbance: bool = True,
    ) -> None:
        super().__init__()
        if win_size < 2:
            raise ValueError("win_size must be at least 2.")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive.")
        if model_dim <= 0 or ff_dim <= 0 or num_blocks <= 0:
            raise ValueError("model_dim, ff_dim, and num_blocks must be positive.")
        if gaussian_bandwidth_init <= 0:
            raise ValueError("gaussian_bandwidth_init must be positive.")

        self.seq_len = int(win_size)
        self.win_size = int(win_size)
        self.num_channels = int(num_channels)
        self.mafd_loss_weight = float(mafd_loss_weight)
        self.aspe_weight = float(aspe_weight)
        self.energy_loss_weight = float(energy_loss_weight)
        self.disturbance_factor = float(disturbance_factor)
        self.langevin_step = float(langevin_step)
        self.langevin_noise = bool(langevin_noise)
        self.sampling_steps = int(sampling_steps) if sampling_steps is not None else None
        self.legacy_disturbance = bool(legacy_disturbance)

        self.schedule = DiffusionSchedule(
            time_steps=int(time_steps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
        )
        self.decomposer = MultiChannelAdaptiveFourierDecomposer(
            max_components=int(mafd_components),
            dictionary_distance=float(dictionary_distance),
            aspe_tolerance=float(aspe_tolerance),
            max_magnitude=float(max_magnitude),
        )
        self.denoiser = PhysDiffDenoiser(
            win_size=self.win_size,
            num_channels=self.num_channels,
            d_model=int(model_dim),
            ff_dim=int(ff_dim),
            attn_dim=int(attn_dim),
            num_heads=int(num_heads),
            num_blocks=int(num_blocks),
            dropout=float(dropout),
        )
        self.potential = PotentialFieldNetwork(
            win_size=self.win_size,
            num_channels=self.num_channels,
            hidden_dim=int(model_dim),
        )
        # Learnable Gaussian bandwidth σ for the paper's frequency-response
        # function A_ω = exp(−ω²/σ²). Stored in log-space so optimisation is
        # unconstrained and σ stays positive after `exp`.
        self.log_gaussian_bandwidth = nn.Parameter(
            torch.tensor(math.log(float(gaussian_bandwidth_init)), dtype=torch.float32)
        )

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    def _training_priors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.legacy_disturbance:
            disturbed = x + torch.rand_like(x) * self.disturbance_factor
        else:
            disturbed = x + torch.randn_like(x) * self.disturbance_factor
        return self.decomposer(disturbed)

    def _energy(
        self,
        x: torch.Tensor,
        create_graph: bool,
        compute_grad_energy: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Lightning predict runs under ``torch.inference_mode()``, but PhysDiff
        # needs local gradients for potential-field guidance during reverse diffusion.
        with torch.inference_mode(False):
            with torch.enable_grad():
                x_req = x.clone().requires_grad_(True)
                potential = self.potential(x_req, self.gaussian_bandwidth())
                first_order_graph = create_graph or compute_grad_energy
                grad_phi = torch.autograd.grad(
                    potential.sum(),
                    x_req,
                    create_graph=first_order_graph,
                    retain_graph=first_order_graph,
                )[0]
                energy = grad_phi.pow(2).sum(dim=(1, 2))
                grad_energy: Optional[torch.Tensor] = None
                if compute_grad_energy:
                    grad_energy = torch.autograd.grad(
                        energy.sum(),
                        x_req,
                        create_graph=create_graph,
                        retain_graph=create_graph,
                    )[0]
        return energy, grad_energy

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> PhysDiffForwardOutput:
        x = inputs[0].contiguous()
        if x.ndim != 3:
            raise ValueError(f"Expected input window tensor [batch, time, channels], got {x.shape}.")
        if x.device.type == "meta":
            batch_size = x.shape[0]
            grad_anchor = self.denoiser.output_head[-1].weight.sum() * 0.0
            predicted_noise = torch.zeros_like(x) + grad_anchor
            return (
                predicted_noise,
                torch.zeros_like(x),
                torch.zeros_like(x),
                torch.zeros(batch_size, device=x.device, dtype=x.dtype),
                torch.zeros(batch_size, device=x.device, dtype=x.dtype),
            )

        high, low, aspe = self._training_priors(x)
        timesteps = torch.randint(
            low=0,
            high=self.schedule.time_steps,
            size=(x.size(0),),
            device=x.device,
        )
        noise = torch.randn_like(x)
        x_t = self.schedule.q_sample(x, timesteps, noise)
        energy, _grad_energy = self._energy(
            x_t,
            create_graph=self.energy_loss_weight > 0.0,
            compute_grad_energy=False,
        )
        mixed_x_t = self._frequency_mix(x_t, high, low, timesteps)
        predicted_noise = self.denoiser(
            x_t=mixed_x_t,
            timesteps=timesteps,
            high=high,
            low=low,
            aspe=aspe,
            energy=energy.detach(),
        )
        return predicted_noise, noise, high, aspe, energy

    def training_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        predictions = self((x,))
        loss = PhysDiffLoss(
            mafd_loss_weight=self.mafd_loss_weight,
            aspe_weight=self.aspe_weight,
            energy_loss_weight=self.energy_loss_weight,
        )
        loss_value = loss(predictions, (x,))
        predicted_noise, noise, high, aspe, energy = predictions
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        mafd_loss = high.pow(2).mean() + self.aspe_weight * aspe.mean()
        stats = {
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "mafd_loss": float(mafd_loss.detach().cpu()),
            "energy_loss": float(energy.mean().detach().cpu()),
        }
        return loss_value, stats

    def _sampling_indices(self) -> List[int]:
        if self.sampling_steps is None or self.sampling_steps >= self.schedule.time_steps:
            return list(range(self.schedule.time_steps - 1, -1, -1))
        grid = torch.linspace(self.schedule.time_steps - 1, 0, steps=self.sampling_steps)
        return sorted({int(round(value.item())) for value in grid}, reverse=True)

    def gaussian_bandwidth(self) -> torch.Tensor:
        """Current Gaussian bandwidth σ used inside ``A_ω = exp(−ω²/σ²)``."""
        return torch.exp(self.log_gaussian_bandwidth)

    def _frequency_guidance_mask(
        self,
        timesteps: torch.Tensor,
        time_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Per-frequency Langevin guidance modulator γ(t, ω) from the paper.

        ``γ(t, ω) = sigmoid(10 · (1 − √α̂_t) · A_ω)``,
        ``A_ω = exp(−|ω|²/σ²)``, with ``ω = k / T`` for rfft bins
        ``k ∈ [0, T/2]``. Returns a tensor of shape ``(B, F)`` where
        ``F = T // 2 + 1``.
        """
        alpha_bar = self.schedule.alphas_cumprod.to(device).gather(0, timesteps)  # (B,)
        sqrt_alpha_bar = torch.sqrt(alpha_bar).to(dtype)

        freq_bins = time_len // 2 + 1
        omega = torch.arange(freq_bins, device=device, dtype=dtype) / float(time_len)
        sigma = self.gaussian_bandwidth().to(device=device, dtype=dtype)
        a_omega = torch.exp(-(omega ** 2) / (sigma ** 2 + 1e-8))  # (F,)

        time_factor = 10.0 * (1.0 - sqrt_alpha_bar).unsqueeze(-1)  # (B, 1)
        return torch.sigmoid(time_factor * a_omega.unsqueeze(0))  # (B, F)

    def _frequency_mix(
        self,
        x_t: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x_freq = torch.fft.rfft(x_t.to(torch.float32), dim=1)
        prior_freq = torch.fft.rfft((high + low).to(torch.float32), dim=1)
        gamma = self._frequency_guidance_mask(
            timesteps, x_t.shape[1], device=x_t.device, dtype=torch.float32
        )
        mixed_freq = (1.0 - gamma.unsqueeze(-1)) * x_freq + gamma.unsqueeze(-1) * prior_freq
        return torch.fft.irfft(mixed_freq, n=x_t.shape[1], dim=1).to(x_t.dtype)

    def _p_sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        aspe: torch.Tensor,
    ) -> torch.Tensor:
        energy, grad_energy = self._energy(x_t, create_graph=False, compute_grad_energy=True)
        if grad_energy is None:
            raise RuntimeError("PhysDiff reverse sampling requires grad_energy guidance.")
        mixed_x_t = self._frequency_mix(x_t, high, low, timesteps)
        predicted_noise = self.denoiser(
            x_t=mixed_x_t,
            timesteps=timesteps,
            high=high,
            low=low,
            aspe=aspe,
            energy=energy.detach(),
        )
        mean = self.schedule.posterior_mean_from_noise(x_t, timesteps, predicted_noise)

        # Apply the paper's Gaussian frequency-response mask A_ω in the
        # frequency domain so the Langevin guidance is band-selective: low
        # frequencies receive stronger guidance, high frequencies less.
        grad_freq = torch.fft.rfft(grad_energy.to(torch.float32), dim=1)  # (B, F, C) complex
        gamma = self._frequency_guidance_mask(
            timesteps, x_t.shape[1], device=x_t.device, dtype=torch.float32
        )  # (B, F)
        grad_modulated = torch.fft.irfft(
            grad_freq * gamma.unsqueeze(-1), n=x_t.shape[1], dim=1
        ).to(x_t.dtype)
        mean = mean - self.langevin_step * grad_modulated

        variance = self.schedule.extract(self.schedule.posterior_variance, timesteps, x_t.shape).to(x_t.dtype)
        nonzero_mask = (timesteps > 0).float().view(-1, 1, 1).to(x_t.dtype)
        noise = torch.randn_like(x_t)
        sample = mean + nonzero_mask * torch.sqrt(variance.clamp_min(1e-8)) * noise
        if self.langevin_noise:
            langevin_noise = torch.randn_like(x_t)
            langevin_scale = math.sqrt(max(2.0 * self.langevin_step, 0.0))
            sample = sample + nonzero_mask * langevin_scale * langevin_noise
        return sample.detach()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        high, low, aspe = self.decomposer(x)
        indices = self._sampling_indices()
        start_t = torch.full((x.size(0),), indices[0], device=x.device, dtype=torch.long)
        x_t = self.schedule.q_sample(x, start_t, torch.randn_like(x))
        for step in indices:
            timesteps = torch.full((x.size(0),), step, device=x.device, dtype=torch.long)
            x_t = self._p_sample(x_t, timesteps, high, low, aspe)
        self.train(was_training)
        return x_t

    def window_score_components(
        self,
        x: torch.Tensor,
        prior_distribution: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_samples = max(1, int(num_samples))
        accumulated = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for _idx in range(num_samples):
            reconstruction = self.reconstruct(x)
            accumulated = accumulated + (x - reconstruction).pow(2).mean(dim=(1, 2))
        reconstruction_error = accumulated / num_samples
        current_distribution = transient_tfd_distribution(x)
        prior = prior_distribution.to(device=x.device, dtype=current_distribution.dtype).unsqueeze(0)
        prior = prior.expand_as(current_distribution)
        kl = (
            current_distribution
            * (
                torch.log(current_distribution.clamp_min(1e-8))
                - torch.log(prior.clamp_min(1e-8))
            )
        ).sum(dim=1)
        return reconstruction_error.detach(), kl.detach()

    def window_scores(
        self,
        x: torch.Tensor,
        prior_distribution: torch.Tensor,
        score_alpha: float,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction_error, kl = self.window_score_components(
            x=x,
            prior_distribution=prior_distribution,
            num_samples=num_samples,
        )
        score = float(score_alpha) * reconstruction_error + (1.0 - float(score_alpha)) * kl
        return score.detach(), reconstruction_error.detach(), kl.detach()


class PhysDiffLoss(Loss):
    """Loss wrapper compatible with ``timesead.optim.loss.Loss``."""

    def __init__(
        self,
        mafd_loss_weight: float = 1.0,
        aspe_weight: float = 0.1,
        energy_loss_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.mafd_loss_weight = float(mafd_loss_weight)
        self.aspe_weight = float(aspe_weight)
        self.energy_loss_weight = float(energy_loss_weight)

    @staticmethod
    def _normalize_predictions(predictions: Tuple[Any, ...]) -> PhysDiffForwardOutput:
        normalized = predictions
        if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
            normalized = tuple(normalized[0])
        if len(normalized) != 5:
            raise ValueError(f"PhysDiffLoss expects 5 prediction tensors, received {len(normalized)}.")
        return normalized  # type: ignore[return-value]

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del targets, args, kwargs
        predicted_noise, noise, high, aspe, energy = self._normalize_predictions(predictions)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        mafd_loss = high.pow(2).mean() + self.aspe_weight * aspe.mean()
        return (
            diffusion_loss
            + self.mafd_loss_weight * mafd_loss
            + self.energy_loss_weight * energy.mean()
        )


class PhysDiffAnomalyDetector(AnomalyDetector):
    """Anomaly detector using PhysDiff reconstruction and transient TFD KL scores."""

    def __init__(
        self,
        model: Optional[PhysDiff] = None,
        score_alpha: float = 0.5,
        smoothing_kernel_size: int = 5,
        spot_q: float = 0.01,
        reconstruction_num_samples: int = 1,
        component_standardize: bool = False,
        score_normalize: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.score_alpha = float(score_alpha)
        self.smoothing_kernel_size = int(smoothing_kernel_size)
        self.spot_q = float(spot_q)
        self.reconstruction_num_samples = int(reconstruction_num_samples)
        self.component_standardize = bool(component_standardize)
        self.score_normalize = bool(score_normalize)
        self.prior_distribution: Optional[torch.Tensor] = None
        self.component_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self.score_stats: Optional[Tuple[float, float]] = None
        self.threshold: Optional[float] = None

    @staticmethod
    def _extract_inputs(batch: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch = batch[0]
        if isinstance(batch, torch.Tensor):
            return (batch,)
        if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            return tuple(batch)
        raise ValueError("PhysDiffAnomalyDetector expected a batch containing input tensors.")

    @staticmethod
    def _robust_stats(values: np.ndarray) -> Tuple[float, float]:
        if values.size == 0:
            return 0.0, 1.0
        location = float(np.median(values))
        mad = float(np.median(np.abs(values - location)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return location, scale

    def _standardize_components(self, mse: torch.Tensor, kl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.component_standardize or self.component_stats is None:
            return mse, kl
        mse_loc, mse_scale = self.component_stats["mse"]
        kl_loc, kl_scale = self.component_stats["kl"]
        return (mse - mse_loc) / mse_scale, (kl - kl_loc) / kl_scale

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.score_normalize or self.score_stats is None:
            return scores
        location, scale = self.score_stats
        return (scores - location) / scale

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs: Any) -> None:
        del kwargs
        if self.model is None:
            raise RuntimeError("PhysDiffAnomalyDetector requires a trained PhysDiff model.")

        model_device = next(self.model.parameters()).device
        distributions: List[torch.Tensor] = []
        windows: List[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                input_batch = self._extract_inputs(batch)[0].to(model_device, dtype=torch.float32)
                distributions.append(transient_tfd_distribution(input_batch).detach().cpu())
                if self.component_standardize or self.score_normalize or self.spot_q > 0:
                    windows.append(input_batch.detach().cpu())

        if distributions:
            prior = torch.cat(distributions, dim=0).mean(dim=0)
        else:
            dummy = torch.ones(
                1,
                self.model.win_size,
                self.model.num_channels,
                device=model_device,
                dtype=torch.float32,
            )
            prior = transient_tfd_distribution(dummy).mean(dim=0).detach().cpu()
        self.prior_distribution = prior.to(model_device)

        if windows:
            self._fit_score_statistics(windows, model_device)
        self.model.train(was_training)

    def _fit_score_statistics(self, windows: List[torch.Tensor], model_device: torch.device) -> None:
        if self.model is None or self.prior_distribution is None:
            return

        mse_parts: List[np.ndarray] = []
        kl_parts: List[np.ndarray] = []
        with torch.no_grad():
            for batch_window in windows:
                input_batch = batch_window.to(model_device, dtype=torch.float32)
                mse, kl = self.model.window_score_components(
                    input_batch,
                    prior_distribution=self.prior_distribution,
                    num_samples=self.reconstruction_num_samples,
                )
                mse_parts.append(mse.cpu().numpy())
                kl_parts.append(kl.cpu().numpy())

        mse_values = np.concatenate(mse_parts, axis=0) if mse_parts else np.empty(0, dtype=np.float64)
        kl_values = np.concatenate(kl_parts, axis=0) if kl_parts else np.empty(0, dtype=np.float64)
        if self.component_standardize:
            self.component_stats = {
                "mse": self._robust_stats(mse_values.astype(np.float64)),
                "kl": self._robust_stats(kl_values.astype(np.float64)),
            }

        if mse_values.size:
            mse_t = torch.as_tensor(mse_values, device=model_device, dtype=torch.float32)
            kl_t = torch.as_tensor(kl_values, device=model_device, dtype=torch.float32)
            mse_t, kl_t = self._standardize_components(mse_t, kl_t)
            scores = self.score_alpha * mse_t + (1.0 - self.score_alpha) * kl_t
            score_values = scores.detach().cpu().numpy().astype(np.float64)
            if self.score_normalize:
                self.score_stats = self._robust_stats(score_values)
            if self.spot_q > 0:
                self.threshold = float(np.quantile(score_values, 1.0 - self.spot_q))

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("PhysDiffAnomalyDetector has no model.")
        if self.prior_distribution is None:
            model_device = next(self.model.parameters()).device
            self.prior_distribution = transient_tfd_distribution(inputs[0].to(model_device)).mean(dim=0)

        input_batch = inputs[0]
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        input_batch = input_batch.to(model_device, dtype=model_dtype)
        mse, kl = self.model.window_score_components(
            x=input_batch,
            prior_distribution=self.prior_distribution,
            num_samples=self.reconstruction_num_samples,
        )
        mse, kl = self._standardize_components(mse, kl)
        score = self.score_alpha * mse + (1.0 - self.score_alpha) * kl
        score = self._normalize_scores(score)

        if self.smoothing_kernel_size > 1 and score.numel() > 1:
            smoothed = smooth_scores(score.detach().cpu().numpy(), self.smoothing_kernel_size)
            score = torch.as_tensor(smoothed, device=input_batch.device, dtype=input_batch.dtype)
        return score.detach().cpu()

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.compute_online_anomaly_score(inputs)

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        return target[:, -1]
