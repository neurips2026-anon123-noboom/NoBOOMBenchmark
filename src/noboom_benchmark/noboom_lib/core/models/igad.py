"""IGAD: Idempotent Generation for Anomaly Detection.

Faithful re-implementation of the module from Sun et al., NeurIPS 2025
(https://openreview.net/pdf?id=w1x9WMMHvi), cross-checked against the official
reference at https://github.com/ProEcho1/Idempotent-Generation-for-Anomaly-Detection-IGAD
("IdempotentLoss" in the integration guide).

IGAD is a training-time module that augments a reconstruction-based MTS anomaly
detector with two extra objectives:

* L_idem  = L1(f'(f(z)), f(z))         — idempotency on Fourier-augmented inputs
* L_tight = -L1(f(f'(z)), f'(z))       — adversarial tightening of the manifold

where ``f`` is the live network being trained, ``f'`` is a frozen copy
re-synchronised with ``f`` on every forward, and ``z`` is a Fourier-resampled
pseudo-normal sample drawn per-batch from the empirical statistics of the input
window's spectrum (Eq. 3-5 of the paper). The tightness term is smoothly
clamped via tanh against the per-window reconstruction loss (Eq. 10).

We pair IGAD with the same flat MLP autoencoder backbone reported in Table 4
of the paper, where AE+IGAD is shown to outperform VAE+IGAD on most datasets
while keeping the parameter footprint tiny.
"""

from __future__ import annotations

import copy
from typing import Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesead.models import BaseModel
from timesead.models.common.anomaly_detector import AnomalyDetector
from timesead.optim.loss import Loss


class MLPAutoEncoder(nn.Module):
    """Flat MLP autoencoder over the (window x channels) input."""

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.window_size = int(window_size)
        self.input_dim = int(input_dim)
        flat_dim = self.window_size * self.input_dim

        encoder_layers: list[nn.Module] = []
        prev_dim = flat_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden))
            encoder_layers.append(nn.GELU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden))
            decoder_layers.append(nn.GELU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        decoder_layers.append(nn.Linear(prev_dim, flat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        latent = self.encoder(flat)
        recon = self.decoder(latent)
        return recon.view(batch_size, self.window_size, self.input_dim)


def _fourier_resample(x: torch.Tensor) -> torch.Tensor:
    """Generate Fourier-domain resampled pseudo-normal samples.

    Mirrors ``IdempotentLoss.get_freq_means_and_stds`` and ``get_noise`` from
    the official IGAD reference, but operates directly on the (B, T, D)
    convention used elsewhere in this benchmark.
    """
    # Match the reference's internal layout: FFT along the time axis.
    perm = x.permute(0, 2, 1).contiguous()  # (B, D, T)
    spectrum = torch.fft.fft(perm, dim=-1)
    real = spectrum.real
    imag = spectrum.imag

    real_mean = real.mean(dim=0, keepdim=True).expand_as(real)
    real_std = real.std(dim=0, keepdim=True).expand_as(real).clamp_min(0.0)
    imag_mean = imag.mean(dim=0, keepdim=True).expand_as(imag)
    imag_std = imag.std(dim=0, keepdim=True).expand_as(imag).clamp_min(0.0)

    freq_real = torch.normal(real_mean, real_std)
    freq_imag = torch.normal(imag_mean, imag_std)
    resampled = torch.fft.ifft(freq_real + 1j * freq_imag, dim=-1).real
    return resampled.permute(0, 2, 1).contiguous()  # (B, T, D)


class IGAD(BaseModel):
    """Reconstruction backbone + IGAD frozen-copy bookkeeping."""

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.input_dim = int(input_dim)

        self.network = MLPAutoEncoder(
            window_size=self.window_size,
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        # Frozen copy used for f'(.) in L_idem and L_tight. We resync it from
        # the live network on each forward instead of training it directly,
        # exactly as the official IdempotentLoss.forward does.
        self.frozen_network = copy.deepcopy(self.network)
        for parameter in self.frozen_network.parameters():
            parameter.requires_grad_(False)

    def grouped_parameters(self) -> Tuple[Iterator[nn.Parameter], ...]:
        # Only the live network is optimised; the frozen copy is resynced via
        # `load_state_dict` in `forward`. Including its parameters in the
        # optimizer breaks `manual_backward(..., inputs=opt_params)` because
        # it tries to retain_grad on tensors with requires_grad=False.
        return (self.network.parameters(),)

    def _zero_like(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(reference)

    def forward(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, = inputs
        recon = self.network(x)

        if x.device.type == "meta":
            zero = self._zero_like(recon)
            return recon, zero, zero, zero, zero

        # Sync frozen copy so f'(.) tracks the current state of f(.).
        with torch.no_grad():
            self.frozen_network.load_state_dict(self.network.state_dict())

        z = _fourier_resample(x)
        fz = self.network(z)
        f_z = fz.detach()
        ff_z = self.network(f_z)
        f_fz = self.frozen_network(fz)
        return recon, fz, ff_z, f_fz, f_z


class IGADLoss(Loss):
    """Combined reconstruction + idempotent + smoothed-tightness objective.

    Equation references follow the paper: L_recon (Eq. 1), L_idem (Eq. 7),
    L*_tight (Eq. 10), and the final aggregation (Eq. 11).
    """

    def __init__(
        self,
        idem_weight: float = 0.1,
        tight_weight: float = 0.1,
        alpha: float = 1.1,
        recon_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.idem_weight = float(idem_weight)
        self.tight_weight = float(tight_weight)
        self.alpha = float(alpha)
        self.recon_weight = float(recon_weight)

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Lightning's setup-time `measure_flops` wraps the network output
        # `(recon, fz, ff_z, f_fz, f_z)` again as `((tuple_of_5,),)` before
        # passing it to losses, so unwrap the extra level if present.
        normalized = predictions
        if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
            normalized = tuple(normalized[0])

        if len(normalized) == 1:
            (recon,) = normalized
            (target,) = targets
            return self.recon_weight * F.mse_loss(recon, target)

        recon, fz, ff_z, f_fz, f_z = normalized
        (target,) = targets
        batch_size = recon.shape[0]

        per_sample_recon = (
            F.mse_loss(recon, target, reduction="none")
            .reshape(batch_size, -1)
            .mean(dim=-1)
        )

        loss_idem = F.l1_loss(f_fz, fz, reduction="mean")
        loss_tight = -(
            F.l1_loss(ff_z, f_z, reduction="none").reshape(batch_size, -1).mean(dim=-1)
        )

        clamp = torch.clamp(self.alpha * per_sample_recon, min=1e-6)
        smoothed_tight = torch.tanh(loss_tight / clamp) * clamp

        recon_loss = per_sample_recon.mean()
        smoothed_tight_loss = smoothed_tight.mean()

        return (
            self.recon_weight * recon_loss
            + self.idem_weight * loss_idem
            + self.tight_weight * smoothed_tight_loss
        )


class IGADAnomalyDetector(AnomalyDetector):
    """MSE reconstruction-error detector for the IGAD-augmented backbone."""

    def __init__(self, model: IGAD, batch_first: bool = True) -> None:
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        return None

    def compute_online_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(inputs)
        recon = outputs[0] if isinstance(outputs, tuple) else outputs
        x, = inputs
        squared_error = torch.mean((x - recon) ** 2, dim=-1)
        return squared_error[:, -1] if self.batch_first else squared_error[-1]

    def compute_offline_anomaly_score(
        self, inputs: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target, = targets
        return target[:, -1] if self.batch_first else target[-1]
