from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesead.models.common.anomaly_detector import AnomalyDetector
from timesead.models.layers.autoformer_encdec import (
    CustomLayerNorm,
    Encoder,
    EncoderLayer,
)
from timesead.models.layers.autocorrelation import AutoCorrelationLayer
from timesead.models.layers.embed import DataEmbedding
from timesead.models.layers.fourier_correlation import FourierBlock
from timesead.optim.loss import Loss

from timesead.models import BaseModel


def _validate_positive_int_sequence(values: Sequence[int], name: str) -> Tuple[int, ...]:
    parsed = tuple(int(value) for value in values)
    if not parsed:
        raise ValueError(f"{name} must contain at least one positive integer.")
    if any(value <= 0 for value in parsed):
        raise ValueError(f"{name} must contain only positive integers.")
    return parsed


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(probs.dtype).eps
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    return entropy.mean()


def _repeat_upsample(values: torch.Tensor, repeat_factor: int, target_len: int) -> torch.Tensor:
    if repeat_factor == 1:
        return values[:, :target_len]
    upsampled = values.repeat_interleave(repeat_factor, dim=1)
    return upsampled[:, :target_len]


def _normalize_predictions(
    predictions: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized: Tuple[torch.Tensor, ...] = predictions
    if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
        normalized = tuple(normalized[0])
    if len(normalized) != 6:
        raise ValueError(
            f"HPAD expects 6 prediction tensors, but received {len(normalized)} entries."
        )
    return normalized  # type: ignore[return-value]


def _fft_top_periods(latent: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return torch.empty(0, dtype=torch.long, device=latent.device)

    _batch_size, seq_len, _channels = latent.shape
    if seq_len < 2:
        return torch.ones(1, dtype=torch.long, device=latent.device)

    spectrum = torch.fft.rfft(latent.transpose(1, 2), dim=-1)
    amplitudes = spectrum.abs().mean(dim=(0, 1))
    amplitudes[0] = -torch.inf

    max_modes = min(int(top_k), int(amplitudes.shape[0] - 1))
    if max_modes <= 0:
        return torch.ones(1, dtype=torch.long, device=latent.device)

    top_bins = torch.topk(amplitudes, k=max_modes, largest=True).indices.clamp_min(1)
    periods = torch.div(seq_len, top_bins, rounding_mode="floor").clamp_min(1)

    seen_periods: list[int] = []
    for value in periods.tolist():
        if value not in seen_periods:
            seen_periods.append(value)
    if not seen_periods:
        seen_periods = [1]
    return torch.as_tensor(seen_periods, dtype=torch.long, device=latent.device)


class HPAD(BaseModel):
    """FEDformer-based H-PAD implementation with prototype branches."""

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        moving_avg: int = 25,
        model_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8,
        fcn_dim: Optional[int] = None,
        activation: str = "gelu",
        encoder_layers: int = 3,
        modes: int = 32,
        fft_norm: str = "forward",
        w_init: str = "randn",
        mode_select: str = "random",
        mode_policy: str = "static",
        topk: int = 8,
        freq_norm_mode: Optional[str] = None,
        lrfop: bool = False,
        shared_attention: bool = False,
        scatter_freq: bool = False,
        patch_scales: Sequence[int] = (1, 2, 4),
        num_patch_prototypes: int = 16,
        top_k_periods: int = 3,
        num_period_prototypes: int = 8,
        prototype_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_patch_prototypes <= 0:
            raise ValueError("num_patch_prototypes must be positive.")
        if num_period_prototypes <= 0:
            raise ValueError("num_period_prototypes must be positive.")
        if top_k_periods <= 0:
            raise ValueError("top_k_periods must be positive.")

        self.seq_len = int(window_size)
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.prototype_dim = int(prototype_dim) if prototype_dim is not None else int(model_dim)
        self.patch_scales = _validate_positive_int_sequence(patch_scales, "patch_scales")
        self.top_k_periods = int(top_k_periods)

        self.enc_embedding = DataEmbedding(input_dim, model_dim, dropout)

        def encoder_self_att() -> FourierBlock:
            return FourierBlock(
                in_channels=model_dim,
                out_channels=model_dim,
                seq_len=self.seq_len,
                num_heads=num_heads,
                modes=modes,
                mode_select_method=mode_select,
                fft_norm=fft_norm,
                w_init=w_init,
                freq_norm_mode=freq_norm_mode,
                lrfop=lrfop,
                mode_policy=mode_policy,
                topk=topk,
                scatter_freq=scatter_freq,
            )

        if shared_attention:
            shared_attention_instance = encoder_self_att()

            def encoder_self_att() -> FourierBlock:
                return shared_attention_instance

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att(), model_dim, num_heads),
                    model_dim,
                    fcn_dim,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(encoder_layers)
            ],
            norm_layer=CustomLayerNorm(model_dim),
        )
        self.projection = nn.Linear(model_dim, input_dim, bias=True)

        self.patch_query_projs = nn.ModuleList(
            [nn.Linear(model_dim, self.prototype_dim) for _ in self.patch_scales]
        )
        self.patch_prototypes = nn.ParameterList(
            [
                nn.Parameter(torch.empty(num_patch_prototypes, self.prototype_dim))
                for _ in self.patch_scales
            ]
        )
        self.period_query_proj = nn.Linear(model_dim, self.prototype_dim)
        self.period_prototypes = nn.Parameter(
            torch.empty(self.top_k_periods, num_period_prototypes, self.prototype_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for prototypes in self.patch_prototypes:
            nn.init.normal_(prototypes, mean=0.0, std=0.02)
        nn.init.normal_(self.period_prototypes, mean=0.0, std=0.02)

    def _build_patch_branch(
        self,
        latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_scores: list[torch.Tensor] = []
        patch_feature_terms: list[torch.Tensor] = []
        patch_entropy_terms: list[torch.Tensor] = []

        latent_channels_first = latent.transpose(1, 2)
        for scale, query_proj, prototypes in zip(
            self.patch_scales,
            self.patch_query_projs,
            self.patch_prototypes,
        ):
            if scale == 1:
                pooled = latent
            else:
                pooled = F.avg_pool1d(
                    latent_channels_first,
                    kernel_size=scale,
                    stride=scale,
                    ceil_mode=True,
                ).transpose(1, 2)

            query = query_proj(pooled)
            logits = torch.einsum("btd,pd->btp", query, prototypes) / math.sqrt(self.prototype_dim)
            weights = torch.softmax(logits, dim=-1)
            reconstructed = torch.einsum("btp,pd->btd", weights, prototypes)
            distance = torch.mean((query - reconstructed) ** 2, dim=-1)

            patch_scores.append(_repeat_upsample(distance, repeat_factor=scale, target_len=self.seq_len))
            patch_feature_terms.append(distance.mean())
            patch_entropy_terms.append(_entropy_from_probs(weights))

        patch_score_seq = torch.stack(patch_scores, dim=0).mean(dim=0)
        patch_feature_term = torch.stack(patch_feature_terms).mean()
        patch_entropy_term = torch.stack(patch_entropy_terms).mean()
        return patch_score_seq, patch_entropy_term, patch_feature_term

    def _build_period_branch(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        periods = _fft_top_periods(latent, self.top_k_periods)
        query_proj = self.period_query_proj(latent)
        period_scores: list[torch.Tensor] = []
        period_feature_terms: list[torch.Tensor] = []

        for slot, period_tensor in enumerate(periods[: self.top_k_periods]):
            period = int(period_tensor.item())
            if period <= 0:
                continue

            padded_len = int(math.ceil(self.seq_len / period) * period)
            if padded_len == self.seq_len:
                padded = query_proj
            else:
                pad_amount = padded_len - self.seq_len
                padded = F.pad(query_proj.transpose(1, 2), (0, pad_amount)).transpose(1, 2)

            segment_count = padded_len // period
            segments = padded.view(padded.shape[0], segment_count, period, self.prototype_dim)
            segment_summary = segments.mean(dim=2)

            prototypes = self.period_prototypes[slot]
            logits = torch.einsum("bsd,pd->bsp", segment_summary, prototypes) / math.sqrt(self.prototype_dim)
            weights = torch.softmax(logits, dim=-1)
            reconstructed_summary = torch.einsum("bsp,pd->bsd", weights, prototypes)
            reconstructed_full = reconstructed_summary.unsqueeze(2).expand_as(segments)
            reconstructed_full = reconstructed_full.reshape(query_proj.shape[0], padded_len, self.prototype_dim)
            reconstructed_full = reconstructed_full[:, : self.seq_len]

            distance = torch.mean((query_proj - reconstructed_full) ** 2, dim=-1)
            period_scores.append(distance)
            period_feature_terms.append(distance.mean())

        if not period_scores:
            zero = latent.new_zeros(())
            return latent.new_zeros(latent.shape[0], self.seq_len), zero

        period_score_seq = torch.stack(period_scores, dim=0).mean(dim=0)
        period_feature_term = torch.stack(period_feature_terms).mean()
        return period_score_seq, period_feature_term

    def forward(
        self,
        inputs: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_enc = inputs[0].contiguous()
        latent = self.enc_embedding(x_enc)
        latent, _attns = self.encoder(latent, attn_mask=None)
        reconstruction = self.projection(latent)

        if latent.device.type == "meta":
            batch_size = latent.shape[0]
            zero_seq = torch.zeros(batch_size, self.seq_len, device=latent.device, dtype=latent.dtype)
            zero_scalar = torch.zeros((), device=latent.device, dtype=latent.dtype)
            return (
                reconstruction,
                zero_seq,
                zero_seq,
                zero_scalar,
                zero_scalar,
                zero_scalar,
            )

        patch_score_seq, patch_entropy_term, patch_feature_term = self._build_patch_branch(latent)
        period_score_seq, period_feature_term = self._build_period_branch(latent)
        return (
            reconstruction,
            patch_score_seq,
            period_score_seq,
            patch_entropy_term,
            patch_feature_term,
            period_feature_term,
        )


class HPADLoss(Loss):
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_patch: float = 1.0,
        lambda_ent: float = 0.01,
        lambda_period: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_rec = float(lambda_rec)
        self.lambda_patch = float(lambda_patch)
        self.lambda_ent = float(lambda_ent)
        self.lambda_period = float(lambda_period)

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        (
            reconstruction,
            _patch_score_seq,
            _period_score_seq,
            patch_entropy_term,
            patch_feature_term,
            period_feature_term,
        ) = _normalize_predictions(predictions)
        (target,) = targets

        reconstruction_loss = F.mse_loss(reconstruction, target)
        return (
            self.lambda_rec * reconstruction_loss
            + self.lambda_patch * patch_feature_term
            + self.lambda_ent * patch_entropy_term
            + self.lambda_period * period_feature_term
        )


class HPADAnomalyDetector(AnomalyDetector):
    def __init__(self, model: HPAD, beta: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.beta = float(beta)

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        return None

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            (
                reconstruction,
                patch_score_seq,
                period_score_seq,
                _patch_entropy_term,
                _patch_feature_term,
                _period_feature_term,
            ) = _normalize_predictions(self.model(inputs))
            input_batch = inputs[0]
            reconstruction_error = F.mse_loss(reconstruction, input_batch, reduction="none").mean(dim=-1)
            proto_weight = torch.softmax(patch_score_seq + self.beta * period_score_seq, dim=1)
            score_seq = proto_weight * reconstruction_error
        return score_seq[:, -1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        (target,) = targets
        return target[:, -1]
