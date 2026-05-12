"""Native RTdetector network, loss, and anomaly detector."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerEncoder

from timesead.models import BaseModel
from timesead.models.common.anomaly_detector import AnomalyDetector
from timesead.optim.loss import Loss


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding used by the RTdetector reference model."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        return self.dropout(x + self.pe[pos : pos + x.size(0)])


class RTdetectorEncoderLayer(nn.Module):
    """Small transformer encoder block used by RTdetector."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del src_mask, src_key_padding_mask, is_causal, kwargs
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(src2)


class RTdetectorDecoderLayer(nn.Module):
    """Small transformer decoder block used by RTdetector."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del (
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            tgt_is_causal,
            memory_is_causal,
            kwargs,
        )
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(tgt2)


class RTdetectorProjector(nn.Module):
    """MLP used by RTdetector to estimate de-stationary factors."""

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        hidden_dims: list[int],
        hidden_layers: int,
        output_dim: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

        layers: list[nn.Module] = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=False))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        return self.backbone(x.view(batch_size, -1))


class RTdetector(BaseModel):
    """Two-pass RTdetector reconstruction network.

    The network receives benchmark windows as ``(B, T, D)`` and reconstructs
    the final point in each window. It returns the two RTdetector passes as
    ``(B, D)`` tensors so the Lightning loss can preserve the original
    epoch-weighted training schedule.
    """

    def __init__(
        self,
        feats: int,
        window_size: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feats <= 0:
            raise ValueError("feats must be positive.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        self.name = "RTdetector"
        self.n_feats = int(feats)
        self.n_window = int(window_size)
        d_model = 2 * self.n_feats
        n_heads = self.n_feats

        self.pos_encoder = PositionalEncoding(d_model, dropout, self.n_window)
        encoder_layer = RTdetectorEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=16,
            dropout=dropout,
        )
        decoder_layer = RTdetectorDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=16,
            dropout=dropout,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, 1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layer, 1)
        self.output_layer = nn.Sequential(nn.Linear(d_model, self.n_feats), nn.Sigmoid())
        self.tau_learner = RTdetectorProjector(
            enc_in=d_model,
            seq_len=self.n_window,
            hidden_dims=[16],
            hidden_layers=1,
            output_dim=1,
        )
        self.delta_learner = RTdetectorProjector(
            enc_in=d_model,
            seq_len=self.n_window,
            hidden_dims=[16],
            hidden_layers=1,
            output_dim=self.n_window,
        )

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    def _encode(
        self,
        src: torch.Tensor,
        cond: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        joined = torch.cat((src, cond), dim=2)
        joined = joined.permute(1, 0, 2)

        mean_enc = joined.mean(1, keepdim=True).detach()
        centered = joined - mean_enc
        std_enc = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / std_enc

        tau = self.tau_learner(normalized, std_enc).exp().unsqueeze(1)
        delta = self.delta_learner(normalized, mean_enc).unsqueeze(-1)

        encoded = normalized.permute(1, 0, 2)
        encoded = self.pos_encoder(encoded)
        memory = self.transformer_encoder(encoded).permute(1, 0, 2)
        memory = memory * tau + delta
        memory = memory.permute(1, 0, 2)

        target = tgt.repeat(1, 1, 2).permute(1, 0, 2)
        target = target * tau + delta.mean(dim=1, keepdim=True)
        return target.permute(1, 0, 2), memory, std_enc.squeeze(1), mean_enc.squeeze(1)

    def _decode(self, src: torch.Tensor, cond: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        target, memory, std_enc, mean_enc = self._encode(src, cond, tgt)
        decoder_out = self.transformer_decoder1(target, memory).squeeze(0)
        decoder_out = decoder_out * std_enc + mean_enc
        return self.output_layer(decoder_out).unsqueeze(0)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, = inputs
        if x.ndim != 3:
            raise ValueError("RTdetector expects input windows with shape [batch, window, features].")
        if x.shape[1] != self.n_window:
            raise ValueError(f"Expected window length {self.n_window}, got {x.shape[1]}.")
        if x.shape[2] != self.n_feats:
            raise ValueError(f"Expected {self.n_feats} features, got {x.shape[2]}.")

        src = x.permute(1, 0, 2)
        tgt = src[-1:].clone()
        cond = torch.zeros_like(src)
        x1 = self._decode(src, cond, tgt)
        cond = (x1 - src) ** 2
        x2 = self._decode(src, cond, tgt)
        return x1.squeeze(0), x2.squeeze(0)


class RTdetectorLoss(Loss):
    """Epoch-weighted RTdetector reconstruction objective."""

    @staticmethod
    def _normalize_predictions(predictions: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        if len(predictions) == 1 and isinstance(predictions[0], (tuple, list)):
            return tuple(predictions[0])
        return predictions

    @staticmethod
    def _last_step(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            return target[:, -1, :]
        if target.ndim == 2:
            return target
        raise ValueError("RTdetector target must have shape [batch, window, features] or [batch, features].")

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...],
        *args,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        del args, kwargs
        normalized = self._normalize_predictions(predictions)
        target = self._last_step(targets[-1])

        if len(normalized) == 1:
            return F.mse_loss(normalized[0], target)
        if len(normalized) != 2:
            raise ValueError("RTdetectorLoss expects one or two prediction tensors.")

        x1, x2 = normalized
        if epoch is None:
            return F.mse_loss(x2, target)

        epoch_index = float(int(epoch) + 1)
        loss_1 = F.mse_loss(x1, target, reduction="none")
        loss_2 = F.mse_loss(x2, target, reduction="none")
        return ((1.0 / epoch_index) * loss_1 + (1.0 - 1.0 / epoch_index) * loss_2).mean()


class RTdetectorAnomalyDetector(AnomalyDetector):
    """Final-step squared reconstruction error detector for RTdetector."""

    def __init__(self, model: Optional[RTdetector] = None) -> None:
        super().__init__()
        self.model = model

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        del dataset, kwargs
        return None

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("RTdetectorAnomalyDetector requires a trained network.")
        x, = inputs
        with torch.no_grad():
            prediction = self.model(inputs)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[-1]
        target = x[:, -1, :]
        return (prediction - target).pow(2).mean(dim=-1)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


RTdetectorAD = RTdetectorAnomalyDetector

__all__ = ["RTdetector", "RTdetectorAD", "RTdetectorAnomalyDetector", "RTdetectorLoss"]
