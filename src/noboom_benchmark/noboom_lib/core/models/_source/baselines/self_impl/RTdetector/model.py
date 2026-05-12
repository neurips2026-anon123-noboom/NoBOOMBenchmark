"""Lean RTdetector model adapted from the official IJCAI 2025 implementation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerEncoder


class PositionalEncoding(nn.Module):
    """Standard sinusoidal encoding copied from the official implementation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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


class TransformerEncoderLayer(nn.Module):
    """Small transformer encoder block used by RTdetector."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
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
        src_mask=None,
        src_key_padding_mask=None,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout2(src2)


class TransformerDecoderLayer(nn.Module):
    """Small transformer decoder block used by RTdetector."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
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
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        return tgt + self.dropout3(tgt2)


class Projector(nn.Module):
    """MLP used by RTdetector to estimate de-stationary factors."""

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        hidden_dims: list[int],
        hidden_layers: int,
        output_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
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


class RTdetectorModel(nn.Module):
    """Core RTdetector architecture.

    The implementation follows the official repo closely but keeps only the
    single RTdetector variant used in the paper benchmark.
    """

    def __init__(self, feats: int, window_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.name = "RTdetector"
        self.n_feats = feats
        self.n_window = window_size
        d_model = 2 * feats
        n_heads = feats

        self.pos_encoder = PositionalEncoding(d_model, dropout, self.n_window)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=16, dropout=dropout)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=16, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layer, 1)
        self.output_layer = nn.Sequential(nn.Linear(d_model, feats), nn.Sigmoid())
        self.tau_learner = Projector(enc_in=d_model, seq_len=self.n_window, hidden_dims=[16], hidden_layers=1, output_dim=1)
        self.delta_learner = Projector(
            enc_in=d_model,
            seq_len=self.n_window,
            hidden_dims=[16],
            hidden_layers=1,
            output_dim=self.n_window,
        )

    def _encode(self, src: torch.Tensor, cond: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = torch.zeros_like(src)
        x1 = self._decode(src, cond, tgt)
        cond = (x1 - src) ** 2
        x2 = self._decode(src, cond, tgt)
        return x1, x2
