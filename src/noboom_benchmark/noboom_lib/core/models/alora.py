"""Native ALoRa network, objective, and detector.

This follows the public ALoRa implementation at commit
fdab4a1d79633a1e0d41bc4c82a29422abface97 while adapting the code to the
NoBoom/TimeSeAD tuple-based reconstruction contract.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timesead.models import BaseModel

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - older TimeSeAD layout.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore

from timesead.optim.loss import Loss


def _extract_window(inputs: Any) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return inputs
    if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
        return inputs[0]
    raise ValueError("ALoRa expects an input tensor or a tuple containing one tensor.")


def _validate_window(
    x: torch.Tensor,
    *,
    win_size: int,
    input_dim: int,
) -> None:
    if x.ndim != 3:
        raise ValueError("ALoRa expects input shape [batch, window, channels].")
    if x.shape[1] != win_size:
        raise ValueError(f"ALoRa expected window length {win_size}, got {x.shape[1]}.")
    if x.shape[2] != input_dim:
        raise ValueError(f"ALoRa expected {input_dim} channels, got {x.shape[2]}.")


def _as_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _normalize_predictions(predictions: Tuple[Any, ...]) -> Tuple[torch.Tensor, list[torch.Tensor]]:
    normalized: Tuple[Any, ...] = predictions
    if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
        normalized = tuple(normalized[0])
    if len(normalized) != 2:
        raise ValueError(f"ALoRa expects reconstruction and attentions, received {len(normalized)} entries.")
    reconstruction, attentions = normalized
    if not isinstance(reconstruction, torch.Tensor):
        raise TypeError("ALoRa reconstruction prediction must be a tensor.")
    if not isinstance(attentions, (tuple, list)):
        raise TypeError("ALoRa attentions prediction must be a list of tensors.")
    return reconstruction, list(attentions)


class ALoRaPositionalEmbedding(nn.Module):
    """Sinusoidal position embedding used by the official ALoRa DataEmbedding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        d_model = _as_int(d_model, "d_model")
        max_len = _as_int(max_len, "max_len")

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[-1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)].to(dtype=x.dtype)


class LightMTSTokenEmbedding(nn.Module):
    """Official LightMTS pair embedding with NoBoom-native pair selection.

    The reference code consumes precomputed ordered top-correlation pairs for
    very high-dimensional datasets and otherwise falls back to combinations.
    This native version keeps the same smoothing and two-scalar pair filter, and
    accepts optional precomputed pair indices without assuming dataset names or
    repository-local preprocessing files.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_size: int = 3,
        top_k_limit: int = 512,
        pair_indices: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = _as_int(input_dim, "input_dim")
        requested_dim = _as_int(d_model, "d_model")
        self.kernel_size = _as_int(kernel_size, "kernel_size")
        self.top_k_limit = _as_int(top_k_limit, "top_k_limit")
        if self.kernel_size % 2 != 1:
            raise ValueError("token_kernel_size must be odd for symmetric circular smoothing.")
        self.padding = self.kernel_size // 2

        selected_pairs = self._select_pairs(pair_indices)
        effective_dim = min(requested_dim, self.top_k_limit, len(selected_pairs))
        if effective_dim <= 0:
            raise ValueError("ALoRa requires at least one feature pair or degenerate single-channel pair.")
        self.pairs = selected_pairs[:effective_dim]
        self.d_model = int(effective_dim)

        self.register_buffer("pairs_idx", torch.tensor(self.pairs, dtype=torch.long))
        self.weights = nn.Parameter(torch.randn(self.d_model, 2))
        avg_kernel = torch.ones(self.input_dim, 1, self.kernel_size, dtype=torch.float32) / self.kernel_size
        self.register_buffer("avg_kernel", avg_kernel)

    def _select_pairs(self, pair_indices: Optional[Sequence[Sequence[int]]]) -> list[Tuple[int, int]]:
        if pair_indices is not None:
            pairs = []
            for pair in pair_indices:
                if len(pair) != 2:
                    raise ValueError("Each ALoRa pair index must contain exactly two feature indices.")
                first, second = int(pair[0]), int(pair[1])
                if first < 0 or second < 0 or first >= self.input_dim or second >= self.input_dim:
                    raise ValueError(f"Invalid ALoRa pair index {(first, second)} for input_dim={self.input_dim}.")
                pairs.append((first, second))
            if not pairs:
                raise ValueError("pair_indices must not be empty when provided.")
            return pairs

        pairs = list(itertools.combinations(range(self.input_dim), 2))
        if pairs:
            return pairs

        # Degenerate but useful for NoBoom datasets with a single channel.
        return [(0, 0)]

    @torch.no_grad()
    def get_weight_contribution_matrix(self) -> torch.Tensor:
        contribution = torch.zeros(
            self.d_model,
            self.input_dim,
            device=self.weights.device,
            dtype=self.weights.dtype,
        )
        contribution.scatter_(1, self.pairs_idx[:, 0:1], self.weights[:, 0:1])
        contribution.scatter_add_(1, self.pairs_idx[:, 1:2], self.weights[:, 1:2])
        return contribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.permute(0, 2, 1).contiguous()
        batch_size, channels, _seq_len = channels_first.shape
        del batch_size
        if channels != self.input_dim:
            raise ValueError(f"ALoRa expected {self.input_dim} channels, got {channels}.")

        padded = F.pad(channels_first, (self.padding, self.padding), mode="circular")
        kernel = self.avg_kernel.to(device=x.device, dtype=x.dtype)
        smoothed = F.conv1d(padded, kernel, groups=channels)

        ch1 = self.pairs_idx[:, 0]
        ch2 = self.pairs_idx[:, 1]
        x1 = smoothed[:, ch1, :]
        x2 = smoothed[:, ch2, :]

        w1 = self.weights[:, 0].view(1, -1, 1).to(dtype=x.dtype)
        w2 = self.weights[:, 1].view(1, -1, 1).to(dtype=x.dtype)
        embedded = w1 * x1 + w2 * x2
        return embedded.permute(0, 2, 1).contiguous()


class LightMTSEmbedding(nn.Module):
    """LightMTS-Embed: pair token embedding plus sinusoidal positions."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.0,
        top_k_limit: int = 512,
        token_kernel_size: int = 3,
        pair_indices: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.value_embedding = LightMTSTokenEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            kernel_size=token_kernel_size,
            top_k_limit=top_k_limit,
            pair_indices=pair_indices,
        )
        self.d_model = self.value_embedding.d_model
        self.position_embedding = ALoRaPositionalEmbedding(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_embedding(x)
        return self.dropout(value + self.position_embedding(value))


class ALoRaAttention(nn.Module):
    """Unmasked self-attention returning [B, H, T, T] attention maps."""

    def __init__(
        self,
        scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _batch_size, _query_len, _heads, dim_per_head = queries.shape
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(dim_per_head)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attention, values)
        return values.contiguous(), attention


class ALoRaAttentionLayer(nn.Module):
    def __init__(
        self,
        attention: ALoRaAttention,
        d_model: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.d_model = _as_int(d_model, "d_model")
        self.n_heads = _as_int(n_heads, "n_heads")
        if self.d_model % self.n_heads != 0:
            raise ValueError("ALoRa effective d_model must be divisible by n_heads.")

        dim_per_head = self.d_model // self.n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(self.d_model, dim_per_head * self.n_heads)
        self.key_projection = nn.Linear(self.d_model, dim_per_head * self.n_heads)
        self.value_projection = nn.Linear(self.d_model, dim_per_head * self.n_heads)
        self.out_projection = nn.Linear(dim_per_head * self.n_heads, self.d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = x.shape
        heads = self.n_heads
        queries = self.query_projection(x).view(batch_size, query_len, heads, -1)
        keys = self.key_projection(x).view(batch_size, query_len, heads, -1)
        values = self.value_projection(x).view(batch_size, query_len, heads, -1)
        out, attention = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(batch_size, query_len, -1)), attention


class ALoRaEncoderLayer(nn.Module):
    """ALoRa residual attention layer with no feed-forward network."""

    def __init__(
        self,
        attention: ALoRaAttentionLayer,
        d_model: int,
        win_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(win_size)
        self.norm2 = nn.LayerNorm(win_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_x, attention = self.attention(x)
        x = x + self.dropout(new_x)

        # The official code normalizes each embedded feature across time.
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = x
        out = self.norm2((x + y).permute(0, 2, 1)).permute(0, 2, 1)
        return out, attention


class ALoRaEncoder(nn.Module):
    def __init__(self, layers: Sequence[ALoRaEncoderLayer], norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        attentions = []
        for layer in self.layers:
            x, attention = layer(x)
            attentions.append(attention)
        if self.norm is not None:
            x = self.norm(x)
        return x, attentions


class ALoRa(BaseModel):
    """ALoRa-Det reconstruction network for [B, T, D] windows."""

    def __init__(
        self,
        win_size: int,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 1,
        e_layers: int = 3,
        dropout: float = 0.0,
        top_k_limit: int = 512,
        token_kernel_size: int = 3,
        pair_indices: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.win_size = _as_int(win_size, "win_size")
        self.seq_len = self.win_size
        self.input_dim = _as_int(input_dim, "input_dim")
        self.requested_d_model = _as_int(d_model, "d_model")
        self.requested_n_heads = _as_int(n_heads, "n_heads")
        self.e_layers = _as_int(e_layers, "e_layers")

        self.embedding = LightMTSEmbedding(
            input_dim=self.input_dim,
            d_model=self.requested_d_model,
            dropout=dropout,
            top_k_limit=top_k_limit,
            token_kernel_size=token_kernel_size,
            pair_indices=pair_indices,
        )
        self.d_model = self.embedding.d_model
        if self.d_model % self.requested_n_heads != 0:
            raise ValueError(
                "ALoRa effective d_model after feature-pair selection "
                f"({self.d_model}) must be divisible by n_heads ({self.requested_n_heads})."
            )

        self.encoder = ALoRaEncoder(
            [
                ALoRaEncoderLayer(
                    ALoRaAttentionLayer(
                        ALoRaAttention(attention_dropout=dropout),
                        d_model=self.d_model,
                        n_heads=self.requested_n_heads,
                    ),
                    d_model=self.d_model,
                    win_size=self.win_size,
                    dropout=dropout,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )
        self.projection = nn.Linear(self.d_model, self.input_dim, bias=True)

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        x = _extract_window(inputs).contiguous()
        _validate_window(x, win_size=self.win_size, input_dim=self.input_dim)
        embedded = self.embedding(x)
        encoded, attentions = self.encoder(embedded)
        reconstruction = self.projection(encoded)
        return reconstruction, attentions


def alora_low_rank_regularizer(attentions: Sequence[torch.Tensor], rank: int = 1) -> torch.Tensor:
    """Paper-style low-rank penalty summed over layers and averaged over batch."""

    if not attentions:
        raise ValueError("ALoRa low-rank regularizer requires at least one attention tensor.")
    rank = int(rank)
    if rank < 0:
        raise ValueError("rank must be non-negative.")

    total = torch.zeros((), dtype=attentions[0].dtype, device=attentions[0].device)
    for attention in attentions:
        if attention.ndim != 4:
            raise ValueError("ALoRa attention tensors must have shape [B, H, T, T].")
        head_average = attention.mean(dim=1)
        singular_values = torch.linalg.svdvals(head_average)
        truncated = singular_values[:, rank:]
        layer_penalty = (truncated / (truncated + 1.0)).sum(dim=-1).mean()
        total = total + layer_penalty
    return total


class ALoRaLoss(Loss):
    """Reconstruction MSE plus ALoRa low-rank attention regularization."""

    def __init__(self, lambda_reg: float = 10.0, rank: int = 1) -> None:
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.rank = int(rank)

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        reconstruction, attentions = _normalize_predictions(predictions)
        target = targets[0].float()
        reconstruction_loss = F.mse_loss(reconstruction, target)
        low_rank_loss = alora_low_rank_regularizer(attentions, rank=self.rank)
        return reconstruction_loss + self.lambda_reg * low_rank_loss


def alora_detection_score(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    attentions: Sequence[torch.Tensor],
    rank_threshold: float = 1.0e-2,
) -> torch.Tensor:
    """Last-step reconstruction error multiplied by final-layer rank count."""

    if not attentions:
        raise ValueError("ALoRa detection scoring requires attention tensors.")
    final_attention = attentions[-1]
    if final_attention.ndim != 4:
        raise ValueError("ALoRa final attention must have shape [B, H, T, T].")
    last_error = F.mse_loss(reconstruction[:, -1, :], x[:, -1, :], reduction="none").mean(dim=-1)
    singular_values = torch.linalg.svdvals(final_attention.mean(dim=1))
    rank_count = (singular_values > float(rank_threshold)).sum(dim=-1).to(dtype=last_error.dtype)
    return last_error * rank_count


class ALoRaAnomalyDetector(AnomalyDetector):
    """ALoRa-Det online scorer; larger scores mean more anomalous windows."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        rank_threshold: float = 1.0e-2,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.model = model
        self.rank_threshold = float(rank_threshold)
        self.batch_size = _as_int(batch_size, "batch_size")

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        return None

    def _model_device(self, fallback: torch.device) -> torch.device:
        if self.model is None:
            return fallback
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return fallback

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("ALoRaAnomalyDetector requires a model.")
        source = _extract_window(inputs)
        if source.ndim != 3:
            raise ValueError("ALoRaAnomalyDetector expects [batch, window, channels] inputs.")
        if source.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32, device=source.device)

        source_device = source.device
        model_device = self._model_device(source_device)
        was_training = self.model.training
        self.model.eval()
        chunks = []
        try:
            with torch.no_grad():
                for start in range(0, source.shape[0], self.batch_size):
                    x = source[start : start + self.batch_size].to(device=model_device, dtype=torch.float32)
                    reconstruction, attentions = _normalize_predictions(self.model((x,)))
                    score = alora_detection_score(
                        x,
                        reconstruction,
                        attentions,
                        rank_threshold=self.rank_threshold,
                    )
                    chunks.append(score.detach().to(source_device))
        finally:
            self.model.train(was_training)
        return torch.cat(chunks, dim=0)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


ALoRaAD = ALoRaAnomalyDetector

__all__ = [
    "ALoRa",
    "ALoRaAD",
    "ALoRaAnomalyDetector",
    "ALoRaLoss",
    "LightMTSEmbedding",
    "LightMTSTokenEmbedding",
    "alora_detection_score",
    "alora_low_rank_regularizer",
]
