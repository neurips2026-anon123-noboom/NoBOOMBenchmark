"""Compact OracleAD modules adapted from the NeurIPS 2025 paper.

This port keeps the paper's main structure:
1. independent per-variable causal encoders,
2. cross-variable self-attention over the causal embeddings,
3. a Stable Latent Structure (SLS) regularizer based on pairwise distances.

The original paper describes per-variable encoder/decoder branches. This
implementation keeps that layout, but trims unrelated experiment code and
root-cause reporting so it fits the benchmark's `detect_fit` / `detect_score`
API cleanly.
"""

from __future__ import annotations

import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F


class OracleTemporalBranch(nn.Module):
    """Per-variable causal encoder/decoder branch.

    Each variable learns its own temporal dynamics. The encoder reads the
    history window, attention-pools it into a single causal embedding, and the
    decoder reconstructs the history while also predicting the current point.
    """

    def __init__(self, hidden_dim: int, history_length: int, num_layers: int) -> None:
        super().__init__()
        self.history_length = history_length
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.init_hidden = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.init_cell = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.reconstruction_head = nn.Linear(hidden_dim, 1)
        self.prediction_head = nn.Linear(hidden_dim, 1)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Encode a `(batch, history)` series into one causal embedding."""

        sequence = history.unsqueeze(-1)
        hidden_states, _ = self.encoder(sequence)
        weights = torch.softmax(self.attention(hidden_states).squeeze(-1), dim=1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

    def decode(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a context vector into reconstructed history + next-point prediction."""

        batch_size = context.size(0)
        hidden = (
            torch.tanh(self.init_hidden(context))
            .view(batch_size, self.num_layers, -1)
            .transpose(0, 1)
            .contiguous()
        )
        cell = (
            torch.tanh(self.init_cell(context))
            .view(batch_size, self.num_layers, -1)
            .transpose(0, 1)
            .contiguous()
        )
        zeros = torch.zeros(batch_size, self.history_length, 1, device=context.device)
        decoded, _ = self.decoder(zeros, (hidden, cell))
        reconstruction = self.reconstruction_head(decoded).squeeze(-1)
        prediction = self.prediction_head(decoded[:, -1]).squeeze(-1)
        return reconstruction, prediction


class OracleADCore(nn.Module):
    """Paper-faithful OracleAD core network."""

    def __init__(
        self,
        num_channels: int,
        hidden_dim: int,
        history_length: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.history_length = history_length
        self.branches = nn.ModuleList(
            OracleTemporalBranch(
                hidden_dim=hidden_dim,
                history_length=history_length,
                num_layers=num_layers,
            )
            for _ in range(num_channels)
        )
        self.relational_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstructed history, predicted point, and refined embeddings.

        Args:
            history: `(batch, history_length, channels)` tensor.
        """

        embeddings = []
        for channel_index, branch in enumerate(self.branches):
            channel_history = history[:, :, channel_index]
            embeddings.append(branch.encode(channel_history))
        embeddings = torch.stack(embeddings, dim=1)

        refined, _ = self.relational_attention(embeddings, embeddings, embeddings)
        refined = self.layer_norm(refined + embeddings)

        reconstructions = []
        predictions = []
        for channel_index, branch in enumerate(self.branches):
            channel_context = refined[:, channel_index]
            reconstruction, prediction = branch.decode(channel_context)
            reconstructions.append(reconstruction)
            predictions.append(prediction)

        reconstruction_tensor = torch.stack(reconstructions, dim=-1)
        prediction_tensor = torch.stack(predictions, dim=-1)
        return reconstruction_tensor, prediction_tensor, refined


def pairwise_l2_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute the per-window pairwise L2 distance matrix across channels."""

    return torch.cdist(embeddings, embeddings, p=2)


def oracle_prediction_score(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average absolute error across channels, matching the paper's point score."""

    return torch.mean(torch.abs(prediction - target), dim=-1)


def oracle_deviation_score(distances: torch.Tensor, sls: torch.Tensor) -> torch.Tensor:
    """Stable Latent Structure deviation via a Frobenius norm."""

    delta = distances - sls.unsqueeze(0)
    return torch.linalg.norm(delta, ord="fro", dim=(-2, -1))


def oracle_loss(
    reconstruction: torch.Tensor,
    history: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    distances: torch.Tensor,
    sls: Optional[torch.Tensor],
    reconstruction_weight: float,
    structure_weight: float,
) -> torch.Tensor:
    """Composite OracleAD objective from reconstruction, prediction, and SLS terms."""

    prediction_loss = F.mse_loss(prediction, target)
    reconstruction_loss = F.mse_loss(reconstruction, history)
    total = prediction_loss + reconstruction_weight * reconstruction_loss
    if sls is not None:
        total = total + structure_weight * F.mse_loss(distances, sls.unsqueeze(0).expand_as(distances))
    return total
