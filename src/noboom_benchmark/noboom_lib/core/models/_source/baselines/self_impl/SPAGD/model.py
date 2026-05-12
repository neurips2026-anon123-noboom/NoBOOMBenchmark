"""Compact SPAGD modules adapted from the NeurIPS 2025 paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal encoding for the reconstruction transformer."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SPAGDReconstructor(nn.Module):
    """Transformer encoder/decoder used for self-perturbation generation."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)
        return self.output_projection(hidden)


def build_sparse_adjacency(x: torch.Tensor, top_k: int) -> torch.Tensor:
    """Build the paper's cosine-similarity graph for each window.

    Args:
        x: `(batch, time, channels)` tensor.
    """

    nodes = x.transpose(1, 2)
    nodes = F.normalize(nodes, p=2, dim=-1)
    similarity = torch.sigmoid(torch.matmul(nodes, nodes.transpose(1, 2)))
    channels = similarity.size(-1)
    if channels == 1:
        return torch.ones_like(similarity)
    effective_k = max(1, min(top_k, channels))
    values, indices = torch.topk(similarity, k=effective_k, dim=-1)
    sparse = torch.zeros_like(similarity)
    sparse.scatter_(-1, indices, values)
    sparse = torch.maximum(sparse, sparse.transpose(1, 2))
    eye = torch.eye(channels, device=x.device).unsqueeze(0)
    return torch.maximum(sparse, eye)


def build_adjusted_adjacency(
    x: torch.Tensor,
    recon: torch.Tensor,
    top_k: int,
    candidate_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust the graph with reconstruction-residual-based anomaly candidates."""

    base = build_sparse_adjacency(x, top_k=top_k)
    residual = torch.mean(torch.abs(x - recon), dim=1)
    channels = residual.size(-1)
    num_candidates = max(1, min(channels, int(math.ceil(channels * candidate_ratio))))
    _, top_indices = torch.topk(residual, k=num_candidates, dim=-1)
    candidate_mask = torch.zeros_like(residual, dtype=torch.bool)
    candidate_mask.scatter_(1, top_indices, True)

    weights = torch.sigmoid(residual)
    row_boost = torch.where(candidate_mask, weights, torch.zeros_like(weights)).unsqueeze(2)
    col_boost = torch.where(candidate_mask, weights, torch.zeros_like(weights)).unsqueeze(1)
    adjusted = base + row_boost + col_boost
    diagonal = torch.diagonal(adjusted, dim1=1, dim2=2)
    diagonal.copy_(torch.diagonal(base, dim1=1, dim2=2) + torch.where(candidate_mask, weights, torch.zeros_like(weights)))
    adjusted = 0.5 * (adjusted + adjusted.transpose(1, 2))
    adjusted = build_sparse_from_similarity(adjusted, top_k=top_k)
    return adjusted, residual


def build_sparse_from_similarity(similarity: torch.Tensor, top_k: int) -> torch.Tensor:
    """Sparsify an already-computed affinity matrix row-wise."""

    channels = similarity.size(-1)
    if channels == 1:
        return torch.ones_like(similarity)
    effective_k = max(1, min(top_k, channels))
    values, indices = torch.topk(similarity, k=effective_k, dim=-1)
    sparse = torch.zeros_like(similarity)
    sparse.scatter_(-1, indices, values)
    sparse = torch.maximum(sparse, sparse.transpose(1, 2))
    eye = torch.eye(channels, device=similarity.device).unsqueeze(0)
    return torch.maximum(sparse, eye)


class GraphAttentionLayer(nn.Module):
    """Small dense GAT layer used for the dual-graph spatial encoder."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.attention = nn.Linear(output_dim * 2, 1, bias=False)
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        batch_size, num_nodes, _ = h.shape
        h_i = h.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        h_j = h.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, -1)
        logits = self.leaky_relu(self.attention(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        scores = logits + torch.log(adjacency.clamp_min(1e-8))
        scores = scores.masked_fill(adjacency <= 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return F.relu(torch.matmul(weights, h))


class SpatioTemporalClassifier(nn.Module):
    """Dual-graph classifier from the SPAGD paper.

    Each time step first receives graph attention across variables. The temporal
    axis is then processed chunk by chunk with a small shared TCN.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_chunks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks
        self.obs_projection = nn.Linear(1, hidden_dim)
        self.gat_layers = nn.ModuleList(
            [
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout),
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout),
            ]
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * num_chunks, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_channels = x.shape
        hidden = self.obs_projection(x.unsqueeze(-1))

        spatial = hidden.reshape(batch_size * time_steps, num_channels, self.hidden_dim)
        adjacency = adjacency.unsqueeze(1).expand(batch_size, time_steps, num_channels, num_channels)
        adjacency = adjacency.reshape(batch_size * time_steps, num_channels, num_channels)
        for layer in self.gat_layers:
            spatial = layer(spatial, adjacency)
        spatial = spatial.reshape(batch_size, time_steps, num_channels, self.hidden_dim)

        # Pool over the variable axis after spatial propagation, then model the
        # time axis chunk by chunk exactly as described in the paper.
        temporal_input = spatial.mean(dim=2)
        chunks = torch.tensor_split(temporal_input, self.num_chunks, dim=1)
        chunk_features = []
        for chunk in chunks:
            features = self.temporal_conv(chunk.transpose(1, 2))
            chunk_features.append(features.mean(dim=-1))
        stacked = torch.cat(chunk_features, dim=-1)
        return self.predictor(stacked).squeeze(-1)


class SPAGDCore(nn.Module):
    """Full SPAGD forward path with self-perturbation and dual-graph scoring."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        top_k: int,
        candidate_ratio: float,
        num_chunks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.candidate_ratio = candidate_ratio
        self.reconstructor = SPAGDReconstructor(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = SpatioTemporalClassifier(
            hidden_dim=model_dim,
            num_chunks=num_chunks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        recon = self.reconstructor(x)
        base_adj = build_sparse_adjacency(x, top_k=self.top_k)
        adjusted_adj, residual = build_adjusted_adjacency(
            x=x,
            recon=recon,
            top_k=self.top_k,
            candidate_ratio=self.candidate_ratio,
        )
        normal_logits = self.classifier(x, base_adj)
        perturbed_logits = self.classifier(recon, adjusted_adj)
        test_logits = self.classifier(x, adjusted_adj)
        return {
            "reconstruction": recon,
            "normal_logits": normal_logits,
            "perturbed_logits": perturbed_logits,
            "test_logits": test_logits,
            "residual": residual,
        }
