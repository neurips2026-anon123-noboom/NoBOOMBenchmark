"""Native SPAGD network, loss, and detector components."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - used by lightweight import tests.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - used by lightweight import tests.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - used by lightweight import tests.
    Loss = nn.modules.loss._Loss  # type: ignore[assignment]


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding for the reconstruction transformer."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[-1]])
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
    """Build the paper's cosine-similarity graph for each window."""

    nodes = x.transpose(1, 2)
    nodes = F.normalize(nodes, p=2, dim=-1)
    similarity = torch.sigmoid(torch.matmul(nodes, nodes.transpose(1, 2)))
    return build_sparse_from_similarity(similarity, top_k=top_k)


def build_sparse_from_similarity(similarity: torch.Tensor, top_k: int) -> torch.Tensor:
    """Sparsify an affinity matrix row-wise while preserving self-loops."""

    channels = similarity.size(-1)
    if channels == 1:
        return torch.ones_like(similarity)
    effective_k = max(1, min(top_k, channels))
    values, indices = torch.topk(similarity, k=effective_k, dim=-1)
    sparse = torch.zeros_like(similarity)
    sparse.scatter_(-1, indices, values)
    sparse = torch.maximum(sparse, sparse.transpose(1, 2))
    eye = torch.eye(channels, device=similarity.device, dtype=similarity.dtype).unsqueeze(0)
    return torch.maximum(sparse, eye)


def build_adjusted_adjacency(
    x: torch.Tensor,
    recon: torch.Tensor,
    top_k: int,
    candidate_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjust the graph with reconstruction-residual anomaly candidates."""

    base = build_sparse_adjacency(x, top_k=top_k)
    residual = torch.mean(torch.abs(x - recon), dim=1)
    channels = residual.size(-1)
    num_candidates = max(1, min(channels, int(math.ceil(channels * candidate_ratio))))
    _, top_indices = torch.topk(residual, k=num_candidates, dim=-1)
    candidate_mask = torch.zeros_like(residual, dtype=torch.bool)
    candidate_mask.scatter_(1, top_indices, True)

    weights = torch.sigmoid(residual)
    candidate_weights = torch.where(candidate_mask, weights, torch.zeros_like(weights))
    adjusted = base + candidate_weights.unsqueeze(2) + candidate_weights.unsqueeze(1)
    diagonal = torch.diagonal(adjusted, dim1=1, dim2=2)
    diagonal.copy_(torch.diagonal(base, dim1=1, dim2=2) + candidate_weights)
    adjusted = 0.5 * (adjusted + adjusted.transpose(1, 2))
    adjusted = build_sparse_from_similarity(adjusted, top_k=top_k)
    return adjusted, residual


class GraphAttentionLayer(nn.Module):
    """Dense graph attention layer used for the dual-graph spatial encoder."""

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
    """Dual-graph classifier from the SPAGD paper."""

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

        temporal_input = spatial.mean(dim=2)
        chunks = torch.tensor_split(temporal_input, self.num_chunks, dim=1)
        chunk_features = []
        for chunk in chunks:
            features = self.temporal_conv(chunk.transpose(1, 2))
            chunk_features.append(features.mean(dim=-1))
        stacked = torch.cat(chunk_features, dim=-1)
        return self.predictor(stacked).squeeze(-1)


class SPAGD(BaseModel):
    """SPAGD network with reconstruction, graph adjustment, and classifier logits."""

    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        model_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        top_k: int = 5,
        candidate_ratio: float = 0.2,
        num_chunks: int = 5,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if win_size is not None and win_size < 2:
            raise ValueError("SPAGD requires win_size >= 2.")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive.")

        self.input_dim = int(input_dim)
        self.win_size = int(win_size) if win_size is not None else None
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.top_k = int(top_k)
        self.candidate_ratio = float(candidate_ratio)
        self.num_chunks = int(num_chunks)

        self.reconstructor = SPAGDReconstructor(
            input_dim=self.input_dim,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.classifier = SpatioTemporalClassifier(
            hidden_dim=self.model_dim,
            num_chunks=self.num_chunks,
            dropout=self.dropout,
        )

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    @staticmethod
    def _extract_input(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        raise ValueError("SPAGD expected input tensor with shape [batch, window, channels].")

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        x = self._extract_input(inputs)
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


class SPAGDLoss(Loss):
    """SPAGD reconstruction plus self-perturbed classifier objective."""

    def __init__(self, beta: float = 1e-2, reduction: str = "mean") -> None:
        try:
            super().__init__(reduction=reduction)
        except TypeError:  # pragma: no cover - lightweight tests may stub Loss as object.
            nn.Module.__init__(self)
        self.reduction = reduction
        self.beta = float(beta)

    @staticmethod
    def _extract_target(targets: Any) -> torch.Tensor:
        if isinstance(targets, torch.Tensor):
            return targets
        if isinstance(targets, (tuple, list)) and targets and isinstance(targets[0], torch.Tensor):
            return targets[0]
        raise ValueError("SPAGDLoss expected reconstruction targets.")

    @staticmethod
    def _extract_outputs(predictions: Any) -> dict[str, torch.Tensor]:
        outputs = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
        if not isinstance(outputs, dict):
            raise TypeError("SPAGDLoss expected SPAGD network outputs.")
        return outputs

    def forward(self, predictions: Any, targets: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        outputs = self._extract_outputs(predictions)
        target = self._extract_target(targets)
        reconstruction_loss = F.mse_loss(
            outputs["reconstruction"],
            target,
            reduction=self.reduction,
        )
        batch_size = outputs["normal_logits"].size(0)
        labels = torch.cat(
            [
                torch.zeros(batch_size, device=target.device, dtype=target.dtype),
                torch.ones(batch_size, device=target.device, dtype=target.dtype),
            ],
            dim=0,
        )
        logits = torch.cat([outputs["normal_logits"], outputs["perturbed_logits"]], dim=0)
        detection_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return reconstruction_loss + self.beta * detection_loss


class SPAGDAnomalyDetector(AnomalyDetector):
    """Detector that scores windows with the trained SPAGD classifier output."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        del kwargs
        try:
            super().__init__()
        except TypeError:  # pragma: no cover - protects mixed timesead/stub imports.
            nn.Module.__init__(self)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model = model
        self.batch_size = int(batch_size)

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        if isinstance(error, torch.OutOfMemoryError):
            return True
        message = str(error).lower()
        return "cuda out of memory" in message or "outofmemoryerror" in message

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        return None

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("SPAGDAnomalyDetector requires a trained SPAGD network.")
        source_device = inputs[0].device
        parameters = getattr(self.model, "parameters", None)
        buffers = getattr(self.model, "buffers", None)
        model_anchor = next(parameters(), None) if callable(parameters) else None
        if model_anchor is None and callable(buffers):
            model_anchor = next(buffers(), None)
        model_device = model_anchor.device if model_anchor is not None else source_device
        source = inputs[0]
        if source.shape[0] == 0:
            return source.new_empty((0,))
        source_cpu = source.detach().cpu()
        was_training = self.model.training
        self.model.eval()
        scores = []
        batch_size = self.batch_size
        start = 0
        try:
            with torch.no_grad():
                while start < source_cpu.shape[0]:
                    try:
                        batch = source_cpu[start : start + batch_size].to(
                            device=model_device,
                            dtype=torch.float32,
                        )
                        outputs = self.model((batch,))
                        scores.append(torch.sigmoid(outputs["test_logits"]).detach().cpu())
                        start += batch_size
                    except Exception as error:
                        if not self._is_cuda_oom(error) or batch_size <= 1:
                            raise
                        if model_device.type == "cuda":
                            torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
            joined_scores = torch.cat(scores, dim=0)
        finally:
            if was_training:
                self.model.train()
        return joined_scores.to(source_device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


SPAGDCore = SPAGD
SPAGDAD = SPAGDAnomalyDetector

__all__ = [
    "SPAGD",
    "SPAGDAD",
    "SPAGDAnomalyDetector",
    "SPAGDCore",
    "SPAGDLoss",
    "build_adjusted_adjacency",
    "build_sparse_adjacency",
    "build_sparse_from_similarity",
]
