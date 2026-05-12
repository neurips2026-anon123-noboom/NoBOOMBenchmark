"""Native ScatterAD network, loss, and detector components."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - lightweight test environments.
    BaseModel = nn.Module  # type: ignore[assignment]
if not issubclass(BaseModel, nn.Module):  # pragma: no cover - test stubs.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - lightweight test environments.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore
if not issubclass(AnomalyDetector, nn.Module):  # pragma: no cover - test stubs.
    AnomalyDetector = nn.Module  # type: ignore[assignment]

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - lightweight test environments.
    Loss = nn.Module  # type: ignore[assignment]
if not issubclass(Loss, nn.Module):  # pragma: no cover - test stubs.
    Loss = nn.Module  # type: ignore[assignment]


def build_temporal_lookback_edges(
    batch_size: int,
    window_size: int,
    tau: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return directed edges ``t-k -> t`` within each window only."""

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")
    if window_size < 0:
        raise ValueError("window_size must be non-negative.")
    if tau < 1:
        raise ValueError("tau must be positive.")

    edges = []
    for batch_index in range(batch_size):
        offset = batch_index * window_size
        for dst in range(window_size):
            max_lag = min(tau, dst)
            for lag in range(1, max_lag + 1):
                edges.append((offset + dst - lag, offset + dst))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def build_temporal_lookback_adjacency(
    window_size: int,
    tau: int = 2,
    device: Optional[torch.device] = None,
    include_self: bool = True,
) -> torch.Tensor:
    """Return a dense incoming-neighbor mask indexed as ``[dst, src]``."""

    if window_size < 0:
        raise ValueError("window_size must be non-negative.")
    if tau < 1:
        raise ValueError("tau must be positive.")
    adjacency = torch.zeros((window_size, window_size), dtype=torch.bool, device=device)
    if include_self and window_size > 0:
        adjacency.fill_diagonal_(True)
    for dst in range(window_size):
        max_lag = min(tau, dst)
        for lag in range(1, max_lag + 1):
            adjacency[dst, dst - lag] = True
    return adjacency


def _sample_center(hidden_dim: int) -> torch.Tensor:
    center = torch.randn(hidden_dim, dtype=torch.float32)
    center = F.normalize(center, p=2, dim=0)
    radius = torch.rand((), dtype=torch.float32).clamp_min(1e-6)
    return center * radius


class CausalTemporalConv(nn.Module):
    """Causal temporal convolution over the window dimension."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=self.kernel_size)
        self.norm = nn.BatchNorm1d(output_dim)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.transpose(1, 2)
        hidden = F.pad(hidden, (self.kernel_size - 1, 0))
        hidden = self.conv(hidden)
        hidden = self.norm(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden.transpose(1, 2)


class DenseTemporalGATLayer(nn.Module):
    """Dependency-free dense multi-head GAT over temporal nodes.

    The mask is a dense equivalent of PyG ``GATConv`` on the paper's directed
    temporal graph: each destination node attends only to its lookback sources
    plus its self-loop.
    """

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = 0.2
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, _hidden_dim = x.shape
        hidden = self.linear(x).view(batch_size, window_size, self.heads, self.head_dim)

        src_logits = (hidden * self.attn_src.view(1, 1, self.heads, self.head_dim)).sum(-1)
        dst_logits = (hidden * self.attn_dst.view(1, 1, self.heads, self.head_dim)).sum(-1)
        logits = dst_logits.permute(0, 2, 1).unsqueeze(-1) + src_logits.permute(0, 2, 1).unsqueeze(-2)
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)

        mask = adjacency.view(1, 1, window_size, window_size)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        attended = torch.einsum("bhij,bjhd->bihd", weights, hidden)
        attended = attended.reshape(batch_size, window_size, self.hidden_dim)
        return F.elu(attended)


class ScatterADEncoder(nn.Module):
    """Temporal convolution plus graph attention encoder from ScatterAD."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.temporal_layers = nn.ModuleList()
        for layer_index in range(num_layers):
            layer_input_dim = input_dim if layer_index == 0 else hidden_dim
            self.temporal_layers.append(
                CausalTemporalConv(
                    input_dim=layer_input_dim,
                    output_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
        self.gat_layers = nn.ModuleList(
            [DenseTemporalGATLayer(hidden_dim=hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.skip_connection = nn.Linear(input_dim, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = self.skip_connection(x)
        hidden = x
        for layer in self.temporal_layers:
            hidden = layer(hidden)
        for layer in self.gat_layers:
            hidden = layer(hidden, adjacency)
        embeddings = hidden + identity
        return embeddings, self.projection(embeddings)


class _ScatterADParameterList(list):
    def __init__(self, parameters: Iterable[nn.Parameter], model: "ScatterAD") -> None:
        super().__init__(parameters)
        self.scatterad_model = model


class ScatterADAdam(torch.optim.Adam):
    """Adam optimizer that applies ScatterAD's EMA update after every step."""

    def __init__(self, params: Iterable[nn.Parameter], *args: Any, **kwargs: Any) -> None:
        self.scatterad_model = getattr(params, "scatterad_model", None)
        super().__init__(params, *args, **kwargs)

    def step(self, closure: Any = None) -> Any:
        result = super().step(closure=closure)
        if self.scatterad_model is not None:
            self.scatterad_model.update_target()
        return result


class ScatterAD(BaseModel):
    """ScatterAD network with online/target encoders and EMA updates."""

    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        tau: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3,
        ema_momentum: float = 0.5,
        score_scatter_weight: float = 1.0,
        score_time_weight: float = 1.0,
        scoring_mode: str = "distance",
        update_target_on_forward: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if win_size is not None and win_size < 2:
            raise ValueError("ScatterAD requires win_size >= 2.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if not 0.0 < ema_momentum < 1.0:
            raise ValueError("ema_momentum must be in (0, 1).")
        if scoring_mode not in {"distance", "official_reciprocal"}:
            raise ValueError("scoring_mode must be one of {'distance', 'official_reciprocal'}.")

        self.input_dim = int(input_dim)
        self.win_size = int(win_size) if win_size is not None else None
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.tau = int(tau)
        self.dropout = float(dropout)
        self.kernel_size = int(kernel_size)
        self.ema_momentum = float(ema_momentum)
        self.score_scatter_weight = float(score_scatter_weight)
        self.score_time_weight = float(score_time_weight)
        self.scoring_mode = str(scoring_mode)
        self.update_target_on_forward = bool(update_target_on_forward)

        self.online_encoder = ScatterADEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout,
            kernel_size=self.kernel_size,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.register_buffer("center", _sample_center(self.hidden_dim))

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        trainable_parameters = _ScatterADParameterList(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            self,
        )
        return (trainable_parameters,)

    @staticmethod
    def _extract_input(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        raise ValueError("ScatterAD expected input tensor with shape [batch, window, channels].")

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("ScatterAD expects input shape [batch, window, channels].")
        if self.win_size is not None and x.shape[1] != self.win_size:
            raise ValueError(f"ScatterAD expected window length {self.win_size}, got {x.shape[1]}.")
        if x.shape[2] != self.input_dim:
            raise ValueError(f"ScatterAD expected {self.input_dim} channels, got {x.shape[2]}.")

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA-update the target encoder from the online encoder."""

        for online_parameter, target_parameter in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_parameter.mul_(self.ema_momentum).add_(online_parameter, alpha=1.0 - self.ema_momentum)
        for online_buffer, target_buffer in zip(
            self.online_encoder.buffers(),
            self.target_encoder.buffers(),
        ):
            if torch.is_floating_point(target_buffer):
                target_buffer.mul_(self.ema_momentum).add_(online_buffer, alpha=1.0 - self.ema_momentum)
            else:
                target_buffer.copy_(online_buffer)

    def _maybe_update_target_for_training(self, x: torch.Tensor) -> None:
        if not self.training or not self.update_target_on_forward or x.device.type == "meta":
            return
        self.update_target()

    def _adjacency(self, window_size: int, device: torch.device) -> torch.Tensor:
        return build_temporal_lookback_adjacency(
            window_size=window_size,
            tau=self.tau,
            device=device,
            include_self=True,
        )

    def _edge_index(self, batch_size: int, window_size: int, device: torch.device) -> torch.Tensor:
        return build_temporal_lookback_edges(
            batch_size=batch_size,
            window_size=window_size,
            tau=self.tau,
            device=device,
        )

    def forward(self, inputs: Any) -> Dict[str, Any]:
        x = self._extract_input(inputs).float()
        self._validate_input(x)
        self._maybe_update_target_for_training(x)

        batch_size, window_size, _input_dim = x.shape
        adjacency = self._adjacency(window_size, x.device)
        edge_index = self._edge_index(batch_size, window_size, x.device)

        online_embeddings, online_projection = self.online_encoder(x, adjacency)
        predicted_embeddings = self.predictor(online_projection)
        target_was_training = self.target_encoder.training
        self.target_encoder.eval()
        with torch.no_grad():
            target_embeddings, _target_projection = self.target_encoder(x, adjacency)
            target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)
        if target_was_training:
            self.target_encoder.train()
        normalized_online_embeddings = F.normalize(online_embeddings, p=2, dim=-1)
        # Value equals the detached target embedding; gradient flows through the online encoder.
        scatter_embeddings = target_embeddings + normalized_online_embeddings - normalized_online_embeddings.detach()

        score_components = self.score_components_from_embeddings(online_embeddings)
        return {
            "online_embeddings": online_embeddings,
            "predicted_embeddings": predicted_embeddings,
            "target_embeddings": target_embeddings,
            "scatter_embeddings": scatter_embeddings,
            "edge_index": edge_index,
            "center": self.center,
            "score_components": score_components,
        }

    def score_components_from_embeddings(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        center = self.center.to(device=embeddings.device, dtype=embeddings.dtype)
        center_distances = torch.linalg.vector_norm(normalized_embeddings - center.view(1, 1, -1), dim=-1)
        if self.scoring_mode == "official_reciprocal":
            scattering_score = center_distances.clamp_min(1e-8).reciprocal()
        else:
            scattering_score = center_distances

        time_inconsistency = embeddings.new_zeros(embeddings.shape[:2])
        if embeddings.shape[1] > 1:
            time_inconsistency[:, 1:] = (embeddings[:, 1:] - embeddings[:, :-1]).square().mean(dim=-1)
        score = self.score_scatter_weight * scattering_score + self.score_time_weight * time_inconsistency
        return {
            "scattering": scattering_score,
            "time_inconsistency": time_inconsistency,
            "score": score,
        }

    def anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            outputs = self((x,))
        finally:
            if was_training:
                self.train()
        return outputs["score_components"]["score"]


class ScatterADLoss(Loss):
    """ScatterAD objective: scattering, temporal consistency, and edge contrast."""

    def __init__(
        self,
        scatter_weight: float = 1.0,
        time_weight: float = 1.0,
        contrast_weight: float = 1.0,
    ) -> None:
        try:
            super().__init__()
        except TypeError:  # pragma: no cover - lightweight tests may stub Loss as object.
            nn.Module.__init__(self)
        self.scatter_weight = float(scatter_weight)
        self.time_weight = float(time_weight)
        self.contrast_weight = float(contrast_weight)

    @staticmethod
    def _extract_outputs(predictions: Any) -> Dict[str, Any]:
        outputs = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
        if not isinstance(outputs, dict):
            raise TypeError("ScatterADLoss expected ScatterAD network outputs.")
        return outputs

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def forward(self, predictions: Any, targets: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        del targets, args, kwargs
        outputs = self._extract_outputs(predictions)
        predicted_embeddings = outputs["predicted_embeddings"]
        target_embeddings = outputs["target_embeddings"]
        scatter_embeddings = outputs.get("scatter_embeddings", target_embeddings)
        center = outputs["center"].to(device=target_embeddings.device, dtype=target_embeddings.dtype)
        edge_index = outputs["edge_index"]

        center_sim = F.cosine_similarity(
            scatter_embeddings,
            center.view(1, 1, -1),
            dim=-1,
        )
        scatter_loss = -torch.logsumexp(center_sim.unsqueeze(-1), dim=-1).mean()

        if predicted_embeddings.shape[1] > 1:
            time_loss = F.mse_loss(predicted_embeddings[:, 1:], predicted_embeddings[:, :-1])
        else:
            time_loss = self._zero_like(predicted_embeddings)

        if edge_index.numel() == 0:
            contrast_loss = self._zero_like(predicted_embeddings)
        else:
            flat_predicted = predicted_embeddings.reshape(-1, predicted_embeddings.shape[-1])
            flat_target = target_embeddings.reshape(-1, target_embeddings.shape[-1])
            src, dst = edge_index
            positive_similarity = F.cosine_similarity(flat_predicted[src], flat_target[dst], dim=-1)
            contrast_loss = -F.logsigmoid(positive_similarity).mean()

        return (
            self.scatter_weight * scatter_loss
            + self.time_weight * time_loss
            + self.contrast_weight * contrast_loss
        )


class ScatterADAnomalyDetector(AnomalyDetector):
    """Detector returning the final timestamp score for each input window."""

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

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        return None

    def _model_device(self, fallback: torch.device) -> torch.device:
        if self.model is None:
            return fallback
        parameters = getattr(self.model, "parameters", None)
        buffers = getattr(self.model, "buffers", None)
        model_anchor = next(parameters(), None) if callable(parameters) else None
        if model_anchor is None and callable(buffers):
            model_anchor = next(buffers(), None)
        return model_anchor.device if model_anchor is not None else fallback

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("ScatterADAnomalyDetector requires a trained ScatterAD network.")
        source = inputs[0]
        source_device = source.device
        if source.shape[0] == 0:
            return source.new_empty((0,))
        model_device = self._model_device(source_device)
        model_inputs = source.to(device=model_device, dtype=torch.float32)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            chunks = []
            for start in range(0, model_inputs.shape[0], self.batch_size):
                batch = model_inputs[start : start + self.batch_size]
                if hasattr(self.model, "anomaly_scores"):
                    scores = self.model.anomaly_scores(batch)
                else:
                    outputs = self.model((batch,))
                    scores = outputs["score_components"]["score"]
                chunks.append(scores[:, -1])
            result = torch.cat(chunks, dim=0)
        if was_training:
            self.model.train()
        return result.to(source_device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


ScatterADAD = ScatterADAnomalyDetector

__all__ = [
    "ScatterAD",
    "ScatterADAD",
    "ScatterADAdam",
    "ScatterADAnomalyDetector",
    "ScatterADEncoder",
    "ScatterADLoss",
    "build_temporal_lookback_adjacency",
    "build_temporal_lookback_edges",
]
