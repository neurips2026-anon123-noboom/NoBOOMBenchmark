"""Native PaAno network, loss, and detector for the benchmark."""

from __future__ import annotations

import random
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - lightweight test environments.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - lightweight test environments.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - lightweight test environments.
    Loss = nn.Module  # type: ignore[assignment]

from .._source.baselines.self_impl.PaAno.model import (
    PatchEncoder,
    TransformerPatchEncoder,
    calculate_patch_scores,
    calculate_patch_scores_by_regime,
    correlation_reference,
    correlation_scores,
    create_memory_bank,
    create_patches,
    create_regime_memory_bank,
    distribute_patch_scores_to_points,
    make_patch_loader,
)
from .._source.baselines.self_impl.recent_utils import transform_frame
from .._windowed_external import dataloader_to_frame, tensor_points_to_frame


class PaAno(BaseModel):
    """PaAno patch encoder exposed as a trainable benchmark network."""

    def __init__(
        self,
        input_dim: int,
        patch_size: Optional[int] = None,
        stride: int = 1,
        projection_dim: int = 256,
        layers: Sequence[int] = (128, 256, 128, 64),
        kernel_sizes: Sequence[int] = (7, 5, 3, 3),
        use_revin: bool = True,
        encoder_type: str = "cnn",
        encoder_d_model: int = 128,
        encoder_num_heads: int = 4,
        encoder_num_layers: int = 2,
        encoder_d_ff: int = 256,
        encoder_dropout: float = 0.1,
        encoder_pooling: str = "mean",
        encoder_patch_len: int = 16,
        encoder_patch_stride: int = 8,
        random_seed: int = 2027,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)
        self.patch_size = int(patch_size) if patch_size is not None else None
        self.stride = int(stride)
        self.projection_dim = int(projection_dim)
        self.layers = tuple(int(value) for value in layers)
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        self.use_revin = bool(use_revin)
        self.encoder_type = str(encoder_type)
        self.encoder_d_model = int(encoder_d_model)
        self.encoder_num_heads = int(encoder_num_heads)
        self.encoder_num_layers = int(encoder_num_layers)
        self.encoder_d_ff = int(encoder_d_ff)
        self.encoder_dropout = float(encoder_dropout)
        self.encoder_pooling = str(encoder_pooling)
        self.encoder_patch_len = int(encoder_patch_len)
        self.encoder_patch_stride = int(encoder_patch_stride)
        self.random_seed = int(random_seed)
        self.encoder = self._build_encoder()

    def _build_encoder(self) -> nn.Module:
        if self.encoder_type == "cnn":
            return PatchEncoder(
                in_channels=self.input_dim,
                projection_dim=self.projection_dim,
                layers=self.layers,
                kernel_sizes=self.kernel_sizes,
                use_revin=self.use_revin,
            )
        if self.patch_size is None:
            raise ValueError("patch_size must be set for PaAno transformer encoders.")
        return TransformerPatchEncoder(
            in_channels=self.input_dim,
            seq_len=self.patch_size,
            projection_dim=self.projection_dim,
            encoder_type=self.encoder_type,
            d_model=self.encoder_d_model,
            num_heads=self.encoder_num_heads,
            num_layers=self.encoder_num_layers,
            d_ff=self.encoder_d_ff,
            dropout=self.encoder_dropout,
            pooling=self.encoder_pooling,
            patch_len=min(self.encoder_patch_len, self.patch_size),
            patch_stride=max(1, min(self.encoder_patch_stride, self.patch_size)),
            use_revin=self.use_revin,
        )

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    @staticmethod
    def _extract_tensor(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        raise ValueError("PaAno expected an input tensor or tuple containing one tensor.")

    def _to_patch_tensor(self, inputs: Any) -> torch.Tensor:
        x = self._extract_tensor(inputs)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError(f"PaAno expects inputs with shape [batch, time, channels], got {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} channels, got {x.shape[-1]}.")
        return x.transpose(1, 2).contiguous()

    def embedding(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError(f"PaAno embeddings expect [batch, channels, time], got {tuple(patches.shape)}.")
        return self.encoder.embedding(patches)

    def projection(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.encoder.projection(embeddings)

    def forward(self, inputs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = self._to_patch_tensor(inputs)
        if patches.shape[-1] < 2:
            positives = patches
        else:
            positives = torch.cat([patches[..., :1], patches[..., :-1]], dim=-1)
        embeddings = self.embedding(patches)
        positive_embeddings = self.embedding(positives)
        projections = self.projection(embeddings)
        positive_projections = self.projection(positive_embeddings)
        negative_embeddings = torch.roll(embeddings, shifts=1, dims=0)
        pretext_logits = self.encoder.classification_head(
            torch.cat([embeddings, positive_embeddings], dim=1)
        ).squeeze(1)
        negative_logits = self.encoder.classification_head(
            torch.cat([embeddings, negative_embeddings], dim=1)
        ).squeeze(1)
        return embeddings, projections, positive_projections, pretext_logits, negative_logits


class PaAnoLoss(Loss):
    """PaAno triplet/pretext objective adapted to Lightning batches."""

    def __init__(
        self,
        radius: int = 2,
        lambda_weight: float = 1.0,
        temperature: float = 1.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.lambda_weight = float(lambda_weight)
        self.temperature = float(temperature)
        self.margin = float(margin)

    @staticmethod
    def _normalize_predictions(predictions: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
        normalized = predictions
        if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
            normalized = tuple(normalized[0])
        return normalized  # type: ignore[return-value]

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        epoch: int = 0,
        num_epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        embeddings, projections, positive_projections, pretext_logits, negative_logits = (
            self._normalize_predictions(predictions)
        )
        batch_size = projections.shape[0]
        if batch_size < 2:
            return projections.sum() * 0.0

        z_anchor = F.normalize(projections, dim=1, eps=1e-12)
        z_pos = F.normalize(positive_projections, dim=1, eps=1e-12)
        similarity = (z_anchor @ z_pos.T) / self.temperature
        positive_similarity = similarity.diag()
        similarity_neg = similarity.clone()
        similarity_neg.fill_diagonal_(float("inf"))
        hard_negative_distance = (1.0 - similarity_neg).max(dim=1).values
        positive_distance = 1.0 - positive_similarity
        target = targets[0] if targets else None
        num_channels = target.shape[-1] if isinstance(target, torch.Tensor) and target.ndim > 0 else 1
        mu = 1 if num_channels != 1 else 10
        triplet_loss = F.relu(positive_distance - hard_negative_distance + self.margin).mean() / mu
        triplet_loss = triplet_loss.detach() + (triplet_loss - triplet_loss.detach()) * 0.01

        if num_epochs is None or num_epochs <= 0:
            progress = 1.0
        else:
            progress = float(epoch) / max(float(num_epochs), 1.0)
        pretext_weight = self.lambda_weight * max(0.0, 1.0 - progress / 0.1) if progress < 0.1 else 0.0
        if pretext_weight <= 0.0:
            return triplet_loss

        labels = torch.cat(
            [
                torch.ones_like(pretext_logits),
                torch.zeros_like(negative_logits),
            ],
            dim=0,
        )
        logits = torch.cat([pretext_logits, negative_logits], dim=0)
        pretext_loss = F.binary_cross_entropy_with_logits(logits, labels)
        return triplet_loss + pretext_weight * pretext_loss


class PaAnoAnomalyDetector(AnomalyDetector):
    """Memory-bank PaAno scorer using a trained native PaAno network."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        patch_size: Optional[int] = None,
        stride: int = 1,
        batch_size: int = 128,
        memory_bank_ratio: Optional[float] = 0.1,
        top_k: int = 3,
        use_revin: bool = True,
        random_seed: int = 2027,
        point_aggregation: str = "mean",
        point_quantile: float = 0.75,
        score_normalization: str = "none",
        regime_count: int = 1,
        structure_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.model = model
        self.patch_size = int(patch_size) if patch_size is not None else None
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.memory_bank_ratio = memory_bank_ratio
        self.top_k = int(top_k)
        self.use_revin = bool(use_revin)
        self.random_seed = int(random_seed)
        self.point_aggregation = str(point_aggregation)
        self.point_quantile = float(point_quantile)
        self.score_normalization = str(score_normalization)
        self.regime_count = int(regime_count)
        self.structure_weight = float(structure_weight)
        self.scaler = None
        self.memory_bank: Optional[torch.Tensor] = None
        self.regime_banks: Optional[list[torch.Tensor]] = None
        self.regime_centroids: Optional[torch.Tensor] = None
        self.structure_reference: Optional[torch.Tensor] = None
        self.patch_score_stats: Optional[tuple[float, float]] = None
        self.structure_score_stats: Optional[tuple[float, float]] = None

    def _set_seed(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    @staticmethod
    def _to_numpy(frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if values.ndim == 1:
            values = values[:, None]
        return values

    @staticmethod
    def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        location = float(np.median(values)) if values.size else 0.0
        scale = float(np.median(np.abs(values - location)) * 1.4826) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(values)) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return location, scale

    def _normalize_scores(self, values: np.ndarray, stats: Optional[tuple[float, float]]) -> np.ndarray:
        if stats is None or self.score_normalization == "none":
            return np.asarray(values, dtype=np.float32)
        location, scale = stats
        return ((np.asarray(values, dtype=np.float32) - location) / scale).astype(np.float32)

    def _device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _prepare_values(self, frame: pd.DataFrame, fit: bool = False) -> np.ndarray:
        if self.use_revin:
            return self._to_numpy(frame)
        from sklearn.preprocessing import StandardScaler

        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self._to_numpy(frame))
        return transform_frame(self.scaler, frame).to_numpy(dtype=np.float32)

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del kwargs
        if self.model is None:
            raise RuntimeError("PaAnoAnomalyDetector requires a trained PaAno model.")
        self._set_seed()
        frame = dataloader_to_frame(dataset)
        train_values = self._prepare_values(frame, fit=True)
        patch_size = self.patch_size
        if patch_size is None:
            patch_size = getattr(self.model, "patch_size", None)
        if patch_size is None:
            raise ValueError("PaAno patch_size must be configured before fitting the detector.")
        self.patch_size = max(2, min(int(patch_size), len(train_values)))
        train_patches, train_indices = create_patches(
            train_values,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        train_loader = make_patch_loader(
            train_patches,
            train_indices,
            batch_size=self.batch_size,
            shuffle=False,
        )
        device = self._device()
        self.model.eval()
        self.memory_bank = create_memory_bank(
            self.model,
            train_loader,
            device,
            compression=self.memory_bank_ratio,
        )
        self.regime_banks = None
        self.regime_centroids = None
        if self.regime_count > 1:
            self.regime_banks, self.regime_centroids = create_regime_memory_bank(
                self.model,
                train_loader,
                device,
                num_regimes=self.regime_count,
                compression=self.memory_bank_ratio,
            )
        train_patch_scores = calculate_patch_scores(
            self.model,
            train_loader,
            self.memory_bank,
            device,
            top_k=self.top_k,
        )
        if self.score_normalization != "none":
            self.patch_score_stats = self._robust_location_scale(train_patch_scores)
        if self.structure_weight > 0.0:
            self.structure_reference = correlation_reference(
                train_patches,
                device,
                batch_size=self.batch_size,
            ).detach().cpu()
            train_structure_scores = correlation_scores(
                train_patches,
                self.structure_reference,
                device,
                batch_size=self.batch_size,
            )
            self.structure_score_stats = self._robust_location_scale(train_structure_scores)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None or self.memory_bank is None or self.patch_size is None:
            raise RuntimeError("PaAnoAnomalyDetector must be fitted before scoring.")
        frame = tensor_points_to_frame(inputs)
        test_values = self._prepare_values(frame, fit=False)
        test_patches, test_indices = create_patches(
            test_values,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        test_loader = make_patch_loader(
            test_patches,
            test_indices,
            batch_size=self.batch_size,
            shuffle=False,
        )
        device = inputs[0].device
        self.model.to(device)
        self.model.eval()
        if self.regime_count > 1:
            if self.regime_banks is None or self.regime_centroids is None:
                raise RuntimeError("PaAno regime memory bank is missing. Fit the detector again.")
            patch_scores = calculate_patch_scores_by_regime(
                self.model,
                test_loader,
                self.regime_banks,
                self.regime_centroids,
                device,
                top_k=self.top_k,
            )
        else:
            patch_scores = calculate_patch_scores(
                self.model,
                test_loader,
                self.memory_bank,
                device,
                top_k=self.top_k,
            )
        patch_scores = self._normalize_scores(patch_scores, self.patch_score_stats)
        if self.structure_weight > 0.0:
            if self.structure_reference is None:
                raise RuntimeError("PaAno structure reference is missing. Fit the detector again.")
            structure_patch_scores = correlation_scores(
                test_patches,
                self.structure_reference,
                device,
                batch_size=self.batch_size,
            )
            patch_scores = patch_scores + self.structure_weight * self._normalize_scores(
                structure_patch_scores,
                self.structure_score_stats,
            )
        point_scores = distribute_patch_scores_to_points(
            patch_scores,
            patch_size=self.patch_size,
            num_points=len(frame),
            aggregation=self.point_aggregation,
            quantile=self.point_quantile,
        )
        return torch.as_tensor(point_scores, dtype=torch.float32, device=inputs[0].device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


PaAnoAD = PaAnoAnomalyDetector

__all__ = ["PaAno", "PaAnoAD", "PaAnoAnomalyDetector", "PaAnoLoss"]
