"""Native CAROTS network, loss, and detector integration."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
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

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - lightweight test environments.
    Loss = nn.Module  # type: ignore[assignment]
if not issubclass(Loss, nn.Module):  # pragma: no cover - test stubs.
    Loss = nn.Module  # type: ignore[assignment]

from .._source.baselines.self_impl.CAROTS.CAROTS import (
    DEFAULT_HYPER_PARAMS,
    _build_training_windows,
    _limit_windows,
)
from .._source.baselines.self_impl.CAROTS.model import (
    CAROTSCore,
    carots_loss,
    distance_to_centroid,
)
from .._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    make_window_loader,
    transform_frame,
)
from .._source.baselines.utils import train_val_split
from .._windowed_external import _align_scores, dataloader_to_frame, tensor_points_to_frame


class CAROTS(BaseModel):
    """CAROTS encoder/projector and CUTS+ causal discoverer as a native network."""

    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        input_step: int = 9,
        pred_step: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        projector_hidden_dim: int = 1024,
        projector_output_dim: int = 512,
        data_dim: int = 1,
        cuts_hidden_dim: int = 32,
        cuts_gru_layers: int = 1,
        cuts_concat_h: bool = True,
        cuts_shared_weights_decoder: bool = False,
        noise_level: float = 0.1,
        bias_candidates: Tuple[float, ...] = DEFAULT_HYPER_PARAMS["bias_candidates"],
        cutoff_probability: float = 0.1,
        disturb_all: bool = False,
        transform_percent: float = 0.5,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if win_size is None:
            raise ValueError("CAROTS requires win_size.")
        if input_step + pred_step != win_size:
            raise ValueError("CAROTS expects input_step + pred_step == win_size.")

        self.input_dim = int(input_dim)
        self.win_size = int(win_size)
        self.input_step = int(input_step)
        self.pred_step = int(pred_step)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.projector_output_dim = int(projector_output_dim)
        self.core = CAROTSCore(
            input_dim=self.input_dim,
            input_step=self.input_step,
            pred_step=self.pred_step,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            projector_hidden_dim=self.projector_hidden_dim,
            projector_output_dim=self.projector_output_dim,
            data_dim=int(data_dim),
            cuts_hidden_dim=int(cuts_hidden_dim),
            cuts_gru_layers=int(cuts_gru_layers),
            cuts_concat_h=bool(cuts_concat_h),
            cuts_shared_weights_decoder=bool(cuts_shared_weights_decoder),
            noise_level=float(noise_level),
            bias_candidates=tuple(float(value) for value in bias_candidates),
            cutoff_probability=float(cutoff_probability),
            disturb_all=bool(disturb_all),
            transform_percent=float(transform_percent),
        )
        self.core.positive_augmentor.set_causal_discoverer(self.core.causal_discoverer)

    @property
    def causal_discoverer(self) -> nn.Module:
        return self.core.causal_discoverer

    @property
    def positive_augmentor(self) -> nn.Module:
        return self.core.positive_augmentor

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    @staticmethod
    def _extract_window(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        raise ValueError("CAROTS expected a window tensor.")

    def forward(
        self,
        inputs: Any,
        positive_augment: bool = True,
        negative_augment: bool = True,
    ) -> torch.Tensor:
        window = self._extract_window(inputs)
        if window.ndim != 3:
            raise ValueError("CAROTS expects input shape [batch, window, channels].")
        if window.shape[1] != self.win_size:
            raise ValueError(f"CAROTS expected window length {self.win_size}, got {window.shape[1]}.")
        if window.shape[2] != self.input_dim:
            raise ValueError(f"CAROTS expected {self.input_dim} channels, got {window.shape[2]}.")
        if window.device.type == "meta":
            parts = [window]
            if positive_augment:
                parts.append(window)
            if negative_augment:
                parts.append(window)
            synthetic = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            return self.core.projector(self.core.encoder(synthetic))
        return self.core(window, positive_augment=positive_augment, negative_augment=negative_augment)


class CAROTSLoss(Loss):
    """CAROTS contrastive objective for the native network output layout."""

    def __init__(
        self,
        sim_threshold: float = 0.5,
        sim_threshold_start: float = 0.5,
        sim_threshold_end: float = 0.9,
        sim_threshold_schedule: bool = True,
        temperature: float = 0.1,
        positive_augment: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.sim_threshold = float(sim_threshold)
        self.sim_threshold_start = float(sim_threshold_start)
        self.sim_threshold_end = float(sim_threshold_end)
        self.sim_threshold_schedule = bool(sim_threshold_schedule)
        self.temperature = float(temperature)
        self.positive_augment = bool(positive_augment)

    @staticmethod
    def _normalize_predictions(predictions: Tuple[Any, ...]) -> torch.Tensor:
        normalized = predictions
        if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
            normalized = tuple(normalized[0])
        if len(normalized) != 1 or not isinstance(normalized[0], torch.Tensor):
            raise ValueError("CAROTSLoss expects one embedding tensor.")
        return normalized[0]

    def current_threshold(self, epoch: int, num_epochs: Optional[int]) -> float:
        if not self.sim_threshold_schedule or num_epochs is None or num_epochs <= 1:
            return self.sim_threshold
        return self.sim_threshold_start + (
            (self.sim_threshold_end - self.sim_threshold_start) * epoch / (num_epochs - 1)
        )

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        epoch: int = 0,
        num_epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del targets, args, kwargs
        embeddings = self._normalize_predictions(predictions)
        if embeddings.device.type == "meta":
            return embeddings.sum() * 0.0
        return carots_loss(
            embeddings,
            sim_threshold=self.current_threshold(int(epoch), num_epochs),
            temperature=self.temperature,
            positive_augment=self.positive_augment,
        )


class CAROTSAnomalyDetector(AnomalyDetector):
    """CAROTS scorer using a native network and the original two-stage training loop.

    CAROTS couples CUTS+ graph discovery, graph-conditioned positive
    augmentation, contrastive encoder training, and centroid scoring. The
    graph stage uses alternating optimizers, sampled Bernoulli/Gumbel graphs,
    and best-validation state restoration, so the detector keeps that custom
    fitting routine instead of delegating it to the shared Lightning loop.
    """

    def __init__(
        self,
        model: Optional[CAROTS] = None,
        loss: Optional[CAROTSLoss] = None,
        win_size: Optional[int] = None,
        input_step: int = 9,
        pred_step: int = 1,
        lr: float = 0.0001,
        num_epochs: int = 30,
        batch_size: int = 256,
        weight_decay: float = 0.0001,
        cuts_num_epochs: int = 50,
        cuts_tau_start: float = 1.0,
        cuts_tau_end: float = 0.1,
        graph_lr_start: float = 0.001,
        graph_lr_end: float = 0.0001,
        graph_lambda_start: float = 0.1,
        graph_lambda_end: float = 0.01,
        data_lr_start: float = 0.01,
        data_lr_end: float = 0.001,
        positive_augment: bool = True,
        sim_threshold: float = 0.5,
        sim_threshold_start: float = 0.5,
        sim_threshold_end: float = 0.9,
        sim_threshold_schedule: bool = True,
        temperature: float = 0.1,
        scorer_type: str = "l2",
        max_train_windows: Optional[int] = None,
        max_eval_windows: Optional[int] = None,
        blend_causal_score: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        nn.Module.__init__(self)
        self.register_buffer("dummy", torch.tensor([]), persistent=False)
        if win_size is None:
            raise ValueError("CAROTS detector requires win_size.")
        if input_step + pred_step != win_size:
            raise ValueError("CAROTS expects input_step + pred_step == win_size.")
        if scorer_type not in {"l2", "cos"}:
            raise ValueError("CAROTS scorer_type must be one of {'l2', 'cos'}.")

        self.model = model
        self.loss = loss or CAROTSLoss(
            sim_threshold=sim_threshold,
            sim_threshold_start=sim_threshold_start,
            sim_threshold_end=sim_threshold_end,
            sim_threshold_schedule=sim_threshold_schedule,
            temperature=temperature,
            positive_augment=positive_augment,
        )
        self.win_size = int(win_size)
        self.input_step = int(input_step)
        self.pred_step = int(pred_step)
        self.lr = float(lr)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.cuts_num_epochs = int(cuts_num_epochs)
        self.cuts_tau_start = float(cuts_tau_start)
        self.cuts_tau_end = float(cuts_tau_end)
        self.graph_lr_start = float(graph_lr_start)
        self.graph_lr_end = float(graph_lr_end)
        self.graph_lambda_start = float(graph_lambda_start)
        self.graph_lambda_end = float(graph_lambda_end)
        self.data_lr_start = float(data_lr_start)
        self.data_lr_end = float(data_lr_end)
        self.positive_augment = bool(positive_augment)
        self.scorer_type = str(scorer_type)
        self.max_train_windows = max_train_windows
        self.max_eval_windows = max_eval_windows
        self.blend_causal_score = bool(blend_causal_score)
        self.scaler = StandardScaler()
        self.cuts_best_state: Optional[dict[str, torch.Tensor]] = None
        self.best_state: Optional[dict[str, torch.Tensor]] = None
        self.centroid_wo_norm: Optional[torch.Tensor] = None
        self.centroid_norm: Optional[torch.Tensor] = None
        self.train_cl_stats = (0.0, 1.0)
        self.train_cd_stats = (0.0, 1.0)

    def _device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _require_model(self) -> CAROTS:
        if self.model is None:
            raise RuntimeError("CAROTSAnomalyDetector requires a CAROTS network.")
        return self.model

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del kwargs
        train_data = dataloader_to_frame(dataset)
        train_values, valid_values = train_val_split(train_data, 0.8, None)
        if len(valid_values) == 0:
            valid_values = train_values.copy()

        model = self._require_model()
        device = self._device()
        model.to(device)
        self.scaler.fit(train_values.to_numpy())
        train_scaled = transform_frame(self.scaler, train_values)
        valid_scaled = transform_frame(self.scaler, valid_values)

        train_windows = _limit_windows(
            _build_training_windows(train_scaled.to_numpy(), self.win_size),
            self.max_train_windows,
        )
        valid_windows = _limit_windows(
            _build_training_windows(valid_scaled.to_numpy(), self.win_size),
            self.max_eval_windows,
        )
        train_loader = make_window_loader(train_windows, batch_size=self.batch_size, shuffle=True)
        valid_loader = make_window_loader(valid_windows, batch_size=self.batch_size, shuffle=False)

        self._train_cuts_plus(train_loader, valid_loader)
        model.positive_augmentor.set_causal_discoverer(model.causal_discoverer)
        self._train_carots(train_loader, valid_loader)
        self._init_score_statistics(train_scaled)

    def _train_cuts_plus(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        model = self._require_model()
        device = self._device()
        causal = model.causal_discoverer
        mse = torch.nn.MSELoss()
        data_parameters = [param for name, param in causal.named_parameters() if name != "GT"]
        data_optimizer = torch.optim.Adam(data_parameters, lr=self.data_lr_start)
        data_gamma = (self.data_lr_end / self.data_lr_start) ** (1 / max(self.cuts_num_epochs, 1))
        data_scheduler = torch.optim.lr_scheduler.StepLR(data_optimizer, step_size=1, gamma=data_gamma)
        graph_gamma = (self.graph_lr_end / self.graph_lr_start) ** (1 / max(self.cuts_num_epochs, 1))
        tau = self.cuts_tau_start
        tau_gamma = (self.cuts_tau_end / self.cuts_tau_start) ** (1 / max(self.cuts_num_epochs, 1))
        lambda_s = self.graph_lambda_start
        lambda_gamma = (self.graph_lambda_end / self.graph_lambda_start) ** (1 / max(self.cuts_num_epochs, 1))
        identity_graph = torch.eye(causal.GT.shape[0], device=device)

        best_val = float("inf")
        for epoch in range(self.cuts_num_epochs):
            graph_optimizer = torch.optim.Adam(
                [causal.GT],
                lr=self.graph_lr_start * (graph_gamma ** epoch),
            )
            causal.train()

            for (batch,) in train_loader:
                batch = batch.to(device)
                x = batch[:, : self.input_step]
                y = batch[:, self.input_step :]
                graph = torch.einsum("nm,ml->nl", identity_graph, torch.sigmoid(causal.GT))
                sampled_graph = self._sample_bernoulli(graph, batch.size(0))
                prediction = causal(x, sampled_graph).transpose(1, 2)
                loss = mse(prediction, y)
                data_optimizer.zero_grad()
                loss.backward()
                data_optimizer.step()

            data_scheduler.step()

            for (batch,) in train_loader:
                batch = batch.to(device)
                x = batch[:, : self.input_step]
                y = batch[:, self.input_step :]
                graph = torch.einsum("nm,ml->nl", identity_graph, torch.sigmoid(causal.GT))
                sampled_graph = self._gumbel_sigmoid_sample(graph, batch.size(0), tau=tau)
                prediction = causal(x, sampled_graph).transpose(1, 2)
                loss_data = mse(prediction, y)
                loss_sparsity = torch.linalg.norm(graph.flatten(), ord=1) / graph.numel()
                loss = loss_data + lambda_s * loss_sparsity
                graph_optimizer.zero_grad()
                loss.backward()
                graph_optimizer.step()

            tau *= tau_gamma
            lambda_s *= lambda_gamma
            val_loss = self._eval_cuts(valid_loader, identity_graph)
            if val_loss < best_val:
                best_val = val_loss
                self.cuts_best_state = copy.deepcopy(causal.state_dict())

        if self.cuts_best_state is not None:
            causal.load_state_dict(self.cuts_best_state)
        causal.eval()

    def _eval_cuts(self, loader: DataLoader, identity_graph: torch.Tensor) -> float:
        causal = self._require_model().causal_discoverer
        device = self._device()
        causal.eval()
        losses = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                x = batch[:, : self.input_step]
                y = batch[:, self.input_step :]
                graph = torch.einsum("nm,ml->nl", identity_graph, torch.sigmoid(causal.GT))
                graph = graph[None].expand(batch.size(0), -1, -1)
                prediction = causal(x, graph).transpose(1, 2)
                losses.append(F.mse_loss(prediction, y).item())
        return float(np.mean(losses)) if losses else 0.0

    def _train_carots(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        model = self._require_model()
        device = self._device()
        for parameter in model.causal_discoverer.parameters():
            parameter.requires_grad = False

        optimizer = torch.optim.Adam(
            [
                param
                for name, param in model.named_parameters()
                if not name.startswith("core.causal_discoverer.")
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val = float("inf")
        for epoch in range(self.num_epochs):
            model.train()
            for (batch,) in train_loader:
                batch = batch.to(device)
                outputs = model(
                    batch,
                    positive_augment=self.positive_augment,
                    negative_augment=True,
                )
                loss = self.loss(
                    (outputs,),
                    (batch,),
                    epoch=epoch,
                    num_epochs=self.num_epochs,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = self._eval_carots(valid_loader, epoch)
            if val_loss < best_val:
                best_val = val_loss
                self.best_state = copy.deepcopy(model.state_dict())

        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        model.eval()

    def _eval_carots(self, loader: DataLoader, epoch: int) -> float:
        model = self._require_model()
        device = self._device()
        model.eval()
        losses = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                outputs = model(
                    batch,
                    positive_augment=self.positive_augment,
                    negative_augment=True,
                )
                losses.append(
                    self.loss(
                        (outputs,),
                        (batch,),
                        epoch=epoch,
                        num_epochs=self.num_epochs,
                    ).item()
                )
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    def _init_score_statistics(self, train_scaled: pd.DataFrame) -> None:
        train_windows = build_causal_windows(train_scaled.to_numpy(), self.win_size)
        self._init_centroids(train_windows)
        train_scores_cl = self._embedding_scores(train_windows)
        self.train_cl_stats = self._mean_std(train_scores_cl)
        if self.blend_causal_score:
            train_scores_cd = self._causal_scores(train_windows)
            self.train_cd_stats = self._mean_std(train_scores_cd)

    def _init_centroids(self, windows: np.ndarray) -> None:
        model = self._require_model()
        device = self._device()
        outputs = []
        loader = make_window_loader(windows, batch_size=self.batch_size, shuffle=False)
        model.eval()
        chunks = 1 + int(self.positive_augment) + 1
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                projected = model(
                    batch,
                    positive_augment=self.positive_augment,
                    negative_augment=True,
                )
                if projected.shape[0] % chunks != 0:
                    raise RuntimeError(
                        "CAROTS centroid init expected projection divisible by "
                        f"{chunks}, got {projected.shape[0]}"
                    )
                anchor_count = projected.shape[0] // chunks
                outputs.append(projected[:anchor_count])
        if outputs:
            merged = torch.cat(outputs, dim=0)
            self.centroid_wo_norm = merged.mean(dim=0, keepdim=True)
            self.centroid_norm = F.normalize(merged, p=2, dim=1).mean(dim=0, keepdim=True)
        else:
            dim = model.projector_output_dim
            self.centroid_wo_norm = torch.zeros((1, dim), device=device)
            self.centroid_norm = torch.zeros((1, dim), device=device)

    def _embedding_scores(self, windows: np.ndarray) -> np.ndarray:
        model = self._require_model()
        if self.centroid_wo_norm is None or self.centroid_norm is None:
            raise RuntimeError("CAROTS centroids are not initialised.")
        device = self._device()
        loader = make_window_loader(windows, batch_size=self.batch_size, shuffle=False)
        scores = []
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                projected = model(batch, positive_augment=False, negative_augment=False)
                batch_scores = distance_to_centroid(
                    projected,
                    centroid_wo_norm=self.centroid_wo_norm,
                    centroid_norm=self.centroid_norm,
                    metric=self.scorer_type,
                )
                scores.append(batch_scores.cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    def _causal_scores(self, windows: np.ndarray) -> np.ndarray:
        causal = self._require_model().causal_discoverer
        device = self._device()
        graph = (causal.causality_mtx > 0.5).float().to(device)
        loader = make_window_loader(windows, batch_size=self.batch_size, shuffle=False)
        scores = []
        causal.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                x = batch[:, : self.input_step]
                y = batch[:, self.input_step :]
                graph_batch = graph[None].expand(batch.size(0), -1, -1)
                prediction = causal(x, graph_batch).transpose(1, 2)
                batch_scores = F.mse_loss(prediction, y, reduction="none").mean(dim=(1, 2))
                scores.append(batch_scores.cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        frame = tensor_points_to_frame(inputs)
        scaled = transform_frame(self.scaler, frame)
        windows = build_causal_windows(scaled.to_numpy(), self.win_size)
        scores_cl = self._embedding_scores(windows)
        normalized_cl = self._normalize(scores_cl, self.train_cl_stats)
        if self.blend_causal_score:
            scores_cd = self._causal_scores(windows)
            normalized_cd = self._normalize(scores_cd, self.train_cd_stats)
            combined = normalized_cl + normalized_cd
        else:
            combined = normalized_cl
        return _align_scores(combined, len(frame)).to(inputs[0].device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]

    @staticmethod
    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        mean = float(np.mean(values)) if len(values) else 0.0
        std = float(np.std(values)) if len(values) else 1.0
        if std == 0.0:
            std = 1.0
        return mean, std

    @staticmethod
    def _normalize(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
        mean, std = stats
        return (values - mean) / std

    @staticmethod
    def _sample_bernoulli(graph: torch.Tensor, batch_size: int) -> torch.Tensor:
        return torch.bernoulli(graph[None].expand(batch_size, -1, -1)).float()

    @staticmethod
    def _gumbel_sigmoid_sample(graph: torch.Tensor, batch_size: int, tau: float) -> torch.Tensor:
        prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
        logits = torch.cat([prob, 1.0 - prob], dim=-1)
        gumbels = -torch.empty_like(logits).exponential_().log()
        soft = ((logits + gumbels) / tau).softmax(dim=-1)
        index = soft.max(dim=-1, keepdim=True)[1]
        hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (hard - soft.detach() + soft)[:, :, :, 0]


CAROTSAD = CAROTSAnomalyDetector

__all__ = ["CAROTS", "CAROTSAD", "CAROTSAnomalyDetector", "CAROTSLoss"]
