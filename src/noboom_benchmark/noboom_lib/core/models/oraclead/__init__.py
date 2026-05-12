"""Native OracleAD network, loss, and detector integration."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - test stubs install this when needed.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - test stubs install this when needed.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - test stubs install this when needed.
    Loss = nn.Module  # type: ignore[assignment]

from .._windowed_external import build_causal_windows
from .._source.baselines.self_impl.OracleAD.model import (
    OracleADCore,
    oracle_deviation_score,
    oracle_loss,
    oracle_prediction_score,
    pairwise_l2_distances,
)


class OracleAD(BaseModel):
    """OracleAD causal per-channel encoder with relational latent attention."""

    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if win_size is None:
            raise ValueError("OracleAD requires win_size.")
        if win_size < 2:
            raise ValueError("OracleAD requires win_size >= 2.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_layers < 1:
            raise ValueError("OracleAD requires num_layers >= 1.")

        self.input_dim = int(input_dim)
        self.win_size = int(win_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.core = OracleADCore(
            num_channels=self.input_dim,
            hidden_dim=self.hidden_dim,
            history_length=self.win_size - 1,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=float(dropout),
        )

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)

    @staticmethod
    def _extract_window(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        raise ValueError("OracleAD expected a window tensor.")

    def forward(self, inputs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self._extract_window(inputs)
        if window.ndim != 3:
            raise ValueError("OracleAD expects input shape [batch, window, channels].")
        if window.shape[1] != self.win_size:
            raise ValueError(f"OracleAD expected window length {self.win_size}, got {window.shape[1]}.")
        if window.shape[2] != self.input_dim:
            raise ValueError(f"OracleAD expected {self.input_dim} channels, got {window.shape[2]}.")
        return self.core(window[:, :-1, :])


class OracleADLoss(Loss):
    """Composite OracleAD objective with lagged epoch-mean SLS regularization."""

    def __init__(
        self,
        reconstruction_weight: float = 0.1,
        structure_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.structure_weight = float(structure_weight)
        self.sls: Optional[torch.Tensor] = None
        self._active_epoch: Optional[int] = None
        self._pending_distance_sum: Optional[torch.Tensor] = None
        self._pending_distance_count = 0

    @staticmethod
    def _normalize_predictions(predictions: Tuple[Any, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = predictions
        if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
            normalized = tuple(normalized[0])
        if len(normalized) != 3:
            raise ValueError(f"OracleADLoss expects 3 prediction tensors, received {len(normalized)}.")
        reconstruction, prediction, refined = normalized
        return reconstruction, prediction, refined

    @staticmethod
    def _target_window(targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim != 3:
            raise ValueError("OracleADLoss expects target shape [batch, window, channels].")
        return target

    def _pending_sls(self) -> Optional[torch.Tensor]:
        if self._pending_distance_sum is None or self._pending_distance_count == 0:
            return None
        return self._pending_distance_sum / self._pending_distance_count

    def current_sls(self) -> Optional[torch.Tensor]:
        """Return the newest available SLS estimate for detector scoring."""

        pending = self._pending_sls()
        return pending if pending is not None else self.sls

    def _roll_epoch(self, epoch: int) -> None:
        if self._active_epoch == epoch:
            return
        pending = self._pending_sls()
        if pending is not None:
            self.sls = pending.detach()
        self._pending_distance_sum = None
        self._pending_distance_count = 0
        self._active_epoch = int(epoch)

    def _accumulate_distances(self, distances: torch.Tensor) -> None:
        detached = distances.detach()
        batch_sum = detached.sum(dim=0)
        if self._pending_distance_sum is None:
            self._pending_distance_sum = batch_sum
        else:
            self._pending_distance_sum = self._pending_distance_sum + batch_sum
        self._pending_distance_count += int(detached.shape[0])

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args
        epoch = kwargs.get("epoch")
        if epoch is not None:
            self._roll_epoch(int(epoch))

        reconstruction, prediction, refined = self._normalize_predictions(predictions)
        target_window = self._target_window(targets)
        history = target_window[:, :-1, :]
        target = target_window[:, -1, :]
        distances = pairwise_l2_distances(refined)
        sls = self.sls if epoch is not None else self.current_sls()
        loss = oracle_loss(
            reconstruction=reconstruction,
            history=history,
            prediction=prediction,
            target=target,
            distances=distances,
            sls=sls.to(distances.device) if sls is not None else None,
            reconstruction_weight=self.reconstruction_weight,
            structure_weight=self.structure_weight,
        )
        if epoch is not None:
            self._accumulate_distances(distances)
        return loss


class OracleADAnomalyDetector(AnomalyDetector):
    """Point scorer using a trained OracleAD network and fitted SLS matrix."""

    def __init__(
        self,
        model: Optional[OracleAD] = None,
        loss: Optional[OracleADLoss] = None,
        win_size: Optional[int] = None,
        batch_size: int = 1024,
        deviation_weight: float = 1.0,
        score_fusion: str = "multiply",
    ) -> None:
        super().__init__()
        if score_fusion not in {"multiply", "add"}:
            raise ValueError("score_fusion must be one of {'multiply', 'add'}.")
        self.model = model
        self.loss = loss
        self.win_size = win_size
        self.batch_size = int(batch_size)
        self.deviation_weight = float(deviation_weight)
        self.score_fusion = score_fusion
        self.sls: Optional[torch.Tensor] = None

    @staticmethod
    def _extract_inputs(batch: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch = batch[0]
        if isinstance(batch, torch.Tensor):
            return (batch,)
        if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            return tuple(batch)
        raise ValueError("OracleADAnomalyDetector expected a batch containing input tensors.")

    def _window_size(self) -> int:
        if self.win_size is not None:
            return int(self.win_size)
        if self.model is not None:
            return int(self.model.win_size)
        raise RuntimeError("OracleADAnomalyDetector requires win_size or model.")

    def _prepare_windows(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        values = inputs[0].detach()
        if values.ndim == 3:
            if values.shape[1] < 2:
                raise ValueError("OracleAD windows must contain at least two steps.")
            return values.to(dtype=torch.float32)
        if values.ndim == 2:
            windows = build_causal_windows(values.cpu().numpy(), self._window_size())
            return torch.as_tensor(windows, dtype=torch.float32, device=values.device)
        raise ValueError("OracleADAnomalyDetector expects [batch, channels] or [batch, window, channels].")

    def _model_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs: Any) -> None:
        del kwargs
        if self.model is None:
            raise RuntimeError("OracleADAnomalyDetector requires a trained OracleAD model.")

        if self.loss is not None and self.loss.current_sls() is not None:
            self.sls = self.loss.current_sls().detach().clone()
            return

        was_training = self.model.training
        self.model.eval()
        distance_sum: Optional[torch.Tensor] = None
        distance_count = 0
        device = self._model_device()
        with torch.no_grad():
            for batch in dataset:
                windows = self._prepare_windows(self._extract_inputs(batch)).to(device)
                _, _, refined = self.model((windows,))
                distances = pairwise_l2_distances(refined).detach().cpu()
                batch_sum = distances.sum(dim=0)
                if distance_sum is None:
                    distance_sum = batch_sum
                else:
                    distance_sum = distance_sum + batch_sum
                distance_count += int(distances.shape[0])
        if was_training:
            self.model.train()
        if distance_sum is None or distance_count == 0:
            raise RuntimeError("OracleADAnomalyDetector could not fit SLS from an empty dataset.")
        self.sls = distance_sum / distance_count

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("OracleADAnomalyDetector requires a trained OracleAD model.")
        if self.sls is None:
            raise RuntimeError("OracleADAnomalyDetector must be fitted before scoring.")

        windows = self._prepare_windows(inputs)
        scores = []
        was_training = self.model.training
        self.model.eval()
        device = self._model_device()
        with torch.no_grad():
            for start in range(0, windows.shape[0], self.batch_size):
                batch = windows[start : start + self.batch_size].to(device)
                _, prediction, refined = self.model((batch,))
                target = batch[:, -1, :]
                distances = pairwise_l2_distances(refined)
                prediction_score = oracle_prediction_score(prediction, target)
                deviation_score = oracle_deviation_score(distances, self.sls.to(device))
                if self.score_fusion == "multiply":
                    batch_scores = prediction_score * deviation_score
                else:
                    batch_scores = prediction_score + self.deviation_weight * deviation_score
                scores.append(batch_scores.detach().to(inputs[0].device))
        if was_training:
            self.model.train()
        if not scores:
            return torch.empty(0, dtype=torch.float32, device=inputs[0].device)
        return torch.cat(scores, dim=0).to(dtype=torch.float32)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


OracleADAD = OracleADAnomalyDetector

__all__ = ["OracleAD", "OracleADAD", "OracleADAnomalyDetector", "OracleADLoss"]
