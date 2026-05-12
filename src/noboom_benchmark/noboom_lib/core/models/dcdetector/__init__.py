"""Native DCdetector network, loss, and detector components."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - test stubs install this when needed.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common.anomaly_detector import AnomalyDetector
except ImportError:  # pragma: no cover - test stubs install this when needed.
    from timesead.models.common import AnomalyDetector  # type: ignore

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - test stubs install this when needed.
    Loss = nn.Module  # type: ignore[assignment]

from .._source.baselines.self_impl.DCdetector.DCdetector_model import DCdetector_model


def _as_patch_sizes(patch_size: Sequence[int]) -> list[int]:
    patch_sizes = [int(value) for value in patch_size]
    if not patch_sizes:
        raise ValueError("patch_size must contain at least one value.")
    if any(value <= 0 for value in patch_sizes):
        raise ValueError("patch_size must contain only positive integers.")
    return patch_sizes


def _validate_divisible(win_size: int, patch_size: Sequence[int]) -> None:
    for ps in patch_size:
        if win_size % ps != 0:
            raise ValueError(
                f"DCdetector: win_size={win_size} is not divisible by "
                f"patch_size element {ps} (win_size % patch_size must equal 0)."
            )


def dc_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL term used by the original DCdetector implementation."""

    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def _normalize_prior(prior: torch.Tensor) -> torch.Tensor:
    return prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).repeat(
        1, 1, 1, prior.shape[-1]
    )


class DCdetector(BaseModel):
    """NoBoom-native wrapper around the DCdetector attention network."""

    def __init__(
        self,
        win_size: int,
        input_dim: int,
        patch_size: Sequence[int] = (5,),
        n_heads: int = 1,
        e_layers: int = 3,
        d_model: int = 256,
        rec_timeseries: bool = True,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        del rec_timeseries
        if win_size <= 0:
            raise ValueError("win_size must be positive.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        self.win_size = int(win_size)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim) if output_dim is not None else self.input_dim
        self.patch_size = _as_patch_sizes(patch_size)
        _validate_divisible(self.win_size, self.patch_size)

        self.network = DCdetector_model(
            win_size=self.win_size,
            enc_in=self.input_dim,
            c_out=self.output_dim,
            n_heads=n_heads,
            d_model=d_model,
            e_layers=e_layers,
            patch_size=self.patch_size,
            channel=self.input_dim,
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        x, = inputs
        if x.ndim != 3:
            raise ValueError("DCdetector expects input shape [batch, window, channels].")
        return self.network(x.float())


class DCdetectorLoss(Loss):
    """Contrastive association discrepancy objective from the source training loop."""

    def __init__(self, win_size: Optional[int] = None) -> None:
        try:
            super().__init__()
        except TypeError:
            nn.Module.__init__(self)
        self.win_size = win_size

    def forward(
        self,
        predictions: Tuple[object, ...],
        targets: Tuple[torch.Tensor, ...],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del targets, args, kwargs
        normalized = predictions
        if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
            normalized = tuple(normalized[0])
        if len(normalized) != 2:
            raise ValueError("DCdetectorLoss expects predictions=(series, prior).")

        series_list, prior_list = normalized
        if len(prior_list) == 0:
            raise ValueError("DCdetectorLoss received no prior attention tensors.")

        series_loss = 0.0
        prior_loss = 0.0
        for series, prior in zip(series_list, prior_list):
            prior_normalized = _normalize_prior(prior)
            series_loss = series_loss + torch.mean(
                dc_kl_loss(series, prior_normalized.detach())
            ) + torch.mean(dc_kl_loss(prior_normalized.detach(), series))
            prior_loss = prior_loss + torch.mean(
                dc_kl_loss(prior_normalized, series.detach())
            ) + torch.mean(dc_kl_loss(series.detach(), prior_normalized))

        return (prior_loss - series_loss) / len(prior_list)


class DCdetectorAnomalyDetector(AnomalyDetector):
    """Attention-energy scorer using a trained DCdetector network."""

    def __init__(
        self,
        model: Optional[DCdetector] = None,
        temperature: float = 50.0,
        batch_size: int = 128,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            nn.Module.__init__(self)
        if not hasattr(self, "dummy"):
            if hasattr(self, "register_buffer"):
                self.register_buffer("dummy", torch.tensor([]), persistent=False)
            else:
                self.dummy = torch.tensor([])
        self.model = model
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        return None

    def _energy(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DCdetectorAnomalyDetector requires a trained model.")

        parameters = getattr(self.model, "parameters", None)
        buffers = getattr(self.model, "buffers", None)
        model_anchor = next(parameters(), None) if callable(parameters) else None
        if model_anchor is None and callable(buffers):
            model_anchor = next(buffers(), None)
        model_device = model_anchor.device if model_anchor is not None else inputs[0].device
        x = inputs[0].to(device=model_device, dtype=torch.float32)
        series_list, prior_list = self.model((x,))
        series_loss = 0.0
        prior_loss = 0.0
        for series, prior in zip(series_list, prior_list):
            prior_normalized = _normalize_prior(prior)
            series_loss = series_loss + (
                dc_kl_loss(series, prior_normalized.detach()) * self.temperature
            )
            prior_loss = prior_loss + (
                dc_kl_loss(prior_normalized, series.detach()) * self.temperature
            )
        return torch.softmax((-series_loss - prior_loss), dim=-1)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        source = inputs[0]
        if source.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32, device=source.device)

        scores = []
        with torch.no_grad():
            for start in range(0, source.shape[0], self.batch_size):
                batch = source[start : start + self.batch_size]
                energy = self._energy((batch,))
                flat_energy = energy.reshape(-1)
                scores.append(flat_energy[-batch.shape[0] :].to(source.device))
        return torch.cat(scores, dim=0)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


DCdetectorAD = DCdetectorAnomalyDetector

__all__ = [
    "DCdetector",
    "DCdetectorAD",
    "DCdetectorAnomalyDetector",
    "DCdetectorLoss",
    "dc_kl_loss",
]
