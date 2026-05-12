"""Native Anomaly Transformer network, loss, and detector."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from timesead.models import BaseModel
from timesead.models.common import AnomalyDetector
from timesead.optim.loss import Loss

from .._source.baselines.self_impl.Anomaly_trans.AnomalyTransformer_model import (
    AnomalyTransformer_model,
)


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL term used by the original Anomaly Transformer implementation."""

    p = p.float()
    q = q.float()
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def _normalize_prior(prior: torch.Tensor, win_size: int) -> torch.Tensor:
    prior = prior.float()
    denominator = torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).clamp_min(torch.finfo(prior.dtype).eps)
    return prior / denominator.repeat(1, 1, 1, win_size)


def _normalize_predictions(
    predictions: Tuple[Any, ...],
) -> Tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    normalized: Tuple[Any, ...] = predictions
    if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
        normalized = tuple(normalized[0])
    if len(normalized) != 4:
        raise ValueError(
            f"AnomalyTransformer expects 4 prediction entries, received {len(normalized)}."
        )
    reconstruction, series, prior, sigmas = normalized
    return reconstruction, list(series), list(prior), list(sigmas)


def association_discrepancy_losses(
    series: list[torch.Tensor],
    prior: list[torch.Tensor],
    win_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the original series/prior discrepancy losses with detach semantics."""

    if len(prior) == 0:
        raise ValueError("AnomalyTransformer returned no attention prior tensors.")

    series_loss = torch.zeros((), dtype=prior[0].dtype, device=prior[0].device)
    prior_loss = torch.zeros((), dtype=prior[0].dtype, device=prior[0].device)
    for series_layer, prior_layer in zip(series, prior):
        normalized_prior = _normalize_prior(prior_layer, win_size)
        series_loss = series_loss + torch.mean(
            my_kl_loss(series_layer, normalized_prior.detach())
        ) + torch.mean(my_kl_loss(normalized_prior.detach(), series_layer))
        prior_loss = prior_loss + torch.mean(
            my_kl_loss(normalized_prior, series_layer.detach())
        ) + torch.mean(my_kl_loss(series_layer.detach(), normalized_prior))

    divisor = len(prior)
    return series_loss / divisor, prior_loss / divisor


def anomaly_transformer_energy(
    x: torch.Tensor,
    predictions: Tuple[Any, ...],
    win_size: int,
    temperature: float = 50.0,
) -> torch.Tensor:
    """Compute the continuous attention-energy score used by detect_score."""

    reconstruction, series, prior, _sigmas = _normalize_predictions(predictions)
    rec_loss = F.mse_loss(reconstruction, x, reduction="none").mean(dim=-1)

    series_loss = torch.zeros_like(rec_loss)
    prior_loss = torch.zeros_like(rec_loss)
    for series_layer, prior_layer in zip(series, prior):
        normalized_prior = _normalize_prior(prior_layer, win_size)
        series_loss = series_loss + my_kl_loss(
            series_layer,
            normalized_prior.detach(),
        ) * temperature
        prior_loss = prior_loss + my_kl_loss(
            normalized_prior,
            series_layer.detach(),
        ) * temperature

    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    return metric * rec_loss


class AnomalyTransformer(BaseModel):
    """NoBoom wrapper around the Anomaly Transformer reconstruction network."""

    def __init__(
        self,
        win_size: int,
        enc_in: int,
        c_out: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        output_attention: bool = True,
    ) -> None:
        super().__init__()
        if win_size <= 0:
            raise ValueError("win_size must be positive.")
        if enc_in <= 0:
            raise ValueError("enc_in must be positive.")
        if c_out <= 0:
            raise ValueError("c_out must be positive.")
        if e_layers <= 0:
            raise ValueError("e_layers must be positive.")

        self.win_size = int(win_size)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.network = AnomalyTransformer_model(
            win_size=self.win_size,
            enc_in=self.enc_in,
            c_out=self.c_out,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=output_attention,
        )

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[Any, ...]:
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        return self.network(x)


class AnomalyTransformerLoss(Loss):
    """Original minimax objective expressed as one Lightning loss scalar."""

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.k = int(k)

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        reconstruction, series, prior, _sigmas = _normalize_predictions(predictions)
        target = targets[0]
        win_size = int(reconstruction.shape[1])
        series_loss, prior_loss = association_discrepancy_losses(series, prior, win_size)
        rec_loss = F.mse_loss(reconstruction, target)
        return 2.0 * rec_loss - self.k * series_loss + self.k * prior_loss


class AnomalyTransformerAnomalyDetector(AnomalyDetector):
    """Continuous attention-energy scorer over a trained Anomaly Transformer."""

    def __init__(
        self,
        model: Optional[AnomalyTransformer] = None,
        temperature: float = 50.0,
        batch_first: bool = True,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = float(temperature)
        self.batch_first = bool(batch_first)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        return None

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("AnomalyTransformerAnomalyDetector requires a model.")
        source_device = inputs[0].device
        parameters = getattr(self.model, "parameters", None)
        buffers = getattr(self.model, "buffers", None)
        model_anchor = next(parameters(), None) if callable(parameters) else None
        if model_anchor is None and callable(buffers):
            model_anchor = next(buffers(), None)
        model_device = model_anchor.device if model_anchor is not None else source_device
        source = inputs[0]
        if source.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32, device=source_device)
        if not self.batch_first:
            x = source.to(device=model_device, dtype=torch.float32)
            with torch.no_grad():
                predictions = self.model((x,))
                scores = anomaly_transformer_energy(
                    x,
                    predictions,
                    win_size=self.model.win_size,
                    temperature=self.temperature,
                )
            return scores[-1].to(source_device)

        chunks = []
        with torch.no_grad():
            for start in range(0, source.shape[0], self.batch_size):
                x = source[start : start + self.batch_size].to(device=model_device, dtype=torch.float32)
                predictions = self.model((x,))
                scores = anomaly_transformer_energy(
                    x,
                    predictions,
                    win_size=self.model.win_size,
                    temperature=self.temperature,
                )
                chunks.append(scores[:, -1].to(source_device))
        return torch.cat(chunks, dim=0)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1] if self.batch_first else target[-1]


AnomalyTransformerAD = AnomalyTransformerAnomalyDetector

__all__ = [
    "AnomalyTransformer",
    "AnomalyTransformerAD",
    "AnomalyTransformerAnomalyDetector",
    "AnomalyTransformerLoss",
    "anomaly_transformer_energy",
    "association_discrepancy_losses",
    "my_kl_loss",
]
