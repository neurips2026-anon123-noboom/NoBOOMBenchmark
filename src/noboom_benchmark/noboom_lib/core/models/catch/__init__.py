"""Native CATCH network, loss, and detector integration."""

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

from .._windowed_external import _align_scores
from .._source.baselines.catch.models.CATCH_model import CATCHModel
from .._source.baselines.catch.utils.fre_rec_loss import (
    frequency_criterion,
    frequency_loss,
)


class CATCHConfig:
    """Attribute config consumed by the vendored CATCH network."""

    def __init__(self, **kwargs: Any) -> None:
        defaults = {
            "Mlr": 1.0e-5,
            "e_layers": 3,
            "n_heads": 2,
            "cf_dim": 64,
            "d_ff": 256,
            "d_model": 128,
            "head_dim": 64,
            "individual": 0,
            "dropout": 0.2,
            "head_dropout": 0.1,
            "auxi_loss": "MAE",
            "auxi_type": "complex",
            "auxi_mode": "fft",
            "regular_lambda": 0.5,
            "temperature": 0.07,
            "patch_stride": 8,
            "patch_size": 16,
            "inference_patch_stride": 1,
            "inference_patch_size": 32,
            "module_first": True,
            "mask": False,
            "pct_start": 0.3,
            "revin": 1,
            "affine": 0,
            "subtract_last": 0,
            "lradj": "type1",
            "task_name": "anomaly_detection",
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)

    @property
    def pred_len(self) -> int:
        return self.seq_len


def _normalize_catch_predictions(
    predictions: Tuple[Any, ...],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = predictions
    if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
        normalized = tuple(normalized[0])
    if len(normalized) != 4:
        raise ValueError(f"CATCHLoss expects 4 prediction tensors, received {len(normalized)}.")
    reconstruction, output_complex, dcloss, normalized_target = normalized
    return reconstruction, output_complex, dcloss, normalized_target


class CATCH(BaseModel):
    """CATCH frequency-domain reconstruction network."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        Mlr: float = 1.0e-5,
        e_layers: int = 3,
        n_heads: int = 2,
        cf_dim: int = 64,
        d_ff: int = 256,
        d_model: int = 128,
        head_dim: int = 64,
        individual: int = 0,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
        auxi_loss: str = "MAE",
        auxi_type: str = "complex",
        auxi_mode: str = "fft",
        regular_lambda: float = 0.5,
        temperature: float = 0.07,
        patch_stride: int = 8,
        patch_size: int = 16,
        inference_patch_stride: int = 1,
        inference_patch_size: int = 32,
        module_first: bool = True,
        mask: bool = False,
        pct_start: float = 0.3,
        revin: int = 1,
        affine: int = 0,
        subtract_last: int = 0,
        lradj: str = "type1",
        **kwargs: Any,
    ) -> None:
        del kwargs, revin
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if seq_len is None:
            raise ValueError("CATCH requires seq_len.")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if patch_size <= 0 or patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be positive.")
        if patch_size > seq_len:
            raise ValueError("patch_size must be <= seq_len.")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.Mlr = float(Mlr)
        self.config = CATCHConfig(
            c_in=self.input_dim,
            enc_in=self.input_dim,
            dec_in=self.input_dim,
            c_out=self.input_dim,
            label_len=48,
            seq_len=self.seq_len,
            Mlr=self.Mlr,
            e_layers=e_layers,
            n_heads=n_heads,
            cf_dim=cf_dim,
            d_ff=d_ff,
            d_model=d_model,
            head_dim=head_dim,
            individual=individual,
            dropout=dropout,
            head_dropout=head_dropout,
            auxi_loss=auxi_loss,
            auxi_type=auxi_type,
            auxi_mode=auxi_mode,
            regular_lambda=regular_lambda,
            temperature=temperature,
            patch_stride=patch_stride,
            patch_size=patch_size,
            inference_patch_stride=inference_patch_stride,
            inference_patch_size=inference_patch_size,
            module_first=module_first,
            mask=mask,
            pct_start=pct_start,
            affine=affine,
            subtract_last=subtract_last,
            lradj=lradj,
        )
        self.network = CATCHModel(self.config)

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        main_params = [
            parameter
            for name, parameter in self.named_parameters()
            if "mask_generator" not in name and parameter.requires_grad
        ]
        mask_params = [
            parameter
            for name, parameter in self.named_parameters()
            if "mask_generator" in name and parameter.requires_grad
        ]
        if not mask_params:
            return (main_params,)
        return (({"params": main_params}, {"params": mask_params, "lr": self.Mlr}),)

    @staticmethod
    def _extract_window(inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
            return inputs[0]
        raise ValueError("CATCH expected a window tensor.")

    def forward(self, inputs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self._extract_window(inputs).float()
        if window.ndim != 3:
            raise ValueError("CATCH expects input shape [batch, window, channels].")
        if window.shape[1] != self.seq_len:
            raise ValueError(f"CATCH expected window length {self.seq_len}, got {window.shape[1]}.")
        if window.shape[2] != self.input_dim:
            raise ValueError(f"CATCH expected {self.input_dim} channels, got {window.shape[2]}.")

        reconstruction, output_complex, dcloss = self.network(window)
        normalized_target = self.network.revin_layer(window, "transform")
        return reconstruction, output_complex, dcloss, normalized_target


class CATCHLoss(Loss):
    """CATCH reconstruction, frequency, and channel-discovering objective."""

    def __init__(
        self,
        auxi_loss: str = "MAE",
        auxi_type: str = "complex",
        auxi_mode: str = "fft",
        auxi_lambda: float = 0.005,
        dc_lambda: float = 0.005,
        module_first: bool = True,
        mask: bool = False,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            nn.Module.__init__(self)
        self.config = CATCHConfig(
            auxi_loss=auxi_loss,
            auxi_type=auxi_type,
            auxi_mode=auxi_mode,
            auxi_lambda=auxi_lambda,
            dc_lambda=dc_lambda,
            module_first=module_first,
            mask=mask,
            seq_len=1,
        )
        self.reconstruction_loss = nn.MSELoss()
        self.frequency_loss = frequency_loss(self.config)
        self.auxi_lambda = float(auxi_lambda)
        self.dc_lambda = float(dc_lambda)

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        reconstruction, output_complex, dcloss, normalized_target = _normalize_catch_predictions(predictions)
        target = targets[0].float()
        rec_loss = self.reconstruction_loss(reconstruction, target)
        auxi_loss = self.frequency_loss(output_complex, normalized_target)
        return rec_loss + self.dc_lambda * dcloss + self.auxi_lambda * auxi_loss


class CATCHFrequencyCriterion(nn.Module):
    """Pointwise frequency reconstruction score used by CATCH detection."""

    def __init__(
        self,
        seq_len: int,
        inference_patch_size: int = 32,
        inference_patch_stride: int = 1,
        auxi_loss: str = "MAE",
        auxi_type: str = "complex",
        auxi_mode: str = "fft",
        module_first: bool = True,
        mask: bool = False,
    ) -> None:
        super().__init__()
        self.config = CATCHConfig(
            seq_len=seq_len,
            inference_patch_size=inference_patch_size,
            inference_patch_stride=inference_patch_stride,
            auxi_loss=auxi_loss,
            auxi_type=auxi_type,
            auxi_mode=auxi_mode,
            module_first=module_first,
            mask=mask,
        )
        self.criterion = frequency_criterion(self.config)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, targets)


class CATCHAnomalyDetector(AnomalyDetector):
    """CATCH scorer using a trained native CATCH network."""

    def __init__(
        self,
        model: Optional[CATCH] = None,
        seq_len: Optional[int] = None,
        batch_size: int = 128,
        score_lambda: float = 0.05,
        inference_patch_stride: int = 1,
        inference_patch_size: int = 32,
        auxi_loss: str = "MAE",
        auxi_type: str = "complex",
        auxi_mode: str = "fft",
        module_first: bool = True,
        mask: bool = False,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            nn.Module.__init__(self)
        self.model = model
        self.seq_len = seq_len
        self.batch_size = int(batch_size)
        self.score_lambda = float(score_lambda)
        self.frequency_criterion = CATCHFrequencyCriterion(
            seq_len=1 if seq_len is None else int(seq_len),
            inference_patch_size=inference_patch_size,
            inference_patch_stride=inference_patch_stride,
            auxi_loss=auxi_loss,
            auxi_type=auxi_type,
            auxi_mode=auxi_mode,
            module_first=module_first,
            mask=mask,
        )
        self._frequency_kwargs = {
            "inference_patch_size": inference_patch_size,
            "inference_patch_stride": inference_patch_stride,
            "auxi_loss": auxi_loss,
            "auxi_type": auxi_type,
            "auxi_mode": auxi_mode,
            "module_first": module_first,
            "mask": mask,
        }

    def _window_size(self) -> int:
        if self.seq_len is not None:
            return int(self.seq_len)
        if self.model is not None:
            return int(self.model.seq_len)
        raise RuntimeError("CATCHAnomalyDetector requires seq_len or model.")

    @staticmethod
    def _extract_inputs(batch: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch = batch[0]
        if isinstance(batch, torch.Tensor):
            return (batch,)
        if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            return tuple(batch)
        raise ValueError("CATCHAnomalyDetector expected a batch containing input tensors.")

    def _points_from_inputs(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        values = inputs[0].detach().to(dtype=torch.float32)
        if values.ndim == 3:
            return values[:, -1, :]
        if values.ndim == 2:
            return values
        raise ValueError("CATCHAnomalyDetector expects [batch, channels] or [batch, window, channels].")

    def _model_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        if self.model is None:
            raise RuntimeError("CATCHAnomalyDetector requires a trained CATCH model.")

    def _make_windows(self, points: torch.Tensor) -> torch.Tensor:
        win_size = self._window_size()
        if points.shape[0] < win_size:
            return points.new_empty((0, win_size, points.shape[-1]))
        return points.unfold(dimension=0, size=win_size, step=1).permute(0, 2, 1).contiguous()

    def _score_windows(self, windows: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("CATCHAnomalyDetector requires a trained CATCH model.")
        device = self._model_device()
        was_training = self.model.training
        self.model.eval()
        scores = []
        temporal_criterion = nn.MSELoss(reduction="none")
        criterion_config = getattr(self.frequency_criterion, "config", None)
        if criterion_config is not None and criterion_config.seq_len != self._window_size():
            self.frequency_criterion = CATCHFrequencyCriterion(
                seq_len=self._window_size(),
                **self._frequency_kwargs,
            )
        freq_criterion = self.frequency_criterion.to(device)
        try:
            with torch.no_grad():
                for start in range(0, windows.shape[0], self.batch_size):
                    batch = windows[start : start + self.batch_size].to(device=device, dtype=torch.float32)
                    reconstruction, _, _, _ = self.model((batch,))
                    temp_score = temporal_criterion(batch, reconstruction).mean(dim=-1)
                    freq_score = freq_criterion(batch, reconstruction).mean(dim=-1)
                    score = temp_score + self.score_lambda * freq_score
                    scores.append(score.detach().cpu())
        finally:
            self.model.train(was_training)
        if not scores:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(scores, dim=0).reshape(-1)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        points = self._points_from_inputs(inputs)
        windows = self._make_windows(points)
        scores = self._score_windows(windows)
        aligned = _align_scores(scores.numpy(), points.shape[0])
        return aligned.to(device=inputs[0].device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


CATCHAD = CATCHAnomalyDetector

__all__ = [
    "CATCH",
    "CATCHAD",
    "CATCHAnomalyDetector",
    "CATCHConfig",
    "CATCHFrequencyCriterion",
    "CATCHLoss",
]
