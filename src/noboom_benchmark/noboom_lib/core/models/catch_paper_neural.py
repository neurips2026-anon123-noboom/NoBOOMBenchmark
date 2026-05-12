"""Neural CATCH-paper baseline adapters for NoBoomBenchmark.

These classes cover the neural baselines requested from the official CATCH
benchmark source:

* DLinear/NLinear: CATCH time_series_library linear anomaly-detection models.
* iTransformer/PatchTST: CATCH time_series_library transformer models.
* ModernTCN/DualTF/TFAD: CATCH self_impl neural baselines.
* AutoEncoder: trainable window autoencoder baseline.

TODO: ModernTCN, DualTF, TFAD, iTransformer, PatchTST, and AutoEncoder are
NoBOOM-native reconstruction adapters rather than line-for-line ports of the
CATCH training harness. They preserve the NoBoom fit/score contract and the
CATCH anomaly-detection reconstruction surface where applicable, but should not
be used for paper-faithfulness claims without a dedicated source-code audit.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch._dynamo import disable as _dynamo_disable
except ImportError:  # pragma: no cover - torch without Dynamo.
    def _dynamo_disable(fn):
        return fn

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - lightweight test environments.
    BaseModel = nn.Module  # type: ignore[assignment]
if not issubclass(BaseModel, nn.Module):  # pragma: no cover - test stubs.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - lightweight test environments.
    try:
        from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore
    except ImportError:
        AnomalyDetector = nn.Module  # type: ignore[assignment]
if not issubclass(AnomalyDetector, nn.Module):  # pragma: no cover - test stubs.
    AnomalyDetector = nn.Module  # type: ignore[assignment]

try:
    from timesead.optim.loss import Loss
except ImportError:  # pragma: no cover - lightweight test environments.
    Loss = nn.Module  # type: ignore[assignment]
if not issubclass(Loss, nn.Module):  # pragma: no cover - test stubs.
    Loss = nn.Module  # type: ignore[assignment]


def _extract_window(inputs: Any) -> torch.Tensor:
    if isinstance(inputs, torch.Tensor):
        return inputs
    if isinstance(inputs, (tuple, list)) and inputs and isinstance(inputs[0], torch.Tensor):
        return inputs[0]
    raise ValueError("Expected an input tensor or tuple containing one tensor.")


def _validate_window(
    x: torch.Tensor,
    *,
    input_dim: int,
    seq_len: Optional[int],
    name: str,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"{name} expects input shape [batch, window, channels].")
    if seq_len is not None and x.shape[1] != seq_len:
        raise ValueError(f"{name} expected window length {seq_len}, got {x.shape[1]}.")
    if x.shape[2] != input_dim:
        raise ValueError(f"{name} expected {input_dim} channels, got {x.shape[2]}.")


def _require_seq_len(seq_len: Optional[int], name: str) -> int:
    if seq_len is None or seq_len <= 0:
        raise ValueError(f"{name} requires positive seq_len.")
    return int(seq_len)


def _require_input_dim(input_dim: int) -> int:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    return int(input_dim)


def _validate_heads(d_model: int, n_heads: int) -> None:
    if d_model <= 0:
        raise ValueError("d_model must be positive.")
    if n_heads <= 0:
        raise ValueError("n_heads must be positive.")
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")


def _position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(dim, 1))
    )
    encoding = torch.zeros(1, length, dim, device=device, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        encoding[0, :, 1::2] = torch.cos(position * div_term[: encoding[0, :, 1::2].shape[-1]])
    return encoding.to(dtype=dtype)


def _normalize_prediction(predictions: Tuple[Any, ...]) -> torch.Tensor:
    normalized = predictions
    if len(normalized) == 1 and isinstance(normalized[0], (tuple, list)):
        normalized = tuple(normalized[0])
    if len(normalized) != 1 or not isinstance(normalized[0], torch.Tensor):
        raise ValueError("NeuralReconstructionLoss expects one tensor prediction.")
    return normalized[0]


@_dynamo_disable
def _irfft_from_real_imag(
    real: torch.Tensor,
    imag: torch.Tensor,
    *,
    n: int,
    dim: int,
    norm: str,
) -> torch.Tensor:
    return torch.fft.irfft(torch.complex(real.contiguous(), imag.contiguous()), n=n, dim=dim, norm=norm)


class _GroupedBase(BaseModel):
    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)


class NeuralReconstructionLoss(Loss):
    """MSE reconstruction loss accepting NoBoom's tuple-packed predictions."""

    def __init__(self, reduction: str = "mean") -> None:
        try:
            super().__init__()
        except TypeError:
            nn.Module.__init__(self)
        self.reduction = reduction

    def forward(
        self,
        predictions: Tuple[Any, ...],
        targets: Tuple[torch.Tensor, ...],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del args, kwargs
        reconstruction = _normalize_prediction(predictions)
        target = targets[0].float()
        if reconstruction.shape != target.shape:
            if reconstruction.ndim == 2 and target.ndim == 3 and reconstruction.shape == target[:, -1, :].shape:
                target = target[:, -1, :]
            else:
                raise ValueError(
                    "Reconstruction shape "
                    f"{tuple(reconstruction.shape)} does not match target {tuple(target.shape)}."
                )
        return F.mse_loss(reconstruction, target, reduction=self.reduction)


class NeuralReconstructionAnomalyDetector(AnomalyDetector):
    """Score windows by last-step reconstruction MSE; larger means more anomalous."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        seq_len: Optional[int] = None,
        batch_size: int = 128,
        reduce_to_last_step: bool = True,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            nn.Module.__init__(self)
        self.model = model
        self.seq_len = seq_len
        self.batch_size = int(batch_size)
        self.reduce_to_last_step = bool(reduce_to_last_step)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        if self.model is None:
            raise RuntimeError("NeuralReconstructionAnomalyDetector requires a model.")

    @staticmethod
    def _extract_inputs(batch: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch = batch[0]
        if isinstance(batch, torch.Tensor):
            return (batch,)
        if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            return tuple(batch)
        raise ValueError("Detector expected a batch containing input tensors.")

    def _window_size(self) -> int:
        if self.seq_len is not None:
            return int(self.seq_len)
        if self.model is not None:
            for attr_name in ("seq_len", "window_size", "win_size", "n_window"):
                value = getattr(self.model, attr_name, None)
                if value is not None:
                    return int(value)
        raise RuntimeError("Detector requires seq_len or a model with a window-size attribute.")

    def _model_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _make_causal_windows(self, points: torch.Tensor) -> torch.Tensor:
        win_size = self._window_size()
        if points.ndim != 2:
            raise ValueError("Point inputs must have shape [batch, channels].")
        if points.shape[0] == 0:
            return points.new_empty((0, win_size, points.shape[-1]))
        windows = []
        first = points[:1]
        for end in range(points.shape[0]):
            start = max(0, end - win_size + 1)
            window = points[start : end + 1]
            if window.shape[0] < win_size:
                pad = first.expand(win_size - window.shape[0], -1)
                window = torch.cat([pad, window], dim=0)
            windows.append(window)
        return torch.stack(windows, dim=0)

    def _score_windows(self, windows: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("NeuralReconstructionAnomalyDetector requires a model.")
        source_device = windows.device
        model_device = self._model_device()
        was_training = self.model.training
        self.model.eval()
        scores = []
        try:
            with torch.no_grad():
                for start in range(0, windows.shape[0], self.batch_size):
                    batch = windows[start : start + self.batch_size].to(
                        device=model_device,
                        dtype=torch.float32,
                    )
                    outputs = self.model((batch,))
                    if isinstance(outputs, (tuple, list)):
                        if not outputs:
                            raise ValueError("Model returned an empty output tuple.")
                        outputs = outputs[0]
                    if not isinstance(outputs, torch.Tensor):
                        raise TypeError("Model output must be a tensor or tensor tuple.")
                    if outputs.shape == batch.shape:
                        per_step = (outputs - batch).pow(2).mean(dim=-1)
                        score = per_step[:, -1] if self.reduce_to_last_step else per_step.mean(dim=-1)
                    elif outputs.ndim == 2 and outputs.shape[0] == batch.shape[0]:
                        score = (outputs - batch[:, -1, :]).pow(2).mean(dim=-1)
                    else:
                        score = outputs.reshape(outputs.shape[0], -1).mean(dim=-1)
                    scores.append(score.detach().to(source_device))
        finally:
            self.model.train(was_training)
        if not scores:
            return torch.empty(0, dtype=torch.float32, device=source_device)
        return torch.cat(scores, dim=0)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        values = self._extract_inputs(inputs)[0].detach()
        if values.ndim == 2:
            windows = self._make_causal_windows(values.to(dtype=torch.float32))
        elif values.ndim == 3:
            windows = values.to(dtype=torch.float32)
        else:
            raise ValueError("Detector expects [batch, channels] or [batch, window, channels].")
        return self._score_windows(windows).to(device=values.device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


class MovingAverageDecomposition(nn.Module):
    """DLinear decomposition block adapted from CATCH's time_series_library."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class DLinear(_GroupedBase):
    """DLinear reconstruction model for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        moving_avg: int = 25,
        individual: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "DLinear")
        self.pred_len = self.seq_len
        self.individual = bool(individual)
        self.decomposition = MovingAverageDecomposition(moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_dim)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_dim)]
            )
            for seasonal, trend in zip(self.linear_seasonal, self.linear_trend):
                seasonal.weight = nn.Parameter(
                    (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
                trend.weight = nn.Parameter(
                    (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal.weight = nn.Parameter(
                (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
            )
            self.linear_trend.weight = nn.Parameter(
                (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
            )

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="DLinear")
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0),
                seasonal_init.size(1),
                self.pred_len,
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros_like(seasonal_output)
            for channel in range(self.input_dim):
                seasonal_output[:, channel, :] = self.linear_seasonal[channel](
                    seasonal_init[:, channel, :]
                )
                trend_output[:, channel, :] = self.linear_trend[channel](trend_init[:, channel, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)


class NLinear(_GroupedBase):
    """NLinear reconstruction model for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        individual: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "NLinear")
        self.pred_len = self.seq_len
        self.individual = bool(individual)
        if self.individual:
            self.linear = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_dim)]
            )
        else:
            self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="NLinear")
        seq_last = x[:, -1:, :].detach()
        centered = x - seq_last
        if self.individual:
            output = torch.zeros(
                centered.size(0),
                self.pred_len,
                centered.size(2),
                dtype=centered.dtype,
                device=centered.device,
            )
            for channel in range(self.input_dim):
                output[:, :, channel] = self.linear[channel](centered[:, :, channel])
        else:
            output = self.linear(centered.permute(0, 2, 1)).permute(0, 2, 1)
        return output + seq_last


class AutoEncoder(_GroupedBase):
    """Plain MLP window autoencoder baseline."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        hidden_dims: Sequence[int] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "AutoEncoder")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        flat_dim = self.seq_len * self.input_dim
        encoder_layers = []
        prev_dim = flat_dim
        for hidden in hidden_dims:
            hidden_dim = int(hidden)
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers.")
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU()])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, int(latent_dim)))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = int(latent_dim)
        for hidden in reversed(tuple(int(value) for value in hidden_dims)):
            decoder_layers.extend([nn.Linear(prev_dim, hidden), nn.GELU()])
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        decoder_layers.append(nn.Linear(prev_dim, flat_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="AutoEncoder")
        batch_size = x.shape[0]
        latent = self.encoder(x.reshape(batch_size, -1))
        return self.decoder(latent).view(batch_size, self.seq_len, self.input_dim)


Autoencoder = AutoEncoder


class ITransformer(_GroupedBase):
    """Inverted-token transformer reconstruction adapter."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        d_model: int = 128,
        e_layers: int = 1,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "ITransformer")
        _validate_heads(int(d_model), int(n_heads))
        if e_layers <= 0:
            raise ValueError("e_layers must be positive.")
        self.d_model = int(d_model)
        self.input_projection = nn.Linear(self.seq_len, self.d_model)
        self.channel_embedding = nn.Parameter(torch.zeros(1, self.input_dim, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff) if d_ff is not None else self.d_model * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(e_layers))
        self.output_projection = nn.Linear(self.d_model, self.seq_len)

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="ITransformer")
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev
        tokens = self.input_projection(normalized.permute(0, 2, 1))
        tokens = tokens + self.channel_embedding
        encoded = self.encoder(tokens)
        decoded = self.output_projection(encoded).permute(0, 2, 1)
        return decoded * stdev + means


iTransformer = ITransformer


def _patch_count(seq_len: int, patch_len: int, stride: int) -> int:
    if seq_len < patch_len:
        return 1
    return int((seq_len - patch_len) / stride + 2)


class PatchTST(_GroupedBase):
    """PatchTST-style patch transformer reconstruction adapter."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        e_layers: int = 2,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "PatchTST")
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive.")
        _validate_heads(int(d_model), int(n_heads))
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.num_patches = _patch_count(self.seq_len, self.patch_len, self.stride)
        self.patch_projection = nn.Linear(self.patch_len, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff) if d_ff is not None else self.d_model * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(e_layers))
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.d_model * self.num_patches, self.seq_len),
            nn.Dropout(dropout),
        )

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.stride
        if x.shape[-1] < self.patch_len:
            padding = self.patch_len - x.shape[-1]
        padded = F.pad(x, (0, padding))
        patches = padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[2] < self.num_patches:
            missing = self.num_patches - patches.shape[2]
            pad = patches[:, :, -1:, :].expand(-1, -1, missing, -1)
            patches = torch.cat([patches, pad], dim=2)
        return patches[:, :, : self.num_patches, :]

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="PatchTST")
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev
        channel_first = normalized.permute(0, 2, 1)
        patches = self._patches(channel_first)
        batch_size, channels, num_patches, _ = patches.shape
        tokens = patches.reshape(batch_size * channels, num_patches, self.patch_len)
        tokens = self.patch_projection(tokens)
        tokens = tokens + self.position_embedding[:, :num_patches, :]
        encoded = self.encoder(tokens)
        encoded = encoded.reshape(batch_size, channels, num_patches, self.d_model).permute(0, 1, 3, 2)
        decoded = self.head(encoded).permute(0, 2, 1)
        return decoded * stdev + means


class _TemporalTransformer(_GroupedBase):
    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int],
        name: str,
        d_model: int,
        e_layers: int,
        n_heads: int,
        d_ff: Optional[int],
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, name)
        _validate_heads(int(d_model), int(n_heads))
        self.name = name
        self.d_model = int(d_model)
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff) if d_ff is not None else self.d_model * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(e_layers))
        self.output_projection = nn.Linear(self.d_model, self.input_dim)

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name=self.name)
        h = self.input_projection(x)
        h = h + _position_encoding(x.shape[1], self.d_model, x.device, h.dtype)
        return self.output_projection(self.encoder(h))


class _ModernTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.pointwise(self.depthwise(x))
        if h.shape[-1] != residual.shape[-1]:
            h = F.interpolate(h, size=residual.shape[-1], mode="linear", align_corners=False)
        return residual + h


class ModernTCN(_GroupedBase):
    """ModernTCN-style temporal convolution reconstruction adapter."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        dims: Sequence[int] = (64, 64),
        num_blocks: Sequence[int] = (1, 1),
        large_size: Sequence[int] = (31, 15),
        small_size: Sequence[int] = (5, 5),
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        del kwargs, small_size
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "ModernTCN")
        hidden_dims = [int(value) for value in dims]
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("dims must contain positive integers.")
        blocks = [int(value) for value in num_blocks] or [1]
        kernels = [int(value) for value in large_size] or [31]
        self.stem = nn.Conv1d(self.input_dim, hidden_dims[0], kernel_size=1)
        layers = []
        current_dim = hidden_dims[0]
        for stage, hidden_dim in enumerate(hidden_dims):
            if hidden_dim != current_dim:
                layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=1))
                current_dim = hidden_dim
            stage_blocks = max(1, blocks[min(stage, len(blocks) - 1)])
            kernel = max(1, kernels[min(stage, len(kernels) - 1)])
            for _ in range(stage_blocks):
                layers.append(_ModernTCNBlock(current_dim, kernel, float(dropout)))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(current_dim, self.input_dim, kernel_size=1)

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="ModernTCN")
        h = self.stem(x.transpose(1, 2))
        h = self.backbone(h)
        h = self.head(h)
        if h.shape[-1] != x.shape[1]:
            h = F.interpolate(h, size=x.shape[1], mode="linear", align_corners=False)
        return h.transpose(1, 2)


class DualTF(_GroupedBase):
    """Dual time/frequency transformer reconstruction adapter."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        d_model: int = 64,
        e_layers: int = 1,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len, "DualTF")
        _validate_heads(int(d_model), int(n_heads))
        self.temporal = _TemporalTransformer(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            name="DualTF",
            d_model=int(d_model),
            e_layers=int(e_layers),
            n_heads=int(n_heads),
            d_ff=d_ff,
            dropout=float(dropout),
            activation=activation,
        )
        freq_len = self.seq_len // 2 + 1
        self.freq_projection = nn.Linear(self.input_dim * 2, int(d_model))
        freq_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_ff) if d_ff is not None else int(d_model) * 4,
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.freq_encoder = nn.TransformerEncoder(freq_layer, num_layers=int(e_layers))
        self.freq_head = nn.Linear(int(d_model), self.input_dim * 2)
        self.freq_position = nn.Parameter(torch.zeros(1, freq_len, int(d_model)))
        self.fusion = nn.Linear(self.input_dim * 2, self.input_dim)

    def _frequency_branch(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "meta":
            return torch.zeros_like(x)
        spectrum = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        features = torch.cat([spectrum.real, spectrum.imag], dim=-1)
        h = self.freq_projection(features) + self.freq_position[:, : features.shape[1], :]
        h = self.freq_head(self.freq_encoder(h))
        real, imag = torch.chunk(h.float(), 2, dim=-1)
        reconstructed = _irfft_from_real_imag(real, imag, n=x.shape[1], dim=1, norm="ortho")
        return reconstructed

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="DualTF")
        temporal = self.temporal((x,))
        frequency = self._frequency_branch(x)
        return self.fusion(torch.cat([temporal, frequency], dim=-1))


class TFAD(_GroupedBase):
    """TFAD-inspired TCN reconstruction adapter for the Lightning path."""

    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        n_window: Optional[int] = None,
        embedding_rep_dim: int = 64,
        tcn_kernel_size: int = 7,
        tcn_layers: int = 3,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.input_dim = _require_input_dim(input_dim)
        self.seq_len = _require_seq_len(seq_len if seq_len is not None else n_window, "TFAD")
        if embedding_rep_dim <= 0 or tcn_layers <= 0:
            raise ValueError("embedding_rep_dim and tcn_layers must be positive.")
        hidden = int(embedding_rep_dim)
        kernel = int(tcn_kernel_size)
        self.input_projection = nn.Conv1d(self.input_dim, hidden, kernel_size=1)
        layers = []
        for layer_idx in range(int(tcn_layers)):
            dilation = 2 ** layer_idx
            padding = dilation * (kernel - 1) // 2
            layers.extend(
                [
                    nn.Conv1d(
                        hidden,
                        hidden,
                        kernel_size=kernel,
                        padding=padding,
                        dilation=dilation,
                        groups=hidden,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.GELU(),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.output_projection = nn.Conv1d(hidden, self.input_dim, kernel_size=1)

    def forward(self, inputs: Any) -> torch.Tensor:
        x = _extract_window(inputs).float()
        _validate_window(x, input_dim=self.input_dim, seq_len=self.seq_len, name="TFAD")
        h = self.input_projection(x.transpose(1, 2))
        h = self.encoder(h)
        if h.shape[-1] != x.shape[1]:
            h = F.interpolate(h, size=x.shape[1], mode="linear", align_corners=False)
        return self.output_projection(h).transpose(1, 2)


ReconstructionAnomalyDetector = NeuralReconstructionAnomalyDetector
ReconstructionLoss = NeuralReconstructionLoss

__all__ = [
    "AutoEncoder",
    "Autoencoder",
    "DLinear",
    "DualTF",
    "ITransformer",
    "MovingAverageDecomposition",
    "ModernTCN",
    "NLinear",
    "NeuralReconstructionAnomalyDetector",
    "NeuralReconstructionLoss",
    "PatchTST",
    "ReconstructionAnomalyDetector",
    "ReconstructionLoss",
    "TFAD",
    "iTransformer",
]
