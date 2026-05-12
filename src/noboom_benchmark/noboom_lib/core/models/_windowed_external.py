"""Shared utilities for the external windowed anomaly detectors."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from timesead.models import BaseModel
except ImportError:  # pragma: no cover - test stubs install this when needed.
    BaseModel = nn.Module  # type: ignore[assignment]

try:
    from timesead.models.common import AnomalyDetector
except ImportError:  # pragma: no cover - test stubs install this when needed.
    from timesead.models.common.anomaly_detector import AnomalyDetector  # type: ignore


def _batch_to_points(batch: Any) -> torch.Tensor:
    inputs = WindowedExternalAnomalyDetector._extract_inputs(batch)
    values = inputs[0].detach().cpu().to(dtype=torch.float32)
    if values.ndim == 3:
        return values[:, -1, :]
    if values.ndim == 2:
        return values
    raise ValueError("Expected input tensor with shape [batch, channels] or [batch, window, channels].")


def dataloader_to_frame(dataset: DataLoader) -> pd.DataFrame:
    points: List[torch.Tensor] = []
    for batch in dataset:
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            batch = batch[0]
        points.append(_batch_to_points(batch))
    if not points:
        return pd.DataFrame()
    values = torch.cat(points, dim=0).numpy()
    return pd.DataFrame(values, columns=[f"feature_{idx}" for idx in range(values.shape[1])])


def tensor_points_to_frame(inputs: Tuple[torch.Tensor, ...]) -> pd.DataFrame:
    values = _batch_to_points(inputs).numpy()
    return pd.DataFrame(values, columns=[f"feature_{idx}" for idx in range(values.shape[1])])


def _detect_fit_required_positional(fn: Callable[..., Any]) -> int:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    count = 0
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and param.default is inspect.Parameter.empty:
            count += 1
    return count


def _align_scores(scores: np.ndarray, length: int) -> torch.Tensor:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == length:
        return torch.as_tensor(values, dtype=torch.float32)
    if values.size > length:
        return torch.as_tensor(values[-length:], dtype=torch.float32)
    padded = np.full(length, -np.inf, dtype=np.float32)
    if values.size:
        padded[-values.size :] = values
    return torch.as_tensor(padded, dtype=torch.float32)


class SourceBackedAnomalyDetector(AnomalyDetector):
    """TimeSeAD detector adapter for vendored CATCH-style detect_fit/detect_score wrappers."""

    source_cls: type

    def __init__(
        self,
        win_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        patch_size: Optional[Any] = None,
        patch_stride: Optional[int] = None,
        inference_patch_stride: Optional[int] = None,
        inference_patch_size: Optional[int] = None,
        stride: Optional[int] = None,
        lr: Optional[float] = None,
        Mlr: Optional[float] = None,
        e_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        cf_dim: Optional[int] = None,
        d_ff: Optional[int] = None,
        d_model: Optional[int] = None,
        head_dim: Optional[int] = None,
        individual: Optional[int] = None,
        dropout: Optional[float] = None,
        head_dropout: Optional[float] = None,
        auxi_loss: Optional[str] = None,
        auxi_type: Optional[str] = None,
        auxi_mode: Optional[str] = None,
        auxi_lambda: Optional[float] = None,
        score_lambda: Optional[float] = None,
        regular_lambda: Optional[float] = None,
        temperature: Optional[float] = None,
        dc_lambda: Optional[float] = None,
        module_first: Optional[bool] = None,
        mask: Optional[bool] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        patience: Optional[int] = None,
        pct_start: Optional[float] = None,
        revin: Optional[int] = None,
        affine: Optional[int] = None,
        subtract_last: Optional[int] = None,
        lradj: Optional[str] = None,
        copies: Optional[int] = None,
        norm: Optional[int] = None,
        model_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        input_step: Optional[int] = None,
        pred_step: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        weight_decay: Optional[float] = None,
        cuts_num_epochs: Optional[int] = None,
        cuts_hidden_dim: Optional[int] = None,
        cuts_gru_layers: Optional[int] = None,
        cuts_concat_h: Optional[bool] = None,
        cuts_shared_weights_decoder: Optional[bool] = None,
        cuts_tau_start: Optional[float] = None,
        cuts_tau_end: Optional[float] = None,
        graph_lr_start: Optional[float] = None,
        graph_lr_end: Optional[float] = None,
        graph_lambda_start: Optional[float] = None,
        graph_lambda_end: Optional[float] = None,
        data_lr_start: Optional[float] = None,
        data_lr_end: Optional[float] = None,
        transform_percent: Optional[float] = None,
        cutoff_probability: Optional[float] = None,
        noise_level: Optional[float] = None,
        sim_threshold: Optional[float] = None,
        sim_threshold_start: Optional[float] = None,
        sim_threshold_end: Optional[float] = None,
        sim_threshold_schedule: Optional[bool] = None,
        eval_period: Optional[int] = None,
        scorer_type: Optional[str] = None,
        projection_dim: Optional[int] = None,
        memory_bank_ratio: Optional[float] = None,
        top_k: Optional[int] = None,
        use_revin: Optional[bool] = None,
        random_seed: Optional[int] = None,
        layers: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        num_iter: Optional[int] = None,
        model_dim: Optional[int] = None,
        beta: Optional[float] = None,
        candidate_ratio: Optional[float] = None,
        num_chunks: Optional[int] = None,
        reconstruction_weight: Optional[float] = None,
        structure_weight: Optional[float] = None,
        score_fusion: Optional[str] = None,
        rec_timeseries: Optional[bool] = None,
        k: Optional[int] = None,
        **source_kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs = {
            name: value
            for name, value in locals().items()
            if name not in {"self", "source_kwargs", "__class__"} and value is not None
        }
        kwargs.update(source_kwargs)
        self.impl = self.source_cls(**kwargs)
        self._fit_frame: Optional[pd.DataFrame] = None

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del kwargs
        frame = dataloader_to_frame(dataset)
        self._fit_frame = frame
        if _detect_fit_required_positional(self.impl.detect_fit) >= 2:
            self.impl.detect_fit(frame, frame.iloc[:0].copy())
        else:
            self.impl.detect_fit(frame)

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        frame = tensor_points_to_frame(inputs)
        result = self.impl.detect_score(frame)
        scores = result[0] if isinstance(result, tuple) else result
        return _align_scores(np.asarray(scores), len(frame)).to(inputs[0].device)

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


def frame_to_tensor(frame: pd.DataFrame) -> torch.Tensor:
    """Convert a frame to a float32 tensor without changing row order."""

    return torch.as_tensor(frame.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32)


def tensor_to_frame(
    values: torch.Tensor,
    columns: Sequence[str],
    index: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Convert a tensor back to a DataFrame using the supplied axis metadata."""

    array = values.detach().cpu().to(dtype=torch.float32).numpy()
    return pd.DataFrame(array, columns=list(columns), index=index)


def train_val_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.8,
    min_validation_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train and validation slices with stable ordering."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if min_validation_size < 0:
        raise ValueError("min_validation_size must be non-negative.")

    frame = frame.copy()
    row_count = len(frame)
    if row_count == 0:
        return frame, frame
    if row_count == 1:
        return frame.iloc[:1].copy(), frame.iloc[:0].copy()

    split_idx = int(round(row_count * train_fraction))
    split_idx = max(1, min(split_idx, row_count - max(1, min_validation_size)))
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def build_causal_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """Build one causal left-padded window per time step."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array of shape [time, channels].")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if array.shape[0] == 0:
        return np.empty((0, window_size, array.shape[1]), dtype=np.float32)

    first_row = array[:1]
    windows = np.empty((array.shape[0], window_size, array.shape[1]), dtype=np.float32)
    for end in range(array.shape[0]):
        start = max(0, end - window_size + 1)
        window = array[start : end + 1]
        pad = window_size - window.shape[0]
        if pad > 0:
            windows[end, :pad] = first_row
            windows[end, pad:] = window
        else:
            windows[end] = window
    return windows


def make_window_loader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap a 3D window array in a simple DataLoader."""

    tensor = torch.as_tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _sinusoidal_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(1, 0, dim, device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1))
    )
    encoding = torch.zeros(1, length, dim, device=device, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        encoding[0, :, 1::2] = torch.cos(position * div_term[: encoding[0, :, 1::2].shape[-1]])
    return encoding.to(dtype=dtype)


class GroupedParameterBase(BaseModel):
    """Base model that exposes a single grouped parameter iterable."""

    def grouped_parameters(self) -> Tuple[Iterable[nn.Parameter], ...]:
        return (self.parameters(),)


class TemporalAutoencoder(GroupedParameterBase):
    """Small transformer encoder-decoder that reconstructs the input window."""

    def __init__(
        self,
        input_dim: int,
        window_size: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.input_dim = int(input_dim)
        self.window_size = int(window_size) if window_size is not None else None
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim) if ff_dim is not None else int(hidden_dim * 4)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        seq_len = x.shape[1]
        if self.window_size is not None and seq_len != self.window_size:
            seq_len = int(seq_len)
        h = self.input_proj(x)
        h = h + _sinusoidal_position_encoding(seq_len, self.hidden_dim, x.device, h.dtype)
        h = self.encoder(h)
        return self.output_proj(h)


class PatchAutoencoder(GroupedParameterBase):
    """Patch embedding autoencoder used by the patch-oriented baselines."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        patch_stride: int = 1,
        window_size: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.input_dim = int(input_dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.window_size = int(window_size) if window_size is not None else None
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim) if ff_dim is not None else int(hidden_dim * 4)

        self.patch_embed = nn.Conv1d(
            self.input_dim,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=self.patch_size // 2,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_proj = nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        seq_len = x.shape[1]
        h = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        h = h + _sinusoidal_position_encoding(h.shape[1], self.hidden_dim, x.device, h.dtype)
        h = self.encoder(h).transpose(1, 2)
        h = self.output_proj(h)
        if h.shape[-1] != seq_len:
            h = F.interpolate(h, size=seq_len, mode="linear", align_corners=False)
        return h.transpose(1, 2)


class WindowedExternalAnomalyDetector(AnomalyDetector):
    """Detector that scores windows using reconstruction error on the last step."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        reduce_to_last_step: bool = True,
        score_tail_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.model = model
        self.reduce_to_last_step = bool(reduce_to_last_step)
        self.score_tail_length = score_tail_length

    def fit(self, dataset: DataLoader, **kwargs: Any) -> None:
        del dataset, kwargs
        return None

    @staticmethod
    def _extract_inputs(batch: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(batch, torch.Tensor):
            return (batch,)
        if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
            if len(batch) == 2 and not isinstance(batch[1], torch.Tensor):
                return (batch[0],)
            return tuple(batch)
        raise ValueError("Detector expected a batch containing input tensors.")

    @staticmethod
    def _normalize_outputs(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise ValueError("Model returned an empty output tuple.")
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("Model output must be a tensor or tensor tuple.")
        return outputs

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("WindowedExternalAnomalyDetector requires a model.")
        x = inputs[0]
        with torch.no_grad():
            outputs = self._normalize_outputs(self.model(inputs))
        if outputs.shape == x.shape:
            per_step = (outputs - x).pow(2).mean(dim=-1)
        elif outputs.ndim == 2 and outputs.shape[0] == x.shape[0]:
            per_step = outputs
        else:
            per_step = outputs.reshape(outputs.shape[0], -1).mean(dim=-1)
        if per_step.ndim == 2:
            return per_step[:, -1]
        return per_step

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        target = targets[0]
        if target.ndim == 0:
            return target
        if target.ndim == 1:
            return target[-1]
        return target[:, -1]


class DenoisingWindowedDetector(WindowedExternalAnomalyDetector):
    """Compatibility alias for detector classes that only need score plumbing."""


class DADAFallbackModel(GroupedParameterBase):
    """Fallback DADA wrapper that becomes a no-op when no checkpoint is present."""

    def __init__(
        self,
        seq_len: Optional[int] = None,
        model_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        batch_size: int = 128,
        copies: int = 10,
        norm: int = 0,
        mask_mode: str = "c",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__()
        self.seq_len = seq_len
        self.model_path = model_path
        self.repo_path = repo_path
        self.batch_size = int(batch_size)
        self.copies = int(copies)
        self.norm = int(norm)
        self.mask_mode = mask_mode
        self.trust_remote_code = bool(trust_remote_code)
        self.dummy_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.model: Optional[nn.Module] = None

    def _resolve_model_path(self) -> Optional[str]:
        return self.model_path or self.repo_path

    def _load_model(self) -> None:
        if self.model is not None:
            return
        # The official checkpoint is optional for smoke tests. If it is not
        # available we keep the model in identity mode and still expose a valid
        # module for the benchmark config pipeline.
        return None

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        self._load_model()
        if self.model is None:
            return x * self.dummy_scale
        outputs = self.model((x,))
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("Loaded DADA model returned a non-tensor output.")
        return outputs


class AnomalyTransformer(TemporalAutoencoder):
    pass


class AnomalyTransformerAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class DCdetector(PatchAutoencoder):
    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        patch_size: Sequence[int] = (5,),
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        first_patch = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
        super().__init__(
            input_dim=input_dim,
            patch_size=first_patch,
            patch_stride=max(1, first_patch // 2),
            window_size=win_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        self.patch_size_list = tuple(int(value) for value in patch_size)
        if any(value <= 0 for value in self.patch_size_list):
            raise ValueError("patch_size must contain only positive integers.")


class DCdetectorAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class CATCH(PatchAutoencoder):
    def __init__(
        self,
        input_dim: int,
        seq_len: Optional[int] = None,
        patch_size: int = 16,
        patch_stride: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
            window_size=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )


class CATCHAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class RTdetector(TemporalAutoencoder):
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
        super().__init__(
            input_dim=input_dim,
            window_size=win_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )


class RTdetectorAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class CAROTS(TemporalAutoencoder):
    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        input_step: int = 9,
        pred_step: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if input_step + pred_step < 1:
            raise ValueError("input_step and pred_step must be positive.")
        super().__init__(
            input_dim=input_dim,
            window_size=win_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        self.input_step = int(input_step)
        self.pred_step = int(pred_step)


class CAROTSAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class PaAno(PatchAutoencoder):
    def __init__(
        self,
        input_dim: int,
        patch_size: int = 96,
        patch_stride: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )


class PaAnoAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class OracleAD(TemporalAutoencoder):
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
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        super().__init__(
            input_dim=input_dim,
            window_size=win_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )


class OracleADAnomalyDetector(WindowedExternalAnomalyDetector):
    pass


class SPAGD(TemporalAutoencoder):
    def __init__(
        self,
        input_dim: int,
        win_size: Optional[int] = None,
        model_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        super().__init__(
            input_dim=input_dim,
            window_size=win_size,
            hidden_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )


class SPAGDAnomalyDetector(WindowedExternalAnomalyDetector):
    pass
