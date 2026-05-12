"""KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks.

Faithful re-implementation of the architecture from Zhou et al., ICML 2025
(https://openreview.net/pdf?id=LWQ4zu9SdQ), cross-checked against the official
reference code at https://github.com/CSTCloudOps/KAN-AD.

Pipeline (per the paper, Section 3):
    1. Mapping phase stacks the input window with two families of fixed
       univariate functions: cosine harmonics of the values (S_n) and
       positional cosine waves over the window index (P_n).
    2. Reducing phase passes the stacked tensor through a stacked 1D-CNN with
       a residual connection between the deep block output and the original
       stack, then a 1x1 down-conv to a single channel.
    3. Projection phase produces the next-step prediction with a single linear
       layer realised via a kernel-sized convolution over the reduced channel.

The Constant Term Elimination (CTE, Section 3.2) happens **upstream in the
data pipeline** via first-order differencing. The model therefore receives the
differenced signal directly and produces a prediction in the same diff space,
matching the paper's "estimating Fourier coefficients A_{1:n}, B_{1:n}"
reformulation while avoiding unstable per-window scaling on flat process
signals.

The official implementation only uses cosine harmonics (S_n is cos-only) and
a single residual chain; we mirror that here. Multivariate inputs are handled
with the channel-independent strategy described in Section 4.7 of the paper.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from timesead.models import BaseModel
from timesead.models.common.anomaly_detector import PredictionAnomalyDetector


class KANADModel(nn.Module):
    """Official KAN-AD network: cosine mapping, residual 1D-CNN, projection."""

    def __init__(self, window: int, order: int) -> None:
        super().__init__()
        if window <= 1:
            raise ValueError("window must be >= 2.")
        if order <= 0:
            raise ValueError("order must be positive.")

        self.window = int(window)
        self.order = int(order)
        self.channels = 2 * self.order + 1

        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine().unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "harmonics",
            torch.arange(1, self.order + 1, dtype=torch.float32).view(1, self.order, 1),
            persistent=False,
        )

        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, self.window, padding=0, stride=1, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_signal = x.unsqueeze(1)
        mapped = self._map_input(x)
        residual_mapped = mapped

        h = self.init_conv(mapped)
        h = self.bn1(h)
        h = self.act(h)
        h = self.inner_conv(h) + residual_mapped
        h = self.bn2(h)
        h = self.act(h)
        h = self.out_conv(h) + residual_signal
        h = self.bn3(h)
        h = self.act(h)
        h = self.final_conv(h)
        return h.squeeze(1)

    def _map_input(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(1)
        mapped = x.new_empty((x.size(0), self.channels, self.window))
        mapped[:, :self.order, :] = self.orders.to(dtype=x.dtype).expand(x.size(0), -1, -1)
        harmonics = self.harmonics.to(device=x.device, dtype=x.dtype)
        mapped[:, self.order:2 * self.order, :] = torch.cos(harmonics * x_expanded)
        mapped[:, -1:, :] = x_expanded
        return mapped

    def _create_custom_periodic_cosine(self) -> torch.Tensor:
        result = torch.empty(self.order, self.window, dtype=torch.float32)
        range_value = torch.arange(self.window, dtype=torch.float32)
        for i, period in enumerate(range(1, self.order + 1)):
            result[i, :] = torch.cos(2 * torch.pi * range_value * period / self.window)
        return result


class KANAD(BaseModel):
    """NoBoom wrapper around the official one-step KAN-AD network."""

    def __init__(
        self,
        window_size: int,
        input_dim: int,
        order: int = 2,
        prediction_horizon: int = 1,
    ) -> None:
        super().__init__()
        if window_size <= 1:
            raise ValueError("window_size must be >= 2.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if order <= 0:
            raise ValueError("order must be positive.")
        if prediction_horizon != 1:
            raise ValueError("KAN-AD official implementation supports prediction_horizon=1 only.")

        self.window_size = int(window_size)
        self.input_dim = int(input_dim)
        self.order = int(order)
        self.prediction_horizon = int(prediction_horizon)
        self.network = KANADModel(window=self.window_size, order=self.order)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, = inputs
        # Inputs arrive in time-major form (T, B, D) already in diff space
        # (CTE differencing applied upstream by the data transform pipeline).
        # Collapse channels into the batch dimension so that each univariate
        # series is modelled independently (paper Section 4.7).
        x_btd = x.permute(1, 2, 0).contiguous()
        batch_size, channel_count, window = x_btd.shape
        x_flat = x_btd.reshape(batch_size * channel_count, window)

        prediction_flat = self.network(x_flat)
        prediction = prediction_flat.view(
            batch_size, channel_count, self.prediction_horizon
        ).permute(2, 0, 1)
        return prediction


class KANADAnomalyDetector(PredictionAnomalyDetector):
    """Anomaly score = mean absolute prediction error at the predicted step."""

    def __init__(self, model: KANAD) -> None:
        super().__init__()
        self.model = model

    def fit(self, dataset: torch.utils.data.DataLoader, **kwargs) -> None:
        return None

    def compute_online_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, target = inputs
        with torch.no_grad():
            prediction = self.model((x,))
        # prediction & target: (prediction_horizon, B, D)
        absolute_error = (prediction - target).abs().mean(dim=-1)
        return absolute_error[-1]

    def compute_offline_anomaly_score(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def format_online_targets(self, targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        label, _target = targets
        return label[-1]
