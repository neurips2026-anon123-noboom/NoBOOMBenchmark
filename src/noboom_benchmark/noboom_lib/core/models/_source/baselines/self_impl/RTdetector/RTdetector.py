"""Minimal RTdetector wrapper for the CATCH anomaly benchmark."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.RTdetector.model import RTdetectorModel
from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    make_window_loader,
    transform_frame,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.utils import train_val_split


DEFAULT_HYPER_PARAMS = {
    "batch_size": 128,
    "dropout": 0.1,
    "lr": 0.001,
    "num_epochs": 15,
    "patience": 5,
    "weight_decay": 1e-5,
    "win_size": 10,
}


class RTdetectorConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class RTdetector:
    """Compact RTdetector integration.

    The training loop keeps the official reconstruction-target objective and the
    two-stage loss schedule, but avoids the unrelated legacy code from the
    original repository.
    """

    def __init__(self, **kwargs):
        self.config = RTdetectorConfig(**kwargs)
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[RTdetectorModel] = None
        self.best_state: Optional[dict] = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> None:
        train_values, valid_values = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_values.to_numpy())
        train_scaled = transform_frame(self.scaler, train_values)
        valid_scaled = transform_frame(self.scaler, valid_values)

        self.model = RTdetectorModel(
            feats=train_data.shape[1],
            window_size=self.config.win_size,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

        train_loader = make_window_loader(
            build_causal_windows(train_scaled.to_numpy(), self.config.win_size),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        valid_loader = make_window_loader(
            build_causal_windows(valid_scaled.to_numpy(), self.config.win_size),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        best_loss = float("inf")
        stale_epochs = 0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                window = batch.permute(1, 0, 2)
                target = window[-1:].clone()
                prediction = self.model(window, target)
                loss = self._loss(prediction, target, epoch + 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            valid_loss = self._evaluate_loader(valid_loader)
            if valid_loss < best_loss:
                best_loss = valid_loss
                stale_epochs = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()

    def _loss(self, prediction: tuple[torch.Tensor, torch.Tensor] | torch.Tensor, target: torch.Tensor, epoch_index: int) -> torch.Tensor:
        criterion = nn.MSELoss(reduction="none")
        if isinstance(prediction, tuple):
            x1, x2 = prediction
            loss_1 = criterion(x1, target)
            loss_2 = criterion(x2, target)
            return ((1.0 / epoch_index) * loss_1 + (1.0 - 1.0 / epoch_index) * loss_2).mean()
        return criterion(prediction, target).mean()

    def _evaluate_loader(self, loader) -> float:
        if self.model is None:
            raise RuntimeError("RTdetector model is not initialised.")

        losses = []
        criterion = nn.MSELoss(reduction="none")
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                window = batch.permute(1, 0, 2)
                target = window[-1:].clone()
                prediction = self.model(window, target)
                if isinstance(prediction, tuple):
                    prediction = prediction[1]
                losses.append(criterion(prediction, target).mean().item())
        self.model.train()
        return float(np.mean(losses)) if losses else 0.0

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("RTdetector must be fitted before calling detect_score.")

        scaled = transform_frame(self.scaler, test)
        windows = build_causal_windows(scaled.to_numpy(), self.config.win_size)
        loader = make_window_loader(windows, batch_size=self.config.batch_size, shuffle=False)

        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                window = batch.permute(1, 0, 2)
                target = window[-1:].clone()
                prediction = self.model(window, target)
                if isinstance(prediction, tuple):
                    prediction = prediction[1]
                batch_score = ((prediction - target) ** 2).squeeze(0).mean(dim=-1)
                scores.append(batch_score.detach().cpu().numpy())
        all_scores = np.concatenate(scores, axis=0)
        return all_scores, all_scores
