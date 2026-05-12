"""Compact SPAGD wrapper for the CATCH anomaly benchmark."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.SPAGD.model import SPAGDCore
from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    make_window_loader,
    transform_frame,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.utils import train_val_split


DEFAULT_HYPER_PARAMS = {
    "batch_size": 128,
    # Match the paper defaults and sensitivity analysis for the full benchmark.
    "beta": 1e-2,
    "candidate_ratio": 0.2,
    "dropout": 0.1,
    "lr": 0.001,
    "model_dim": 256,
    "num_chunks": 5,
    "num_epochs": 20,
    "num_heads": 8,
    "num_layers": 3,
    "patience": 5,
    "random_seed": 2026,
    "top_k": 5,
    "weight_decay": 1e-5,
    "win_size": 100,
}


@dataclass
class SPAGDConfig:
    batch_size: int = 128
    beta: float = 1e-2
    candidate_ratio: float = 0.2
    dropout: float = 0.1
    lr: float = 0.001
    model_dim: int = 256
    num_chunks: int = 5
    num_epochs: int = 20
    num_heads: int = 8
    num_layers: int = 3
    patience: int = 5
    random_seed: int = 2026
    top_k: int = 5
    weight_decay: float = 1e-5
    win_size: int = 100

    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class SPAGD:
    """Lean SPAGD benchmark integration.

    This keeps the three core paper components:
    - a transformer reconstructor that produces self-perturbed windows,
    - anomaly-aware dynamic graph construction,
    - a classifier trained to separate normal windows from self-perturbed ones.
    """

    def __init__(self, **kwargs) -> None:
        self.config = SPAGDConfig(**kwargs)
        if self.config.win_size < 2:
            raise ValueError("SPAGD requires win_size >= 2.")
        if self.config.model_dim % self.config.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SPAGDCore] = None
        self.best_state: Optional[dict] = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def _set_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def _make_loader(self, frame: pd.DataFrame, shuffle: bool):
        windows = build_causal_windows(frame.to_numpy(dtype=np.float32), self.config.win_size)
        return make_window_loader(
            windows=windows,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> None:
        self._set_seed()
        train_values, valid_values = train_val_split(train_data, 0.8, None)
        if len(valid_values) == 0:
            valid_values = train_values.copy()

        self.scaler.fit(train_values.to_numpy())
        train_scaled = transform_frame(self.scaler, train_values)
        valid_scaled = transform_frame(self.scaler, valid_values)

        self.model = SPAGDCore(
            input_dim=train_data.shape[1],
            model_dim=self.config.model_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            top_k=self.config.top_k,
            candidate_ratio=self.config.candidate_ratio,
            num_chunks=self.config.num_chunks,
            dropout=self.config.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._make_loader(train_scaled, shuffle=True)
        valid_loader = self._make_loader(valid_scaled, shuffle=False)

        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(self.config.num_epochs):
            self._run_epoch(train_loader, optimizer=optimizer)
            valid_loss = self._run_epoch(valid_loader, optimizer=None)
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

    def _run_epoch(self, loader, optimizer):
        if self.model is None:
            raise RuntimeError("SPAGD model is not initialised.")

        training = optimizer is not None
        self.model.train(training)
        losses = []
        for (batch,) in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            reconstruction_loss = F.mse_loss(outputs["reconstruction"], batch)
            labels = torch.cat(
                [
                    torch.zeros(batch.size(0), device=self.device),
                    torch.ones(batch.size(0), device=self.device),
                ],
                dim=0,
            )
            logits = torch.cat([outputs["normal_logits"], outputs["perturbed_logits"]], dim=0)
            detection_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = reconstruction_loss + self.config.beta * detection_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.detach().item())
        return float(np.mean(losses)) if losses else 0.0

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("SPAGD must be fitted before calling detect_score.")

        scaled = transform_frame(self.scaler, test)
        loader = self._make_loader(scaled, shuffle=False)

        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                scores.append(torch.sigmoid(outputs["test_logits"]).detach().cpu().numpy())
        all_scores = np.concatenate(scores, axis=0)
        return all_scores, all_scores
