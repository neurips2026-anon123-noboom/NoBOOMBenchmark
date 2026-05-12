"""Compact OracleAD wrapper for the CATCH anomaly benchmark."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.OracleAD.model import (
    OracleADCore,
    oracle_deviation_score,
    oracle_loss,
    oracle_prediction_score,
    pairwise_l2_distances,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    make_window_loader,
    transform_frame,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.utils import train_val_split


DEFAULT_HYPER_PARAMS = {
    # Use the paper defaults where they are reported explicitly.
    "batch_size": 1024,
    "deviation_weight": 1.0,
    "dropout": 0.1,
    "hidden_dim": 64,
    "lr": 5e-4,
    "num_epochs": 20,
    "num_heads": 4,
    "num_layers": 2,
    "patience": 5,
    "random_seed": 2026,
    "reconstruction_weight": 0.1,
    "score_fusion": "multiply",
    "structure_weight": 3.0,
    "weight_decay": 1e-5,
    "win_size": 10,
}


@dataclass
class OracleADConfig:
    batch_size: int = 1024
    deviation_weight: float = 1.0
    dropout: float = 0.1
    hidden_dim: int = 64
    lr: float = 5e-4
    num_epochs: int = 20
    num_heads: int = 4
    num_layers: int = 2
    patience: int = 5
    random_seed: int = 2026
    reconstruction_weight: float = 0.1
    score_fusion: str = "multiply"
    structure_weight: float = 3.0
    weight_decay: float = 1e-5
    win_size: int = 10

    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class OracleAD:
    """Lean OracleAD benchmark integration.

    The wrapper follows the paper closely while staying compact:
    - only normal train windows are used for fitting,
    - SLS is updated from latent distance matrices each epoch,
    - inference returns one anomaly score per point via causal windows.
    """

    def __init__(self, **kwargs) -> None:
        self.config = OracleADConfig(**kwargs)
        if self.config.win_size < 2:
            raise ValueError("OracleAD requires win_size >= 2.")
        if self.config.hidden_dim % self.config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.config.num_layers < 1:
            raise ValueError("OracleAD requires num_layers >= 1.")
        if self.config.score_fusion not in {"multiply", "add"}:
            raise ValueError("score_fusion must be one of {'multiply', 'add'}.")
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[OracleADCore] = None
        self.best_state: Optional[dict] = None
        self.sls: Optional[torch.Tensor] = None

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

        self.model = OracleADCore(
            num_channels=train_data.shape[1],
            hidden_dim=self.config.hidden_dim,
            history_length=self.config.win_size - 1,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
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
        current_sls = None
        for _ in range(self.config.num_epochs):
            train_loss, next_sls = self._run_epoch(train_loader, optimizer=optimizer, sls=current_sls)
            current_sls = next_sls
            valid_loss, _ = self._run_epoch(valid_loader, optimizer=None, sls=current_sls)
            if valid_loss < best_loss:
                best_loss = valid_loss
                stale_epochs = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.sls = current_sls.detach().clone() if current_sls is not None else None
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()

    def _run_epoch(self, loader, optimizer, sls: Optional[torch.Tensor]):
        if self.model is None:
            raise RuntimeError("OracleAD model is not initialised.")

        training = optimizer is not None
        self.model.train(training)
        losses = []
        distance_sum: Optional[torch.Tensor] = None
        distance_count = 0

        for (batch,) in loader:
            batch = batch.to(self.device)
            history = batch[:, :-1, :]
            target = batch[:, -1, :]
            reconstruction, prediction, refined = self.model(history)
            distances = pairwise_l2_distances(refined)
            loss = oracle_loss(
                reconstruction=reconstruction,
                history=history,
                prediction=prediction,
                target=target,
                distances=distances,
                sls=sls,
                reconstruction_weight=self.config.reconstruction_weight,
                structure_weight=self.config.structure_weight,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.detach().item())
            detached_distances = distances.detach()
            batch_distance_sum = detached_distances.sum(dim=0)
            if distance_sum is None:
                distance_sum = batch_distance_sum
            else:
                distance_sum = distance_sum + batch_distance_sum
            distance_count += detached_distances.size(0)

        next_sls = None
        if distance_count:
            next_sls = distance_sum / distance_count
        return float(np.mean(losses)) if losses else 0.0, next_sls

    def detect_score(self, test: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("OracleAD must be fitted before calling detect_score.")
        if self.sls is None:
            raise RuntimeError("OracleAD has no Stable Latent Structure after fitting.")

        scaled = transform_frame(self.scaler, test)
        loader = self._make_loader(scaled, shuffle=False)

        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                history = batch[:, :-1, :]
                target = batch[:, -1, :]
                _, prediction, refined = self.model(history)
                distances = pairwise_l2_distances(refined)
                prediction_score = oracle_prediction_score(prediction, target)
                deviation_score = oracle_deviation_score(distances, self.sls.to(self.device))
                if self.config.score_fusion == "multiply":
                    batch_scores = prediction_score * deviation_score
                else:
                    batch_scores = prediction_score + self.config.deviation_weight * deviation_score
                scores.append(batch_scores.detach().cpu().numpy())

        all_scores = np.concatenate(scores, axis=0)
        return all_scores, all_scores
