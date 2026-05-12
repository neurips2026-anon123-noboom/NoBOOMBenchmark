"""Faithful CAROTS benchmark wrapper with the official CUTS+ stage retained."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.CAROTS.model import (
    CAROTSCore,
    carots_loss,
    distance_to_centroid,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.recent_utils import (
    build_causal_windows,
    make_window_loader,
    transform_frame,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.utils import train_val_split


DEFAULT_HYPER_PARAMS = {
    # Follow the official CAROTS config defaults for the benchmark path.
    "batch_size": 256,
    "bias_candidates": (0.5, 0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5),
    "cuts_concat_h": True,
    "cuts_hidden_dim": 32,
    "cuts_gru_layers": 1,
    "cuts_num_epochs": 50,
    "cuts_shared_weights_decoder": False,
    "cuts_tau_start": 1.0,
    "cuts_tau_end": 0.1,
    "cutoff_probability": 0.1,
    "data_dim": 1,
    "data_lr_end": 0.001,
    "data_lr_start": 0.01,
    "disturb_all": False,
    "dropout": 0.0,
    "eval_period": 5,
    "graph_lr_end": 0.0001,
    "graph_lr_start": 0.001,
    "graph_lambda_end": 0.01,
    "graph_lambda_start": 0.1,
    "hidden_dim": 512,
    "input_step": 9,
    "lr": 0.0001,
    "max_eval_windows": None,
    "max_train_windows": None,
    "noise_level": 0.1,
    "num_epochs": 30,
    "num_layers": 1,
    "pred_step": 1,
    "projector_hidden_dim": 1024,
    "projector_output_dim": 512,
    "positive_augment": True,
    "scorer_type": "l2",
    "sim_threshold": 0.5,
    "sim_threshold_end": 0.9,
    "sim_threshold_schedule": True,
    "sim_threshold_start": 0.5,
    "transform_percent": 0.5,
    "temperature": 0.1,
    "weight_decay": 0.0001,
    "win_size": 10,
    # Paper §3.4 (arXiv:2506.03964) defines the anomaly score as centroid
    # distance only. The official reference (kimanki/CAROTS,
    # models/carots/scorer_carots.py:48-67) ships separate scorer types
    # ("l2"/"cos" centroid-only OR "causal_discoverer" only) — never a sum
    # of the two. We default to centroid-only and gate the legacy blended
    # CUTS+ reconstruction term behind this opt-in flag for parity studies.
    "blend_causal_score": False,
}


class CAROTSConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.input_step + self.pred_step != self.win_size:
            raise ValueError("CAROTS expects input_step + pred_step == win_size.")


def _build_training_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """Use full sliding windows for training, but fall back to causal padding when data is short."""

    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] <= 0:
        return np.empty((0, window_size, array.shape[1]), dtype=np.float32)
    if array.shape[0] < window_size:
        return build_causal_windows(array, window_size)
    return np.stack(
        [array[start : start + window_size] for start in range(array.shape[0] - window_size + 1)],
        axis=0,
    ).astype(np.float32)


def _limit_windows(windows: np.ndarray, limit: Optional[int]) -> np.ndarray:
    if limit is None or len(windows) <= limit:
        return windows
    return windows[:limit]


class CAROTS:
    """CAROTS wrapper that keeps the official CUTS+ pretraining stage.

    The implementation stays close to the official code path:
    1. train CUTS+ to discover a causal graph,
    2. freeze it and train the contrastive encoder/projector,
    3. combine centroid-distance and causal-reconstruction scores.
    """

    def __init__(self, **kwargs):
        self.config = CAROTSConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CAROTSCore] = None
        self.cuts_best_state: Optional[dict] = None
        self.best_state: Optional[dict] = None
        self.centroid_wo_norm: Optional[torch.Tensor] = None
        self.centroid_norm: Optional[torch.Tensor] = None
        self.train_cl_stats = (0.0, 1.0)
        self.train_cd_stats = (0.0, 1.0)

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> None:
        train_values, valid_values = train_val_split(train_data, 0.8, None)
        if len(valid_values) == 0:
            valid_values = train_values.copy()

        self.scaler.fit(train_values.to_numpy())
        train_scaled = transform_frame(self.scaler, train_values)
        valid_scaled = transform_frame(self.scaler, valid_values)

        self.model = CAROTSCore(
            input_dim=train_data.shape[1],
            input_step=self.config.input_step,
            pred_step=self.config.pred_step,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            projector_hidden_dim=self.config.projector_hidden_dim,
            projector_output_dim=self.config.projector_output_dim,
            data_dim=self.config.data_dim,
            cuts_hidden_dim=self.config.cuts_hidden_dim,
            cuts_gru_layers=self.config.cuts_gru_layers,
            cuts_concat_h=self.config.cuts_concat_h,
            cuts_shared_weights_decoder=self.config.cuts_shared_weights_decoder,
            noise_level=self.config.noise_level,
            bias_candidates=self.config.bias_candidates,
            cutoff_probability=self.config.cutoff_probability,
            disturb_all=self.config.disturb_all,
            transform_percent=self.config.transform_percent,
        ).to(self.device)

        train_windows = _limit_windows(
            _build_training_windows(train_scaled.to_numpy(), self.config.win_size),
            self.config.max_train_windows,
        )
        valid_windows = _limit_windows(
            _build_training_windows(valid_scaled.to_numpy(), self.config.win_size),
            self.config.max_eval_windows,
        )
        train_loader = make_window_loader(train_windows, batch_size=self.config.batch_size, shuffle=True)
        valid_loader = make_window_loader(valid_windows, batch_size=self.config.batch_size, shuffle=False)

        self._train_cuts_plus(train_loader, valid_loader)
        self.model.positive_augmentor.set_causal_discoverer(self.model.causal_discoverer)
        self._train_carots(train_loader, valid_loader)
        self._init_score_statistics(train_scaled)

    def _train_cuts_plus(self, train_loader, valid_loader) -> None:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        causal = self.model.causal_discoverer
        mse = torch.nn.MSELoss()
        data_parameters = [param for name, param in causal.named_parameters() if name != "GT"]
        data_optimizer = torch.optim.Adam(data_parameters, lr=self.config.data_lr_start)
        data_gamma = (self.config.data_lr_end / self.config.data_lr_start) ** (1 / max(self.config.cuts_num_epochs, 1))
        data_scheduler = torch.optim.lr_scheduler.StepLR(data_optimizer, step_size=1, gamma=data_gamma)
        graph_gamma = (self.config.graph_lr_end / self.config.graph_lr_start) ** (1 / max(self.config.cuts_num_epochs, 1))
        tau = self.config.cuts_tau_start
        tau_gamma = (self.config.cuts_tau_end / self.config.cuts_tau_start) ** (1 / max(self.config.cuts_num_epochs, 1))
        lambda_s = self.config.graph_lambda_start
        lambda_gamma = (self.config.graph_lambda_end / self.config.graph_lambda_start) ** (1 / max(self.config.cuts_num_epochs, 1))
        identity_graph = torch.eye(causal.GT.shape[0], device=self.device)

        best_val = float("inf")
        for epoch in range(self.config.cuts_num_epochs):
            graph_optimizer = torch.optim.Adam(
                [causal.GT],
                lr=self.config.graph_lr_start * (graph_gamma ** epoch),
            )
            causal.train()

            for (batch,) in train_loader:
                batch = batch.to(self.device)
                x = batch[:, : self.config.input_step]
                y = batch[:, self.config.input_step :]

                graph = torch.einsum("nm,ml->nl", identity_graph, torch.sigmoid(causal.GT))
                sampled_graph = self._sample_bernoulli(graph, batch.size(0))
                prediction = causal(x, sampled_graph).transpose(1, 2)
                loss = mse(prediction, y)

                data_optimizer.zero_grad()
                loss.backward()
                data_optimizer.step()

            data_scheduler.step()

            for (batch,) in train_loader:
                batch = batch.to(self.device)
                x = batch[:, : self.config.input_step]
                y = batch[:, self.config.input_step :]

                graph = torch.einsum("nm,ml->nl", identity_graph, torch.sigmoid(causal.GT))
                sampled_graph = self._gumbel_sigmoid_sample(graph, batch.size(0), tau=tau)
                prediction = causal(x, sampled_graph).transpose(1, 2)
                loss_data = mse(prediction, y)
                loss_sparsity = torch.linalg.norm(graph.flatten(), ord=1) / graph.numel()
                loss = loss_data + lambda_s * loss_sparsity

                graph_optimizer.zero_grad()
                loss.backward()
                graph_optimizer.step()

            tau *= tau_gamma
            lambda_s *= lambda_gamma
            val_loss = self._eval_cuts(valid_loader, identity_graph)
            if val_loss < best_val:
                best_val = val_loss
                self.cuts_best_state = copy.deepcopy(causal.state_dict())

        if self.cuts_best_state is not None:
            causal.load_state_dict(self.cuts_best_state)
        causal.eval()

    def _eval_cuts(self, loader, identity_graph: torch.Tensor) -> float:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        causal = self.model.causal_discoverer
        causal.eval()
        losses = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                x = batch[:, : self.config.input_step]
                y = batch[:, self.config.input_step :]
                graph = torch.einsum("nm,ml->nl", identity_graph, torch.sigmoid(causal.GT))
                graph = graph[None].expand(batch.size(0), -1, -1)
                prediction = causal(x, graph).transpose(1, 2)
                losses.append(F.mse_loss(prediction, y).item())
        return float(np.mean(losses)) if losses else 0.0

    def _train_carots(self, train_loader, valid_loader) -> None:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        for parameter in self.model.causal_discoverer.parameters():
            parameter.requires_grad = False

        optimizer = torch.optim.Adam(
            [
                param
                for name, param in self.model.named_parameters()
                if not name.startswith("causal_discoverer.")
            ],
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_val = float("inf")
        for epoch in range(self.config.num_epochs):
            self.model.train()
            threshold = self._current_threshold(epoch)
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                outputs = self.model(
                    batch,
                    positive_augment=self.config.positive_augment,
                    negative_augment=True,
                )
                loss = carots_loss(
                    outputs,
                    sim_threshold=threshold,
                    temperature=self.config.temperature,
                    positive_augment=self.config.positive_augment,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = self._eval_carots(valid_loader, threshold)
            if val_loss < best_val:
                best_val = val_loss
                self.best_state = copy.deepcopy(self.model.state_dict())

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.eval()

    def _eval_carots(self, loader, threshold: float) -> float:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        self.model.eval()
        losses = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                outputs = self.model(
                    batch,
                    positive_augment=self.config.positive_augment,
                    negative_augment=True,
                )
                losses.append(
                    carots_loss(
                        outputs,
                        sim_threshold=threshold,
                        temperature=self.config.temperature,
                        positive_augment=self.config.positive_augment,
                    ).item()
                )
        self.model.train()
        return float(np.mean(losses)) if losses else 0.0

    def _current_threshold(self, epoch: int) -> float:
        if not self.config.sim_threshold_schedule or self.config.num_epochs <= 1:
            return self.config.sim_threshold
        return self.config.sim_threshold_start + (
            (self.config.sim_threshold_end - self.config.sim_threshold_start) * epoch / (self.config.num_epochs - 1)
        )

    def _init_score_statistics(self, train_scaled: pd.DataFrame) -> None:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        train_windows = build_causal_windows(train_scaled.to_numpy(), self.config.win_size)
        self._init_centroids(train_windows)
        train_scores_cl = self._embedding_scores(train_windows)
        self.train_cl_stats = self._mean_std(train_scores_cl)
        if getattr(self.config, "blend_causal_score", False):
            train_scores_cd = self._causal_scores(train_windows)
            self.train_cd_stats = self._mean_std(train_scores_cd)

    def _init_centroids(self, windows: np.ndarray) -> None:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        outputs = []
        loader = make_window_loader(windows, batch_size=self.config.batch_size, shuffle=False)
        self.model.eval()
        # Layout produced by the patched CAROTSCore.forward (paper §3.3):
        # [anchors; positives?; negatives]. Centroid is computed over anchors
        # only. The official reference (kimanki/CAROTS,
        # models/carots/scorer_carots.py:30) takes ``output[: len(output)//2]``
        # which under the official 4B layout corresponds to anchors+positives —
        # not pure anchors. Aligning with the paper requires the explicit
        # anchor slice below.
        chunks = 1 + int(self.config.positive_augment) + 1  # +negatives
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                projected = self.model(
                    batch,
                    positive_augment=self.config.positive_augment,
                    negative_augment=True,
                )
                if projected.shape[0] % chunks != 0:
                    raise RuntimeError(
                        "CAROTS centroid init expected projection divisible by "
                        f"{chunks}, got {projected.shape[0]}"
                    )
                anchor_count = projected.shape[0] // chunks
                outputs.append(projected[:anchor_count])
        if outputs:
            merged = torch.cat(outputs, dim=0)
            self.centroid_wo_norm = merged.mean(dim=0, keepdim=True)
            self.centroid_norm = F.normalize(merged, p=2, dim=1).mean(dim=0, keepdim=True)
        else:
            dim = self.config.projector_output_dim
            self.centroid_wo_norm = torch.zeros((1, dim), device=self.device)
            self.centroid_norm = torch.zeros((1, dim), device=self.device)

    def _embedding_scores(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None or self.centroid_wo_norm is None or self.centroid_norm is None:
            raise RuntimeError("CAROTS centroids are not initialised.")

        loader = make_window_loader(windows, batch_size=self.config.batch_size, shuffle=False)
        scores = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                projected = self.model(batch, positive_augment=False, negative_augment=False)
                batch_scores = distance_to_centroid(
                    projected,
                    centroid_wo_norm=self.centroid_wo_norm,
                    centroid_norm=self.centroid_norm,
                    metric=self.config.scorer_type,
                )
                scores.append(batch_scores.cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    def _causal_scores(self, windows: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CAROTS model is not initialised.")

        causal = self.model.causal_discoverer
        graph = (causal.causality_mtx > 0.5).float().to(self.device)
        loader = make_window_loader(windows, batch_size=self.config.batch_size, shuffle=False)
        scores = []
        causal.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                x = batch[:, : self.config.input_step]
                y = batch[:, self.config.input_step :]
                graph_batch = graph[None].expand(batch.size(0), -1, -1)
                prediction = causal(x, graph_batch).transpose(1, 2)
                batch_scores = F.mse_loss(prediction, y, reduction="none").mean(dim=(1, 2))
                scores.append(batch_scores.cpu().numpy())
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    def detect_score(self, test: pd.DataFrame):
        # Paper §3.4 (arXiv:2506.03964) defines the anomaly score purely as
        # the centroid distance in projector space. The official reference
        # (kimanki/CAROTS, models/carots/scorer_carots.py:48-67) likewise
        # exposes the centroid scorers ("l2"/"cos") and the causal scorer as
        # *mutually exclusive* SCORER.TYPE choices — never as a sum.
        # ``blend_causal_score`` is an opt-in flag for ablation parity with
        # the legacy benchmark wiring; default False matches the paper.
        scaled = transform_frame(self.scaler, test)
        windows = build_causal_windows(scaled.to_numpy(), self.config.win_size)
        scores_cl = self._embedding_scores(windows)
        normalized_cl = self._normalize(scores_cl, self.train_cl_stats)
        if getattr(self.config, "blend_causal_score", False):
            scores_cd = self._causal_scores(windows)
            normalized_cd = self._normalize(scores_cd, self.train_cd_stats)
            combined = normalized_cl + normalized_cd
        else:
            combined = normalized_cl
        return combined, combined

    @staticmethod
    def _mean_std(values: np.ndarray) -> tuple[float, float]:
        mean = float(np.mean(values)) if len(values) else 0.0
        std = float(np.std(values)) if len(values) else 1.0
        if std == 0.0:
            std = 1.0
        return mean, std

    @staticmethod
    def _normalize(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
        mean, std = stats
        return (values - mean) / std

    @staticmethod
    def _sample_bernoulli(graph: torch.Tensor, batch_size: int) -> torch.Tensor:
        return torch.bernoulli(graph[None].expand(batch_size, -1, -1)).float()

    @staticmethod
    def _gumbel_sigmoid_sample(graph: torch.Tensor, batch_size: int, tau: float) -> torch.Tensor:
        prob = graph[None, :, :, None].expand(batch_size, -1, -1, -1)
        logits = torch.cat([prob, 1.0 - prob], dim=-1)
        gumbels = -torch.empty_like(logits).exponential_().log()
        soft = ((logits + gumbels) / tau).softmax(dim=-1)
        index = soft.max(dim=-1, keepdim=True)[1]
        hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return (hard - soft.detach() + soft)[:, :, :, 0]
