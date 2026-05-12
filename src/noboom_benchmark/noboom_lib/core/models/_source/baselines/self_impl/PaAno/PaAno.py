"""Lean benchmark wrapper for the official PaAno anomaly detector."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.PaAno.model import (
    PaAnoTrainConfig,
    PatchEncoder,
    TransformerPatchEncoder,
    calculate_patch_scores,
    calculate_patch_scores_by_regime,
    correlation_reference,
    correlation_scores,
    create_memory_bank,
    create_patches,
    create_regime_memory_bank,
    distribute_patch_scores_to_points,
    make_patch_loader,
    train_encoder,
)
from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.recent_utils import transform_frame


DEFAULT_HYPER_PARAMS = {
    "batch_size": 512,
    "encoder_d_ff": 256,
    "encoder_d_model": 128,
    "encoder_dropout": 0.1,
    "encoder_num_heads": 4,
    "encoder_num_layers": 2,
    "encoder_patch_len": 16,
    "encoder_patch_stride": 8,
    "encoder_pooling": "mean",
    "encoder_type": "cnn",
    "kernel_sizes": (7, 5, 3, 3),
    "layers": (128, 256, 128, 64),
    "lr": 1e-4,
    "memory_bank_ratio": 0.1,
    "num_iter": 200,
    # NoBOOM is multivariate, so use the official multivariate run script preset.
    "patch_size": 96,
    "point_aggregation": "mean",
    "point_quantile": 0.75,
    "pretext_step": None,
    "projection_dim": 256,
    "random_seed": 2027,
    "regime_count": 1,
    "score_normalization": "none",
    "sequence_score_postprocess": "none",
    "sparse_derivative_weight": 1.0,
    "sparse_top_r": 4,
    "sparse_weight": 0.0,
    "stride": 1,
    "structure_weight": 0.0,
    "top_k": 3,
    "use_revin": True,
    "drift_corr_weight": 1.0,
    "drift_mean_weight": 1.0,
    "drift_weight": 0.0,
    "drift_window_lengths": (64, 128, 256),
    "drift_auto_blocks": False,
    "drift_block_min_size": 2,
    "drift_block_similarity_threshold": 0.6,
    "drift_block_window_length": 128,
    "drift_regime_channel_auto": False,
    "drift_regime_count": 1,
    "drift_regime_top_k": 4,
    "drift_regime_window_length": 128,
    "relative_drift_weight": 0.0,
    "relative_drift_slack": 0.5,
    "relative_drift_top_r": 4,
    "relative_drift_channel_top_k": 8,
    "relative_drift_name_prefixes": ("T",),
    "duration_pool_lengths": (),
    "duration_pool_positive_only": True,
}


@dataclass
class PaAnoConfig:
    batch_size: int = 512
    encoder_d_ff: int = 256
    encoder_d_model: int = 128
    encoder_dropout: float = 0.1
    encoder_num_heads: int = 4
    encoder_num_layers: int = 2
    encoder_patch_len: int = 16
    encoder_patch_stride: int = 8
    encoder_pooling: str = "mean"
    encoder_type: str = "cnn"
    kernel_sizes: tuple[int, ...] = (7, 5, 3, 3)
    layers: tuple[int, ...] = (128, 256, 128, 64)
    lr: float = 1e-4
    memory_bank_ratio: float = 0.1
    num_iter: int = 200
    patch_size: int = 96
    point_aggregation: str = "mean"
    point_quantile: float = 0.75
    pretext_step: Optional[int] = None
    projection_dim: int = 256
    random_seed: int = 2027
    regime_count: int = 1
    score_normalization: str = "none"
    sequence_score_postprocess: str = "none"
    sparse_derivative_weight: float = 1.0
    sparse_top_r: int = 4
    sparse_weight: float = 0.0
    stride: int = 1
    structure_weight: float = 0.0
    top_k: int = 3
    use_revin: bool = True
    drift_corr_weight: float = 1.0
    drift_mean_weight: float = 1.0
    drift_weight: float = 0.0
    drift_window_lengths: tuple[int, ...] = (64, 128, 256)
    drift_auto_blocks: bool = False
    drift_block_min_size: int = 2
    drift_block_similarity_threshold: float = 0.6
    drift_block_window_length: int = 128
    drift_regime_channel_auto: bool = False
    drift_regime_count: int = 1
    drift_regime_top_k: int = 4
    drift_regime_window_length: int = 128
    relative_drift_weight: float = 0.0
    relative_drift_slack: float = 0.5
    relative_drift_top_r: int = 4
    relative_drift_channel_top_k: int = 8
    relative_drift_name_prefixes: tuple[str, ...] = ("T",)
    duration_pool_lengths: tuple[int, ...] = ()
    duration_pool_positive_only: bool = True

    def __init__(self, **kwargs) -> None:
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.kernel_sizes = tuple(self.kernel_sizes)
        self.layers = tuple(self.layers)
        self.drift_window_lengths = tuple(int(length) for length in self.drift_window_lengths)
        self.drift_block_window_length = int(self.drift_block_window_length)
        self.drift_regime_window_length = int(self.drift_regime_window_length)
        self.relative_drift_name_prefixes = tuple(str(prefix) for prefix in self.relative_drift_name_prefixes)
        self.duration_pool_lengths = tuple(int(length) for length in self.duration_pool_lengths)


class PaAno:
    """Patch-based anomaly detector from the official PaAno codebase.

    The original release scores a whole CSV directory at once. The benchmark
    API needs `detect_fit` / `detect_score`, so this wrapper keeps only the
    train-on-normal / score-test path.
    """

    def __init__(self, **kwargs) -> None:
        self.config = PaAnoConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Union[PatchEncoder, TransformerPatchEncoder]] = None
        self.feature_names: Optional[list[str]] = None
        self.memory_bank: Optional[torch.Tensor] = None
        self.regime_banks: Optional[list[torch.Tensor]] = None
        self.regime_centroids: Optional[torch.Tensor] = None
        self.patch_size: int = int(self.config.patch_size)
        self.structure_reference: Optional[torch.Tensor] = None
        self.patch_score_stats: Optional[tuple[float, float]] = None
        self.channel_diff_mean: Optional[np.ndarray] = None
        self.channel_diff_std: Optional[np.ndarray] = None
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        self.drift_reference_corr: Optional[np.ndarray] = None
        self.drift_blocks: Optional[list[np.ndarray]] = None
        self.drift_reference_bundle: Optional[dict[int, dict[str, object]]] = None
        self.drift_regime_channel_indices: Optional[np.ndarray] = None
        self.drift_regime_centroids: Optional[np.ndarray] = None
        self.drift_regime_feature_mean: Optional[np.ndarray] = None
        self.drift_regime_feature_scale: Optional[np.ndarray] = None
        self.drift_score_stats: Optional[tuple[float, float]] = None
        self.relative_drift_channels: Optional[np.ndarray] = None
        self.sparse_score_stats: Optional[tuple[float, float]] = None
        self.structure_score_stats: Optional[tuple[float, float]] = None

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def _set_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    @staticmethod
    def _to_numpy(frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if values.ndim == 1:
            values = values[:, None]
        return values

    @staticmethod
    def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        location = float(np.median(values)) if values.size else 0.0
        scale = float(np.median(np.abs(values - location)) * 1.4826) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(values)) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return location, scale

    def _normalize_scores(self, values: np.ndarray, stats: Optional[tuple[float, float]]) -> np.ndarray:
        if stats is None or self.config.score_normalization == "none":
            return np.asarray(values, dtype=np.float32)
        location, scale = stats
        return ((np.asarray(values, dtype=np.float32) - location) / scale).astype(np.float32)

    @staticmethod
    def _safe_corrcoef(values: np.ndarray) -> np.ndarray:
        if values.shape[0] <= 1:
            return np.zeros((values.shape[1], values.shape[1]), dtype=np.float32)
        corr = np.corrcoef(values.T)
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _normalize_feature_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = values.mean(axis=0, dtype=np.float32)
        scale = values.std(axis=0, dtype=np.float32)
        scale[scale < 1e-6] = 1.0
        normalized = (values - mean) / scale
        return normalized.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)

    @staticmethod
    def _connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
        num_nodes = int(adjacency.shape[0])
        visited = np.zeros(num_nodes, dtype=bool)
        components: list[np.ndarray] = []
        for start in range(num_nodes):
            if visited[start]:
                continue
            stack = [start]
            component = []
            visited[start] = True
            while stack:
                node = stack.pop()
                component.append(node)
                neighbours = np.flatnonzero(adjacency[node])
                for neighbour in neighbours:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        stack.append(int(neighbour))
            components.append(np.array(sorted(component), dtype=np.int64))
        return components

    def _window_summary_features(
        self,
        values: np.ndarray,
        window_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return trailing window mean/std/slope summaries for each point."""

        values = np.asarray(values, dtype=np.float32)
        num_points, num_channels = values.shape
        if num_points == 0:
            empty = np.zeros((0, num_channels), dtype=np.float32)
            return empty, empty, empty
        window_length = max(2, min(int(window_length), num_points))
        prefix = np.vstack(
            [np.zeros((1, num_channels), dtype=np.float32), np.cumsum(values, axis=0, dtype=np.float32)]
        )
        prefix_sq = np.vstack(
            [np.zeros((1, num_channels), dtype=np.float32), np.cumsum(values * values, axis=0, dtype=np.float32)]
        )
        indices = np.arange(num_points, dtype=np.int64)
        starts = np.maximum(0, indices - window_length + 1)
        counts = (indices - starts + 1).astype(np.float32)
        sums = prefix[indices + 1] - prefix[starts]
        sums_sq = prefix_sq[indices + 1] - prefix_sq[starts]
        means = sums / counts[:, None]
        variances = np.clip(sums_sq / counts[:, None] - means * means, 0.0, None)
        stds = np.sqrt(variances, dtype=np.float32)
        denom = np.maximum(counts - 1.0, 1.0)
        slopes = (values[indices] - values[starts]) / denom[:, None]
        return means.astype(np.float32), stds.astype(np.float32), slopes.astype(np.float32)

    def _discover_channel_blocks(self, train_values: np.ndarray) -> list[np.ndarray]:
        """Group channels by shared train-only window statistics.

        This avoids hand-picking thermal blocks. Channels whose window means,
        slopes, and variances evolve together become one block.
        """

        num_channels = train_values.shape[1]
        if num_channels <= 1:
            return [np.arange(num_channels, dtype=np.int64)]

        means, stds, slopes = self._window_summary_features(
            train_values,
            self.config.drift_block_window_length,
        )
        summaries = []
        for summary in (means, stds, slopes):
            normalized, _, _ = self._normalize_feature_columns(summary)
            summaries.append(normalized)
        channel_signatures = np.concatenate(
            [summary.T for summary in summaries],
            axis=1,
        )
        similarity = np.abs(np.corrcoef(channel_signatures))
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(similarity, 1.0)
        adjacency = similarity >= float(self.config.drift_block_similarity_threshold)
        components = self._connected_components(adjacency)
        blocks: list[np.ndarray] = []
        for component in components:
            if len(component) >= int(self.config.drift_block_min_size):
                blocks.append(component.astype(np.int64, copy=False))
        used = {int(index) for block in blocks for index in block}
        for index in range(num_channels):
            if index not in used:
                blocks.append(np.array([index], dtype=np.int64))
        return blocks or [np.arange(num_channels, dtype=np.int64)]

    def _regime_feature_matrix(
        self,
        values: np.ndarray,
        window_length: int,
        channel_indices: np.ndarray,
    ) -> np.ndarray:
        means, stds, slopes = self._window_summary_features(values, window_length)
        return np.concatenate(
            [
                means[:, channel_indices],
                stds[:, channel_indices],
                slopes[:, channel_indices],
            ],
            axis=1,
        ).astype(np.float32)

    def _regime_separation_scores(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        slopes: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Score channels by between-regime / within-regime separation."""

        num_channels = means.shape[1]
        scores = np.zeros(num_channels, dtype=np.float32)
        unique_labels = np.unique(labels)
        for channel_index in range(num_channels):
            between = 0.0
            within = 0.0
            for summary in (means[:, channel_index], stds[:, channel_index], slopes[:, channel_index]):
                global_mean = float(summary.mean())
                for label in unique_labels:
                    mask = labels == label
                    if not np.any(mask):
                        continue
                    values = summary[mask]
                    cluster_mean = float(values.mean())
                    between += float(mask.sum()) * (cluster_mean - global_mean) ** 2
                    within += float(np.square(values - cluster_mean).sum())
            scores[channel_index] = between / (within + 1e-6)
        return scores

    def _fit_regime_selector(self, train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select regime channels and fit train-only operating-point centroids."""

        num_channels = train_values.shape[1]
        window_length = max(2, min(int(self.config.drift_regime_window_length), len(train_values)))
        means, stds, slopes = self._window_summary_features(train_values, window_length)
        full_features = np.concatenate([means, stds, slopes], axis=1).astype(np.float32)
        normalized_full, _, _ = self._normalize_feature_columns(full_features)
        n_clusters = max(1, min(int(self.config.drift_regime_count), len(normalized_full)))
        if n_clusters <= 1 or len(normalized_full) <= 1:
            channel_indices = np.arange(num_channels, dtype=np.int64)
            selected_features = self._regime_feature_matrix(train_values, window_length, channel_indices)
            normalized_selected, feature_mean, feature_scale = self._normalize_feature_columns(selected_features)
            centroids = normalized_selected[:1].mean(axis=0, keepdims=True)
            return channel_indices, centroids.astype(np.float32), feature_mean.astype(np.float32), feature_scale.astype(np.float32)

        initial_kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=self.config.random_seed,
            batch_size=max(512, n_clusters),
            max_iter=100,
            n_init=3,
            reassignment_ratio=0.01,
        )
        initial_labels = initial_kmeans.fit_predict(normalized_full)
        separation = self._regime_separation_scores(means, stds, slopes, initial_labels)
        top_k = max(1, min(int(self.config.drift_regime_top_k), num_channels))
        ranked = np.argsort(separation)[::-1]
        channel_indices = np.sort(ranked[:top_k]).astype(np.int64)
        selected_features = self._regime_feature_matrix(train_values, window_length, channel_indices)
        normalized_selected, feature_mean, feature_scale = self._normalize_feature_columns(selected_features)
        final_kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=self.config.random_seed,
            batch_size=max(512, n_clusters),
            max_iter=100,
            n_init=3,
            reassignment_ratio=0.01,
        )
        final_kmeans.fit(normalized_selected)
        return (
            channel_indices,
            final_kmeans.cluster_centers_.astype(np.float32),
            feature_mean.astype(np.float32),
            feature_scale.astype(np.float32),
        )

    def _assign_regimes(self, values: np.ndarray) -> np.ndarray:
        if (
            self.drift_regime_channel_indices is None
            or self.drift_regime_centroids is None
            or self.drift_regime_feature_mean is None
            or self.drift_regime_feature_scale is None
        ):
            return np.zeros(len(values), dtype=np.int64)
        features = self._regime_feature_matrix(
            values,
            self.config.drift_regime_window_length,
            self.drift_regime_channel_indices,
        )
        normalized = (features - self.drift_regime_feature_mean) / self.drift_regime_feature_scale
        distances = np.sum(
            (normalized[:, None, :] - self.drift_regime_centroids[None, :, :]) ** 2,
            axis=2,
            dtype=np.float32,
        )
        return np.argmin(distances, axis=1).astype(np.int64)

    def _rolling_correlation_matrices(self, values: np.ndarray, window_length: int) -> np.ndarray:
        """Return trailing rolling correlation matrices for one channel block."""

        num_points, num_channels = values.shape
        if num_channels <= 1:
            return np.ones((num_points, num_channels, num_channels), dtype=np.float32)
        values = values.astype(np.float32, copy=False)
        prefix = np.vstack(
            [np.zeros((1, num_channels), dtype=np.float32), np.cumsum(values, axis=0, dtype=np.float32)]
        )
        pairwise = (values[:, :, None] * values[:, None, :]).reshape(num_points, num_channels * num_channels)
        pairwise_prefix = np.vstack(
            [
                np.zeros((1, num_channels * num_channels), dtype=np.float32),
                np.cumsum(pairwise, axis=0, dtype=np.float32),
            ]
        )
        indices = np.arange(num_points, dtype=np.int64)
        starts = np.maximum(0, indices - window_length + 1)
        counts = (indices - starts + 1).astype(np.float32)
        sums = prefix[indices + 1] - prefix[starts]
        means = sums / counts[:, None]
        pairwise_sums = pairwise_prefix[indices + 1] - pairwise_prefix[starts]
        exx = (pairwise_sums / counts[:, None]).reshape(num_points, num_channels, num_channels)
        cov = exx - means[:, :, None] * means[:, None, :]
        std = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 1e-6, None)).astype(np.float32)
        denom = std[:, :, None] * std[:, None, :]
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-6)
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _fit_drift_references(self, train_values: np.ndarray) -> None:
        """Fit blockwise and optional regime-conditioned drift references."""

        num_channels = train_values.shape[1]
        if self.config.drift_auto_blocks:
            self.drift_blocks = self._discover_channel_blocks(train_values)
        else:
            self.drift_blocks = [np.arange(num_channels, dtype=np.int64)]

        if self.config.drift_regime_channel_auto:
            (
                self.drift_regime_channel_indices,
                self.drift_regime_centroids,
                self.drift_regime_feature_mean,
                self.drift_regime_feature_scale,
            ) = self._fit_regime_selector(train_values)
        else:
            self.drift_regime_channel_indices = None
            self.drift_regime_centroids = None
            self.drift_regime_feature_mean = None
            self.drift_regime_feature_scale = None

        regime_labels = self._assign_regimes(train_values)
        num_regimes = int(regime_labels.max()) + 1 if regime_labels.size else 1
        self.drift_reference_bundle = {}
        for window_length in self.config.drift_window_lengths:
            if window_length <= 1:
                continue
            means, _, _ = self._window_summary_features(train_values, window_length)
            window_refs = []
            for block in self.drift_blocks:
                block_values = train_values[:, block]
                block_means = means[:, block]
                block_corr = self._rolling_correlation_matrices(block_values, window_length)
                mean_ref = np.zeros((num_regimes, len(block)), dtype=np.float32)
                mean_scale = np.ones((num_regimes, len(block)), dtype=np.float32)
                corr_ref = np.zeros((num_regimes, len(block), len(block)), dtype=np.float32)
                for regime_index in range(num_regimes):
                    mask = regime_labels == regime_index
                    if not np.any(mask):
                        mask = np.ones_like(regime_labels, dtype=bool)
                    regime_means = block_means[mask]
                    mean_ref[regime_index] = regime_means.mean(axis=0, dtype=np.float32)
                    regime_scale = regime_means.std(axis=0, dtype=np.float32)
                    regime_scale[regime_scale < 1e-6] = 1.0
                    mean_scale[regime_index] = regime_scale
                    corr_ref[regime_index] = block_corr[mask].mean(axis=0, dtype=np.float32)
                window_refs.append(
                    {
                        "channels": block,
                        "mean_ref": mean_ref,
                        "mean_scale": mean_scale,
                        "corr_ref": corr_ref,
                    }
                )
            self.drift_reference_bundle[int(window_length)] = {
                "regime_labels": regime_labels.copy(),
                "window_refs": window_refs,
            }

    def _channel_zscores(self, values: np.ndarray) -> np.ndarray:
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError("PaAno channel statistics are missing. Fit the model again.")
        return np.abs((values - self.channel_mean) / self.channel_std).astype(np.float32)

    def _channel_diff_zscores(self, values: np.ndarray) -> np.ndarray:
        if self.channel_diff_mean is None or self.channel_diff_std is None:
            raise RuntimeError("PaAno derivative statistics are missing. Fit the model again.")
        diffs = np.diff(values, axis=0, prepend=values[:1])
        return np.abs((diffs - self.channel_diff_mean) / self.channel_diff_std).astype(np.float32)

    def _sparse_channel_scores(self, values: np.ndarray) -> np.ndarray:
        """Return a top-r sparse-channel anomaly score per point.

        PaAno's multivariate patch score can dilute a one-or-few-channel event.
        This branch stays simple on purpose: combine level and derivative z-scores
        per channel, then keep only the strongest channels at each point.
        """

        point_z = self._channel_zscores(values)
        diff_z = self._channel_diff_zscores(values)
        combined = point_z + float(self.config.sparse_derivative_weight) * diff_z
        top_r = max(1, min(int(self.config.sparse_top_r), combined.shape[1]))
        if top_r >= combined.shape[1]:
            return combined.mean(axis=1).astype(np.float32)
        partition = np.partition(combined, combined.shape[1] - top_r, axis=1)
        return partition[:, -top_r:].mean(axis=1).astype(np.float32)

    def _select_relative_drift_channels(self, train_values: np.ndarray) -> np.ndarray:
        """Pick slow subsystem channels for within-run drift scoring.

        Prefer explicit process-style prefixes when they exist. Otherwise fall
        back to smooth channels with high lag-one correlation and low roughness.
        """

        num_channels = train_values.shape[1]
        top_k = max(1, min(int(self.config.relative_drift_channel_top_k), num_channels))
        if self.feature_names:
            matched = [
                index
                for index, name in enumerate(self.feature_names)
                if any(str(name).startswith(prefix) for prefix in self.config.relative_drift_name_prefixes)
            ]
            if len(matched) >= min(3, top_k):
                return np.array(matched[:top_k], dtype=np.int64)

        if len(train_values) <= 2:
            return np.arange(top_k, dtype=np.int64)

        prev = train_values[:-1]
        curr = train_values[1:]
        prev_centered = prev - prev.mean(axis=0, dtype=np.float32)
        curr_centered = curr - curr.mean(axis=0, dtype=np.float32)
        denom = np.sqrt(
            np.clip(np.sum(prev_centered * prev_centered, axis=0) * np.sum(curr_centered * curr_centered, axis=0), 1e-6, None)
        )
        lag_corr = np.divide(
            np.sum(prev_centered * curr_centered, axis=0),
            denom,
            out=np.zeros(num_channels, dtype=np.float32),
            where=denom > 1e-6,
        )
        level_std = train_values.std(axis=0, dtype=np.float32)
        level_std[level_std < 1e-6] = 1.0
        diff_std = np.diff(train_values, axis=0).std(axis=0, dtype=np.float32)
        smoothness = lag_corr - (diff_std / level_std)
        ranked = np.argsort(smoothness)[::-1]
        return np.sort(ranked[:top_k]).astype(np.int64)

    def _relative_drift_scores(self, values: np.ndarray) -> np.ndarray:
        """Score sustained within-run drift on slow channels with CUSUM.

        Smooth temperature drifts do not create sharp patch novelty peaks. A
        one-sided cumulative residual catches the same broad directional change
        that an operator would notice on a trend plot.
        """

        if self.relative_drift_channels is None or len(self.relative_drift_channels) == 0:
            return np.zeros(len(values), dtype=np.float32)

        signed_z = ((values - self.channel_mean) / self.channel_std).astype(np.float32)
        selected = signed_z[:, self.relative_drift_channels]
        top_r = max(1, min(int(self.config.relative_drift_top_r), selected.shape[1]))
        positive = np.zeros(selected.shape[1], dtype=np.float32)
        negative = np.zeros(selected.shape[1], dtype=np.float32)
        raw_scores = np.zeros(len(selected), dtype=np.float32)
        slack = float(self.config.relative_drift_slack)
        for index, point in enumerate(selected):
            positive = np.maximum(0.0, positive + point - slack)
            negative = np.maximum(0.0, negative - point - slack)
            current = np.maximum(positive, negative)
            if top_r >= current.shape[0]:
                raw_scores[index] = float(current.mean())
            else:
                partition = np.partition(current, current.shape[0] - top_r)
                raw_scores[index] = float(partition[-top_r:].mean())
        location, scale = self._robust_location_scale(raw_scores)
        return ((raw_scores - location) / scale).astype(np.float32)

    def _rolling_mean_scores(self, z_values: np.ndarray, window_length: int) -> np.ndarray:
        """Return trailing rolling mean drift scores for one window length."""

        num_points, num_channels = z_values.shape
        prefix = np.vstack(
            [np.zeros((1, num_channels), dtype=np.float32), np.cumsum(z_values, axis=0, dtype=np.float32)]
        )
        indices = np.arange(num_points, dtype=np.int64)
        starts = np.maximum(0, indices - window_length + 1)
        counts = (indices - starts + 1).astype(np.float32)
        sums = prefix[indices + 1] - prefix[starts]
        return (sums / counts[:, None]).mean(axis=1).astype(np.float32)

    def _rolling_correlation_scores(self, values: np.ndarray, window_length: int) -> np.ndarray:
        """Return trailing rolling correlation shift scores for one window length."""

        if self.drift_reference_corr is None:
            raise RuntimeError("PaAno drift reference is missing. Fit the model again.")

        num_points, num_channels = values.shape
        values = values.astype(np.float32, copy=False)
        prefix = np.vstack(
            [np.zeros((1, num_channels), dtype=np.float32), np.cumsum(values, axis=0, dtype=np.float32)]
        )
        pairwise = (values[:, :, None] * values[:, None, :]).reshape(num_points, num_channels * num_channels)
        pairwise_prefix = np.vstack(
            [
                np.zeros((1, num_channels * num_channels), dtype=np.float32),
                np.cumsum(pairwise, axis=0, dtype=np.float32),
            ]
        )
        indices = np.arange(num_points, dtype=np.int64)
        starts = np.maximum(0, indices - window_length + 1)
        counts = (indices - starts + 1).astype(np.float32)

        sums = prefix[indices + 1] - prefix[starts]
        means = sums / counts[:, None]
        pairwise_sums = pairwise_prefix[indices + 1] - pairwise_prefix[starts]
        exx = (pairwise_sums / counts[:, None]).reshape(num_points, num_channels, num_channels)
        cov = exx - means[:, :, None] * means[:, None, :]
        std = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 1e-6, None))
        denom = std[:, :, None] * std[:, None, :]
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-6)
        return np.mean(np.abs(corr - self.drift_reference_corr[None, :, :]), axis=(1, 2)).astype(np.float32)

    def _slow_drift_scores(self, values: np.ndarray) -> np.ndarray:
        """Return one broad multivariate drift score per point.

        The hard NoBOOM failures are smooth, temperature-dominated drifts across
        many channels. This branch looks at larger trailing windows and combines
        broad mean drift with correlation shift, then keeps the strongest window.
        """

        z_values = self._channel_zscores(values)
        regime_labels = self._assign_regimes(values)
        branch_scores = []
        for window_length in self.config.drift_window_lengths:
            if window_length <= 1:
                continue
            bundle = self.drift_reference_bundle.get(int(window_length)) if self.drift_reference_bundle else None
            if bundle is None:
                mean_score = self._rolling_mean_scores(z_values, window_length)
                corr_score = self._rolling_correlation_scores(values, window_length)
                branch_scores.append(
                    self.config.drift_mean_weight * mean_score
                    + self.config.drift_corr_weight * corr_score
                )
                continue

            means, _, _ = self._window_summary_features(values, window_length)
            block_scores = []
            for block_ref in bundle["window_refs"]:
                block = block_ref["channels"]
                block_means = means[:, block]
                mean_ref = block_ref["mean_ref"][regime_labels]
                mean_scale = block_ref["mean_scale"][regime_labels]
                mean_score = np.mean(
                    np.abs((block_means - mean_ref) / mean_scale),
                    axis=1,
                    dtype=np.float32,
                )
                block_values = values[:, block]
                corr = self._rolling_correlation_matrices(block_values, window_length)
                corr_ref = block_ref["corr_ref"][regime_labels]
                corr_score = np.mean(np.abs(corr - corr_ref), axis=(1, 2), dtype=np.float32)
                block_scores.append(
                    self.config.drift_mean_weight * mean_score
                    + self.config.drift_corr_weight * corr_score
                )
            if block_scores:
                branch_scores.append(np.max(np.stack(block_scores, axis=0), axis=0).astype(np.float32))
        if not branch_scores:
            return np.zeros(len(values), dtype=np.float32)
        return np.max(np.stack(branch_scores, axis=0), axis=0).astype(np.float32)

    def _duration_pool_scores(self, point_scores: np.ndarray) -> np.ndarray:
        """Emphasize sustained alarms over isolated peaks after score fusion."""

        if not self.config.duration_pool_lengths:
            return np.asarray(point_scores, dtype=np.float32)

        base = np.asarray(point_scores, dtype=np.float32)
        if self.config.duration_pool_positive_only:
            base = np.maximum(base, 0.0)
        prefix = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(base, dtype=np.float32)])
        pooled = base.copy()
        indices = np.arange(len(base), dtype=np.int64)
        for window_length in self.config.duration_pool_lengths:
            if window_length <= 1:
                continue
            starts = np.maximum(0, indices - window_length + 1)
            counts = (indices - starts + 1).astype(np.float32)
            sums = prefix[indices + 1] - prefix[starts]
            sustained = sums / np.sqrt(counts)
            pooled = np.maximum(pooled, sustained.astype(np.float32))
        return pooled.astype(np.float32)

    def _build_encoder(self, in_channels: int) -> PatchEncoder | TransformerPatchEncoder:
        if self.config.encoder_type == "cnn":
            return PatchEncoder(
                in_channels=in_channels,
                projection_dim=self.config.projection_dim,
                layers=tuple(self.config.layers),
                kernel_sizes=tuple(self.config.kernel_sizes),
                use_revin=self.config.use_revin,
            ).to(self.device)
        return TransformerPatchEncoder(
            in_channels=in_channels,
            seq_len=self.patch_size,
            projection_dim=self.config.projection_dim,
            encoder_type=self.config.encoder_type,
            d_model=self.config.encoder_d_model,
            num_heads=self.config.encoder_num_heads,
            num_layers=self.config.encoder_num_layers,
            d_ff=self.config.encoder_d_ff,
            dropout=self.config.encoder_dropout,
            pooling=self.config.encoder_pooling,
            patch_len=min(self.config.encoder_patch_len, self.patch_size),
            patch_stride=max(1, min(self.config.encoder_patch_stride, self.patch_size)),
            use_revin=self.config.use_revin,
        ).to(self.device)

    def detect_fit(self, train_data: pd.DataFrame, train_label: Optional[pd.DataFrame] = None) -> None:
        self._set_seed()
        self.feature_names = list(train_data.columns)
        train_values = self._to_numpy(train_data)
        self.channel_mean = train_values.mean(axis=0, dtype=np.float32)
        self.channel_std = train_values.std(axis=0, dtype=np.float32)
        self.channel_std[self.channel_std < 1e-6] = 1.0
        train_diffs = np.diff(train_values, axis=0, prepend=train_values[:1])
        self.channel_diff_mean = train_diffs.mean(axis=0, dtype=np.float32)
        self.channel_diff_std = train_diffs.std(axis=0, dtype=np.float32)
        self.channel_diff_std[self.channel_diff_std < 1e-6] = 1.0
        self.drift_reference_corr = self._safe_corrcoef(train_values)
        self.drift_blocks = None
        self.drift_reference_bundle = None
        self.drift_regime_channel_indices = None
        self.drift_regime_centroids = None
        self.drift_regime_feature_mean = None
        self.drift_regime_feature_scale = None
        self.patch_size = max(2, min(int(self.config.patch_size), len(train_values)))
        self.relative_drift_channels = None

        if self.config.use_revin:
            train_scaled = train_values
        else:
            self.scaler.fit(train_values)
            train_scaled = transform_frame(self.scaler, train_data).to_numpy(dtype=np.float32)

        train_patches, train_indices = create_patches(
            train_scaled,
            patch_size=self.patch_size,
            stride=self.config.stride,
        )
        train_loader = make_patch_loader(
            train_patches,
            train_indices,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.model = self._build_encoder(train_patches.shape[1])

        train_encoder(
            self.model,
            train_loader,
            train_patches,
            self.device,
            PaAnoTrainConfig(
                num_iter=self.config.num_iter,
                pretext_step=self.config.pretext_step or self.patch_size,
                lr=self.config.lr,
            ),
        )
        self.memory_bank = create_memory_bank(
            self.model,
            train_loader,
            self.device,
            compression=self.config.memory_bank_ratio,
        )
        train_eval_loader = make_patch_loader(
            train_patches,
            train_indices,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        self.regime_banks = None
        self.regime_centroids = None
        if self.config.regime_count > 1:
            self.regime_banks, self.regime_centroids = create_regime_memory_bank(
                self.model,
                train_eval_loader,
                self.device,
                num_regimes=int(self.config.regime_count),
                compression=self.config.memory_bank_ratio,
            )
        train_patch_scores = calculate_patch_scores(
            self.model,
            train_eval_loader,
            self.memory_bank,
            self.device,
            top_k=self.config.top_k,
        )
        if self.config.score_normalization != "none":
            self.patch_score_stats = self._robust_location_scale(train_patch_scores)
        if self.config.sparse_weight > 0.0:
            train_sparse_scores = self._sparse_channel_scores(train_values)
            self.sparse_score_stats = self._robust_location_scale(train_sparse_scores)
        if self.config.drift_weight > 0.0:
            self._fit_drift_references(train_values)
            train_drift_scores = self._slow_drift_scores(train_values)
            self.drift_score_stats = self._robust_location_scale(train_drift_scores)
        if self.config.relative_drift_weight > 0.0:
            self.relative_drift_channels = self._select_relative_drift_channels(train_values)
        if self.config.structure_weight > 0.0:
            self.structure_reference = correlation_reference(
                train_patches,
                self.device,
                batch_size=self.config.batch_size,
            ).detach().cpu()
            train_structure_scores = correlation_scores(
                train_patches,
                self.structure_reference,
                self.device,
                batch_size=self.config.batch_size,
            )
            self.structure_score_stats = self._robust_location_scale(train_structure_scores)
        self.model.eval()

    def detect_score(self, test: pd.DataFrame):
        if self.model is None or self.memory_bank is None:
            raise RuntimeError("PaAno must be fitted before calling detect_score.")

        if self.config.use_revin:
            test_values = self._to_numpy(test)
        else:
            test_values = transform_frame(self.scaler, test).to_numpy(dtype=np.float32)

        test_patches, test_indices = create_patches(
            test_values,
            patch_size=self.patch_size,
            stride=self.config.stride,
        )
        test_loader = make_patch_loader(
            test_patches,
            test_indices,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        if self.config.regime_count > 1:
            if self.regime_banks is None or self.regime_centroids is None:
                raise RuntimeError("PaAno regime memory bank is missing. Fit the model again.")
            patch_scores = calculate_patch_scores_by_regime(
                self.model,
                test_loader,
                self.regime_banks,
                self.regime_centroids,
                self.device,
                top_k=self.config.top_k,
            )
        else:
            patch_scores = calculate_patch_scores(
                self.model,
                test_loader,
                self.memory_bank,
                self.device,
                top_k=self.config.top_k,
            )
        patch_scores = self._normalize_scores(patch_scores, self.patch_score_stats)
        if self.config.sparse_weight > 0.0:
            sparse_scores = self._normalize_scores(
                self._sparse_channel_scores(test_values),
                self.sparse_score_stats,
            )
        else:
            sparse_scores = None
        if self.config.drift_weight > 0.0:
            drift_scores = self._normalize_scores(
                self._slow_drift_scores(test_values),
                self.drift_score_stats,
            )
        else:
            drift_scores = None
        if self.config.structure_weight > 0.0:
            if self.structure_reference is None:
                raise RuntimeError("PaAno structure reference is missing. Fit the model again.")
            structure_patch_scores = correlation_scores(
                test_patches,
                self.structure_reference,
                self.device,
                batch_size=self.config.batch_size,
            )
            structure_patch_scores = self._normalize_scores(
                structure_patch_scores,
                self.structure_score_stats,
            )
            patch_scores = patch_scores + self.config.structure_weight * structure_patch_scores
        point_scores = distribute_patch_scores_to_points(
            patch_scores,
            patch_size=self.patch_size,
            num_points=len(test),
            aggregation=self.config.point_aggregation,
            quantile=self.config.point_quantile,
        )
        if sparse_scores is not None:
            point_scores = point_scores + self.config.sparse_weight * sparse_scores
        if drift_scores is not None:
            point_scores = point_scores + self.config.drift_weight * drift_scores
        if self.config.relative_drift_weight > 0.0:
            relative_scores = self._relative_drift_scores(test_values)
            point_scores = point_scores + self.config.relative_drift_weight * relative_scores
        point_scores = self._duration_pool_scores(point_scores)
        if self.config.sequence_score_postprocess == "robust_zscore":
            point_scores = self._normalize_scores(
                point_scores,
                self._robust_location_scale(point_scores),
            )
        return point_scores, point_scores
