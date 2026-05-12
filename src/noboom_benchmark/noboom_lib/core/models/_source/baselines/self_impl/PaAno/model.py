"""Minimal PaAno modules adapted from the official ICLR 2026 release.

The official repository is compact already. This port keeps the same
patch-based encoder, pretext + triplet training objective, and memory-bank
scoring path, while trimming unrelated dataset runners and file I/O.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from numpy.lib.stride_tricks import sliding_window_view

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.NeutralAD.encoders import PatchTSTEncoder


class RevIN1d(nn.Module):
    """Small 1D RevIN port used by the official PaAno encoder."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        min_sigma: float = 1e-5,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.min_sigma = min_sigma
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self._mu = None
        self._sigma = None

    @torch.no_grad()
    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return mu, sigma

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        self._mu, self._sigma = self._stats(x)
        x_hat = (x - self._mu) / self._sigma
        if self.affine:
            x_hat = x_hat * self.weight + self.bias
        return x_hat


class PatchEncoder(nn.Module):
    """Official PaAno convolutional patch encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        projection_dim: int = 256,
        layers: tuple[int, ...] = (128, 256, 128, 64),
        kernel_sizes: tuple[int, ...] = (7, 5, 3, 3),
        use_revin: bool = True,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        revin_min_sigma: float = 1e-5,
    ) -> None:
        super().__init__()
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine,
            )
        self.convblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        layers[i - 1] if i > 0 else in_channels,
                        layers[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        padding=kernel_sizes[i] // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(layers[i]),
                    nn.ReLU(inplace=True),
                )
                for i in range(len(layers))
            ]
        )
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(layers[-1], projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.classification_head = nn.Linear(layers[-1] * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_projection: bool = False,
    ) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin.norm(x)
        for block in self.convblocks:
            x = block(x)
        embedding = self.fc_embedding(x).flatten(start_dim=1)
        if return_embedding:
            return embedding
        if return_projection:
            return self.projection_head(embedding)
        raise ValueError("PatchEncoder.forward only supports embedding/projection paths.")

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)

    def projection(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection_head(embedding)


class TransformerPatchEncoder(nn.Module):
    """PaAno encoder variant backed by a lightweight transformer encoder.

    This keeps PaAno's metric-learning objective and memory-bank scoring, but
    swaps the local feature extractor. The goal is to test whether the missed
    NoBOOM event is a representation issue rather than a scoring issue.
    """

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        projection_dim: int = 256,
        encoder_type: str = "patchtst",
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
        patch_len: int = 16,
        patch_stride: int = 8,
        use_revin: bool = True,
        revin_affine: bool = False,
        revin_eps: float = 1e-5,
        revin_min_sigma: float = 1e-5,
    ) -> None:
        super().__init__()
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(
                num_channels=in_channels,
                eps=revin_eps,
                min_sigma=revin_min_sigma,
                affine=revin_affine,
            )
        if encoder_type == "patchtst":
            self.encoder = PatchTSTEncoder(
                seq_len=seq_len,
                patch_len=min(patch_len, seq_len),
                patch_stride=max(1, min(patch_stride, patch_len)),
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                dropout=dropout,
                pooling=pooling,
            )
        else:
            raise ValueError(f"Unsupported PaAno transformer encoder_type: {encoder_type}")
        self.embedding_dim = d_model
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.classification_head = nn.Linear(d_model * 2, 1)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin.norm(x)
        return self.encoder(x)

    def projection(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection_head(embedding)


class IndexedPatchDataset(Dataset):
    """Dataset returning a patch and its relative patch index."""

    def __init__(self, patches: torch.Tensor, indices: np.ndarray) -> None:
        self.patches = patches
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patches[idx], self.indices[idx]


def create_patches(values: np.ndarray, patch_size: int, stride: int) -> tuple[torch.Tensor, np.ndarray]:
    """Create `(N, C, L)` sliding patches from a `(T, C)` sequence."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("PaAno expects a 2D array of shape [time, channels].")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")
    if array.shape[0] == 0:
        raise ValueError("PaAno cannot build patches from an empty sequence.")

    if array.shape[0] < patch_size:
        pad = np.repeat(array[:1], patch_size - array.shape[0], axis=0)
        array = np.concatenate([pad, array], axis=0)

    positions = np.arange(0, array.shape[0] - patch_size + 1, stride, dtype=np.int64)
    patch_array = np.stack([array[pos : pos + patch_size].T for pos in positions], axis=0)
    patches = torch.tensor(patch_array, dtype=torch.float32)
    return patches, positions


def make_patch_loader(
    patches: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = IndexedPatchDataset(patches, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def gradual_triplet_gradient(loss: torch.Tensor, factor: float = 0.01) -> torch.Tensor:
    """Small gradient scaling helper copied from the official training loop."""

    return loss.detach() + (loss - loss.detach()) * factor


@dataclass
class PaAnoTrainConfig:
    num_iter: int = 200
    pretext_step: int = 64
    lr: float = 1e-4
    radius: int = 2
    lambda_weight: float = 1.0
    temperature: float = 1.0
    num_rand_patches: int = 5


def train_encoder(
    model: PatchEncoder,
    train_loader: DataLoader,
    train_patches: torch.Tensor,
    device: torch.device,
    config: PaAnoTrainConfig,
) -> None:
    """Train the official PaAno encoder on sliding train patches."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion_pretext = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device),
        reduction="none",
    )

    offsets = torch.tensor(
        [*range(-config.radius, 0), *range(1, config.radius + 1)],
        dtype=torch.long,
    )
    train_patches = train_patches.cpu()
    total_len = len(train_patches)

    initial_lr = config.lr
    final_lr = config.lr / 10.0

    def cosine_annealed_lr(iteration: int) -> float:
        t = min(iteration, config.num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / config.num_iter))
        return final_lr + (initial_lr - final_lr) * cosine_factor

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    loader_iter = iter(train_loader)

    for iteration in range(1, config.num_iter + 1):
        try:
            batch_data, batch_indices = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch_data, batch_indices = next(loader_iter)

        lr = cosine_annealed_lr(iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch_data = batch_data.to(device, non_blocking=True)
        batch_indices = batch_indices.view(-1).cpu()
        batch_size = batch_data.shape[0]
        if batch_size < 2:
            continue
        mu = 1 if batch_data.shape[1] != 1 else 10

        candidates = batch_indices.view(-1, 1) + offsets.view(1, -1)
        valid = (candidates >= 0) & (candidates < total_len)
        sample_noise = torch.rand(candidates.shape, dtype=torch.float32)
        sample_noise = torch.where(valid, sample_noise, torch.full_like(sample_noise, -1.0))
        choice = sample_noise.argmax(dim=1)
        positive_indices = candidates.gather(1, choice.view(-1, 1)).squeeze(1)
        no_valid = valid.sum(dim=1) == 0
        positive_indices[no_valid] = batch_indices[no_valid]
        positives = train_patches[positive_indices].to(device, non_blocking=True)

        pretext_weight = 0.0
        if iteration < (config.num_iter / 10):
            pretext_weight = config.lambda_weight * (1 - iteration / (config.num_iter / 10))

        if pretext_weight > 0.0:
            pretext_indices = batch_indices - config.pretext_step
            valid_pretext = (pretext_indices >= 0) & (pretext_indices < total_len)
            safe_pretext_indices = pretext_indices.clamp(0, total_len - 1)
            pretext_patches = train_patches[safe_pretext_indices].clone()
            pretext_patches[~valid_pretext] = 0.0
            pretext_patches = pretext_patches.to(device, non_blocking=True)
            all_patches = torch.cat([batch_data, positives, pretext_patches], dim=0)
            embeddings = model.embedding(all_patches)
            h_anchor = embeddings[:batch_size]
            h_pos = embeddings[batch_size : 2 * batch_size]
            h_pretext = embeddings[2 * batch_size : 3 * batch_size]
        else:
            valid_pretext = None
            embeddings = model.embedding(torch.cat([batch_data, positives], dim=0))
            h_anchor = embeddings[:batch_size]
            h_pos = embeddings[batch_size : 2 * batch_size]
            h_pretext = None

        z_anchor = F.normalize(model.projection(h_anchor), dim=1)
        z_pos = F.normalize(model.projection(h_pos), dim=1)

        similarity = (z_anchor @ z_pos.T) / config.temperature
        positive_similarity = similarity.diag()
        similarity_neg = similarity.clone()
        similarity_neg.fill_diagonal_(float("inf"))
        hard_negative_distance = (1.0 - similarity_neg).max(dim=1).values
        positive_distance = 1.0 - positive_similarity
        triplet_loss = F.relu(positive_distance - hard_negative_distance + 0.5).mean() / mu
        triplet_loss = gradual_triplet_gradient(triplet_loss)

        if pretext_weight > 0.0 and h_pretext is not None and valid_pretext is not None:
            valid_pretext = valid_pretext.to(device)
            h_pre = h_pretext[valid_pretext]
            h_anchor_pre = h_anchor[valid_pretext]
            h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)

            anchor_indices = torch.arange(batch_size, device=device).repeat_interleave(
                config.num_rand_patches
            )
            rand_offsets = torch.randint(
                1,
                batch_size,
                (batch_size * config.num_rand_patches,),
                device=device,
            )
            unadj_indices = (anchor_indices + rand_offsets) % batch_size
            h_unadj = h_anchor[unadj_indices]
            h_anchor_unadj = h_anchor.repeat_interleave(config.num_rand_patches, dim=0)
            h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)

            pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
            pretext_labels = torch.cat(
                [
                    torch.ones(h_concat_pre.size(0), device=device),
                    torch.zeros(h_concat_unadj.size(0), device=device),
                ]
            )
            pretext_logits = model.classification_head(pretext_features).squeeze(1)
            pretext_loss_all = criterion_pretext(pretext_logits, pretext_labels)
            loss_adjacent = pretext_loss_all[: h_concat_pre.size(0)].mean()
            loss_unadj = pretext_loss_all[h_concat_pre.size(0) :].mean()
            pretext_loss = loss_adjacent + loss_unadj
        else:
            pretext_loss = torch.tensor(0.0, device=device)

        final_loss = triplet_loss + pretext_weight * pretext_loss

        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        optimizer.step()

        if final_loss.item() < best_loss:
            best_loss = final_loss.item()
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)


def create_memory_bank(
    model: PatchEncoder,
    data_loader: DataLoader,
    device: torch.device,
    compression: float | Optional[int],
) -> torch.Tensor:
    """Build PaAno's memory bank from train patch embeddings."""

    model.eval()
    embeddings = []
    with torch.no_grad():
        for data, batch_indices in data_loader:
            feats = model.embedding(data.to(device))
            embeddings.append(feats.detach().cpu().float())

    embedding_tensor = torch.cat(embeddings, dim=0)
    num_samples = embedding_tensor.size(0)

    if compression is None:
        return embedding_tensor
    if isinstance(compression, float):
        if compression <= 0:
            return embedding_tensor
        target_cores = int(round(compression * num_samples))
    else:
        target_cores = int(compression)
    target_cores = max(1, min(target_cores, num_samples - 1))
    if target_cores >= num_samples:
        return embedding_tensor

    flat_embeddings = F.normalize(embedding_tensor.view(num_samples, -1), p=2, dim=1)
    kmeans = MiniBatchKMeans(
        n_clusters=target_cores,
        init="k-means++",
        random_state=42,
        batch_size=max(8192, target_cores),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    )
    kmeans.fit(flat_embeddings.numpy())

    centers = torch.tensor(kmeans.cluster_centers_, dtype=flat_embeddings.dtype)
    distances = torch.cdist(flat_embeddings, centers, p=2)
    core_indices = torch.argmin(distances, dim=0)
    return embedding_tensor[core_indices.long()]


@torch.no_grad()
def calculate_patch_scores(
    model: PatchEncoder,
    data_loader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    top_k: int = 3,
) -> np.ndarray:
    """Score each patch by its cosine distance to the memory bank."""

    model.eval()
    memory = F.normalize(memory_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)
    all_scores = []
    for data, _ in data_loader:
        feats = model.embedding(data.to(device, non_blocking=True))
        feats = F.normalize(torch.nan_to_num(feats), dim=1, eps=1e-12)
        similarities = torch.nan_to_num(feats @ memory.T, nan=-1.0, posinf=1.0, neginf=-1.0)
        k = min(top_k, similarities.shape[1])
        topk_similarity, _ = torch.topk(similarities, k=k, dim=1, largest=True)
        patch_score = 1.0 - topk_similarity.mean(dim=1)
        all_scores.append(torch.nan_to_num(patch_score, nan=1.0).cpu().numpy())
    return np.concatenate(all_scores, axis=0)


@torch.no_grad()
def calculate_patch_scores_by_regime(
    model: PatchEncoder,
    data_loader: DataLoader,
    regime_banks: list[torch.Tensor],
    regime_centroids: torch.Tensor,
    device: torch.device,
    top_k: int = 3,
) -> np.ndarray:
    """Score each patch against a mode-selected memory bank.

    Hard regime assignment is intentionally simple: assign each test patch to
    the nearest normal regime centroid in embedding space, then score against
    that regime's compressed memory bank only.
    """

    model.eval()
    centroids = F.normalize(regime_centroids.to(device, dtype=torch.float32), dim=1, eps=1e-12)
    banks = [
        F.normalize(bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)
        for bank in regime_banks
    ]
    all_scores = []
    for data, _ in data_loader:
        feats = model.embedding(data.to(device, non_blocking=True))
        feats = F.normalize(torch.nan_to_num(feats), dim=1, eps=1e-12)
        centroid_sims = torch.nan_to_num(feats @ centroids.T, nan=-1.0, posinf=1.0, neginf=-1.0)
        assignments = centroid_sims.argmax(dim=1)
        batch_scores = torch.empty(feats.shape[0], device=device, dtype=torch.float32)
        for regime_index, bank in enumerate(banks):
            member_mask = assignments == regime_index
            if not torch.any(member_mask):
                continue
            member_feats = feats[member_mask]
            sims = torch.nan_to_num(member_feats @ bank.T, nan=-1.0, posinf=1.0, neginf=-1.0)
            k = min(top_k, sims.shape[1])
            topk_similarity, _ = torch.topk(sims, k=k, dim=1, largest=True)
            batch_scores[member_mask] = 1.0 - topk_similarity.mean(dim=1)
        all_scores.append(torch.nan_to_num(batch_scores, nan=1.0).cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def create_regime_memory_bank(
    model: PatchEncoder,
    data_loader: DataLoader,
    device: torch.device,
    num_regimes: int,
    compression: float | Optional[int],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Cluster train embeddings into regimes and build one bank per regime."""

    if num_regimes <= 1:
        raise ValueError("Regime memory bank requires num_regimes > 1.")

    model.eval()
    embeddings = []
    with torch.no_grad():
        for data, _ in data_loader:
            feats = model.embedding(data.to(device))
            embeddings.append(feats.detach().cpu().float())
    embedding_tensor = torch.cat(embeddings, dim=0)
    num_samples = embedding_tensor.size(0)
    if num_samples < num_regimes:
        num_regimes = num_samples
    flat_embeddings = F.normalize(embedding_tensor.view(num_samples, -1), p=2, dim=1)
    kmeans = MiniBatchKMeans(
        n_clusters=num_regimes,
        init="k-means++",
        random_state=42,
        batch_size=max(8192, num_regimes),
        max_iter=100,
        n_init=3,
        reassignment_ratio=0.01,
    )
    labels = kmeans.fit_predict(flat_embeddings.numpy())
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=flat_embeddings.dtype)

    regime_banks: list[torch.Tensor] = []
    for regime_index in range(num_regimes):
        regime_members = embedding_tensor[labels == regime_index]
        if len(regime_members) == 0:
            regime_banks.append(embedding_tensor[:1].clone())
            continue
        if compression is None:
            regime_banks.append(regime_members)
            continue
        if isinstance(compression, float):
            target_cores = int(round(compression * len(regime_members)))
        else:
            target_cores = int(compression)
        target_cores = max(1, min(target_cores, max(1, len(regime_members) - 1)))
        if target_cores >= len(regime_members):
            regime_banks.append(regime_members)
            continue
        normalized_members = F.normalize(regime_members.view(len(regime_members), -1), p=2, dim=1)
        member_kmeans = MiniBatchKMeans(
            n_clusters=target_cores,
            init="k-means++",
            random_state=42,
            batch_size=max(4096, target_cores),
            max_iter=50,
            n_init=1,
            reassignment_ratio=0.01,
        )
        member_kmeans.fit(normalized_members.numpy())
        centers = torch.tensor(member_kmeans.cluster_centers_, dtype=normalized_members.dtype)
        distances = torch.cdist(normalized_members, centers, p=2)
        core_indices = torch.argmin(distances, dim=0)
        regime_banks.append(regime_members[core_indices.long()])
    return regime_banks, centroids


def distribute_patch_scores_to_points(
    patch_scores: np.ndarray,
    patch_size: int,
    num_points: int,
    aggregation: str = "mean",
    quantile: float = 0.75,
) -> np.ndarray:
    """Pool overlapping patch scores back to one score per time step.

    The official PaAno runner uses a plain mean over all overlapping patch
    scores. That remains the default here. The benchmark wrapper optionally
    exposes sharper pooling modes because family-level NoBOOM thresholding is
    sensitive to one missed event, and mean pooling can dilute short anomalies.
    """

    patch_scores = np.nan_to_num(np.asarray(patch_scores, dtype=np.float32))
    if aggregation == "mean":
        kernel = np.ones(patch_size, dtype=np.float32)
        sums = np.convolve(patch_scores, kernel, mode="full")[:num_points]
        counts = np.convolve(np.ones_like(patch_scores), kernel, mode="full")[:num_points]
        point_scores = np.divide(
            sums,
            counts,
            out=np.zeros(num_points, dtype=np.float32),
            where=counts != 0,
        )
        return np.nan_to_num(point_scores)

    if aggregation not in {"max", "quantile"}:
        raise ValueError(f"Unsupported PaAno point aggregation: {aggregation}")

    pad_width = max(0, patch_size - 1)
    padded = np.pad(
        patch_scores,
        (pad_width, pad_width),
        mode="constant",
        constant_values=np.nan,
    )
    windows = sliding_window_view(padded, patch_size)[:num_points]
    if aggregation == "max":
        return np.nan_to_num(np.nanmax(windows, axis=1), nan=0.0)
    return np.nan_to_num(np.nanquantile(windows, quantile, axis=1), nan=0.0)


def correlation_reference(
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate one train-set correlation template from raw multivariate patches.

    This is a light-weight structural baseline inspired by OracleAD/CAROTS:
    if a process anomaly mostly changes inter-variable relationships rather than
    absolute amplitudes, PaAno's nearest-neighbour score can miss it while the
    correlation pattern still moves away from the normal train manifold.
    """

    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape [N, C, L], got {patches.shape}.")

    accum = None
    total = 0
    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size].to(device, non_blocking=True)
        corr = _patch_correlations(batch, eps=eps)
        batch_sum = corr.sum(dim=0)
        accum = batch_sum if accum is None else accum + batch_sum
        total += corr.shape[0]
    if accum is None or total == 0:
        raise ValueError("Cannot estimate a structural reference from zero patches.")
    return accum / total


@torch.no_grad()
def correlation_scores(
    patches: torch.Tensor,
    reference: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return one structural deviation score per patch."""

    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape [N, C, L], got {patches.shape}.")

    scores = []
    reference = reference.to(device)
    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size].to(device, non_blocking=True)
        corr = _patch_correlations(batch, eps=eps)
        delta = corr - reference.unsqueeze(0)
        batch_scores = torch.linalg.norm(delta, dim=(-2, -1))
        scores.append(batch_scores.cpu().numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)


def _patch_correlations(patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute per-patch channel correlation matrices for `(N, C, L)` tensors."""

    centered = patches - patches.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    scaled = centered / variance.add(eps).sqrt()
    length = max(1, patches.shape[-1])
    corr = torch.matmul(scaled, scaled.transpose(1, 2)) / float(length)
    return torch.nan_to_num(corr)
