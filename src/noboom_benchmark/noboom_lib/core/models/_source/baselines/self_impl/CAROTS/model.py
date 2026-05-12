"""Minimal CAROTS modules with the official CUTS+ stage kept intact.

This file ports the default LSTM-based CAROTS path from the official release:
the CUTS+ causal discoverer, the two augmentors, the projector head, and the
contrastive loss/scoring helpers. Unused encoder variants and unrelated
training utilities are intentionally omitted to keep the benchmark readable.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AddBias(nn.Module):
    """Official CAROTS negative transform: perturb a subset of channels in time."""

    def __init__(self, bias_candidates: tuple[float, ...], percent: float):
        super().__init__()
        self.bias_candidates = bias_candidates
        self.percent = percent

    def forward(self, series: torch.Tensor, indices: list[int]) -> torch.Tensor:
        biased = series.clone().detach()
        if not indices:
            return biased
        n_steps = max(1, int(series.shape[1] * self.percent))
        chosen_steps = torch.randperm(series.shape[1], device=series.device)[:n_steps]
        bias = torch.tensor(
            random.choices(self.bias_candidates, k=series.shape[0] * n_steps * len(indices)),
            device=series.device,
            dtype=series.dtype,
        ).view(series.shape[0], n_steps, len(indices))
        for offset, step in enumerate(chosen_steps.tolist()):
            biased[:, step, indices] += bias[:, offset, :]
        return biased


class NegativeAugmentor(nn.Module):
    """Official graph-aware negative augmentor with device-safe tensor handling."""

    def __init__(
        self,
        bias_candidates: tuple[float, ...],
        cutoff_probability: float,
        disturb_all: bool,
        transform_percent: float,
    ):
        super().__init__()
        self.cutoff_probability = cutoff_probability
        self.disturb_all = disturb_all
        self.transform = AddBias(bias_candidates=bias_candidates, percent=transform_percent)

    @torch.no_grad()
    def _indices_to_disturb(self, causality_mtx: torch.Tensor) -> list[int]:
        if self.disturb_all:
            return list(range(len(causality_mtx)))

        def dfs(node: int, visited: list[bool], subgraph: list[int]) -> None:
            visited[node] = True
            subgraph.append(node)
            for neighbor, is_causal in enumerate(causality_mtx[node].tolist()):
                if is_causal and not visited[neighbor]:
                    if random.random() > self.cutoff_probability:
                        dfs(neighbor, visited, subgraph)
                    else:
                        break

        visited = [False] * len(causality_mtx)
        subgraph: list[int] = []
        dfs(random.randrange(len(causality_mtx)), visited, subgraph)
        return subgraph

    def forward(self, series: torch.Tensor, causality_mtx: torch.Tensor) -> torch.Tensor:
        binary_graph = (causality_mtx > 0.5).int()
        indices = self._indices_to_disturb(binary_graph)
        return self.transform(series, indices).clone().detach()


class PositiveAugmentor(nn.Module):
    """Official CAROTS positive augmentor driven by the learned CUTS+ graph."""

    def __init__(self, input_step: int, pred_step: int, noise_level: float):
        super().__init__()
        self.input_step = input_step
        self.pred_step = pred_step
        self.noise_level = noise_level
        self.causal_discoverer: Optional[CUTSPlusNet] = None

    def set_causal_discoverer(self, causal_discoverer: "CUTSPlusNet") -> None:
        self.causal_discoverer = causal_discoverer

    def forward(self, series: torch.Tensor, causality_mtx: torch.Tensor) -> torch.Tensor:
        if self.causal_discoverer is None:
            raise RuntimeError("PositiveAugmentor requires a fitted causal discoverer.")

        positive = series.clone()
        binary_graph = (causality_mtx > 0.5).int()
        start_node = random.randrange(len(binary_graph))
        effects = [idx for idx, value in enumerate(binary_graph[start_node].tolist()) if value == 1]
        noise = torch.randn(series.size(0), device=series.device, dtype=series.dtype) * self.noise_level
        positive[:, self.input_step, start_node] += noise

        with torch.no_grad():
            graph = (self.causal_discoverer.causality_mtx > 0.5).float().to(series.device)
            graph = graph[None].expand(series.size(0), -1, -1)
            predicted = self.causal_discoverer(positive[:, : self.input_step], graph).transpose(1, 2)

        if effects:
            positive[:, -self.pred_step, effects] = predicted[:, -self.pred_step, effects]
        return positive


class MPNN(nn.Module):
    """Message-passing layer from CUTS+."""

    def __init__(self, c_in: int, c_out: int, concat_h: bool = True):
        super().__init__()
        self.concat_h = concat_h
        self.mlp = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        _, _, n_nodes = x.shape
        x_repeat = x[:, :, :, None].expand(-1, -1, -1, n_nodes)
        x_messages = torch.einsum("bcmn,bmn->bcmn", x_repeat, graph)
        x_messages = rearrange(x_messages, "b c m n -> b (c m) n")
        if self.concat_h:
            return self.mlp(torch.cat([x_messages, h], dim=1))
        return self.mlp(x_messages)


class GRUCell(nn.Module):
    """CUTS+ recurrent cell with graph-conditioned message passing."""

    def __init__(self, d_in: int, num_units: int, n_nodes: int, concat_h: bool = False, activation: str = "tanh"):
        super().__init__()
        self.activation_fn = getattr(torch, activation)
        mpnn_channels = d_in * n_nodes + num_units if concat_h else d_in * n_nodes
        self.forget_gate = MPNN(c_in=mpnn_channels, c_out=num_units, concat_h=concat_h)
        self.update_gate = MPNN(c_in=mpnn_channels, c_out=num_units, concat_h=concat_h)
        self.c_gate = MPNN(c_in=mpnn_channels, c_out=num_units, concat_h=concat_h)

    def forward(self, x: torch.Tensor, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        reset = torch.sigmoid(self.forget_gate(x, h, adj))
        update = torch.sigmoid(self.update_gate(x, h, adj))
        candidate = self.activation_fn(self.c_gate(x, reset * h, adj))
        return update * h + (1.0 - update) * candidate


class LocalConv1D(nn.Module):
    """Node-wise decoder used by the official CUTS+ release."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_nodes: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
                for _ in range(n_nodes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, n_nodes = x.shape
        out = torch.zeros((batch, self.out_channels, n_nodes), device=x.device, dtype=x.dtype)
        for node in range(n_nodes):
            local_out = self.conv_list[node](x[..., node].unsqueeze(-1))
            out[..., node] = local_out.squeeze(-1)
        return out


class CUTSPlusNet(nn.Module):
    """Device-safe port of the official CUTS+ predictor."""

    def __init__(
        self,
        n_nodes: int,
        data_dim: int = 1,
        hidden_dim: int = 32,
        gru_layers: int = 1,
        concat_h: bool = True,
        shared_weights_decoder: bool = False,
    ):
        super().__init__()
        self.in_ch = data_dim
        self.hidden_ch = hidden_dim
        self.n_layers = gru_layers

        self.conv_encoder1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv_encoder2 = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)
        if shared_weights_decoder:
            self.decoder = nn.Sequential(nn.Conv1d(2 * hidden_dim, data_dim, kernel_size=1))
        else:
            self.decoder = nn.Sequential(LocalConv1D(2 * hidden_dim, data_dim, kernel_size=1, n_nodes=n_nodes))
        self.act = nn.LeakyReLU()
        self.cells = nn.ModuleList(
            [
                GRUCell(
                    d_in=data_dim if index == 0 else hidden_dim,
                    num_units=hidden_dim,
                    n_nodes=n_nodes,
                    concat_h=concat_h,
                )
                for index in range(gru_layers)
            ]
        )
        self.h0 = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim, n_nodes)) for _ in range(gru_layers)]
        )
        self.GT = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    @property
    def causality_mtx(self) -> torch.Tensor:
        return torch.sigmoid(self.GT)

    def update_state(self, x: torch.Tensor, h: list[torch.Tensor], graph: torch.Tensor) -> list[torch.Tensor]:
        rnn_in = x
        for layer, cell in enumerate(self.cells):
            rnn_in = h[layer] = cell(rnn_in, h[layer], graph)
        return h

    def forward(self, x: torch.Tensor, fwd_graph: torch.Tensor) -> torch.Tensor:
        # Official code expects [batch, input_step, channels] and immediately
        # swaps it to [batch, channels, input_step] internally.
        x = rearrange(x, "b n t -> b t n")
        batch, n_nodes, steps = x.shape
        h = [state.to(x.device).expand(batch, -1, -1) for state in self.h0]
        predictions = []
        for step in range(steps):
            current = x[..., step].unsqueeze(1)
            h = self.update_state(current, h, fwd_graph)
            h_now = h[-1]
            x_repr = self.act(self.conv_encoder1(h_now))
            x_repr = self.act(self.conv_encoder2(torch.cat([x_repr, h_now], dim=1)))
            x_repr = torch.cat([x_repr, h_now], dim=1)
            predictions.append(self.decoder(x_repr))
        pred = torch.stack(predictions, dim=-1)
        pred = rearrange(pred, "b c n s -> b n s c").squeeze(-1)
        return pred[:, :, -1:]


class LSTMEncoder(nn.Module):
    """Default CAROTS encoder from the official configuration."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        hidden = hidden.permute(1, 0, 2)
        return hidden.reshape(x.size(0), -1)


class CAROTSCore(nn.Module):
    """Small CAROTS model containing only the default official encoder path."""

    def __init__(
        self,
        input_dim: int,
        input_step: int,
        pred_step: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        projector_hidden_dim: int,
        projector_output_dim: int,
        data_dim: int,
        cuts_hidden_dim: int,
        cuts_gru_layers: int,
        cuts_concat_h: bool,
        cuts_shared_weights_decoder: bool,
        noise_level: float,
        bias_candidates: tuple[float, ...],
        cutoff_probability: float,
        disturb_all: bool,
        transform_percent: float,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, projector_output_dim),
        )
        self.causal_discoverer = CUTSPlusNet(
            n_nodes=input_dim,
            data_dim=data_dim,
            hidden_dim=cuts_hidden_dim,
            gru_layers=cuts_gru_layers,
            concat_h=cuts_concat_h,
            shared_weights_decoder=cuts_shared_weights_decoder,
        )
        self.positive_augmentor = PositiveAugmentor(
            input_step=input_step,
            pred_step=pred_step,
            noise_level=noise_level,
        )
        self.negative_augmentor = NegativeAugmentor(
            bias_candidates=bias_candidates,
            cutoff_probability=cutoff_probability,
            disturb_all=disturb_all,
            transform_percent=transform_percent,
        )

    def forward(self, x: torch.Tensor, positive_augment: bool = True, negative_augment: bool = True) -> torch.Tensor:
        # Paper layout (arXiv:2506.03964 §3.3): anchors, positives from the
        # learned causal positive augmentor, and negatives produced by applying
        # the negative augmentor to anchors only. The official reference
        # (kimanki/CAROTS, models/carots/modeling_carots.py:66-77) chains
        # ``negative_augmentor`` over ``[anchors; positives]`` which yields a
        # 4B mixed-source layout; we follow the paper and keep negatives as a
        # function of anchors so the loss/centroids remain interpretable.
        parts = [x]
        if positive_augment:
            positive = self.positive_augmentor(x, self.causal_discoverer.causality_mtx)
            parts.append(positive)
        if negative_augment:
            negative = self.negative_augmentor(x, self.causal_discoverer.causality_mtx)
            parts.append(negative)
        x_all = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        embeddings = self.encoder(x_all)
        return self.projector(embeddings)


def carots_loss(
    embeddings: torch.Tensor,
    sim_threshold: float,
    temperature: float,
    positive_augment: bool = True,
) -> torch.Tensor:
    """CAROTS InfoNCE-style loss on the 3B ``[anchors; positives; negatives]``
    layout (paper arXiv:2506.03964 §3.3, Eq. 4-5).

    The official reference (kimanki/CAROTS, models/carots/loss.py:5-37) was
    written against a 4B mixed layout produced by the official forward pass;
    we re-derive it for the 3B layout used here. Symmetric similarity
    filtering still applies: if anchor i and positive j fall *below* the
    similarity threshold they are masked out everywhere they appear (the
    positive panel and the corresponding negative panel), so negatives in
    the denominator only contribute when (i, j) qualifies as a positive
    pair. This is the symmetric analogue of the `indices`/`indices_neg`
    masking on lines 14-21 of the official ``loss.py``.

    When ``positive_augment`` is False the layout collapses to 2B
    ``[anchors; negatives]`` and the positive panel falls back to the
    nearest off-diagonal neighbour among anchors so the objective stays
    finite for smoke runs (off-paper safeguard).
    """

    if embeddings.numel() == 0:
        return embeddings.new_tensor(0.0)

    chunks = 3 if positive_augment else 2
    if embeddings.size(0) % chunks != 0:
        raise ValueError(
            f"carots_loss expected embeddings divisible by {chunks} for the "
            f"{'3B' if positive_augment else '2B'} layout, got {embeddings.size(0)}"
        )
    n = embeddings.size(0) // chunks
    if n == 0:
        return embeddings.new_tensor(0.0)

    normalized = F.normalize(embeddings, p=2, dim=1)
    sim = normalized @ normalized.T  # (chunks*n, chunks*n)

    if positive_augment:
        # Anchor-to-positive similarity panel (paper Eq. 4): rows are anchors
        # i in [0, n), columns are positives j in [n, 2n).
        sim_pos = sim[:n, n : 2 * n].clone()
    else:
        # Without a positive augmentor the anchors themselves serve as their
        # own positive bank.
        sim_pos = sim[:n, :n].clone()
        sim_pos.fill_diagonal_(float("-inf"))

    positive_mask = sim_pos >= sim_threshold
    if positive_augment is False:
        # Avoid the diagonal contributing as a self-positive in the 2B layout.
        positive_mask.fill_diagonal_(False)

    if not torch.any(positive_mask):
        # Tiny smoke batches may have no off-diagonal pairs above the
        # scheduled threshold. Fallback to nearest neighbour keeps the
        # objective usable without altering the canonical path.
        sim_pos_eff = sim_pos.clone()
        if positive_augment is False:
            sim_pos_eff.fill_diagonal_(float("-inf"))
        nearest = sim_pos_eff.argmax(dim=1)
        positive_mask = torch.zeros_like(sim_pos, dtype=torch.bool)
        positive_mask[torch.arange(n, device=sim.device), nearest] = True

    # Symmetric similarity filtering: pairs that fall below the threshold are
    # excluded from both the positive panel and the matching negative-column
    # contribution (analogous to the official indices/indices_neg masking).
    below_mask = ~positive_mask
    masked = sim.clone()
    if positive_augment:
        # Positive panel anchors→positives.
        masked[:n, n : 2 * n] = torch.where(
            below_mask, masked.new_full(below_mask.shape, float("-inf")), masked[:n, n : 2 * n]
        )
        # Symmetric negative-column panel anchors→negatives matched index-wise
        # to the positives that were filtered out.
        masked[:n, 2 * n : 3 * n] = torch.where(
            below_mask, masked.new_full(below_mask.shape, float("-inf")), masked[:n, 2 * n : 3 * n]
        )
        neg_start = 2 * n
    else:
        masked[:n, :n] = torch.where(
            below_mask, masked.new_full(below_mask.shape, float("-inf")), masked[:n, :n]
        )
        masked[:n, n : 2 * n] = torch.where(
            below_mask, masked.new_full(below_mask.shape, float("-inf")), masked[:n, n : 2 * n]
        )
        neg_start = n

    logits = torch.exp(masked / temperature)
    # Drop self-similarity from the diagonal as in the official loss.py:24.
    logits = logits - torch.diag_embed(torch.diagonal(logits))

    # Denominator: positive logit + sum of negative-column logits respecting
    # the symmetric filtering applied above.
    neg_sum = logits[:n, neg_start : neg_start + n].sum(dim=1, keepdim=True)
    if positive_augment:
        positive_panel = logits[:n, n : 2 * n]
    else:
        positive_panel = logits[:n, :n]
    denominator = positive_panel + neg_sum
    positive_logits = positive_panel[positive_mask]
    positive_denominator = denominator[positive_mask]
    return (-torch.log(positive_logits / (positive_denominator + 1e-12) + 1e-12)).mean()


def distance_to_centroid(
    embeddings: torch.Tensor,
    centroid_wo_norm: torch.Tensor,
    centroid_norm: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    if metric == "l2":
        return torch.cdist(embeddings, centroid_wo_norm).squeeze(-1)
    if metric == "cos":
        centroid = F.normalize(centroid_norm, dim=1)
        points = F.normalize(embeddings, dim=1)
        return -torch.matmul(points, centroid.t()).squeeze(-1) * 0.5 + 0.5
    raise ValueError(f"Unsupported CAROTS scorer: {metric}")
