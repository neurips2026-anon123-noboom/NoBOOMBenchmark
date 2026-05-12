"""PatchTST encoder head used by the vendored PaAno source port."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from noboom_benchmark.noboom_lib.core.models._source.baselines.self_impl.NeutralAD.blocks import TransformerBlock


def _pool_tokens(x: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "mean":
        return x.mean(dim=1)
    if pooling == "max":
        return x.max(dim=1).values
    if pooling == "first":
        return x[:, 0]
    raise ValueError(f"Unknown pooling: {pooling}")


def _compute_padded_len(seq_len: int, patch_len: int, patch_stride: int) -> int:
    if seq_len < patch_len:
        return patch_len
    remainder = (seq_len - patch_len) % patch_stride
    if remainder == 0:
        return seq_len
    return seq_len + (patch_stride - remainder)


def _compute_num_patches(seq_len: int, patch_len: int, patch_stride: int) -> int:
    padded_len = _compute_padded_len(seq_len, patch_len, patch_stride)
    return ((padded_len - patch_len) // patch_stride) + 1


def _pad_for_patching(x: torch.Tensor, patch_len: int, patch_stride: int) -> torch.Tensor:
    seq_len = x.shape[-1]
    if seq_len < patch_len:
        pad = patch_len - seq_len
    else:
        remainder = (seq_len - patch_len) % patch_stride
        pad = 0 if remainder == 0 else patch_stride - remainder
    if pad == 0:
        return x
    return F.pad(x, (0, pad))


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        norm_first: bool = True,
        pooling: str = "mean",
        token_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.pooling = pooling
        self.token_chunk_size = token_chunk_size

        max_patches = _compute_num_patches(seq_len, patch_len, patch_stride)
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_for_patching(x, self.patch_len, self.patch_stride)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        bsz, channels, num_patches, _ = patches.shape
        patches = patches.reshape(bsz * channels, num_patches, self.patch_len)
        tokens = self.patch_proj(patches)
        pos_embed = self.pos_embed[:, :num_patches, :]
        tokens = self.dropout(tokens + pos_embed)

        tokens_in = torch.tensor_split(tokens, math.ceil(tokens.shape[0] / 65535), dim=0)
        tokens_out = []
        for token_batch in tokens_in:
            for block in self.blocks:
                if self.token_chunk_size is not None and token_batch.shape[1] > self.token_chunk_size:
                    token_chunks = []
                    for token_chunk in token_batch.split(self.token_chunk_size, dim=1):
                        token_chunks.append(block(token_chunk)[0])
                    token_batch = torch.cat(token_chunks, dim=1)
                else:
                    token_batch, _ = block(token_batch)
            tokens_out.append(token_batch)
        tokens = torch.cat(tokens_out, dim=0)

        if self.pooling == "base":
            return tokens.reshape(bsz, channels * num_patches, self.d_model)

        pooled_patches = _pool_tokens(tokens, self.pooling)
        pooled_patches = pooled_patches.reshape(bsz, channels, self.d_model)
        pooled_channels = _pool_tokens(pooled_patches, self.pooling)
        return pooled_channels


def _pad_or_trim(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    current_len = x.shape[-1]
    if current_len == seq_len:
        return x
    if current_len < seq_len:
        pad = seq_len - current_len
        return F.pad(x, (0, pad))
    return x[..., -seq_len:].contiguous()

