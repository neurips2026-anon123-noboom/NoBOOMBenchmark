"""Local compatibility helpers for vendored CATCH benchmark wrappers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def train_val_split(
    train_data: pd.DataFrame,
    ratio: float,
    seq_len: Optional[int],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    external_validation = getattr(train_data, "attrs", {}).get("_external_validation_data")
    if external_validation is not None and len(external_validation) > 0:
        return train_data, external_validation
    if ratio == 1:
        return train_data, None

    border = int(train_data.shape[0] * ratio)
    if seq_len is None:
        return train_data.iloc[:border].copy(), train_data.iloc[border:].copy()

    return (
        train_data.iloc[:border].copy(),
        train_data.iloc[max(0, border - seq_len) :].copy(),
    )


class _AnomalyWindowDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, win_size: int, step: int = 1) -> None:
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if values.ndim != 2:
            raise ValueError("Expected a 2D frame.")
        self.values = values
        self.win_size = int(win_size)
        self.step = int(step)
        if self.win_size <= 0:
            raise ValueError("win_size must be positive.")
        self.length = max(0, (len(values) - self.win_size) // self.step + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.step
        window = self.values[start : start + self.win_size]
        return torch.as_tensor(window, dtype=torch.float32), torch.zeros(self.win_size, dtype=torch.float32)


def anomaly_detection_data_provider(
    data: pd.DataFrame,
    batch_size: int,
    win_size: int,
    step: int = 1,
    mode: str = "train",
) -> DataLoader:
    del mode
    dataset = _AnomalyWindowDataset(data, win_size=win_size, step=step)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
