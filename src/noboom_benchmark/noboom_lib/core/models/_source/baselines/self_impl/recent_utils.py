"""Small utilities shared by the lean 2025+ anomaly baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader, TensorDataset


def transform_frame(scaler: TransformerMixin, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted sklearn-style scaler while preserving the DataFrame shape."""

    values = scaler.transform(frame.to_numpy())
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)


def build_causal_windows(values: np.ndarray, window_size: int) -> np.ndarray:
    """Build one left-padded causal window per time step.

    The official code for both DADA and RTdetector operates on fixed windows.
    The benchmark expects one score per point, so we map each time step to the
    window ending at that position and left-pad the warm-up prefix with the
    first observation.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array of shape [time, channels].")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if array.shape[0] == 0:
        return np.empty((0, window_size, array.shape[1]), dtype=np.float32)

    first_row = array[:1]
    windows = np.empty((array.shape[0], window_size, array.shape[1]), dtype=np.float32)
    for end in range(array.shape[0]):
        start = max(0, end - window_size + 1)
        window = array[start : end + 1]
        pad = window_size - window.shape[0]
        if pad > 0:
            windows[end, :pad] = first_row
            windows[end, pad:] = window
        else:
            windows[end] = window
    return windows


def make_window_loader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap a window tensor in a minimal DataLoader."""

    tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    """Return the first existing path from a small candidate list."""

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None
