from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATASET_LABELS, OptionalFigureSkipped, repo_root, save_figure


LABEL_COLUMNS = ["label", "labels", "y", "target", "phase", "anomaly", "is_anomaly"]
PREDICTION_COLUMNS = ["alarm", "prediction", "pred", "is_alarm", "score", "anomaly_score"]
PREFERRED_CHANNELS: Dict[str, List[str]] = {
    "cont_ome": ["PIC101", "PDIC101", "T106"],
    "batch_bpw": ["PDI701", "PDI702", "LS701"],
}
DATASET_TOKENS: Dict[str, List[str]] = {
    "cont_ome": ["cont_ome", "cont reactive ome", "cont_reactive_ome"],
    "batch_bpw": ["batch_bpw", "batch bpw", "1_butanol_2_propanol_water"],
    "cont_ind": ["cont_ind", "cont ind", "industry_process"],
}


def _candidate_paths() -> List[Path]:
    root = repo_root()
    dirs = ["data", "datasets", "noboom", "raw", "results", "outputs", "predictions", "scores", "alarms", "artifacts"]
    suffixes = {".csv", ".tsv", ".parquet"}
    paths: List[Path] = []
    for dirname in dirs:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() in suffixes and path.stat().st_size < 75_000_000:
                paths.append(path)
    return paths


def _read_frame(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception:
        return None
    return None


def _matches_dataset(path: Path, dataset: str) -> bool:
    text = str(path).lower()
    return any(token in text for token in DATASET_TOKENS[dataset])


def _column_by_name(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _is_binary(series: pd.Series) -> bool:
    values = pd.Series(series.dropna().unique())
    if values.empty:
        return False
    if len(values) > 8:
        return False
    return set(values.astype(str).str.lower()).issubset({"0", "1", "0.0", "1.0", "false", "true"})


def _label_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).to_numpy() != 0
    text = series.astype(str).str.lower()
    return text.str.contains("anom|fault|failure|attack|abnormal", regex=True).to_numpy()


def _numeric_channels(df: pd.DataFrame, excluded: Iterable[str]) -> List[str]:
    exclude = set(excluded)
    channels = []
    for column in df.columns:
        if column in exclude:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().mean() > 0.80 and values.nunique(dropna=True) > 5:
            channels.append(column)
    return channels


def _event_window(mask: np.ndarray, context: int = 250) -> Optional[Tuple[int, int, int, int]]:
    event_indices = np.flatnonzero(mask)
    if event_indices.size == 0:
        return None
    start = int(event_indices[0])
    end = start
    while end + 1 < len(mask) and mask[end + 1]:
        end += 1
    left = max(0, start - context)
    right = min(len(mask) - 1, end + context)
    return left, start, end, right


def _robust_z(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    median = float(np.nanmedian(numeric))
    q75, q25 = np.nanpercentile(numeric, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr if iqr > 1e-12 else float(np.nanstd(numeric))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return ((numeric - median) / scale).to_numpy()


def _select_channels(df: pd.DataFrame, dataset: str, mask: np.ndarray, label_column: str, pred_column: str) -> List[str]:
    channels = _numeric_channels(df, excluded=[label_column, pred_column])
    preferred = [channel for channel in PREFERRED_CHANNELS.get(dataset, []) if channel in channels]
    if preferred:
        return preferred[:3]
    if dataset != "cont_ind":
        return channels[:3]
    event = _event_window(mask, context=500)
    if event is None:
        return channels[:3]
    left, start, end, _ = event
    normal_slice = slice(left, start)
    event_slice = slice(start, end + 1)
    scores = []
    for channel in channels:
        values = pd.to_numeric(df[channel], errors="coerce").astype(float)
        normal = values.iloc[normal_slice]
        event_values = values.iloc[event_slice]
        q75, q25 = np.nanpercentile(normal, [75, 25])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 1e-12:
            continue
        effect = abs(float(np.nanmedian(event_values)) - float(np.nanmedian(normal))) / iqr
        scores.append((effect, channel))
    scores.sort(reverse=True)
    return [channel for _, channel in scores[:3]] or channels[:3]


def _find_dataset_frame(dataset: str) -> Optional[Tuple[pd.DataFrame, str, str, Path]]:
    for path in _candidate_paths():
        if not _matches_dataset(path, dataset):
            continue
        df = _read_frame(path)
        if df is None or df.empty:
            continue
        label = _column_by_name(df.columns, LABEL_COLUMNS)
        pred = _column_by_name(df.columns, PREDICTION_COLUMNS)
        if label is None or pred is None:
            continue
        channels = _numeric_channels(df, excluded=[label, pred])
        if not channels:
            continue
        return df, label, pred, path
    return None


def _plot_dataset(dataset: str, outdir: Path) -> Optional[List[Path]]:
    found = _find_dataset_frame(dataset)
    if found is None:
        return None
    df, label_column, pred_column, _ = found
    mask = _label_mask(df[label_column])
    window = _event_window(mask)
    if window is None:
        return None
    left, start, end, right = window
    view = df.iloc[left : right + 1].reset_index(drop=True)
    view_mask = mask[left : right + 1]
    channels = _select_channels(df, dataset, mask, label_column, pred_column)
    if not channels:
        return None

    x = np.arange(len(view))
    fig, axes = plt.subplots(2, 1, figsize=(6.75, 2.75), sharex=True, gridspec_kw={"height_ratios": [1.4, 0.8]})
    ax_signal, ax_alarm = axes
    for channel in channels[:3]:
        ax_signal.plot(x, _robust_z(view[channel]), linewidth=0.8, label=channel)
    for ax in axes:
        ax.axvspan(start - left, end - left, color="#fdd0a2", alpha=0.45, linewidth=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_signal.set_ylabel("robust z")
    ax_signal.set_title(f"{DATASET_LABELS[dataset]} representative alarm context")
    ax_signal.legend(frameon=False, ncol=min(3, len(channels)), loc="upper right")

    pred = view[pred_column]
    if _is_binary(pred):
        ax_alarm.step(x, pd.to_numeric(pred, errors="coerce").fillna(0), where="mid", color="#D55E00", linewidth=0.9)
        ax_alarm.set_ylabel("alarm")
        ax_alarm.set_ylim(-0.08, 1.08)
    else:
        ax_alarm.plot(x, pd.to_numeric(pred, errors="coerce"), color="#D55E00", linewidth=0.9)
        ax_alarm.set_ylabel("score")
    ax_alarm.fill_between(x, 0, view_mask.astype(float), step="mid", color="#999999", alpha=0.25, transform=ax_alarm.get_xaxis_transform())
    ax_alarm.set_xlabel("time index")
    fig.tight_layout()
    return list(save_figure(fig, outdir, f"fig_alarm_timeline_{dataset}"))


def plot(outdir: Path, strict: bool = False) -> List[Path]:
    generated: List[Path] = []
    for dataset in ["cont_ome", "batch_bpw", "cont_ind"]:
        paths = _plot_dataset(dataset, outdir)
        if paths:
            generated.extend(paths)
    if generated:
        return generated
    message = "Skipping timeline figures: raw time series, labels, and real alarms/scores were not found together."
    if strict:
        raise FileNotFoundError(message)
    raise OptionalFigureSkipped(message)

