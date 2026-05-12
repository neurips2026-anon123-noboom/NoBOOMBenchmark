from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DATASET_LABELS,
    DOUBLE_COLUMN_WIDTH,
    OptionalFigureSkipped,
    best_method,
    clean_axis,
    completed_alarm,
    load_alarm_results,
    load_seed_results,
    rounded_axis_limit,
    save_figure,
)


def _aggregate_lookup(alarm_df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[float, float]]:
    lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for _, row in completed_alarm(alarm_df).iterrows():
        std = float(row["alarm_std"]) if pd.notna(row["alarm_std"]) else 0.0
        lookup[(str(row["dataset"]), str(row["method"]))] = (float(row["alarm_mean"]), std)
    return lookup


def _panel_items(alarm_df: pd.DataFrame) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    threshold_led = []
    for dataset in ["batch_bpw", "cont_wat", "cont_but", "cont_ome"]:
        deep = str(best_method(alarm_df, dataset, family="deep")["method"])
        threshold_led.append((dataset, "Threshold", f"{DATASET_LABELS[dataset]}\nThreshold"))
        threshold_led.append((dataset, deep, f"{DATASET_LABELS[dataset]}\n{deep}"))
    return [
        (
            "batch abm contenders",
            [
                ("batch_abm", "IGAD", "IGAD"),
                ("batch_abm", "K-Means", "K-Means"),
                ("batch_abm", "GMM-HMM", "GMM-HMM"),
                ("batch_abm", "Threshold", "Threshold"),
                ("batch_abm", "LSTM-AE", "LSTM-AE"),
            ],
        ),
        (
            "cont ind contenders",
            [
                ("cont_ind", "LSTM-AE", "LSTM-AE"),
                ("cont_ind", "RTdetector", "RTdetector"),
                ("cont_ind", "H-PAD", "H-PAD"),
                ("cont_ind", "OracleAD", "OracleAD"),
                ("cont_ind", "GDN", "GDN"),
                ("cont_ind", "Threshold", "Threshold"),
            ],
        ),
        ("Threshold-led datasets", threshold_led),
    ]


def _plot_item(
    ax: plt.Axes,
    y: float,
    dataset: str,
    method: str,
    aggregate: Dict[Tuple[str, str], Tuple[float, float]],
    seeds: pd.DataFrame,
) -> bool:
    seed_rows = seeds[seeds["dataset"].eq(dataset) & seeds["method"].eq(method)] if not seeds.empty else pd.DataFrame()
    if not seed_rows.empty:
        values = seed_rows.sort_values("seed")["alarm"].astype(float).to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(values, y + jitter, s=10, color="#4d4d4d", alpha=0.65, linewidth=0, zorder=2)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ax.errorbar(
            mean,
            y,
            xerr=std,
            fmt="D",
            markersize=4,
            color="#0072B2",
            ecolor="#0072B2",
            elinewidth=0.7,
            capsize=2,
            zorder=3,
        )
        return True

    if (dataset, method) not in aggregate:
        return False
    mean, std = aggregate[(dataset, method)]
    ax.errorbar(
        mean,
        y,
        xerr=std,
        fmt="o",
        markersize=4,
        markerfacecolor="white",
        markeredgecolor="#D55E00",
        ecolor="#D55E00",
        elinewidth=0.7,
        capsize=2,
        zorder=3,
    )
    return True


def plot(data_dir: Path, outdir: Path, results_path: Optional[Path] = None, strict: bool = False) -> List[Path]:
    alarm_df, _ = load_alarm_results(data_dir=data_dir, results_path=results_path)
    aggregate = _aggregate_lookup(alarm_df)
    if not aggregate:
        message = "Skipping uncertainty figure: ALARM mean/std data not found."
        if strict:
            raise FileNotFoundError(message)
        raise OptionalFigureSkipped(message)

    try:
        seed_df, _ = load_seed_results(data_dir=data_dir, strict=False)
    except OptionalFigureSkipped:
        seed_df = pd.DataFrame(columns=["method", "family", "dataset", "seed", "alarm", "status"])

    panels = _panel_items(alarm_df)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 3.20))
    xmax = rounded_axis_limit(max([value[0] + value[1] for value in aggregate.values()] + [1.0]))

    for ax, (title, items) in zip(axes, panels):
        visible_labels: List[str] = []
        visible_y: List[float] = []
        y_positions = np.arange(len(items))
        for y, (dataset, method, label) in zip(y_positions, items):
            if _plot_item(ax, float(y), dataset, method, aggregate, seed_df):
                visible_labels.append(label)
                visible_y.append(float(y))
        ax.set_yticks(visible_y)
        ax.set_yticklabels(visible_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(title)
        ax.set_xlabel("Normalized ALARM (%)")
        clean_axis(ax)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.4, alpha=0.7)
        ax.grid(axis="y", visible=False)

    if seed_df.empty:
        note = "Mean \u00b1 seed SD from aggregate table"
    else:
        note = "Per-seed points where available; hollow marks use aggregate mean \u00b1 seed SD"
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=1.0)

    paths = save_figure(fig, outdir, "fig_seed_uncertainty")
    return list(paths)

