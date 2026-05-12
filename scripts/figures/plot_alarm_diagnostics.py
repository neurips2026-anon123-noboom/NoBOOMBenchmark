from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DATASET_LABELS,
    DATASET_ORDER,
    DOUBLE_COLUMN_WIDTH,
    OptionalFigureSkipped,
    best_method,
    clean_axis,
    load_alarm_results,
    load_metric_diagnostics,
    save_figure,
    threshold_row,
)


ROLE_COLORS: Dict[str, str] = {
    "Threshold": "#4d4d4d",
    "Best overall": "#D55E00",
    "Best deep": "#0072B2",
    "Best shallow": "#009E73",
}


def _selected_methods(alarm_df: pd.DataFrame, dataset: str) -> List[Tuple[str, str]]:
    candidates = [
        ("Threshold", str(threshold_row(alarm_df, dataset)["method"])),
        ("Best overall", str(best_method(alarm_df, dataset)["method"])),
        ("Best deep", str(best_method(alarm_df, dataset, family="deep")["method"])),
        ("Best shallow", str(best_method(alarm_df, dataset, family="shallow", exclude=["Threshold"])["method"])),
    ]
    seen = set()
    selected: List[Tuple[str, str]] = []
    for role, method in candidates:
        if method in seen:
            continue
        seen.add(method)
        selected.append((role, method))
    return selected


def plot(data_dir: Path, outdir: Path, results_path: Optional[Path] = None, strict: bool = False) -> List[Path]:
    alarm_df, _ = load_alarm_results(data_dir=data_dir, results_path=results_path)
    diagnostics, _ = load_metric_diagnostics(data_dir=data_dir, strict=strict)

    metrics = ["AAF", "EDF", "LDF"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 2.95), sharex=True)
    width = 0.18
    offsets = {"Threshold": -0.30, "Best overall": -0.10, "Best deep": 0.10, "Best shallow": 0.30}
    x = np.arange(len(DATASET_ORDER))

    legend_handles = {}
    for ax, metric, panel in zip(axes, metrics, ["A", "B", "C"]):
        metric_df = diagnostics[diagnostics["metric"].eq(metric)]
        for dataset_index, dataset in enumerate(DATASET_ORDER):
            for role, method in _selected_methods(alarm_df, dataset):
                row = metric_df[
                    metric_df["dataset"].eq(dataset)
                    & metric_df["method"].eq(method)
                    & metric_df["status"].eq("complete")
                ]
                if row.empty:
                    continue
                value = float(row["mean"].iloc[0])
                std = row["std"].iloc[0]
                yerr = float(std) if pd.notna(std) else None
                bar = ax.bar(
                    dataset_index + offsets[role],
                    value,
                    width=width,
                    yerr=yerr,
                    capsize=1.5 if yerr is not None else 0,
                    color=ROLE_COLORS[role],
                    edgecolor="#222222",
                    linewidth=0.3,
                    error_kw={"linewidth": 0.55, "capthick": 0.55},
                    label=role,
                )
                if role not in legend_handles:
                    legend_handles[role] = bar
        ax.set_title(f"{panel}. {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER], rotation=35, ha="right")
        ax.set_ylabel(metric)
        clean_axis(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1.08)
    axes[2].set_ylim(0, 1.08)
    fig.suptitle("ALARM diagnostic components, not standalone ranking metrics", y=0.98, fontsize=8)
    fig.legend(
        [legend_handles[key] for key in ROLE_COLORS if key in legend_handles],
        [key for key in ROLE_COLORS if key in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93], w_pad=1.0)

    if not legend_handles:
        message = "Skipping diagnostic figure: AAF/EDF/LDF data not found."
        if strict:
            raise FileNotFoundError(message)
        raise OptionalFigureSkipped(message)

    paths = save_figure(fig, outdir, "fig_alarm_diagnostics")
    return list(paths)

