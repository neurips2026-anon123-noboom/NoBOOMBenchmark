from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DATASET_LABELS,
    DATASET_ORDER,
    SINGLE_COLUMN_WIDTH,
    best_method,
    clean_axis,
    completed_alarm,
    load_alarm_results,
    rounded_axis_limit,
    save_figure,
    threshold_row,
)


MARKERS: Dict[str, str] = {
    "Threshold": "s",
    "Best deep": "o",
    "Best shallow": "^",
    "Best overall": "*",
}

COLORS: Dict[str, str] = {
    "Threshold": "#4d4d4d",
    "Best deep": "#0072B2",
    "Best shallow": "#009E73",
    "Best overall": "#D55E00",
}


def _comparison_rows(df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        threshold = threshold_row(df, dataset)
        deep = best_method(df, dataset, family="deep")
        shallow = best_method(df, dataset, family="shallow", exclude=["Threshold"])
        overall = best_method(df, dataset)
        rows.extend(
            [
                {"dataset": dataset, "category": "Threshold", "method": threshold["method"], "value": threshold["alarm_mean"]},
                {"dataset": dataset, "category": "Best deep", "method": deep["method"], "value": deep["alarm_mean"]},
                {"dataset": dataset, "category": "Best shallow", "method": shallow["method"], "value": shallow["alarm_mean"]},
                {"dataset": dataset, "category": "Best overall", "method": overall["method"], "value": overall["alarm_mean"]},
            ]
        )
    return rows


def plot(data_dir: Path, outdir: Path, results_path: Optional[Path] = None) -> List[Path]:
    df, _ = load_alarm_results(data_dir=data_dir, results_path=results_path)
    rows = pd.DataFrame(_comparison_rows(df))
    max_value = max(float(completed_alarm(df)["alarm_mean"].max()), 1.0)
    xmax = 90.0 if max_value <= 90 else rounded_axis_limit(max_value)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.10))
    y_base = np.arange(len(DATASET_ORDER))
    offsets = {"Threshold": -0.24, "Best deep": -0.08, "Best shallow": 0.08, "Best overall": 0.24}

    for dataset_index, dataset in enumerate(DATASET_ORDER):
        sub = rows[rows["dataset"].eq(dataset)]
        span_values = sub["value"].astype(float)
        ax.hlines(dataset_index, span_values.min(), span_values.max(), color="#c7c7c7", linewidth=0.8, zorder=1)

    for category in ["Threshold", "Best deep", "Best shallow", "Best overall"]:
        sub = rows[rows["category"].eq(category)]
        y = np.array([DATASET_ORDER.index(dataset) for dataset in sub["dataset"]]) + offsets[category]
        ax.scatter(
            sub["value"].astype(float),
            y,
            marker=MARKERS[category],
            s=24 if category != "Best overall" else 40,
            color=COLORS[category],
            edgecolor="white",
            linewidth=0.3,
            label=category,
            zorder=3,
        )

    overall = rows[rows["category"].eq("Best overall")]
    for _, row in overall.iterrows():
        y = DATASET_ORDER.index(row["dataset"]) + offsets["Best overall"]
        label = f"{row['method']} {float(row['value']):.2f}"
        ax.text(float(row["value"]) + 1.4, y, label, ha="left", va="center", fontsize=6.7)

    ax.set_yticks(y_base)
    ax.set_yticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax + 16)
    ax.set_xlabel("Normalized ALARM (%)")
    ax.set_title("Dataset winners and gaps")
    clean_axis(ax)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.4, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.9)

    paths = save_figure(fig, outdir, "fig_winner_gap")
    return list(paths)

