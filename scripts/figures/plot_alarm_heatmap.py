from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DATASET_LABELS,
    DATASET_ORDER,
    DOUBLE_COLUMN_WIDTH,
    METHOD_ORDER,
    completed_alarm,
    load_alarm_results,
    rounded_axis_limit,
    save_figure,
)


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot(index="method", columns="dataset", values="alarm_mean")
    return table.reindex(index=METHOD_ORDER, columns=DATASET_ORDER)


def plot(data_dir: Path, outdir: Path, results_path: Optional[Path] = None) -> List[Path]:
    df, _ = load_alarm_results(data_dir=data_dir, results_path=results_path)
    matrix = _matrix(df)
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    vmax_observed = float(np.nanmax(values))
    vmax = 90.0 if vmax_observed <= 90.0 else rounded_axis_limit(vmax_observed)

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 5.65))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[name] for name in DATASET_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, method in enumerate(METHOD_ORDER):
        for j, dataset in enumerate(DATASET_ORDER):
            value = matrix.loc[method, dataset]
            if pd.isna(value):
                ax.text(j, i, "\u2014", ha="center", va="center", color="#666666", fontsize=7)
            else:
                text_color = "white" if value > 0.55 * vmax else "#111111"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=6.4)

    complete = completed_alarm(df)
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        sub = complete[complete["dataset"].eq(dataset)]
        if sub.empty:
            continue
        best = sub.loc[sub["alarm_mean"].idxmax()]
        method_index = METHOD_ORDER.index(best["method"])
        ax.add_patch(
            patches.Rectangle(
                (dataset_index - 0.5, method_index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#111111",
                linewidth=1.3,
            )
        )
        ax.scatter(
            [dataset_index + 0.33],
            [method_index - 0.31],
            marker="*",
            s=18,
            c="#111111",
            linewidths=0.0,
            zorder=4,
        )

    divider_y = 17.5
    ax.axhline(divider_y, color="#111111", linewidth=0.8)
    ax.text(-1.85, 8.5, "Deep", rotation=90, ha="center", va="center", fontweight="bold")
    ax.text(-1.85, 20.5, "Shallow", rotation=90, ha="center", va="center", fontweight="bold")

    colorbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("Normalized ALARM (%)")
    colorbar.ax.tick_params(labelsize=7, length=2)

    ax.set_title("ALARM performance by method and dataset")
    paths = save_figure(fig, outdir, "fig_alarm_heatmap")
    return list(paths)
