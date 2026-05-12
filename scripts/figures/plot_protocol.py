from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

from common import DOUBLE_COLUMN_WIDTH, save_figure


def _box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    text: str,
    width: float,
    height: float,
    face: str,
    fontsize: float = 7.1,
) -> None:
    patch = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0.8,
        edgecolor="#333333",
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        linespacing=1.15,
        color="#111111",
        fontsize=fontsize,
    )


def _arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = "#444444") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot(outdir: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    flow = [
        ("Raw\nmultivariate\ntime series", 0.020, "#e8eef6"),
        ("Normalization\n/ windowing", 0.160, "#eef3f7"),
        ("Detector\nmodel", 0.300, "#e8f1ec"),
        ("Continuous\nanomaly score\ns(t)", 0.435, "#f1f5e8"),
        ("Thresholding\n+ alarm\npostprocessing", 0.575, "#f7f0df"),
        ("Binary alarm\nsegments\np(t)", 0.725, "#f3e9e6"),
        ("Event-level\nALARM\nevaluation", 0.860, "#e9e6f3"),
    ]
    width = 0.110
    height = 0.30
    y = 0.50
    for text, x, face in flow:
        _box(ax, (x, y), text, width, height, face)
    for idx in range(len(flow) - 1):
        x0 = flow[idx][1] + width
        x1 = flow[idx + 1][1]
        _arrow(ax, (x0, y + height / 2), (x1, y + height / 2))

    _box(ax, (0.205, 0.12), "Validation labels\n/ tuning objective", 0.160, 0.22, "#f4f4f4")
    _box(ax, (0.425, 0.12), "Hyperparameter\nand threshold\nselection", 0.165, 0.22, "#f4f4f4")
    _arrow(ax, (0.365, 0.23), (0.425, 0.23))
    _arrow(ax, (0.505, 0.34), (0.360, 0.50), "#666666")
    _arrow(ax, (0.565, 0.34), (0.640, 0.50), "#666666")

    notes = [
        (0.08, 0.92, "Report normalized ALARM"),
        (0.43, 0.92, "Diagnostics: AAF, EDF, LDF"),
        (0.73, 0.92, "Specify validation vs oracle tuning"),
    ]
    for x, y_note, text in notes:
        ax.text(
            x,
            y_note,
            text,
            ha="left",
            va="center",
            fontsize=7,
            color="#222222",
            bbox={"boxstyle": "square,pad=0.18", "facecolor": "#ffffff", "edgecolor": "#bdbdbd", "linewidth": 0.5},
        )

    paths = save_figure(fig, outdir, "fig_protocol")
    return list(paths)
