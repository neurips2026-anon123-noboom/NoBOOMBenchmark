from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from common import DOUBLE_COLUMN_WIDTH, clean_axis, load_dataset_metadata, save_figure


def plot(data_dir: Path, outdir: Path) -> List[Path]:
    df, _ = load_dataset_metadata(data_dir=data_dir)
    datasets = list(df["dataset"].astype(str))
    labels = ["bpw", "abm", "wat", "but", "ome", "ind"]
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 2.70), gridspec_kw={"width_ratios": [1.45, 0.9, 1.0]})
    ax_steps, ax_features, ax_anoms = axes

    width = 0.36
    ax_steps.bar(x - width / 2, df["train_steps"], width=width, label="Train", color="#6BAED6", edgecolor="#333333", linewidth=0.35)
    ax_steps.bar(x + width / 2, df["test_steps"], width=width, label="Test", color="#FD8D3C", edgecolor="#333333", linewidth=0.35)
    ax_steps.set_yscale("log")
    ax_steps.set_xticks(x)
    ax_steps.set_xticklabels(labels, rotation=40, ha="right")
    ax_steps.set_ylabel("Time steps")
    ax_steps.set_title("A. Train/test size")
    ax_steps.legend(frameon=False, loc="upper left")
    clean_axis(ax_steps)

    feature_colors = ["#9ECAE1" if dataset != "cont_ind" else "#A1D99B" for dataset in datasets]
    edge_widths = [0.35 if dataset != "cont_ind" else 1.5 for dataset in datasets]
    bars = ax_features.bar(x, df["features"], color=feature_colors, edgecolor="#111111", linewidth=edge_widths)
    ax_features.set_yscale("log")
    ax_features.set_xticks(x)
    ax_features.set_xticklabels(labels, rotation=40, ha="right")
    ax_features.set_ylabel("Sensors")
    ax_features.set_title("B. Feature count")
    clean_axis(ax_features)
    cont_idx = datasets.index("cont_ind")
    for bar, value in zip(bars, df["features"]):
        ax_features.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{int(value)}", ha="center", va="bottom", fontsize=6.4)

    anomaly_colors = ["#BCBDDC" if dataset != "cont_ind" else "#A1D99B" for dataset in datasets]
    edge_widths = [0.35 if dataset != "cont_ind" else 1.5 for dataset in datasets]
    bars = ax_anoms.bar(x, df["test_anomalies"], color=anomaly_colors, edgecolor="#111111", linewidth=edge_widths)
    ax_anoms.set_xticks(x)
    ax_anoms.set_xticklabels(labels, rotation=40, ha="right")
    ax_anoms.set_ylabel("Test anomalies")
    ax_anoms.set_title("C. Event load")
    clean_axis(ax_anoms)
    top = float(df["test_anomalies"].max()) * 1.20
    ax_anoms.set_ylim(0, top)
    for bar, pct in zip(bars, df["test_anomaly_pct"]):
        ax_anoms.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * 0.025,
            f"{int(pct)}%",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.3,
        )

    fig.tight_layout(w_pad=1.1)
    paths = save_figure(fig, outdir, "fig_dataset_regimes")
    return list(paths)
