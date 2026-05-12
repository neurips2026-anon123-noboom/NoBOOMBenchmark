from __future__ import annotations

from pathlib import Path

import yaml


KAN_AD_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
    / "models"
    / "kan_ad.yaml"
)


def test_shipped_model_configs_use_packaged_module_paths() -> None:
    config_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "noboom_cluster"
        / "cluster_files"
        / "configs"
        / "models"
    )

    offending_files = []
    for config_path in sorted(config_root.glob("*.yaml")):
        content = config_path.read_text(encoding="utf-8")
        if "class_path: noboom_lib." in content:
            offending_files.append(config_path.name)

    assert offending_files == []


def test_kan_ad_config_offsets_differenced_labels_to_raw_endpoint() -> None:
    content = KAN_AD_CONFIG_PATH.read_text(encoding="utf-8")

    assert "label_index_offset: 1" in content


def test_kan_ad_uses_difference_prediction_pipeline_without_per_window_scaling() -> None:
    with KAN_AD_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    expected_classes = [
        "noboom_benchmark.noboom_lib.core.benchmark_utils.noboom_transforms.FirstOrderDifferenceTransform",
        "timesead.data.transforms.PredictionTargetTransform",
    ]
    replace_labels_by_split = {
        "train_transform": True,
        "val_transform": True,
        "test_transform": False,
    }

    for split, expected_replace_labels in replace_labels_by_split.items():
        steps = config["data"][split]
        assert [step["class_path"] for step in steps] == expected_classes
        assert steps[1]["init_args"]["replace_labels"] is expected_replace_labels
