from pathlib import Path

import yaml


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
PARAM_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params"


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_hpad_model_config_uses_local_classes_and_expected_defaults() -> None:
    hpad_config = _load_yaml(MODEL_CONFIG_DIR / "hpad.yaml")

    assert hpad_config["model"]["network"]["class_path"] == "noboom_benchmark.noboom_lib.core.models.hpad.HPAD"
    assert hpad_config["model"]["detector"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.models.hpad.HPADAnomalyDetector"
    )
    assert hpad_config["model"]["losses"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.models.hpad.HPADLoss"
    )

    init_args = hpad_config["model"]["network"]["init_args"]
    assert init_args["patch_scales"] == [1, 2, 4]
    assert init_args["num_patch_prototypes"] == 16
    assert init_args["top_k_periods"] == 3
    assert init_args["num_period_prototypes"] == 8
    assert init_args["prototype_dim"] is None

    detector_args = hpad_config["model"]["detector"]["init_args"]
    assert detector_args["beta"] == 1.0

    loss_args = hpad_config["model"]["losses"]["init_args"]
    assert set(loss_args) == {"lambda_rec", "lambda_patch", "lambda_ent", "lambda_period"}


def test_hpad_param_config_exposes_fedformer_and_hpad_search_axes() -> None:
    hpad_params = _load_yaml(PARAM_CONFIG_DIR / "hpad.yaml")
    network_args = hpad_params["search_space"]["model"]["network"]

    assert {"modes", "model_dim", "dropout", "num_heads", "encoder_layers"} <= set(network_args)
    assert {"patch_scales", "num_patch_prototypes", "top_k_periods", "num_period_prototypes"} <= set(network_args)

    loss_args = hpad_params["search_space"]["model"]["losses"]["init_args"]
    assert set(loss_args) == {"lambda_patch", "lambda_ent", "lambda_period"}

    detector_args = hpad_params["search_space"]["model"]["detector"]["init_args"]
    assert set(detector_args) == {"beta"}
