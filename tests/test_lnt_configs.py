from pathlib import Path

import yaml


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
PARAM_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params"


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_lnt_model_config_uses_safe_encoder_defaults() -> None:
    lnt_config = _load_yaml(MODEL_CONFIG_DIR / "lnt.yaml")

    assert lnt_config["model"]["compile_torch_model"] is True
    assert lnt_config["model"]["compile_torch_model_mode"] == "max-autotune-no-cudagraphs"

    init_args = lnt_config["model"]["network"]["init_args"]
    assert init_args["encoder_type"] == "sensorformer_time"

    encoder_cfg = init_args["encoder_cfg"]
    assert "num_layers" not in encoder_cfg
    assert "token_chunk_size" not in encoder_cfg
    assert encoder_cfg["global_patches"] == 16
    assert encoder_cfg["attention_chunk_size"] is None
    assert encoder_cfg["kernel_sizes"] == [7, 7, 15, 15]
    assert encoder_cfg["dilations"] == [1, 2, 4, 8]
    assert encoder_cfg["ff_mult"] == 4
    assert len(encoder_cfg["strides"]) == len(encoder_cfg["filters"]) == len(encoder_cfg["padding"])
    assert encoder_cfg["upsampler"] == "linear"


def test_lnt_param_config_excludes_bosch_cpc_search_axes() -> None:
    lnt_params = _load_yaml(PARAM_CONFIG_DIR / "lnt.yaml")

    init_args = lnt_params["search_space"]["model"]["network"]["init_args"]
    assert set(init_args["encoder_type"]["choices"]) == {"cnn", "modern_tcn", "sensorformer_time"}

    encoder_cfg = init_args["encoder_cfg"]
    assert "num_layers" not in encoder_cfg
    assert "global_patches" not in encoder_cfg
    assert "attention_chunk_size" not in encoder_cfg
    assert encoder_cfg["kernel_sizes"]["choices"] == [[5, 5, 9, 9], [7, 7, 15, 15]]
    assert encoder_cfg["dilations"]["choices"] == [[1, 1, 2, 4], [1, 2, 4, 8]]
    assert encoder_cfg["ff_mult"]["choices"] == [2, 4]
    assert "enc_hidden" not in encoder_cfg
    assert "strides" not in encoder_cfg
    assert "filters" not in encoder_cfg
    assert "padding" not in encoder_cfg
    assert "upsampler" not in encoder_cfg
