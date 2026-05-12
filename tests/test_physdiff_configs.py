import importlib.util
from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs"
MODEL_CONFIG = CONFIG_ROOT / "models" / "physdiff.yaml"
PARAM_CONFIG = CONFIG_ROOT / "params" / "physdiff.yaml"
MODEL_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "model_utils.py"
)


def _load_model_utils():
    spec = importlib.util.spec_from_file_location("physdiff_test_model_utils", MODEL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_physdiff_configs_exist_and_use_local_class_paths() -> None:
    assert MODEL_CONFIG.exists()
    assert PARAM_CONFIG.exists()

    content = MODEL_CONFIG.read_text(encoding="utf-8")
    assert "timesead_ext" not in content

    config = _load_yaml(MODEL_CONFIG)
    assert config["model"]["network"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.physdiff.PhysDiff"
    )
    assert config["model"]["losses"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.physdiff.PhysDiffLoss"
    )
    assert config["model"]["detector"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.physdiff.PhysDiffAnomalyDetector"
    )


def test_physdiff_model_config_defaults() -> None:
    config = _load_yaml(MODEL_CONFIG)

    assert config["window_size"] == 64
    assert config["trainer"]["max_epochs"] == 25
    assert config["trainer"]["precision"] == 32
    assert config["data"]["batch_dim"] == 0

    init_args = config["model"]["network"]["init_args"]
    assert init_args["time_steps"] == 1000
    assert init_args["model_dim"] == 512
    assert init_args["ff_dim"] == 2048
    assert init_args["attn_dim"] == 64
    assert init_args["num_heads"] == 8
    assert init_args["sampling_steps"] == 64
    assert init_args["legacy_disturbance"] is True
    assert init_args["langevin_noise"] is True
    assert init_args["aspe_tolerance"] == 1e-3

    detector_args = config["model"]["detector"]["init_args"]
    assert detector_args == {
        "score_alpha": 0.5,
        "smoothing_kernel_size": 5,
        "spot_q": 0.01,
        "reconstruction_num_samples": 1,
        "component_standardize": False,
        "score_normalize": False,
    }

    transforms = config["data"]["train_transform"]
    assert [step["class_path"] for step in transforms] == [
        "timesead.data.transforms.WindowTransformIfNotWindow",
        "timesead.data.transforms.ReconstructionTargetTransform",
    ]


def test_physdiff_param_config_uses_nested_search_paths() -> None:
    params = _load_yaml(PARAM_CONFIG)
    search_space = params["search_space"]

    optimizer_args = search_space["model"]["optimizer"]["init_args"]
    assert optimizer_args["lr"] == {
        "distribution": "loguniform",
        "low": 1e-5,
        "high": 1e-3,
    }
    assert optimizer_args["weight_decay"] == {
        "distribution": "loguniform",
        "low": 1e-6,
        "high": 1e-3,
    }

    network_args = search_space["model"]["network"]["init_args"]
    assert network_args["model_dim"]["choices"] == [128, 256, 512]
    assert network_args["ff_dim"]["choices"] == [512, 1024, 2048]
    assert network_args["num_blocks"] == {"distribution": "int", "low": 1, "high": 3}
    assert network_args["sampling_steps"]["choices"] == [32, 64, 128]

    detector_args = search_space["model"]["detector"]["init_args"]
    assert detector_args["score_alpha"]["choices"] == [0.25, 0.5, 0.75]
    assert detector_args["smoothing_kernel_size"]["choices"] == [1, 3, 5, 7]
    assert detector_args["spot_q"]["choices"] == [0.005, 0.01, 0.02]
    assert search_space["window_size"]["choices"] == [32, 64, 128]


def test_physdiff_model_utils_registers_required_links() -> None:
    model_utils = _load_model_utils()
    assert (
        "data.num_features",
        "model.network.init_args.num_channels",
        "instantiate",
    ) in model_utils.get_args("physdiff")
    assert (
        "window_size",
        "model.network.init_args.win_size",
        "parse",
    ) in model_utils.get_args("physdiff")
