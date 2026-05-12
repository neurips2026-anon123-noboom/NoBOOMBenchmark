from pathlib import Path

import yaml

from noboom_benchmark.noboom_lib.core.tune_constants import CPU_ONLY_MODELS


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
PARAM_CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params"


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_hmm_variants_are_cpu_only_models() -> None:
    assert "hmm" in CPU_ONLY_MODELS
    assert "gmmhmm" in CPU_ONLY_MODELS


def test_hmm_model_configs_use_latest_timesead_baselines() -> None:
    hmm_config = _load_yaml(MODEL_CONFIG_DIR / "hmm.yaml")
    gmmhmm_config = _load_yaml(MODEL_CONFIG_DIR / "gmmhmm.yaml")

    assert hmm_config["model"]["detector"]["class_path"] == "timesead.models.baselines.HMMAnomalyDetector"
    assert gmmhmm_config["model"]["detector"]["class_path"] == "timesead.models.baselines.GMMHMMAnomalyDetector"

    assert hmm_config["trainer"]["accelerator"] == "cpu"
    assert gmmhmm_config["trainer"]["accelerator"] == "cpu"
    assert hmm_config["trainer"]["max_epochs"] == 0
    assert gmmhmm_config["trainer"]["max_epochs"] == 0
    assert hmm_config["data"]["batch_dim"] == 0
    assert gmmhmm_config["data"]["batch_dim"] == 0


def test_hmm_param_configs_expose_expected_search_spaces() -> None:
    hmm_params = _load_yaml(PARAM_CONFIG_DIR / "hmm.yaml")
    gmmhmm_params = _load_yaml(PARAM_CONFIG_DIR / "gmmhmm.yaml")

    hmm_init_args = hmm_params["search_space"]["model"]["detector"]["init_args"]
    gmmhmm_init_args = gmmhmm_params["search_space"]["model"]["detector"]["init_args"]

    assert set(hmm_init_args) == {"n_components", "covariance_type", "n_iter"}
    assert set(gmmhmm_init_args) == {"n_components", "n_mix", "n_iter"}
