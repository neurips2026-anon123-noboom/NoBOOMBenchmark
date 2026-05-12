from __future__ import annotations

import importlib
import importlib.util
from argparse import Namespace
from pathlib import Path
import re

import yaml

from noboom_benchmark.noboom_lib.core.tune_constants import CPU_ONLY_MODELS
from noboom_benchmark.noboom_lib.core.tune_utils import create_extra_params_for_lightning


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_ROOT = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
PARAM_CONFIG_ROOT = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params"
MODEL_UTILS_PATH = (
    ROOT
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "model_utils.py"
)

REQUIRED_MODEL_NAMES = {
    "itransformer",
    "modern_tcn",
    "dualtf",
    "patchtst",
    "dlinear",
    "nlinear",
    "tfad",
    "autoencoder",
    "ocsvm",
    "pca",
    "hbos",
}
DEEP_MODEL_NAMES = REQUIRED_MODEL_NAMES - {"ocsvm", "pca", "hbos"}
SEQ_LEN_MODELS = DEEP_MODEL_NAMES


def _load_get_args():
    spec = importlib.util.spec_from_file_location("new_model_registry_model_utils", MODEL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_args


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def _resolve_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def test_required_new_model_names_are_snake_case_and_have_configs() -> None:
    snake_case = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

    assert all(snake_case.fullmatch(model_name) for model_name in REQUIRED_MODEL_NAMES)
    for model_name in REQUIRED_MODEL_NAMES:
        assert (MODEL_CONFIG_ROOT / f"{model_name}.yaml").is_file()
        assert (PARAM_CONFIG_ROOT / f"{model_name}.yaml").is_file()


def test_new_model_class_paths_are_importable_and_avoid_isolation_forest() -> None:
    for model_name in REQUIRED_MODEL_NAMES:
        config_text = (MODEL_CONFIG_ROOT / f"{model_name}.yaml").read_text(encoding="utf-8")
        params_text = (PARAM_CONFIG_ROOT / f"{model_name}.yaml").read_text(encoding="utf-8")
        assert "IsolationForest" not in config_text
        assert "IsolationForest" not in params_text

        config = _load_yaml(MODEL_CONFIG_ROOT / f"{model_name}.yaml")
        detector = config["model"]["detector"]
        _resolve_class(detector["class_path"])
        network = config["model"].get("network")
        if network is not None:
            _resolve_class(network["class_path"])


def test_deep_new_models_link_num_features_and_window_size() -> None:
    get_args = _load_get_args()
    for model_name in DEEP_MODEL_NAMES:
        mappings = set(get_args(model_name))
        assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings

    for model_name in SEQ_LEN_MODELS:
        assert (
            "window_size",
            "model.network.init_args.seq_len",
            "parse",
        ) in set(get_args(model_name))


def test_classical_new_models_are_cpu_only() -> None:
    assert {"ocsvm", "pca", "hbos"}.issubset(CPU_ONLY_MODELS)


def test_classical_new_models_disable_lightning_training_batches(tmp_path: Path) -> None:
    args = Namespace(
        model_name="ocsvm",
        dataset_name="cont_reactive_ome",
        temp_dir=str(tmp_path),
        study_params={"metrics": ["alarm_score"]},
        prepared_manifest_paths={"real": "/prepared/real/manifest.json"},
    )

    hp_path = create_extra_params_for_lightning(
        args,
        logger_params={"run_id": "seed-run-id"},
        seed=42,
        hp_params={},
    )
    payload = yaml.safe_load(Path(hp_path).read_text(encoding="utf-8"))

    assert payload["trainer.max_epochs"] == 1
    assert payload["trainer.limit_train_batches"] == 0
    assert payload["trainer.accelerator"] == "cpu"
