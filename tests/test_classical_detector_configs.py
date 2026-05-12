from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from noboom_benchmark.noboom_lib.core.tune_constants import CPU_ONLY_MODELS


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DIR = REPO_ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models"
PARAM_CONFIG_DIR = REPO_ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params"
CLASSICAL_DETECTORS = {
    "ocsvm": "OCSVMAD",
    "pca": "PCAAD",
    "hbos": "HBOSAD",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_class(class_path: str) -> type:
    module_name, _, class_name = class_path.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def test_classical_detector_configs_are_cpu_only_detector_style() -> None:
    for model_name, class_name in CLASSICAL_DETECTORS.items():
        config = _load_yaml(MODEL_CONFIG_DIR / f"{model_name}.yaml")
        detector_config = config["model"]["detector"]

        assert model_name in CPU_ONLY_MODELS
        assert "network" not in config["model"]
        assert "losses" not in config["model"]
        assert detector_config["class_path"] == (
            f"noboom_benchmark.noboom_lib.core.models.detectors.{class_name}"
        )
        assert config["trainer"]["accelerator"] == "cpu"
        assert config["trainer"]["max_epochs"] == 0
        assert config["data"]["batch_dim"] == 0


def test_classical_detector_configs_expose_valid_init_args() -> None:
    for model_name in CLASSICAL_DETECTORS:
        config = _load_yaml(MODEL_CONFIG_DIR / f"{model_name}.yaml")
        detector_config = config["model"]["detector"]
        detector_cls = _resolve_class(detector_config["class_path"])
        accepted_args = set(inspect.signature(detector_cls.__init__).parameters)
        declared_args = set(detector_config.get("init_args", {}))

        assert declared_args <= accepted_args


def test_classical_detector_param_configs_expose_expected_search_spaces() -> None:
    expected_args = {
        "ocsvm": {"nu", "gamma", "kernel"},
        "pca": {"n_components", "whiten"},
        "hbos": {"n_bins", "alpha", "bin_tol"},
    }

    for model_name, init_arg_names in expected_args.items():
        params = _load_yaml(PARAM_CONFIG_DIR / f"{model_name}.yaml")
        init_args = params["search_space"]["model"]["detector"]["init_args"]

        assert set(init_args) == init_arg_names
        assert params["search_space"]["window_size"]["low"] == 8
        assert params["search_space"]["window_size"]["high"] == 128


def test_classical_detectors_fit_score_and_are_repeatable() -> None:
    from noboom_benchmark.noboom_lib.core.models.detectors import HBOSAD, OCSVMAD, PCAAD

    generator = torch.Generator().manual_seed(7)
    train_windows = torch.randn(24, 4, 3, generator=generator)
    score_windows = torch.cat([train_windows[:4], torch.full((1, 4, 3), 8.0)], dim=0)
    labels = torch.zeros(len(train_windows), 4)
    loader = DataLoader(TensorDataset(train_windows, labels), batch_size=6, shuffle=False)

    detectors = [
        OCSVMAD(nu=0.2, gamma="scale", random_state=11),
        PCAAD(n_components=0.9, random_state=11),
        HBOSAD(n_bins=5, random_state=11),
    ]

    for detector in detectors:
        detector.fit(loader)
        first = detector.compute_online_anomaly_score((score_windows,))
        second = detector.compute_online_anomaly_score((score_windows,))

        assert first.shape == (len(score_windows),)
        assert torch.isfinite(first).all()
        assert torch.allclose(first, second)
