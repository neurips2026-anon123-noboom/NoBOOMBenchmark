from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from noboom_benchmark.noboom_lib.core.benchmark_utils.model_utils import get_args
from noboom_benchmark.noboom_lib.core.tune_constants import CPU_ONLY_MODELS, uses_lightning_optimizer


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs"
MODEL_CONFIG_DIR = CONFIG_ROOT / "models"
PARAM_CONFIG_DIR = CONFIG_ROOT / "params"

NEURAL_MODELS = [
    "itransformer",
    "modern_tcn",
    "dualtf",
    "patchtst",
    "dlinear",
    "nlinear",
    "tfad",
    "autoencoder",
]
DETECTOR_MODELS = ["ocsvm", "pca", "hbos"]
CATCH_PAPER_SCOPE = NEURAL_MODELS + DETECTOR_MODELS


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict)
    return config


def test_catch_paper_scope_configs_and_params_exist_without_isolationforest() -> None:
    for model_name in CATCH_PAPER_SCOPE:
        assert (MODEL_CONFIG_DIR / f"{model_name}.yaml").is_file()
        assert (PARAM_CONFIG_DIR / f"{model_name}.yaml").is_file()

    shipped_names = {path.stem.lower() for path in MODEL_CONFIG_DIR.glob("*.yaml")}
    assert "isolationforest" not in shipped_names
    assert "isolation_forest" not in shipped_names


def test_catch_paper_scope_model_configs_follow_expected_training_style() -> None:
    for model_name in NEURAL_MODELS:
        config = _load_yaml(MODEL_CONFIG_DIR / f"{model_name}.yaml")
        assert "network" in config["model"]
        assert "losses" in config["model"]
        assert "optimizer" in config["model"]
        assert config["model"]["detector"]["class_path"] == (
            "noboom_benchmark.noboom_lib.core.models.catch_paper_neural."
            "NeuralReconstructionAnomalyDetector"
        )
        assert uses_lightning_optimizer(model_name)

    for model_name in DETECTOR_MODELS:
        config = _load_yaml(MODEL_CONFIG_DIR / f"{model_name}.yaml")
        assert "network" not in config["model"]
        assert "losses" not in config["model"]
        assert "optimizer" not in config["model"]
        assert model_name in CPU_ONLY_MODELS
        assert not uses_lightning_optimizer(model_name)


def test_catch_paper_scope_registry_wires_neural_dimensions_only() -> None:
    for model_name in NEURAL_MODELS:
        mappings = get_args(model_name)
        assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
        assert any(source == "window_size" for source, _target, _apply_on in mappings)

    for model_name in DETECTOR_MODELS:
        assert get_args(model_name) == []


def test_catch_paper_neural_adapters_forward_and_score_tiny_windows() -> None:
    from noboom_benchmark.noboom_lib.core.models.catch_paper_neural import (
        AutoEncoder,
        DLinear,
        DualTF,
        ITransformer,
        ModernTCN,
        NLinear,
        PatchTST,
        NeuralReconstructionAnomalyDetector,
        TFAD,
    )

    x = torch.randn(4, 16, 3, generator=torch.Generator().manual_seed(17))
    specs = [
        (ITransformer, {"input_dim": 3, "seq_len": 16, "d_model": 16, "n_heads": 4, "d_ff": 32}),
        (ModernTCN, {"input_dim": 3, "seq_len": 16, "dims": [8], "large_size": [5]}),
        (DualTF, {"input_dim": 3, "seq_len": 16, "d_model": 16, "n_heads": 4}),
        (
            PatchTST,
            {"input_dim": 3, "seq_len": 16, "patch_len": 4, "stride": 2, "d_model": 16, "n_heads": 4},
        ),
        (DLinear, {"input_dim": 3, "seq_len": 16, "moving_avg": 5}),
        (NLinear, {"input_dim": 3, "seq_len": 16}),
        (TFAD, {"input_dim": 3, "seq_len": 16, "embedding_rep_dim": 16}),
        (AutoEncoder, {"input_dim": 3, "seq_len": 16, "hidden_dims": [16], "latent_dim": 8}),
    ]

    for model_cls, kwargs in specs:
        model = model_cls(**kwargs)
        output = model((x,))
        detector = NeuralReconstructionAnomalyDetector(model)
        scores = detector.compute_online_anomaly_score((x,))

        assert output.shape == x.shape
        assert scores.shape == (x.shape[0],)
        assert torch.isfinite(scores).all()
