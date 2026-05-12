from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from noboom_benchmark.noboom_lib.core.benchmark_utils.model_utils import get_args
from noboom_benchmark.noboom_lib.core.models import catch_paper_neural as neural


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs"

MODEL_CASES = {
    "dlinear": (
        neural.DLinear,
        {"input_dim": 3, "seq_len": 16, "moving_avg": 3, "individual": False},
    ),
    "nlinear": (
        neural.NLinear,
        {"input_dim": 3, "seq_len": 16, "individual": False},
    ),
    "itransformer": (
        neural.ITransformer,
        {"input_dim": 3, "seq_len": 16, "d_model": 8, "e_layers": 1, "n_heads": 1, "d_ff": 16},
    ),
    "patchtst": (
        neural.PatchTST,
        {
            "input_dim": 3,
            "seq_len": 16,
            "patch_len": 4,
            "stride": 2,
            "d_model": 8,
            "e_layers": 1,
            "n_heads": 1,
            "d_ff": 16,
        },
    ),
    "modern_tcn": (
        neural.ModernTCN,
        {
            "input_dim": 3,
            "seq_len": 16,
            "dims": [8],
            "num_blocks": [1],
            "large_size": [3],
            "small_size": [3],
        },
    ),
    "dualtf": (
        neural.DualTF,
        {"input_dim": 3, "seq_len": 16, "d_model": 8, "e_layers": 1, "n_heads": 1, "d_ff": 16},
    ),
    "tfad": (
        neural.TFAD,
        {"input_dim": 3, "seq_len": 16, "embedding_rep_dim": 8, "tcn_kernel_size": 3, "tcn_layers": 1},
    ),
    "autoencoder": (
        neural.AutoEncoder,
        {"input_dim": 3, "seq_len": 16, "hidden_dims": [8], "latent_dim": 4},
    ),
}


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


@pytest.mark.parametrize("model_name", sorted(MODEL_CASES))
def test_neural_model_registry_links_feature_and_window_args(model_name: str) -> None:
    mappings = get_args(model_name)

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.seq_len", "parse") in mappings
    assert ("window_size", "model.detector.init_args.seq_len", "parse") in mappings


@pytest.mark.parametrize("model_name", sorted(MODEL_CASES))
def test_neural_model_configs_resolve_to_declared_classes(model_name: str) -> None:
    config = _load_yaml(CONFIG_ROOT / "models" / f"{model_name}.yaml")
    params = _load_yaml(CONFIG_ROOT / "params" / f"{model_name}.yaml")

    assert "search_space" in params
    assert config["model"]["compile_torch_model"] is True
    assert config["model"]["compile_torch_model_mode"] == "max-autotune-no-cudagraphs"
    assert config["model"]["compile_torch_model_fullgraph"] is False
    assert config["trainer"]["precision"] == "bf16-mixed"

    for section in ("network", "detector", "losses"):
        section_config = config["model"][section]
        cls = _resolve_class(section_config["class_path"])
        accepted_args = set(inspect.signature(cls.__init__).parameters)
        declared_args = set(section_config.get("init_args", {}))
        assert declared_args <= accepted_args


@pytest.mark.parametrize("model_name", sorted(MODEL_CASES))
def test_neural_models_reconstruct_window_shape_and_loss(model_name: str) -> None:
    cls, init_args = MODEL_CASES[model_name]
    model = cls(**init_args)
    model.eval()
    x = torch.randn(4, 16, 3)

    output = model((x,))
    loss = neural.NeuralReconstructionLoss()((output,), (x,))

    assert output.shape == x.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("model_name", sorted(MODEL_CASES))
def test_neural_detector_returns_one_score_per_window(model_name: str) -> None:
    cls, init_args = MODEL_CASES[model_name]
    model = cls(**init_args)
    detector = neural.NeuralReconstructionAnomalyDetector(model=model, seq_len=16, batch_size=2)
    x = torch.randn(5, 16, 3)

    scores = detector.compute_online_anomaly_score((x,))

    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()


def test_neural_detector_score_increases_with_last_step_reconstruction_error() -> None:
    class ZeroReconstructor(torch.nn.Module):
        seq_len = 4

        def forward(self, inputs):
            x = inputs[0]
            return torch.zeros_like(x)

    detector = neural.NeuralReconstructionAnomalyDetector(
        model=ZeroReconstructor(),
        seq_len=4,
        batch_size=2,
    )
    normal = torch.zeros(3, 4, 2)
    anomalous = normal.clone()
    anomalous[:, -1, :] = 10.0

    normal_scores = detector.compute_online_anomaly_score((normal,))
    anomalous_scores = detector.compute_online_anomaly_score((anomalous,))

    assert normal_scores.shape == (3,)
    assert anomalous_scores.shape == (3,)
    assert torch.all(anomalous_scores > normal_scores)
