from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import types
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models" / "alora.yaml"
PARAM_CONFIG = ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "params" / "alora.yaml"
MODEL_UTILS_PATH = (
    ROOT
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "model_utils.py"
)


def _install_timesead_stubs() -> None:
    timesead_module = sys.modules.setdefault("timesead", types.ModuleType("timesead"))
    models_module = sys.modules.setdefault("timesead.models", types.ModuleType("timesead.models"))
    common_module = sys.modules.setdefault("timesead.models.common", types.ModuleType("timesead.models.common"))
    anomaly_detector_module = sys.modules.setdefault(
        "timesead.models.common.anomaly_detector",
        types.ModuleType("timesead.models.common.anomaly_detector"),
    )
    optim_module = sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))
    loss_module = sys.modules.setdefault("timesead.optim.loss", types.ModuleType("timesead.optim.loss"))

    class BaseModel(torch.nn.Module):
        def grouped_parameters(self):
            return (self.parameters(),)

    class AnomalyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    class Loss(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            super().__init__()

    models_module.BaseModel = BaseModel
    common_module.AnomalyDetector = AnomalyDetector
    anomaly_detector_module.AnomalyDetector = AnomalyDetector
    loss_module.Loss = Loss

    timesead_module.models = models_module
    timesead_module.optim = optim_module
    models_module.common = common_module
    common_module.anomaly_detector = anomaly_detector_module
    optim_module.loss = loss_module


def _ensure_timesead_imports() -> None:
    try:
        import timesead.models  # noqa: F401
        import timesead.optim.loss  # noqa: F401
        try:
            from timesead.models.common import AnomalyDetector  # noqa: F401
        except ImportError:
            from timesead.models.common.anomaly_detector import AnomalyDetector  # noqa: F401
    except ImportError:
        _install_timesead_stubs()


_ensure_timesead_imports()

from noboom_benchmark.noboom_lib.core.models import ALoRa as ExportedALoRa  # noqa: E402
from noboom_benchmark.noboom_lib.core.models import ALoRaAnomalyDetector as ExportedDetector  # noqa: E402
from noboom_benchmark.noboom_lib.core.models import ALoRaLoss as ExportedLoss  # noqa: E402
from noboom_benchmark.noboom_lib.core.models.alora import (  # noqa: E402
    ALoRa,
    ALoRaAnomalyDetector,
    ALoRaLoss,
    alora_detection_score,
    alora_low_rank_regularizer,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def _load_get_args():
    spec = importlib.util.spec_from_file_location("alora_model_utils", MODEL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_args


def test_alora_imports_from_model_registry() -> None:
    assert ExportedALoRa is ALoRa
    assert ExportedDetector is ALoRaAnomalyDetector
    assert ExportedLoss is ALoRaLoss


def test_alora_forward_returns_reconstruction_and_attention_stack() -> None:
    torch.manual_seed(0)
    model = ALoRa(
        win_size=8,
        input_dim=4,
        d_model=6,
        n_heads=2,
        e_layers=3,
        dropout=0.0,
        top_k_limit=6,
    )
    x = torch.randn(3, 8, 4)

    reconstruction, attentions = model((x,))

    assert reconstruction.shape == x.shape
    assert len(attentions) == 3
    assert all(attention.shape == (3, 2, 8, 8) for attention in attentions)
    assert torch.isfinite(reconstruction).all()
    assert all(torch.isfinite(attention).all() for attention in attentions)


def test_alora_low_rank_penalty_matches_hand_computable_svd() -> None:
    layer_1 = torch.tensor([[[[2.0, 0.0], [0.0, 0.5]]]])
    layer_2 = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    penalty = alora_low_rank_regularizer([layer_1, layer_2], rank=1)

    expected = torch.tensor(0.5 / 1.5 + 1.0 / 2.0)
    assert torch.allclose(penalty, expected, atol=1.0e-6)


def test_alora_loss_is_finite_and_includes_low_rank_regularizer() -> None:
    reconstruction = torch.ones(1, 2, 1)
    target = torch.zeros(1, 2, 1)
    attention = torch.tensor([[[[2.0, 0.0], [0.0, 0.5]]]])
    loss = ALoRaLoss(lambda_reg=2.0, rank=1)

    actual = loss((reconstruction, [attention]), (target,))

    expected_regularizer = torch.tensor(0.5 / 1.5)
    expected = torch.tensor(1.0) + 2.0 * expected_regularizer
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_alora_loss_unwraps_nested_predictions_from_flops_probe() -> None:
    reconstruction = torch.zeros(1, 2, 1)
    target = torch.zeros(1, 2, 1)
    attention = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    loss = ALoRaLoss(lambda_reg=1.0, rank=1)
    predictions = (reconstruction, [attention])

    assert torch.allclose(loss(predictions, (target,)), loss((predictions,), (target,)))


def test_alora_detection_score_direction_increases_with_error_and_rank() -> None:
    x = torch.zeros(3, 3, 1)
    reconstruction = torch.zeros_like(x)
    reconstruction[:, -1, 0] = torch.tensor([1.0, 2.0, 1.0])
    attention = torch.zeros(3, 1, 3, 3)
    attention[:, 0, 0, 0] = 1.0
    attention[2, 0, 1, 1] = 1.0

    scores = alora_detection_score(x, reconstruction, [attention], rank_threshold=0.5)

    assert scores.shape == (3,)
    assert scores[1] > scores[0]
    assert scores[2] > scores[0]


def test_alora_detector_returns_per_window_scores() -> None:
    class ScoreDirectionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            batch = inputs[0]
            reconstruction = torch.zeros_like(batch) * self.anchor
            reconstruction[:, -1, 0] = batch[:, 0, 0]
            attention = torch.zeros(batch.shape[0], 1, batch.shape[1], batch.shape[1], device=batch.device)
            attention[:, 0, 0, 0] = 1.0
            rank_two = batch[:, 0, 1] > 0.0
            attention[rank_two, 0, 1, 1] = 1.0
            return reconstruction, [attention]

    windows = torch.zeros(3, 3, 2)
    windows[:, 0, 0] = torch.tensor([1.0, 2.0, 1.0])
    windows[:, 0, 1] = torch.tensor([0.0, 0.0, 1.0])
    detector = ALoRaAnomalyDetector(model=ScoreDirectionModel(), rank_threshold=0.5, batch_size=2)

    scores = detector.compute_online_anomaly_score((windows,))

    assert scores.shape == (3,)
    assert scores[1] > scores[0]
    assert scores[2] > scores[0]


def test_alora_configs_and_model_utils_links_are_constructable() -> None:
    model_config = _load_yaml(MODEL_CONFIG)
    params_config = _load_yaml(PARAM_CONFIG)

    assert params_config["search_space"]
    assert model_config["model"]["predict_on_end"] is True
    assert model_config["model"]["network"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.models.alora.ALoRa"
    )
    assert model_config["model"]["detector"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.models.alora.ALoRaAnomalyDetector"
    )
    assert model_config["model"]["losses"]["class_path"] == (
        "noboom_benchmark.noboom_lib.core.models.alora.ALoRaLoss"
    )

    for section in ("network", "detector", "losses"):
        section_config = model_config["model"][section]
        module_name, class_name = section_config["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        accepted_args = set(inspect.signature(cls.__init__).parameters)
        declared_args = set(section_config.get("init_args", {}))
        assert declared_args <= accepted_args

    mappings = set(_load_get_args()("alora"))
    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.win_size", "parse") in mappings


def test_alora_tiny_synthetic_train_and_validation_step() -> None:
    torch.manual_seed(4)
    model = ALoRa(
        win_size=6,
        input_dim=4,
        d_model=6,
        n_heads=2,
        e_layers=1,
        dropout=0.0,
        top_k_limit=6,
    )
    loss = ALoRaLoss(lambda_reg=0.01, rank=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x = torch.randn(4, 6, 4)

    model.train()
    optimizer.zero_grad()
    train_loss = loss(model((x,)), (x,))
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss(model((x,)), (x,))

    assert torch.isfinite(train_loss)
    assert torch.isfinite(val_loss)
