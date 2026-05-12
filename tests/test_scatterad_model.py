from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import torch
import yaml

from noboom_benchmark.noboom_lib.core.models import ScatterAD as RegistryScatterAD
from noboom_benchmark.noboom_lib.core.models.scatterad import (
    ScatterAD,
    ScatterADAdam,
    ScatterADAnomalyDetector,
    ScatterADLoss,
    build_temporal_lookback_adjacency,
    build_temporal_lookback_edges,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_UTILS_PATH = (
    ROOT
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
    / "model_utils.py"
)
MODEL_CONFIG_PATH = (
    ROOT
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
    / "models"
    / "scatterad.yaml"
)
PARAM_CONFIG_PATH = (
    ROOT
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
    / "params"
    / "scatterad.yaml"
)


def _make_model() -> ScatterAD:
    torch.manual_seed(0)
    return ScatterAD(
        input_dim=3,
        win_size=5,
        hidden_dim=16,
        num_layers=1,
        heads=4,
        tau=2,
        dropout=0.0,
        ema_momentum=0.5,
    )


def _resolve_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _load_model_utils():
    spec = importlib.util.spec_from_file_location("scatterad_model_utils", MODEL_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registry_exports_scatterad_classes() -> None:
    assert RegistryScatterAD is ScatterAD
    detector_cls = _resolve_class(
        "noboom_benchmark.noboom_lib.core.models.scatterad.ScatterADAnomalyDetector"
    )
    loss_cls = _resolve_class("noboom_benchmark.noboom_lib.core.models.scatterad.ScatterADLoss")
    optimizer_cls = _resolve_class("noboom_benchmark.noboom_lib.core.models.scatterad.ScatterADAdam")

    assert detector_cls is ScatterADAnomalyDetector
    assert loss_cls is ScatterADLoss
    assert optimizer_cls is ScatterADAdam


def test_forward_outputs_embedding_and_score_shapes() -> None:
    model = _make_model()
    x = torch.randn(2, 5, 3)

    outputs = model((x,))

    assert outputs["online_embeddings"].shape == (2, 5, 16)
    assert outputs["predicted_embeddings"].shape == (2, 5, 16)
    assert outputs["target_embeddings"].shape == (2, 5, 16)
    assert outputs["scatter_embeddings"].shape == (2, 5, 16)
    assert outputs["edge_index"].shape == (2, 14)
    assert outputs["score_components"]["scattering"].shape == (2, 5)
    assert outputs["score_components"]["time_inconsistency"].shape == (2, 5)
    assert outputs["score_components"]["score"].shape == (2, 5)


def test_loss_is_scalar_and_finite() -> None:
    model = _make_model()
    loss_fn = ScatterADLoss()
    x = torch.randn(2, 5, 3)

    loss = loss_fn((model((x,)),), (x,))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_scatter_loss_backpropagates_to_online_encoder_only() -> None:
    model = _make_model()
    loss_fn = ScatterADLoss(time_weight=0.0, contrast_weight=0.0)
    x = torch.randn(2, 5, 3)

    loss = loss_fn((model((x,)),), (x,))
    loss.backward()

    online_grad_norm = sum(
        parameter.grad.detach().abs().sum()
        for parameter in model.online_encoder.parameters()
        if parameter.grad is not None
    )
    assert online_grad_norm > 0
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_detector_returns_last_step_window_scores_and_distance_direction() -> None:
    model = _make_model()
    detector = ScatterADAnomalyDetector(model=model, batch_size=2)
    x = torch.randn(3, 5, 3)

    scores = detector.compute_online_anomaly_score((x,))
    with torch.no_grad():
        expected = model.anomaly_scores(x)[:, -1]

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(scores, expected)

    with torch.no_grad():
        model.center.copy_(torch.tensor([0.5] + [0.0] * 15))
    near_center = torch.tensor([[[1.0] + [0.0] * 15]])
    far_from_center = torch.tensor([[[-1.0] + [0.0] * 15]])

    near_score = model.score_components_from_embeddings(near_center)["score"]
    far_score = model.score_components_from_embeddings(far_from_center)["score"]

    assert far_score.item() > near_score.item()


def test_temporal_graph_edges_are_directed_lookbacks_with_no_cross_window_edges() -> None:
    edges = build_temporal_lookback_edges(batch_size=2, window_size=5, tau=2)

    assert edges.shape == (2, 14)
    src, dst = edges
    assert torch.all(dst > src)
    assert torch.all((dst - src) <= 2)
    assert torch.equal(src // 5, dst // 5)

    adjacency = build_temporal_lookback_adjacency(window_size=5, tau=2, include_self=False)
    expected = torch.tensor(
        [
            [False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, True, True, False, False],
            [False, False, True, True, False],
        ]
    )
    torch.testing.assert_close(adjacency, expected)


def test_center_is_fixed_buffer_and_target_encoder_ema_updates() -> None:
    model = _make_model()

    parameter_names = {name for name, _parameter in model.named_parameters()}
    buffer_names = {name for name, _buffer in model.named_buffers()}
    assert "center" not in parameter_names
    assert "center" in buffer_names
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())

    online_parameter = next(model.online_encoder.parameters())
    target_parameter = next(model.target_encoder.parameters())
    before_target = target_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(2.0)

    model.update_target()

    expected = before_target * 0.5 + online_parameter.detach() * 0.5
    torch.testing.assert_close(target_parameter, expected)


def test_config_construction_and_model_utils_links() -> None:
    model_utils = _load_model_utils()
    mappings = set(model_utils.get_args("scatterad"))

    assert ("data.num_features", "model.network.init_args.input_dim", "instantiate") in mappings
    assert ("window_size", "model.network.init_args.win_size", "parse") in mappings

    config = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    params = yaml.safe_load(PARAM_CONFIG_PATH.read_text(encoding="utf-8"))
    network_config = config["model"]["network"]
    detector_config = config["model"]["detector"]
    loss_config = config["model"]["losses"]
    optimizer_config = config["model"]["optimizer"]

    assert config["model"]["predict_on_end"] is True
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["data"]["batch_size"] == 128
    assert network_config["init_args"]["tau"] == 2
    assert network_config["init_args"]["ema_momentum"] == 0.5
    assert optimizer_config["class_path"] == "noboom_benchmark.noboom_lib.core.models.scatterad.ScatterADAdam"
    assert params["search_space"]["window_size"]["choices"] == [100]

    network_cls = _resolve_class(network_config["class_path"])
    detector_cls = _resolve_class(detector_config["class_path"])
    loss_cls = _resolve_class(loss_config["class_path"])
    optimizer_cls = _resolve_class(optimizer_config["class_path"])
    init_args = dict(network_config["init_args"])
    init_args.update(input_dim=3, win_size=5, hidden_dim=16, num_layers=1, dropout=0.0)

    network = network_cls(**init_args)
    detector = detector_cls(model=network, **detector_config["init_args"])
    loss = loss_cls(**loss_config["init_args"])
    optimizer = optimizer_cls(network.grouped_parameters()[0], **optimizer_config["init_args"])

    assert isinstance(network, ScatterAD)
    assert isinstance(detector, ScatterADAnomalyDetector)
    assert isinstance(loss, ScatterADLoss)
    assert isinstance(optimizer, ScatterADAdam)


def test_tiny_synthetic_train_and_validation_step() -> None:
    model = _make_model()
    loss_fn = ScatterADLoss()
    optimizer = ScatterADAdam(model.grouped_parameters()[0], lr=1.0e-3)
    x = torch.randn(4, 5, 3)
    target_before_step = next(model.target_encoder.parameters()).detach().clone()

    model.train()
    outputs = model((x,))
    loss = loss_fn((outputs,), (x,))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    target_after_step = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(target_after_step, target_before_step)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn((model((x,)),), (x,))

    assert torch.isfinite(val_loss)
