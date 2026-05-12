from __future__ import annotations

import importlib
from pathlib import Path

import torch
import yaml

from noboom_benchmark.noboom_lib.core.benchmark_utils.model_utils import get_args
from noboom_benchmark.noboom_lib.core.models._windowed_external import SourceBackedAnomalyDetector
from noboom_benchmark.noboom_lib.core.models.rtdetector import (
    RTdetectorAnomalyDetector,
    RTdetectorLoss,
)


ROOT = Path(__file__).resolve().parents[1]
RTDETECTOR_CONFIG = (
    ROOT / "src" / "noboom_cluster" / "cluster_files" / "configs" / "models" / "rtdetector.yaml"
)


def _set_dotted(config: dict, path: str, value) -> None:
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _resolve_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def test_rtdetector_config_instantiates_native_components() -> None:
    with RTDETECTOR_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for source, target, apply_on in get_args("rtdetector"):
        if source == "window_size" and apply_on == "parse":
            _set_dotted(config, target, 4)
        elif source == "data.num_features" and apply_on == "instantiate":
            _set_dotted(config, target, 2)

    network_config = config["model"]["network"]
    detector_config = config["model"]["detector"]
    loss_config = config["model"]["losses"]
    network_cls = _resolve_class(network_config["class_path"])
    detector_cls = _resolve_class(detector_config["class_path"])
    loss_cls = _resolve_class(loss_config["class_path"])

    network = network_cls(**network_config["init_args"])
    detector = detector_cls(network)
    loss = loss_cls()

    assert network_cls.__name__ == "RTdetector"
    assert detector_cls.__name__ == "RTdetectorAnomalyDetector"
    assert loss_cls.__name__ == "RTdetectorLoss"
    assert isinstance(network, network_cls)
    assert isinstance(detector, detector_cls)
    assert isinstance(loss, loss_cls)
    assert not issubclass(type(detector), SourceBackedAnomalyDetector)

    x = torch.rand(3, 4, 2)
    outputs = network((x,))
    assert len(outputs) == 2
    assert outputs[0].shape == (3, 2)
    assert outputs[1].shape == (3, 2)


def test_rtdetector_loss_preserves_epoch_weighted_two_pass_schedule() -> None:
    loss = RTdetectorLoss()
    target = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 4.0]],
            [[1.0, 1.0], [3.0, 5.0]],
        ]
    )
    first_pass = torch.zeros(2, 2)
    second_pass = torch.ones(2, 2)

    first_epoch = loss((first_pass, second_pass), (target,), epoch=0)
    second_epoch = loss((first_pass, second_pass), (target,), epoch=1)
    validation_epoch = loss((first_pass, second_pass), (target,))

    expected_first = torch.nn.functional.mse_loss(first_pass, target[:, -1, :])
    expected_second = (
        0.5 * torch.nn.functional.mse_loss(first_pass, target[:, -1, :], reduction="none")
        + 0.5 * torch.nn.functional.mse_loss(second_pass, target[:, -1, :], reduction="none")
    ).mean()
    expected_validation = torch.nn.functional.mse_loss(second_pass, target[:, -1, :])

    assert torch.allclose(first_epoch, expected_first)
    assert torch.allclose(second_epoch, expected_second)
    assert torch.allclose(validation_epoch, expected_validation)


def test_rtdetector_detector_scores_final_step_with_larger_error_higher() -> None:
    class StubRTdetector(torch.nn.Module):
        def forward(self, inputs):
            x, = inputs
            final = x[:, -1, :]
            return final, final + torch.tensor([[1.0, 3.0], [0.5, 0.5]])

    detector = RTdetectorAnomalyDetector(StubRTdetector())
    x = torch.zeros(2, 4, 2)
    labels = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]])

    scores = detector.compute_online_anomaly_score((x,))
    formatted = detector.format_online_targets((labels, x))

    assert torch.allclose(scores, torch.tensor([5.0, 0.25]))
    assert scores[0] > scores[1]
    assert torch.equal(formatted, torch.tensor([0, 1]))
