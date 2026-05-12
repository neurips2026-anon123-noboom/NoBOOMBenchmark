from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
import yaml

from noboom_benchmark.noboom_lib.core.benchmark_utils.prediction_stream import (
    CatchMinimalLegacyTailPredictionStream,
    CausalPointRewindowPredictionStream,
    PatchPointDelayPredictionStream,
    WindowAtomicPredictionStream,
)

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_benchmark"
    / "noboom_lib"
    / "core"
    / "benchmark_utils"
)
PACKAGE_NAME = "noboom_benchmark.noboom_lib.core.benchmark_utils"
CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "noboom_cluster"
    / "cluster_files"
    / "configs"
    / "models"
)

PREDICT_ON_END_MODELS = (
    "eif",
    "alora",
    "scatterad",
    "anomaly_transformer",
    "dcdetector",
    "oraclead",
    "dada",
    "carots",
    "paano",
    "catch",
)
PREDICTION_STREAM_MODULE = "noboom_benchmark.noboom_lib.core.benchmark_utils.prediction_stream"
WINDOW_ATOMIC_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.WindowAtomicPredictionStream"
CAUSAL_POINT_REWINDOW_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.CausalPointRewindowPredictionStream"
PATCH_POINT_DELAY_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.PatchPointDelayPredictionStream"
CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.CatchMinimalLegacyTailPredictionStream"
EXPECTED_CLASS_PATH_BY_MODEL = {
    "eif": WINDOW_ATOMIC_CLASS_PATH,
    "alora": WINDOW_ATOMIC_CLASS_PATH,
    "scatterad": WINDOW_ATOMIC_CLASS_PATH,
    "anomaly_transformer": WINDOW_ATOMIC_CLASS_PATH,
    "dcdetector": WINDOW_ATOMIC_CLASS_PATH,
    "oraclead": WINDOW_ATOMIC_CLASS_PATH,
    "dada": CAUSAL_POINT_REWINDOW_CLASS_PATH,
    "carots": CAUSAL_POINT_REWINDOW_CLASS_PATH,
    "paano": PATCH_POINT_DELAY_CLASS_PATH,
    "catch": CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH,
}
EXPECTED_ADAPTER_CLASS_BY_CLASS_PATH = {
    WINDOW_ATOMIC_CLASS_PATH: WindowAtomicPredictionStream,
    CAUSAL_POINT_REWINDOW_CLASS_PATH: CausalPointRewindowPredictionStream,
    PATCH_POINT_DELAY_CLASS_PATH: PatchPointDelayPredictionStream,
    CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH: CatchMinimalLegacyTailPredictionStream,
}
EXPECTED_CHUNK_ATTR_BY_CLASS_PATH = {
    WINDOW_ATOMIC_CLASS_PATH: "window_chunk_size",
    CAUSAL_POINT_REWINDOW_CLASS_PATH: "point_chunk_size",
    PATCH_POINT_DELAY_CLASS_PATH: "point_chunk_size",
    CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH: "window_chunk_size",
}


def _install_benchmark_model_stubs() -> None:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules[PACKAGE_NAME] = package
    sys.modules.pop(f"{PACKAGE_NAME}.benchmark_model", None)

    style_transfer_module = types.ModuleType(f"{PACKAGE_NAME}.style_transfer")

    class _StyleTransfer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def setup(self) -> None:
            return None

        def state_dict(self, prefix: str = "") -> dict[str, Any]:
            del prefix
            return {}

    style_transfer_module.StyleTransfer = _StyleTransfer
    sys.modules[style_transfer_module.__name__] = style_transfer_module

    training_ingredient_module = types.ModuleType("timesead_experiments.utils.training_ingredient")
    training_ingredient_module.instantiate_loss = lambda loss: loss
    sys.modules[training_ingredient_module.__name__] = training_ingredient_module
    sys.modules.setdefault("timesead_experiments", types.ModuleType("timesead_experiments"))
    sys.modules.setdefault("timesead_experiments.utils", types.ModuleType("timesead_experiments.utils"))

    timesead_models_module = types.ModuleType("timesead.models")

    class _BaseModel(torch.nn.Module):
        def grouped_parameters(self):
            return (self.parameters(),)

    timesead_models_module.BaseModel = _BaseModel
    sys.modules["timesead.models"] = timesead_models_module

    timesead_common_module = types.ModuleType("timesead.models.common")

    class _AnomalyDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, inputs):
            return self.compute_online_anomaly_score(inputs)

    timesead_common_module.AnomalyDetector = _AnomalyDetector
    sys.modules["timesead.models.common"] = timesead_common_module

    timesead_loss_module = types.ModuleType("timesead.optim.loss")

    class _Loss(torch.nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            super().__init__()

    timesead_loss_module.Loss = _Loss
    sys.modules["timesead.optim.loss"] = timesead_loss_module
    sys.modules.setdefault("timesead.optim", types.ModuleType("timesead.optim"))

    timesead_utils_module = types.ModuleType("timesead.utils.utils")

    def _pack_tuple(value):
        return value if isinstance(value, tuple) else (value,)

    timesead_utils_module.pack_tuple = _pack_tuple
    sys.modules["timesead.utils.utils"] = timesead_utils_module
    sys.modules.setdefault("timesead.utils", types.ModuleType("timesead.utils"))


_install_benchmark_model_stubs()
benchmark_model_module = importlib.import_module(f"{PACKAGE_NAME}.benchmark_model")
BenchmarkModel = benchmark_model_module.BenchmarkModel
for module_name in [
    PACKAGE_NAME,
    f"{PACKAGE_NAME}.benchmark_model",
    f"{PACKAGE_NAME}.style_transfer",
    "timesead_experiments.utils.training_ingredient",
    "timesead.models",
    "timesead.models.common",
    "timesead.optim.loss",
    "timesead.utils.utils",
]:
    sys.modules.pop(module_name, None)


class _LastValueDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scored_shapes: list[tuple[int, ...]] = []

    def forward(self, inputs):
        return self.compute_online_anomaly_score(inputs)

    def compute_online_anomaly_score(self, inputs):
        windows = inputs[0]
        self.scored_shapes.append(tuple(windows.shape))
        return windows[:, -1, 0].detach().to(dtype=torch.float32, device=windows.device)


def _load_config(model_name: str) -> dict[str, Any]:
    with (CONFIG_ROOT / f"{model_name}.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _prediction_stream_config(
    class_path: str = WINDOW_ATOMIC_CLASS_PATH,
    *,
    chunk_size: int = 1024,
    enabled: Optional[bool] = None,
) -> dict[str, Any]:
    init_args: dict[str, Any] = {"chunk_size": chunk_size}
    if class_path in (CAUSAL_POINT_REWINDOW_CLASS_PATH, CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH):
        init_args["window_size"] = 3
    if class_path == PATCH_POINT_DELAY_CLASS_PATH:
        init_args["patch_size"] = 3
        init_args["stride"] = 1
    config = {
        "class_path": class_path,
        "init_args": init_args,
    }
    if enabled is not None:
        config["enabled"] = enabled
    return config


def _model(
    *,
    detector: _LastValueDetector,
    streaming: bool,
    model_name: str = "anomaly_transformer",
    prediction_horizon: Optional[int] = None,
    label_index_offset: int = 0,
    class_path: str = WINDOW_ATOMIC_CLASS_PATH,
    prediction_stream: Optional[dict[str, Any]] = None,
) -> BenchmarkModel:
    kwargs: dict[str, Any] = {
        "detector": detector,
        "batch_dim": 0,
        "window_size": 3,
        "prediction_horizon": prediction_horizon,
        "label_index_offset": label_index_offset,
        "metrics": [],
        "model_name": model_name,
        "predict_on_end": True,
    }
    if streaming:
        kwargs["prediction_stream"] = prediction_stream or _prediction_stream_config(class_path)
    return BenchmarkModel(**kwargs)


def _sequence_windows(sequence: torch.Tensor, window_size: int) -> torch.Tensor:
    return torch.stack(
        [sequence[start : start + window_size] for start in range(sequence.shape[0] - window_size + 1)]
    )


def _window_batches(*, batch_sizes: tuple[int, ...] = (2, 3, 2)) -> list[tuple[tuple[torch.Tensor], tuple[torch.Tensor]]]:
    first = torch.arange(6, dtype=torch.float32).view(6, 1)
    second = torch.arange(20, 25, dtype=torch.float32).view(5, 1)
    windows = torch.cat([_sequence_windows(first, 3), _sequence_windows(second, 3)], dim=0)
    labels = torch.zeros(windows.shape[0], 3, 1, dtype=torch.float32)
    batches = []
    start = 0
    for size in batch_sizes:
        stop = start + size
        batches.append(((windows[start:stop],), (labels[start:stop],)))
        start = stop
    assert start == windows.shape[0]
    return batches


def _offset_window_batches() -> list[tuple[tuple[torch.Tensor], tuple[torch.Tensor]]]:
    sequence = torch.arange(40, 48, dtype=torch.float32).view(8, 1)
    windows = _sequence_windows(sequence, 3)
    labels = torch.zeros(2, 3, 1, dtype=torch.float32)
    return [((windows[:1],), (labels[:1],)), ((windows[1:2],), (labels[1:2],))]


def _trainer(predictions: list[torch.Tensor], labels: list[torch.Tensor], seq_len: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        world_size=1,
        checkpoint_callback=None,
        predict_loop=SimpleNamespace(predictions=predictions),
        datamodule=SimpleNamespace(
            dataset_name="streaming_unit",
            seq_len=lambda split: seq_len,
            predict_orig_dataloader=lambda: [
                (None, (label.view(1, -1),)) for label in labels
            ],
        ),
    )


def _run_predict_epoch(
    model: BenchmarkModel,
    batches: list[tuple[tuple[torch.Tensor], tuple[torch.Tensor]]],
    *,
    labels: list[torch.Tensor],
    seq_len: list[int],
) -> list[torch.Tensor]:
    trainer = _trainer([], labels, seq_len)
    model._trainer = trainer
    model._detector_fitted = True
    model.on_predict_start()
    predictions = [model.predict_step(batch, idx).detach().cpu() for idx, batch in enumerate(batches)]
    trainer.predict_loop.predictions = predictions
    model.on_predict_epoch_end()
    return predictions


def _start_streaming_model(model: BenchmarkModel) -> None:
    model._trainer = _trainer([], [torch.zeros(3, dtype=torch.float32)], [3])
    model._detector_fitted = True
    model.on_predict_start()


def _disable_thresholding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark_model_module,
        "select_threshold",
        lambda *_args, **_kwargs: SimpleNamespace(best_threshold=float("inf")),
    )


def _prediction_stream_policy(model: BenchmarkModel) -> object:
    policy = getattr(model, "_prediction_stream_policy", None)
    if policy is not None:
        return policy
    stream_state = getattr(model, "_prediction_stream_state", None)
    policy = getattr(stream_state, "policy", None)
    if policy is not None:
        return policy
    raise AssertionError("streaming BenchmarkModel did not expose a prediction stream policy")


@pytest.mark.parametrize("model_name", PREDICT_ON_END_MODELS)
def test_existing_predict_on_end_configs_still_instantiate_without_streaming(model_name: str) -> None:
    config = _load_config(model_name)

    model = BenchmarkModel(
        detector=_LastValueDetector(),
        batch_dim=0,
        window_size=3,
        metrics=[],
        model_name=model_name,
        predict_on_end=config["model"]["predict_on_end"],
    )

    assert model.predict_on_end is True


def test_prediction_stream_config_parses_and_enables_streaming_policy() -> None:
    model = _model(detector=_LastValueDetector(), streaming=True)
    _start_streaming_model(model)

    assert "enabled" in str(getattr(model, "prediction_stream", getattr(model, "_prediction_stream", ""))).lower()
    assert model.predict_on_end_streaming is True
    assert model.predict_on_end_stream_class_path == WINDOW_ATOMIC_CLASS_PATH
    assert isinstance(_prediction_stream_policy(model), WindowAtomicPredictionStream)


def test_prediction_streaming_without_class_path_fails_closed() -> None:
    model = BenchmarkModel(
        detector=_LastValueDetector(),
        batch_dim=0,
        window_size=3,
        metrics=[],
        model_name="anomaly_transformer",
        predict_on_end=True,
        prediction_stream={"enabled": True, "chunk_size": 1024},
    )

    with pytest.raises((RuntimeError, ValueError), match="adapter|class_path"):
        _start_streaming_model(model)


def test_prediction_stream_adapter_fails_closed() -> None:
    model = _model(
        detector=_LastValueDetector(),
        streaming=True,
        prediction_stream={
            "enabled": True,
            "adapter": "window_atomic",
            "init_args": {"chunk_size": 1024},
        },
    )

    with pytest.raises((RuntimeError, ValueError), match="adapter"):
        _start_streaming_model(model)


def test_unknown_prediction_stream_class_path_fails_closed() -> None:
    model = _model(
        detector=_LastValueDetector(),
        streaming=True,
        prediction_stream={
            "enabled": True,
            "class_path": "collections.Counter",
            "init_args": {"chunk_size": 1024},
        },
    )

    with pytest.raises((RuntimeError, ValueError), match="collections.Counter|class_path|Unsupported"):
        _start_streaming_model(model)


@pytest.mark.parametrize(
    ("class_path", "expected_class"),
    sorted(EXPECTED_ADAPTER_CLASS_BY_CLASS_PATH.items()),
)
def test_prediction_stream_class_path_instantiates_expected_class(
    class_path: str,
    expected_class: type[object],
) -> None:
    model = _model(
        detector=_LastValueDetector(),
        streaming=True,
        prediction_stream=_prediction_stream_config(class_path),
    )

    _start_streaming_model(model)

    assert isinstance(_prediction_stream_policy(model), expected_class)


@pytest.mark.parametrize(
    ("class_path", "chunk_attr"),
    sorted(EXPECTED_CHUNK_ATTR_BY_CLASS_PATH.items()),
)
def test_prediction_stream_init_args_chunk_size_maps_to_concrete_field(
    class_path: str,
    chunk_attr: str,
) -> None:
    model = _model(
        detector=_LastValueDetector(),
        streaming=True,
        prediction_stream=_prediction_stream_config(class_path, chunk_size=7),
    )

    _start_streaming_model(model)

    assert getattr(_prediction_stream_policy(model), chunk_attr) == 7


def test_prediction_stream_unknown_class_path_is_not_imported() -> None:
    model = _model(
        detector=_LastValueDetector(),
        streaming=True,
        prediction_stream={
            "enabled": True,
            "class_path": "tests.test_benchmark_prediction_streaming._LastValueDetector",
            "init_args": {"chunk_size": 1024},
        },
    )

    with pytest.raises((RuntimeError, ValueError), match="class_path|Unsupported"):
        _start_streaming_model(model)


@pytest.mark.parametrize("model_name", PREDICT_ON_END_MODELS)
def test_predict_on_end_configs_enable_expected_streaming_class_path(model_name: str) -> None:
    config = _load_config(model_name)
    expected_class_path = EXPECTED_CLASS_PATH_BY_MODEL[model_name]
    prediction_stream = config["model"].get("prediction_stream") or {}

    assert config["model"]["predict_on_end"] is True
    assert "predict_on_end_streaming" not in config["model"]
    assert prediction_stream.get("class_path") == expected_class_path


def test_all_streamed_predict_on_end_yaml_configs_declare_prediction_stream_class_path() -> None:
    missing = []
    for config_path in sorted(CONFIG_ROOT.glob("*.yaml")):
        config = _load_config(config_path.stem)
        model_config = config.get("model") or {}
        if not model_config.get("predict_on_end") or not model_config.get("prediction_stream"):
            continue
        prediction_stream = model_config.get("prediction_stream") or {}
        if not prediction_stream.get("class_path"):
            missing.append(config_path.name)

    assert missing == []


def test_no_streamed_yaml_config_uses_adapter_or_policy_name() -> None:
    offenders = []
    for config_path in sorted(CONFIG_ROOT.glob("*.yaml")):
        config = _load_config(config_path.stem)
        model_config = config.get("model") or {}
        if not model_config.get("predict_on_end") or not model_config.get("prediction_stream"):
            continue
        prediction_stream = model_config.get("prediction_stream") or {}
        forbidden = sorted(set(prediction_stream) & {"adapter", "policy_name"})
        if forbidden:
            offenders.append(f"{config_path.name}: {', '.join(forbidden)}")

    assert offenders == []


def test_predict_on_end_streaming_false_allows_missing_stream_config() -> None:
    detector = _LastValueDetector()
    model = BenchmarkModel(
        detector=detector,
        batch_dim=0,
        window_size=3,
        metrics=[],
        model_name="legacy_predict_on_end",
        predict_on_end=True,
        predict_on_end_streaming=False,
    )
    batch = _window_batches(batch_sizes=(2, 3, 2))[0]

    _start_streaming_model(model)
    prediction = model.predict_step(batch, 0).detach().cpu()

    torch.testing.assert_close(prediction, batch[0][0])
    assert detector.scored_shapes == []


def test_predict_on_end_streaming_false_disables_configured_stream() -> None:
    detector = _LastValueDetector()
    model = BenchmarkModel(
        detector=detector,
        batch_dim=0,
        window_size=3,
        metrics=[],
        model_name="legacy_predict_on_end",
        predict_on_end=True,
        predict_on_end_streaming=False,
        prediction_stream=_prediction_stream_config(WINDOW_ATOMIC_CLASS_PATH),
    )
    batch = _window_batches(batch_sizes=(2, 3, 2))[0]

    _start_streaming_model(model)
    prediction = model.predict_step(batch, 0).detach().cpu()

    assert model.predict_on_end_streaming is False
    torch.testing.assert_close(prediction, batch[0][0])
    assert detector.scored_shapes == []


def test_predict_step_streams_scores_and_returns_tiny_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_thresholding(monkeypatch)
    detector = _LastValueDetector()
    model = _model(detector=detector, streaming=True)
    batches = _window_batches()
    _start_streaming_model(model)

    predictions = [model.predict_step(batch, idx).detach().cpu() for idx, batch in enumerate(batches)]

    assert sum(prediction.numel() for prediction in predictions) <= len(predictions)
    assert sum(prediction.numel() for prediction in predictions) < sum(batch[0][0].numel() for batch in batches)
    assert detector.scored_shapes == []


def test_streaming_epoch_end_matches_legacy_alignment_with_exact_separators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_thresholding(monkeypatch)
    labels = [
        torch.tensor([0, 1, 0, 0, 2, 0], dtype=torch.float32),
        torch.tensor([0, 0, 3, 0, 0], dtype=torch.float32),
    ]
    seq_len = [6, 5]
    batches = _window_batches()

    legacy = _model(detector=_LastValueDetector(), streaming=False)
    _run_predict_epoch(legacy, batches, labels=labels, seq_len=seq_len)
    streaming = _model(detector=_LastValueDetector(), streaming=True)
    streaming_predictions = _run_predict_epoch(streaming, batches, labels=labels, seq_len=seq_len)

    expected_scores = np.array(
        [-np.inf, -np.inf, -np.inf, 2.0, 3.0, 4.0, 5.0, -np.inf, 22.0, -np.inf, -np.inf, 23.0, 24.0],
        dtype=np.float32,
    )
    expected_labels = np.array([0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0], dtype=np.float32)

    assert sum(prediction.numel() for prediction in streaming_predictions) <= len(streaming_predictions)
    np.testing.assert_allclose(streaming.anomaly_scores, legacy.anomaly_scores)
    np.testing.assert_allclose(streaming.anomaly_labels, legacy.anomaly_labels)
    np.testing.assert_allclose(streaming.anomaly_scores, expected_scores)
    np.testing.assert_allclose(streaming.anomaly_labels, expected_labels)


def test_streaming_epoch_end_preserves_prediction_horizon_and_label_offset_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_thresholding(monkeypatch)
    labels = [torch.tensor([0, 1, 0, 2, 0, 3, 0, 4], dtype=torch.float32)]
    seq_len = [8]
    batches = _offset_window_batches()

    legacy = _model(
        detector=_LastValueDetector(),
        streaming=False,
        prediction_horizon=2,
        label_index_offset=1,
    )
    _run_predict_epoch(legacy, batches, labels=labels, seq_len=seq_len)
    streaming = _model(
        detector=_LastValueDetector(),
        streaming=True,
        prediction_horizon=2,
        label_index_offset=1,
    )
    _run_predict_epoch(streaming, batches, labels=labels, seq_len=seq_len)

    expected_scores = np.array(
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 42.0, 43.0, -np.inf],
        dtype=np.float32,
    )
    expected_labels = np.array([0, 0, 1, 0, 2, 0, 3, 0, 4], dtype=np.float32)

    np.testing.assert_allclose(streaming.anomaly_scores, legacy.anomaly_scores)
    np.testing.assert_allclose(streaming.anomaly_labels, legacy.anomaly_labels)
    np.testing.assert_allclose(streaming.anomaly_scores, expected_scores)
    np.testing.assert_allclose(streaming.anomaly_labels, expected_labels)


@pytest.mark.parametrize("model_name", PREDICT_ON_END_MODELS)
def test_model_specific_streaming_parity_matches_legacy_predict_on_end(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_thresholding(monkeypatch)
    labels = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32),
        torch.tensor([0, 2, 0, 0, 0], dtype=torch.float32),
    ]
    seq_len = [6, 5]
    batches = _window_batches(batch_sizes=(3, 1, 3))

    legacy = _model(detector=_LastValueDetector(), streaming=False, model_name=model_name)
    _run_predict_epoch(legacy, batches, labels=labels, seq_len=seq_len)
    streaming = _model(detector=_LastValueDetector(), streaming=True, model_name=model_name)
    _run_predict_epoch(streaming, batches, labels=labels, seq_len=seq_len)

    np.testing.assert_allclose(streaming.anomaly_scores, legacy.anomaly_scores)
    np.testing.assert_allclose(streaming.anomaly_labels, legacy.anomaly_labels)
