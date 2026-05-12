from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

from noboom_benchmark.noboom_lib.core.benchmark_utils.prediction_stream import (
    CatchMinimalLegacyTailPredictionStream,
    CausalPointRewindowPredictionStream,
    PatchPointDelayPredictionStream,
    PredictionStreamConfig,
    WindowAtomicPredictionStream,
    create_prediction_stream_policy,
)

PREDICTION_STREAM_MODULE = "noboom_benchmark.noboom_lib.core.benchmark_utils.prediction_stream"
WINDOW_ATOMIC_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.WindowAtomicPredictionStream"
CAUSAL_POINT_REWINDOW_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.CausalPointRewindowPredictionStream"
PATCH_POINT_DELAY_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.PatchPointDelayPredictionStream"
CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH = f"{PREDICTION_STREAM_MODULE}.CatchMinimalLegacyTailPredictionStream"


class WindowDetector:
    def __call__(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        windows = inputs[0]
        weights = torch.arange(1, windows.shape[1] + 1, dtype=windows.dtype).view(1, -1, 1)
        return (windows * weights).sum(dim=(1, 2))


class CausalWindowDetector:
    def __init__(self, window_size: int) -> None:
        self.seq_len = window_size

    def __call__(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        points = inputs[0]
        windows = _causal_windows(points, self.seq_len)
        weights = torch.arange(1, self.seq_len + 1, dtype=points.dtype).view(1, -1, 1)
        return (windows * weights).sum(dim=(1, 2))


class PatchPoolingDetector:
    def __init__(self, patch_size: int, stride: int = 1) -> None:
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        points = inputs[0]
        patch_scores = _patch_scores(points, self.patch_size, self.stride)
        return _distribute_patch_scores(patch_scores, self.patch_size, len(points))


class FakeCatchDetector:
    def __init__(self, window_size: int) -> None:
        self.seq_len = window_size

    def __call__(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        points = inputs[0][:, -1, :] if inputs[0].ndim == 3 else inputs[0]
        windows = _catch_windows(points, self.seq_len)
        scores = self._score_windows(windows)
        return _legacy_align(scores, len(points))

    def _score_windows(self, windows: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(1, self.seq_len + 1, dtype=windows.dtype).view(1, -1, 1)
        return (windows * weights).sum(dim=2).reshape(-1)


def test_window_atomic_chunked_scoring_equals_one_full_detector_call() -> None:
    detector = WindowDetector()
    windows = torch.arange(17 * 4 * 2, dtype=torch.float32).reshape(17, 4, 2)
    policy = WindowAtomicPredictionStream(detector, window_chunk_size=5)

    streamed = _collect_stream(policy, [windows[:3], windows[3:9], windows[9:12], windows[12:]])
    expected = detector((windows,))

    assert torch.allclose(streamed, expected)


def test_window_atomic_class_path_can_be_selected_explicitly() -> None:
    detector = WindowDetector()
    policy = create_prediction_stream_policy(
        detector,
        PredictionStreamConfig(class_path=WINDOW_ATOMIC_CLASS_PATH, init_args={"window_chunk_size": 2}),
    )

    assert isinstance(policy, WindowAtomicPredictionStream)
    assert policy.window_chunk_size == 2


def test_dada_carots_carry_logic_equals_full_public_detector_call() -> None:
    detector = CausalWindowDetector(window_size=5)
    points = torch.arange(23 * 2, dtype=torch.float32).reshape(23, 2)
    predict_windows = _endpoint_wrapped(points, window_size=5)
    policy = CausalPointRewindowPredictionStream(detector, window_size=5, point_chunk_size=7)

    streamed = _collect_stream(
        policy,
        [predict_windows[:4], predict_windows[4:10], predict_windows[10:18], predict_windows[18:]],
    )
    expected = detector((points,))

    assert torch.allclose(streamed, expected)


def test_causal_point_class_path_can_be_selected_explicitly() -> None:
    detector = CausalWindowDetector(window_size=3)
    policy = create_prediction_stream_policy(
        detector,
        PredictionStreamConfig(
            class_path=CAUSAL_POINT_REWINDOW_CLASS_PATH,
            init_args={"window_size": 3, "point_chunk_size": 2},
        ),
    )

    assert isinstance(policy, CausalPointRewindowPredictionStream)
    assert policy.point_chunk_size == 2


def test_paano_delayed_logic_equals_full_public_detector_call() -> None:
    detector = PatchPoolingDetector(patch_size=4, stride=1)
    points = torch.linspace(-2.0, 3.0, 29).view(29, 1)
    predict_windows = _endpoint_wrapped(points, window_size=6)
    policy = PatchPointDelayPredictionStream(detector, patch_size=4, stride=1, point_chunk_size=6)

    streamed = _collect_stream(
        policy,
        [predict_windows[:5], predict_windows[5:11], predict_windows[11:19], predict_windows[19:]],
    )
    expected = detector((points,))

    assert torch.allclose(streamed, expected)


def test_patch_point_class_path_can_be_selected_explicitly() -> None:
    detector = PatchPoolingDetector(patch_size=4)
    policy = create_prediction_stream_policy(
        detector,
        PredictionStreamConfig(
            class_path=PATCH_POINT_DELAY_CLASS_PATH,
            init_args={"patch_size": 4, "point_chunk_size": 2},
        ),
    )

    assert isinstance(policy, PatchPointDelayPredictionStream)
    assert policy.point_chunk_size == 2


def test_catch_minimal_legacy_tail_equals_full_detector_while_retaining_fewer_rows() -> None:
    detector = FakeCatchDetector(window_size=10)
    points = torch.arange(1009 * 2, dtype=torch.float32).reshape(1009, 2)
    predict_windows = _endpoint_wrapped(points, window_size=10)
    policy = CatchMinimalLegacyTailPredictionStream(detector, window_size=10, window_chunk_size=11)

    streamed = _collect_stream(
        policy,
        [
            predict_windows[:131],
            predict_windows[131:443],
            predict_windows[443:777],
            predict_windows[777:],
        ],
    )
    expected = detector((predict_windows,))

    assert torch.allclose(streamed, expected)
    assert policy.max_retained_points < points.shape[0] // 5


def test_catch_short_sequence_flushes_negative_infinity_scores() -> None:
    detector = FakeCatchDetector(window_size=8)
    points = torch.arange(5, dtype=torch.float32).view(5, 1)
    policy = CatchMinimalLegacyTailPredictionStream(detector, window_size=8)

    streamed = _collect_stream(policy, [_endpoint_wrapped(points, window_size=8)])

    assert streamed.shape == (5,)
    assert torch.isneginf(streamed).all()


def test_catch_class_path_can_be_selected_explicitly() -> None:
    detector = FakeCatchDetector(window_size=4)
    policy = create_prediction_stream_policy(
        detector,
        PredictionStreamConfig(
            class_path=CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH,
            init_args={"window_size": 4, "window_chunk_size": 2},
        ),
    )

    assert isinstance(policy, CatchMinimalLegacyTailPredictionStream)
    assert policy.window_chunk_size == 2


@pytest.mark.parametrize(
    ("class_path", "expected_class", "init_args"),
    [
        (WINDOW_ATOMIC_CLASS_PATH, WindowAtomicPredictionStream, {"chunk_size": 7}),
        (
            CAUSAL_POINT_REWINDOW_CLASS_PATH,
            CausalPointRewindowPredictionStream,
            {"window_size": 4, "chunk_size": 7},
        ),
        (
            PATCH_POINT_DELAY_CLASS_PATH,
            PatchPointDelayPredictionStream,
            {"patch_size": 4, "chunk_size": 7},
        ),
        (
            CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH,
            CatchMinimalLegacyTailPredictionStream,
            {"window_size": 4, "chunk_size": 7},
        ),
    ],
)
def test_each_whitelisted_class_path_instantiates_expected_adapter(
    class_path: str,
    expected_class: type[object],
    init_args: dict[str, Any],
) -> None:
    detector = CausalWindowDetector(window_size=4)

    policy = create_prediction_stream_policy(
        detector,
        PredictionStreamConfig(class_path=class_path, init_args=init_args),
    )

    assert isinstance(policy, expected_class)


@pytest.mark.parametrize(
    ("class_path", "chunk_attr", "init_args"),
    [
        (WINDOW_ATOMIC_CLASS_PATH, "window_chunk_size", {"chunk_size": 7}),
        (
            CAUSAL_POINT_REWINDOW_CLASS_PATH,
            "point_chunk_size",
            {"window_size": 4, "chunk_size": 7},
        ),
        (
            PATCH_POINT_DELAY_CLASS_PATH,
            "point_chunk_size",
            {"patch_size": 4, "chunk_size": 7},
        ),
        (
            CATCH_MINIMAL_LEGACY_TAIL_CLASS_PATH,
            "window_chunk_size",
            {"window_size": 4, "chunk_size": 7},
        ),
    ],
)
def test_init_args_chunk_size_maps_to_concrete_chunk_size_field(
    class_path: str,
    chunk_attr: str,
    init_args: dict[str, Any],
) -> None:
    detector = CausalWindowDetector(window_size=4)

    policy = create_prediction_stream_policy(
        detector,
        PredictionStreamConfig(class_path=class_path, init_args=init_args),
    )

    assert getattr(policy, chunk_attr) == 7


def _collect_stream(policy: object, batches: List[torch.Tensor]) -> torch.Tensor:
    outputs = []
    for batch in batches:
        outputs.append(policy.update(batch))
    outputs.append(policy.finish())
    return torch.cat([output for output in outputs if output.numel() > 0], dim=0)


def _endpoint_wrapped(points: torch.Tensor, window_size: int) -> torch.Tensor:
    prefix = points[:1].repeat(window_size - 1, 1)
    padded = torch.cat([prefix, points], dim=0)
    windows = []
    for index in range(points.shape[0]):
        windows.append(padded[index : index + window_size])
    return torch.stack(windows, dim=0)


def _causal_windows(points: torch.Tensor, window_size: int) -> torch.Tensor:
    windows = []
    first = points[:1]
    for end in range(points.shape[0]):
        start = max(0, end - window_size + 1)
        window = points[start : end + 1]
        pad = window_size - window.shape[0]
        if pad > 0:
            window = torch.cat([first.repeat(pad, 1), window], dim=0)
        windows.append(window)
    return torch.stack(windows, dim=0)


def _patch_scores(points: torch.Tensor, patch_size: int, stride: int) -> np.ndarray:
    values = points.detach().cpu().numpy().astype(np.float32)
    if len(values) < patch_size:
        pad = np.repeat(values[:1], patch_size - len(values), axis=0)
        values = np.concatenate([pad, values], axis=0)
    starts = np.arange(0, len(values) - patch_size + 1, stride, dtype=np.int64)
    return np.asarray([values[start : start + patch_size].sum() for start in starts], dtype=np.float32)


def _distribute_patch_scores(patch_scores: np.ndarray, patch_size: int, num_points: int) -> torch.Tensor:
    kernel = np.ones(patch_size, dtype=np.float32)
    sums = np.convolve(patch_scores, kernel, mode="full")[:num_points]
    counts = np.convolve(np.ones_like(patch_scores), kernel, mode="full")[:num_points]
    point_scores = np.divide(
        sums,
        counts,
        out=np.zeros(num_points, dtype=np.float32),
        where=counts != 0,
    )
    return torch.as_tensor(point_scores, dtype=torch.float32)


def _catch_windows(points: torch.Tensor, window_size: int) -> torch.Tensor:
    if points.shape[0] < window_size:
        return points.new_empty((0, window_size, points.shape[-1]))
    return points.unfold(dimension=0, size=window_size, step=1).permute(0, 2, 1).contiguous()


def _legacy_align(scores: torch.Tensor, length: int) -> torch.Tensor:
    values = scores.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if values.size == length:
        return torch.as_tensor(values, dtype=torch.float32)
    if values.size > length:
        return torch.as_tensor(values[-length:], dtype=torch.float32)
    padded = np.full(length, -np.inf, dtype=np.float32)
    if values.size:
        padded[-values.size :] = values
    return torch.as_tensor(padded, dtype=torch.float32)
