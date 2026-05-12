"""Streaming policies for benchmark-level ``predict_on_end`` scoring.

The policies in this module preserve the legacy benchmark contract: prediction
batches are concatenated in dataloader order, and detector/model code remains
unchanged.  Each policy only changes how much input the benchmark keeps before
calling the existing detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import torch


DEFAULT_WINDOW_CHUNK_SIZE = 16384
DEFAULT_POINT_CHUNK_SIZE = 65536

WINDOW_ATOMIC_POLICY = "window_atomic"
CAUSAL_POINT_REWINDOW_POLICY = "causal_point_rewindow"
PATCH_POINT_DELAY_POLICY = "patch_point_delay"
CATCH_MINIMAL_LEGACY_TAIL_POLICY = "catch_minimal_legacy_tail"


PREDICT_ON_END_STREAM_POLICIES: Dict[str, Tuple[str, ...]] = {
    WINDOW_ATOMIC_POLICY: ("WindowAtomicPredictionStream",),
    CAUSAL_POINT_REWINDOW_POLICY: ("CausalPointRewindowPredictionStream",),
    PATCH_POINT_DELAY_POLICY: ("PatchPointDelayPredictionStream",),
    CATCH_MINIMAL_LEGACY_TAIL_POLICY: ("CatchMinimalLegacyTailPredictionStream",),
}


@dataclass(frozen=True)
class PredictionStreamConfig:
    """Class-path and constructor arguments for streamed prediction."""

    adapter: Optional[str] = None
    class_path: Optional[str] = None
    init_args: Optional[Dict[str, Any]] = None
    policy_name: Optional[str] = None
    window_size: Optional[int] = None
    patch_size: Optional[int] = None
    stride: Optional[int] = None
    window_chunk_size: int = DEFAULT_WINDOW_CHUNK_SIZE
    point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE


class PredictionStreamState:
    """Benchmark-owned accumulator around a concrete prediction stream policy."""

    def __init__(
        self,
        policy: "BasePredictionStreamPolicy",
    ) -> None:
        self.policy = policy
        self.policy_name = policy.policy_name
        self._score_chunks: List[torch.Tensor] = []
        self._closed = False

    def add(self, inputs: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("Cannot add prediction inputs after the stream has been closed.")
        scores = self.policy.update(inputs)
        self._append_scores(scores)

    def finalize(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Cannot finalize prediction stream after it has been closed.")
        self._append_scores(self.policy.finish())
        return _cat_scores(self._score_chunks)

    def close(self) -> None:
        self._score_chunks.clear()
        self.policy.close()
        self._closed = True

    def _append_scores(self, scores: torch.Tensor) -> None:
        if scores.numel() == 0:
            return
        self._score_chunks.append(scores.detach().cpu().reshape(-1))


def detector_class_path(detector: Any) -> str:
    """Return the import path used by the explicit policy registry."""

    detector_type = type(detector)
    return f"{detector_type.__module__}.{detector_type.__name__}"


def infer_prediction_stream_policy_name(detector: Any) -> str:
    """Deprecated: implicit detector-name streaming inference is disabled."""

    raise ValueError(
        "Implicit predict_on_end streaming policy inference is disabled. "
        "Configure model.prediction_stream.class_path explicitly."
    )


def create_prediction_stream_policy(
    detector: Any,
    config: Optional[PredictionStreamConfig] = None,
) -> "BasePredictionStreamPolicy":
    """Build the prediction stream selected explicitly by whitelisted class path."""

    resolved = config or PredictionStreamConfig()
    if resolved.adapter is not None:
        raise ValueError(
            "model.prediction_stream.adapter is no longer supported for "
            "predict_on_end_streaming=True. Use model.prediction_stream.class_path."
        )
    if resolved.policy_name is not None:
        raise ValueError(
            "model.prediction_stream.policy_name is no longer supported for "
            "predict_on_end_streaming=True. Use model.prediction_stream.class_path."
        )
    if resolved.class_path is None:
        raise ValueError(
            "predict_on_end_streaming=True requires model.prediction_stream.class_path."
        )
    return _validate_prediction_stream_interface(
        _create_stream_by_class_path(str(resolved.class_path), detector, resolved)
    )


def create_prediction_stream_state(
    detector: Any,
    chunk_size: Optional[int] = None,
    device_type: str = "cpu",
    config: Optional[PredictionStreamConfig] = None,
) -> PredictionStreamState:
    """Create the state object consumed by ``BenchmarkModel``.

    ``device_type`` is accepted for API compatibility with ``BenchmarkModel``;
    policy implementations intentionally keep buffered prediction inputs on CPU.
    """

    del device_type
    resolved = config or PredictionStreamConfig()
    if chunk_size is not None and resolved.init_args is None:
        resolved = PredictionStreamConfig(
            adapter=resolved.adapter,
            class_path=resolved.class_path,
            init_args={"chunk_size": int(chunk_size)},
            policy_name=resolved.policy_name,
            window_size=resolved.window_size,
            patch_size=resolved.patch_size,
            stride=resolved.stride,
            window_chunk_size=resolved.window_chunk_size,
            point_chunk_size=resolved.point_chunk_size,
        )
    return PredictionStreamState(create_prediction_stream_policy(detector, resolved))


class BasePredictionStreamPolicy:
    """Common interface consumed by benchmark prediction aggregation."""

    policy_name: str

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self._score_chunks: List[torch.Tensor] = []
        self._closed = False

    def add(self, windows: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("Cannot add prediction inputs after the stream has been closed.")
        scores = self.update(windows)
        if scores.numel() != 0:
            self._score_chunks.append(scores.detach().cpu().reshape(-1))

    def finalize(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Cannot finalize prediction stream after it has been closed.")
        scores = self.finish()
        if scores.numel() != 0:
            self._score_chunks.append(scores.detach().cpu().reshape(-1))
        return _cat_scores(self._score_chunks)

    def close(self) -> None:
        self._score_chunks.clear()
        self._clear_buffers()
        self._closed = True

    def _clear_buffers(self) -> None:
        return None

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def finish(self) -> torch.Tensor:
        raise NotImplementedError


class WindowAtomicPredictionStream(BasePredictionStreamPolicy):
    """Score 3D window batches atomically and concatenate detector scores."""

    policy_name = WINDOW_ATOMIC_POLICY

    def __init__(
        self,
        detector: Any,
        window_chunk_size: int = DEFAULT_WINDOW_CHUNK_SIZE,
    ) -> None:
        super().__init__(detector)
        self.window_chunk_size = _positive_chunk_size(window_chunk_size, "window_chunk_size")
        self._pending: List[torch.Tensor] = []
        self._pending_count = 0

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        windows = _require_tensor(inputs, "window_atomic inputs").detach().cpu()
        if windows.ndim != 3:
            raise ValueError(f"window_atomic expects [batch, window, features], got {tuple(windows.shape)}.")
        self._pending.append(windows)
        self._pending_count += int(windows.shape[0])
        if self._pending_count < self.window_chunk_size:
            return _empty_scores(windows)
        return self._score_pending(limit=self.window_chunk_size)

    def finish(self) -> torch.Tensor:
        return self._score_pending(limit=None)

    def _score_pending(self, limit: Optional[int]) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        while self._pending_count and (limit is None or self._pending_count >= limit):
            take = self._pending_count if limit is None else limit
            chunk, remainder = _take_from_tensor_list(self._pending, take)
            self._pending = remainder
            self._pending_count -= int(chunk.shape[0])
            outputs.append(_call_public_detector(self.detector, chunk))
            if limit is None:
                break
        return _cat_scores(outputs, device=_first_pending_device(self._pending))

    def _clear_buffers(self) -> None:
        self._pending.clear()
        self._pending_count = 0


class CausalPointRewindowPredictionStream(BasePredictionStreamPolicy):
    """Stream endpoint points and let causal detectors rebuild local windows."""

    policy_name = CAUSAL_POINT_REWINDOW_POLICY

    def __init__(
        self,
        detector: Any,
        window_size: int,
        point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE,
    ) -> None:
        super().__init__(detector)
        self.window_size = _positive_chunk_size(window_size, "window_size")
        self.point_chunk_size = _positive_chunk_size(point_chunk_size, "point_chunk_size")
        self._buffer = _TensorBuffer()
        self._emitted_count = 0

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        self._buffer.append(_endpoint_points(inputs))
        return self._emit_available(final=False)

    def finish(self) -> torch.Tensor:
        return self._emit_available(final=True)

    def _emit_available(self, final: bool) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        total_seen = self._buffer.end
        while self._emitted_count < total_seen:
            available = total_seen - self._emitted_count
            if not final and available < self.point_chunk_size:
                break
            emit_end = total_seen if final else self._emitted_count + self.point_chunk_size
            emit_start = self._emitted_count
            region_start = max(0, emit_start - self.window_size + 1)
            region = self._buffer.slice(region_start, emit_end)
            scores = _call_public_detector(self.detector, region)
            local_start = emit_start - region_start
            local_end = emit_end - region_start
            outputs.append(scores[local_start:local_end].detach().cpu())
            self._emitted_count = emit_end
            self._buffer.drop_before(max(0, self._emitted_count - self.window_size + 1))
        return _cat_scores(outputs, device=self._buffer.device)

    def _clear_buffers(self) -> None:
        self._buffer = _TensorBuffer()
        self._emitted_count = 0


class PatchPointDelayPredictionStream(BasePredictionStreamPolicy):
    """Stream endpoint points and emit PaAno scores only after right context is stable."""

    policy_name = PATCH_POINT_DELAY_POLICY

    def __init__(
        self,
        detector: Any,
        patch_size: int,
        stride: int = 1,
        point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE,
    ) -> None:
        super().__init__(detector)
        self.patch_size = _positive_chunk_size(patch_size, "patch_size")
        self.stride = _positive_chunk_size(stride, "stride")
        self.point_chunk_size = _positive_chunk_size(point_chunk_size, "point_chunk_size")
        self.right_delay = max(0, self.patch_size - 1)
        self._buffer = _TensorBuffer()
        self._emitted_count = 0

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        self._buffer.append(_endpoint_points(inputs))
        return self._emit_available(final=False)

    def finish(self) -> torch.Tensor:
        return self._emit_available(final=True)

    def _emit_available(self, final: bool) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        total_seen = self._buffer.end
        stable_end = total_seen if final else max(self._emitted_count, total_seen - self.right_delay)
        while self._emitted_count < stable_end:
            available = stable_end - self._emitted_count
            if not final and available < self.point_chunk_size:
                break
            emit_start = self._emitted_count
            emit_end = stable_end if final else min(stable_end, emit_start + self.point_chunk_size)
            region_start = self._aligned_patch_region_start(emit_start)
            region_end = total_seen if final else self._patch_region_end(emit_end)
            region = self._buffer.slice(region_start, region_end)
            scores = _call_public_detector(self.detector, region)
            local_start = emit_start - region_start
            local_end = emit_end - region_start
            outputs.append(scores[local_start:local_end].detach().cpu())
            self._emitted_count = emit_end
            self._buffer.drop_before(self._aligned_patch_region_start(self._emitted_count))
        return _cat_scores(outputs, device=self._buffer.device)

    def _aligned_patch_region_start(self, emit_start: int) -> int:
        earliest_patch_start = max(0, emit_start - self.patch_size + 1)
        return (earliest_patch_start // self.stride) * self.stride

    def _patch_region_end(self, emit_end: int) -> int:
        if emit_end <= 0:
            return 0
        last_point = emit_end - 1
        last_patch_start = (last_point // self.stride) * self.stride
        return min(self._buffer.end, last_patch_start + self.patch_size)

    def _clear_buffers(self) -> None:
        self._buffer = _TensorBuffer()
        self._emitted_count = 0


class CatchMinimalLegacyTailPredictionStream(BasePredictionStreamPolicy):
    """Preserve legacy CATCH tail flattening without storing all input windows."""

    policy_name = CATCH_MINIMAL_LEGACY_TAIL_POLICY

    def __init__(
        self,
        detector: Any,
        window_size: int,
        window_chunk_size: int = DEFAULT_WINDOW_CHUNK_SIZE,
    ) -> None:
        super().__init__(detector)
        self.window_size = _positive_chunk_size(window_size, "window_size")
        self.window_chunk_size = _positive_chunk_size(window_chunk_size, "window_chunk_size")
        self._points = _TensorBuffer()
        self._retained_start = 0
        self.max_retained_points = 0

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        self._points.append(_endpoint_points(inputs))
        self._drop_unneeded_prefix()
        self.max_retained_points = max(self.max_retained_points, self._points.length)
        return _empty_scores(self._points.tensor)

    def finish(self) -> torch.Tensor:
        total = self._points.end
        if total < self.window_size:
            return torch.full((total,), -torch.inf, dtype=torch.float32, device=self._points.device)
        num_windows = total - self.window_size + 1
        flat_len = num_windows * self.window_size
        tail_start = flat_len - total
        first_window = tail_start // self.window_size
        first_offset = tail_start % self.window_size
        if self._points.start > first_window:
            raise RuntimeError("CATCH stream dropped endpoint rows required for legacy tail scoring.")
        self._points.drop_before(first_window)
        retained_window_count = num_windows - first_window
        flat_scores = self._score_retained_windows(retained_window_count)
        return flat_scores[first_offset : first_offset + total].detach().cpu()

    def _drop_unneeded_prefix(self) -> None:
        total = self._points.end
        if total < self.window_size:
            self._retained_start = 0
            return
        num_windows = total - self.window_size + 1
        flat_len = num_windows * self.window_size
        tail_start = flat_len - total
        self._retained_start = tail_start // self.window_size
        self._points.drop_before(self._retained_start)

    def _score_retained_windows(self, retained_window_count: int) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for start in range(0, retained_window_count, self.window_chunk_size):
            count = min(self.window_chunk_size, retained_window_count - start)
            point_start = self._points.start + start
            point_end = point_start + count + self.window_size - 1
            points = self._points.slice(point_start, point_end)
            windows = _make_sliding_windows(points, self.window_size, count)
            outputs.append(_score_catch_windows(self.detector, windows))
        return _cat_scores(outputs, device=self._points.device)

    def _clear_buffers(self) -> None:
        self._points = _TensorBuffer()
        self._retained_start = 0
        self.max_retained_points = 0


ALLOWED_PREDICTION_STREAM_CLASS_PATHS: Dict[str, Type[BasePredictionStreamPolicy]] = {
    f"{stream_cls.__module__}.{stream_cls.__name__}": stream_cls
    for stream_cls in (
        WindowAtomicPredictionStream,
        CausalPointRewindowPredictionStream,
        PatchPointDelayPredictionStream,
        CatchMinimalLegacyTailPredictionStream,
    )
}


def _resolve_stream_class(class_path: str) -> Type[BasePredictionStreamPolicy]:
    try:
        return ALLOWED_PREDICTION_STREAM_CLASS_PATHS[class_path]
    except KeyError as exc:
        allowed = ", ".join(sorted(ALLOWED_PREDICTION_STREAM_CLASS_PATHS))
        raise ValueError(
            f"Unsupported prediction_stream.class_path {class_path!r}. "
            f"Allowed class paths: {allowed}."
        ) from exc


def _create_stream_by_class_path(
    class_path: str,
    detector: Any,
    config: PredictionStreamConfig,
) -> BasePredictionStreamPolicy:
    stream_cls = _resolve_stream_class(class_path)
    init_args = _stream_init_args(config)
    if stream_cls is WindowAtomicPredictionStream:
        _reject_unknown_init_args(init_args, {"chunk_size", "window_chunk_size"}, class_path)
        return WindowAtomicPredictionStream(
            detector,
            window_chunk_size=_init_arg(
                init_args,
                "window_chunk_size",
                _init_arg(init_args, "chunk_size", config.window_chunk_size),
            ),
        )
    if stream_cls is CausalPointRewindowPredictionStream:
        _reject_unknown_init_args(
            init_args,
            {"chunk_size", "window_size", "point_chunk_size"},
            class_path,
        )
        return CausalPointRewindowPredictionStream(
            detector,
            window_size=_resolve_positive_int(
                detector,
                _init_arg(init_args, "window_size", config.window_size),
                ("seq_len", "win_size", "window_size"),
            ),
            point_chunk_size=_init_arg(
                init_args,
                "point_chunk_size",
                _init_arg(init_args, "chunk_size", config.point_chunk_size),
            ),
        )
    if stream_cls is PatchPointDelayPredictionStream:
        _reject_unknown_init_args(
            init_args,
            {"chunk_size", "patch_size", "point_chunk_size", "stride"},
            class_path,
        )
        return PatchPointDelayPredictionStream(
            detector,
            patch_size=_resolve_positive_int(
                detector,
                _init_arg(init_args, "patch_size", config.patch_size),
                ("patch_size",),
            ),
            stride=_resolve_positive_int(
                detector,
                _init_arg(init_args, "stride", config.stride),
                ("stride",),
                default=1,
            ),
            point_chunk_size=_init_arg(
                init_args,
                "point_chunk_size",
                _init_arg(init_args, "chunk_size", config.point_chunk_size),
            ),
        )
    if stream_cls is CatchMinimalLegacyTailPredictionStream:
        _reject_unknown_init_args(
            init_args,
            {"chunk_size", "window_size", "window_chunk_size"},
            class_path,
        )
        return CatchMinimalLegacyTailPredictionStream(
            detector,
            window_size=_resolve_positive_int(
                detector,
                _init_arg(init_args, "window_size", config.window_size),
                ("seq_len", "win_size", "window_size"),
            ),
            window_chunk_size=_init_arg(
                init_args,
                "window_chunk_size",
                _init_arg(init_args, "chunk_size", config.window_chunk_size),
            ),
        )

    raise TypeError(f"Unsupported prediction stream class {stream_cls.__name__}.")


def _validate_prediction_stream_interface(
    adapter: BasePredictionStreamPolicy,
) -> BasePredictionStreamPolicy:
    missing = [
        method_name
        for method_name in ("add", "finalize", "close")
        if not callable(getattr(adapter, method_name, None))
    ]
    if missing:
        raise TypeError(
            f"Prediction stream adapter {type(adapter).__name__} lacks required methods: "
            f"{', '.join(missing)}."
        )
    return adapter


def _init_arg(init_args: Mapping[str, Any], name: str, default: Any) -> Any:
    if name in init_args:
        return init_args[name]
    return default


def _stream_init_args(config: PredictionStreamConfig) -> Dict[str, Any]:
    init_args = config.init_args
    if init_args is None:
        return {}
    if not isinstance(init_args, Mapping):
        raise TypeError("model.prediction_stream.init_args must be a mapping.")
    return dict(init_args)


def _reject_unknown_init_args(
    init_args: Mapping[str, Any],
    allowed: Iterable[str],
    class_path: str,
) -> None:
    unknown = sorted(set(init_args) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unsupported init_args for prediction_stream.class_path {class_path!r}: "
            f"{', '.join(unknown)}."
        )


class _TensorBuffer:
    def __init__(self) -> None:
        self.tensor: Optional[torch.Tensor] = None
        self.start = 0

    @property
    def length(self) -> int:
        if self.tensor is None:
            return 0
        return int(self.tensor.shape[0])

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def device(self) -> torch.device:
        if self.tensor is None:
            return torch.device("cpu")
        return self.tensor.device

    def append(self, values: torch.Tensor) -> None:
        detached = _require_tensor(values, "buffer values").detach().cpu()
        if detached.ndim != 2:
            raise ValueError(f"stream point buffers expect [time, features], got {tuple(detached.shape)}.")
        if detached.shape[0] == 0:
            return
        if self.tensor is None:
            self.tensor = detached
        else:
            self.tensor = torch.cat([self.tensor, detached], dim=0)

    def slice(self, start: int, end: int) -> torch.Tensor:
        if self.tensor is None:
            raise ValueError("Cannot slice an empty prediction stream buffer.")
        if start < self.start or end > self.end or start > end:
            raise ValueError(f"Requested stream slice [{start}, {end}) outside [{self.start}, {self.end}).")
        local_start = start - self.start
        local_end = end - self.start
        return self.tensor[local_start:local_end]

    def drop_before(self, start: int) -> None:
        if self.tensor is None or start <= self.start:
            return
        if start >= self.end:
            self.tensor = None
            self.start = start
            return
        local_start = start - self.start
        self.tensor = self.tensor[local_start:].contiguous()
        self.start = start


def _endpoint_points(inputs: torch.Tensor) -> torch.Tensor:
    values = _require_tensor(inputs, "predict_on_end inputs").detach().cpu().to(dtype=torch.float32)
    if values.ndim == 3:
        return values[:, -1, :]
    if values.ndim == 2:
        return values
    raise ValueError(f"Expected [batch, window, features] or [batch, features], got {tuple(values.shape)}.")


def _call_public_detector(detector: Any, values: torch.Tensor) -> torch.Tensor:
    scores = detector((values,)) if callable(detector) else None
    if scores is None and hasattr(detector, "compute_online_anomaly_score"):
        scores = detector.compute_online_anomaly_score((values,))
    if scores is None:
        return _empty_scores(values)
    return _as_score_tensor(scores)


def _score_catch_windows(detector: Any, windows: torch.Tensor) -> torch.Tensor:
    if not hasattr(detector, "_score_windows"):
        raise RuntimeError("CATCH streaming policy requires detector._score_windows().")
    return _as_score_tensor(detector._score_windows(windows))


def _as_score_tensor(scores: Any) -> torch.Tensor:
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().to(dtype=torch.float32).reshape(-1)
    return torch.as_tensor(scores, dtype=torch.float32).reshape(-1)


def _cat_scores(scores: Sequence[torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    if not scores:
        return torch.empty(0, dtype=torch.float32, device=device or torch.device("cpu"))
    return torch.cat([score.detach().cpu().reshape(-1) for score in scores], dim=0)


def _empty_scores(reference: Optional[torch.Tensor]) -> torch.Tensor:
    device = reference.device if isinstance(reference, torch.Tensor) else torch.device("cpu")
    return torch.empty(0, dtype=torch.float32, device=device)


def _make_sliding_windows(points: torch.Tensor, window_size: int, count: int) -> torch.Tensor:
    if count == 0:
        features = int(points.shape[-1]) if points.ndim == 2 else 0
        return points.new_empty((0, window_size, features))
    windows = points.unfold(dimension=0, size=window_size, step=1)
    windows = windows.permute(0, 2, 1).contiguous()
    return windows[:count]


def _take_from_tensor_list(values: List[torch.Tensor], count: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    taken: List[torch.Tensor] = []
    remaining: List[torch.Tensor] = []
    left = count
    consumed_prefix = True
    for tensor in values:
        if consumed_prefix and left > 0:
            if tensor.shape[0] <= left:
                taken.append(tensor)
                left -= int(tensor.shape[0])
                continue
            taken.append(tensor[:left])
            remaining.append(tensor[left:].contiguous())
            left = 0
            consumed_prefix = False
        else:
            remaining.append(tensor)
    if left != 0:
        raise ValueError("Not enough tensors buffered for requested chunk.")
    return torch.cat(taken, dim=0), remaining


def _first_pending_device(values: Iterable[torch.Tensor]) -> torch.device:
    for value in values:
        return value.device
    return torch.device("cpu")


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return value


def _positive_chunk_size(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _resolve_positive_int(
    detector: Any,
    explicit: Optional[int],
    attribute_names: Tuple[str, ...],
    default: Optional[int] = None,
) -> int:
    if explicit is not None:
        return _positive_chunk_size(explicit, "explicit stream size")
    for attribute_name in attribute_names:
        value = getattr(detector, attribute_name, None)
        if value is not None:
            return _positive_chunk_size(value, attribute_name)
    if default is not None:
        return _positive_chunk_size(default, "default stream size")
    raise ValueError(
        "Could not infer a positive stream size from detector "
        f"{detector_class_path(detector)} attributes {attribute_names}."
    )
