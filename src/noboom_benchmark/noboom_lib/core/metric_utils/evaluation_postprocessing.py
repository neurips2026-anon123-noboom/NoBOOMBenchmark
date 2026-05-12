from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .alarm_threshold_search import DATASET_FRACTIONS, select_threshold
from .metrics import get_metric_by_name, is_metric_binary


_FLOAT_SENTINEL = -np.inf
_LABEL_SENTINEL = 0


@dataclass(frozen=True)
class EvaluationPostprocessingConfig:
    enabled: bool = False
    short_window: int = 5
    long_ewma_span: int = 25
    hysteresis_delta_mad: float = 0.5
    enter_consecutive: int = 2
    exit_consecutive: int = 3
    merge_gap: int = 3
    min_event_length: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class ProcessedEvaluationResult:
    raw_scores: np.ndarray
    processed_scores: np.ndarray
    raw_labels: np.ndarray
    binary_labels: np.ndarray
    predictions: np.ndarray
    metric_values: Dict[str, float]
    threshold: float
    threshold_on: float
    threshold_off: float
    robust_scale: float
    sequence_lengths: Tuple[int, ...]


def resolve_evaluation_postprocessing_config(
    value: Optional[Mapping[str, Any]],
) -> EvaluationPostprocessingConfig:
    if isinstance(value, EvaluationPostprocessingConfig):
        return value

    payload = dict(value or {})
    config = EvaluationPostprocessingConfig(
        enabled=bool(payload.get("enabled", False)),
        short_window=int(payload.get("short_window", 5)),
        long_ewma_span=int(payload.get("long_ewma_span", 25)),
        hysteresis_delta_mad=float(payload.get("hysteresis_delta_mad", 0.5)),
        enter_consecutive=int(payload.get("enter_consecutive", 2)),
        exit_consecutive=int(payload.get("exit_consecutive", 3)),
        merge_gap=int(payload.get("merge_gap", 3)),
        min_event_length=int(payload.get("min_event_length", 2)),
    )
    _validate_config(config)
    return config


def _validate_config(config: EvaluationPostprocessingConfig) -> None:
    if config.short_window < 1:
        raise ValueError("evaluation_postprocessing.short_window must be >= 1")
    if config.long_ewma_span < 1:
        raise ValueError("evaluation_postprocessing.long_ewma_span must be >= 1")
    if config.enter_consecutive < 1:
        raise ValueError("evaluation_postprocessing.enter_consecutive must be >= 1")
    if config.exit_consecutive < 1:
        raise ValueError("evaluation_postprocessing.exit_consecutive must be >= 1")
    if config.merge_gap < 0:
        raise ValueError("evaluation_postprocessing.merge_gap must be >= 0")
    if config.min_event_length < 1:
        raise ValueError("evaluation_postprocessing.min_event_length must be >= 1")
    if config.hysteresis_delta_mad < 0:
        raise ValueError("evaluation_postprocessing.hysteresis_delta_mad must be >= 0")


def causal_rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float64).ravel()
    if sequence.size == 0:
        return sequence.copy()

    result = np.empty(sequence.shape[0], dtype=np.float64)
    candidates: deque[int] = deque()
    for index, value in enumerate(sequence):
        while candidates and sequence[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(index)

        window_start = index - window + 1
        while candidates[0] < window_start:
            candidates.popleft()
        result[index] = sequence[candidates[0]]
    return result


def causal_ewma(values: np.ndarray, span: int) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float64).ravel()
    if sequence.size == 0:
        return sequence.copy()

    result = np.empty(sequence.shape[0], dtype=np.float64)
    alpha = 2.0 / (float(span) + 1.0)
    result[0] = sequence[0]
    for index in range(1, sequence.shape[0]):
        result[index] = alpha * sequence[index] + (1.0 - alpha) * result[index - 1]
    return result


def compute_decision_scores(
    sequence_scores: np.ndarray,
    config: EvaluationPostprocessingConfig,
) -> np.ndarray:
    scores = np.asarray(sequence_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return scores.copy()

    decision_scores = np.full(scores.shape[0], _FLOAT_SENTINEL, dtype=np.float64)
    valid_mask = np.isfinite(scores)
    for start, end in _true_segments(valid_mask):
        segment = scores[start:end]
        decision_scores[start:end] = np.maximum(
            causal_rolling_max(segment, config.short_window),
            causal_ewma(segment, config.long_ewma_span),
        )
    return decision_scores


def build_hysteresis_predictions(
    decision_scores: np.ndarray,
    threshold_on: float,
    threshold_off: float,
    config: EvaluationPostprocessingConfig,
) -> np.ndarray:
    scores = np.asarray(decision_scores, dtype=np.float64).ravel()
    predictions = np.zeros(scores.shape[0], dtype=np.int32)
    valid_mask = np.isfinite(scores)
    for start, end in _true_segments(valid_mask):
        predictions[start:end] = _build_hysteresis_predictions_for_segment(
            scores[start:end],
            threshold_on=threshold_on,
            threshold_off=threshold_off,
            config=config,
        )
    return predictions


def aggregate_prediction_events(
    predictions: np.ndarray,
    decision_scores: np.ndarray,
    config: EvaluationPostprocessingConfig,
) -> np.ndarray:
    aggregated = np.asarray(predictions, dtype=np.int32).ravel().copy()
    valid_mask = np.isfinite(np.asarray(decision_scores, dtype=np.float64).ravel())
    aggregated[~valid_mask] = 0
    for start, end in _true_segments(valid_mask):
        segment = aggregated[start:end]
        segment = _fill_small_gaps(segment, config.merge_gap)
        segment = _drop_short_events(segment, config.min_event_length)
        aggregated[start:end] = segment
    return aggregated


def evaluate_sequences_with_optional_postprocessing(
    score_sequences: Sequence[np.ndarray],
    label_sequences: Sequence[np.ndarray],
    *,
    metric_names: Sequence[str],
    config: EvaluationPostprocessingConfig,
    fix_threshold: bool = False,
    dataset_name: Optional[str] = None,
) -> ProcessedEvaluationResult:
    normalized_scores, normalized_labels = _normalize_sequence_inputs(score_sequences, label_sequences)
    sequence_lengths = tuple(sequence.shape[0] for sequence in normalized_scores)

    raw_scores = _flatten_float_sequences(normalized_scores)
    raw_labels = _flatten_label_sequences(normalized_labels)
    binary_labels = (raw_labels != 0).astype(np.int32, copy=False)

    if not config.enabled:
        if fix_threshold:
            threshold = _fixed_threshold(raw_scores, dataset_name)
        else:
            threshold = select_threshold(raw_scores, binary_labels).best_threshold
        predictions = (raw_scores > threshold).astype(np.int32)
        metric_values = _evaluate_metrics(metric_names, predictions, binary_labels, raw_labels)
        return ProcessedEvaluationResult(
            raw_scores=raw_scores,
            processed_scores=raw_scores.copy(),
            raw_labels=raw_labels,
            binary_labels=binary_labels,
            predictions=predictions,
            metric_values=metric_values,
            threshold=float(threshold),
            threshold_on=float(threshold),
            threshold_off=float(threshold),
            robust_scale=0.0,
            sequence_lengths=sequence_lengths,
        )

    decision_score_sequences = [
        compute_decision_scores(sequence_scores, config)
        for sequence_scores in normalized_scores
    ]
    processed_scores = _flatten_float_sequences(decision_score_sequences)
    if fix_threshold:
        threshold = _fixed_threshold(processed_scores, dataset_name)
    else:
        threshold = select_threshold(processed_scores, binary_labels).best_threshold

    robust_scale = _robust_scale(processed_scores)
    threshold_off = float(threshold) - config.hysteresis_delta_mad * robust_scale
    prediction_sequences = []
    for decision_scores in decision_score_sequences:
        predictions = build_hysteresis_predictions(
            decision_scores,
            threshold_on=float(threshold),
            threshold_off=threshold_off,
            config=config,
        )
        prediction_sequences.append(aggregate_prediction_events(predictions, decision_scores, config))

    flattened_predictions = _flatten_int_sequences(prediction_sequences)
    metric_values = _evaluate_metrics(metric_names, flattened_predictions, binary_labels, raw_labels)
    return ProcessedEvaluationResult(
        raw_scores=raw_scores,
        processed_scores=processed_scores,
        raw_labels=raw_labels,
        binary_labels=binary_labels,
        predictions=flattened_predictions,
        metric_values=metric_values,
        threshold=float(threshold),
        threshold_on=float(threshold),
        threshold_off=float(threshold_off),
        robust_scale=float(robust_scale),
        sequence_lengths=sequence_lengths,
    )


def _normalize_sequence_inputs(
    score_sequences: Sequence[np.ndarray],
    label_sequences: Sequence[np.ndarray],
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    if len(score_sequences) != len(label_sequences):
        raise ValueError("score_sequences and label_sequences must have the same number of sequences")

    normalized_scores = []
    normalized_labels = []
    for score_sequence, label_sequence in zip(score_sequences, label_sequences):
        score_array = np.asarray(score_sequence, dtype=np.float64).ravel()
        label_array = np.asarray(label_sequence).ravel()
        if score_array.shape[0] != label_array.shape[0]:
            raise ValueError("score and label sequence lengths must match")
        normalized_scores.append(score_array)
        normalized_labels.append(label_array)
    return tuple(normalized_scores), tuple(normalized_labels)


def _flatten_float_sequences(sequences: Sequence[np.ndarray]) -> np.ndarray:
    if not sequences:
        return np.empty(0, dtype=np.float64)
    flattened_parts = []
    for sequence in sequences:
        flattened_parts.append(np.array([_FLOAT_SENTINEL], dtype=np.float64))
        flattened_parts.append(np.asarray(sequence, dtype=np.float64).ravel())
    return np.concatenate(flattened_parts)


def _flatten_label_sequences(sequences: Sequence[np.ndarray]) -> np.ndarray:
    if not sequences:
        return np.empty(0, dtype=np.float64)
    flattened_parts = []
    for sequence in sequences:
        sequence_array = np.asarray(sequence).ravel()
        flattened_parts.append(np.array([_LABEL_SENTINEL], dtype=sequence_array.dtype))
        flattened_parts.append(sequence_array)
    return np.concatenate(flattened_parts)


def _flatten_int_sequences(sequences: Sequence[np.ndarray]) -> np.ndarray:
    if not sequences:
        return np.empty(0, dtype=np.int32)
    flattened_parts = []
    for sequence in sequences:
        flattened_parts.append(np.array([_LABEL_SENTINEL], dtype=np.int32))
        flattened_parts.append(np.asarray(sequence, dtype=np.int32).ravel())
    return np.concatenate(flattened_parts)


def _fixed_threshold(scores: np.ndarray, dataset_name: Optional[str]) -> float:
    if dataset_name is None:
        raise ValueError("dataset_name is required when fix_threshold=True")
    return float(np.quantile(scores, 1 - DATASET_FRACTIONS[dataset_name]))


def _robust_scale(scores: np.ndarray) -> float:
    finite_scores = np.asarray(scores, dtype=np.float64)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        return 1e-12
    median = np.median(finite_scores)
    mad = np.median(np.abs(finite_scores - median))
    return float(max(1e-12, 1.4826 * mad))


def _evaluate_metrics(
    metric_names: Sequence[str],
    predictions: np.ndarray,
    binary_labels: np.ndarray,
    raw_labels: np.ndarray,
) -> Dict[str, float]:
    metric_values: Dict[str, float] = {}
    for metric_name in metric_names:
        metric_fn = get_metric_by_name(metric_name)
        metric_targets = binary_labels if is_metric_binary(metric_name) else raw_labels
        metric_values[metric_name] = float(metric_fn(predictions, metric_targets))
    return metric_values


def _build_hysteresis_predictions_for_segment(
    decision_scores: np.ndarray,
    *,
    threshold_on: float,
    threshold_off: float,
    config: EvaluationPostprocessingConfig,
) -> np.ndarray:
    segment_scores = np.asarray(decision_scores, dtype=np.float64).ravel()
    predictions = np.zeros(segment_scores.shape[0], dtype=np.int32)
    active = False
    above_count = 0
    below_count = 0
    for index, score in enumerate(segment_scores):
        if not active:
            if score >= threshold_on:
                above_count += 1
            else:
                above_count = 0

            if above_count >= config.enter_consecutive:
                active = True
                start = index - config.enter_consecutive + 1
                predictions[start:index + 1] = 1
                below_count = 0
            continue

        predictions[index] = 1
        if score < threshold_off:
            below_count += 1
        else:
            below_count = 0

        if below_count >= config.exit_consecutive:
            active = False
            above_count = 0
            below_count = 0

    return predictions


def _fill_small_gaps(predictions: np.ndarray, merge_gap: int) -> np.ndarray:
    if merge_gap <= 0:
        return np.asarray(predictions, dtype=np.int32).ravel().copy()

    merged = np.asarray(predictions, dtype=np.int32).ravel().copy()
    zero_mask = merged == 0
    for start, end in _true_segments(zero_mask):
        if start == 0 or end == merged.shape[0]:
            continue
        if merged[start - 1] == 1 and merged[end] == 1 and (end - start) <= merge_gap:
            merged[start:end] = 1
    return merged


def _drop_short_events(predictions: np.ndarray, min_event_length: int) -> np.ndarray:
    filtered = np.asarray(predictions, dtype=np.int32).ravel().copy()
    if min_event_length <= 1:
        return filtered

    for start, end in _true_segments(filtered != 0):
        if (end - start) < min_event_length:
            filtered[start:end] = 0
    return filtered


def _true_segments(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    bool_mask = np.asarray(mask, dtype=bool).ravel()
    if bool_mask.size == 0:
        return ()

    starts = np.flatnonzero(bool_mask & ~np.r_[False, bool_mask[:-1]])
    ends = np.flatnonzero(bool_mask & ~np.r_[bool_mask[1:], False]) + 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))
