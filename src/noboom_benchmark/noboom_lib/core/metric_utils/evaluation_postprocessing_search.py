from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .evaluation_postprocessing import (
    EvaluationPostprocessingConfig,
    evaluate_sequences_with_optional_postprocessing,
    resolve_evaluation_postprocessing_config,
)


DEFAULT_SEARCH_ORDER = (
    "short_window",
    "long_ewma_span",
    "hysteresis_delta_mad",
    "enter_consecutive",
    "exit_consecutive",
    "merge_gap",
    "min_event_length",
)

DEFAULT_SEARCH_SPACE = {
    "short_window": (3, 5, 7, 9),
    "long_ewma_span": (15, 25, 35, 50),
    "hysteresis_delta_mad": (0.25, 0.5, 0.75, 1.0),
    "enter_consecutive": (1, 2, 3),
    "exit_consecutive": (2, 3, 5),
    "merge_gap": (0, 1, 3, 5),
    "min_event_length": (1, 2, 3, 5),
}

MINIMIZE_METRICS = frozenset({"aaf", "edf", "ldf"})


@dataclass(frozen=True)
class PostprocessingSearchTrial:
    config: Dict[str, Any]
    metric_values: Dict[str, float]
    threshold: float
    threshold_on: float
    threshold_off: float
    robust_scale: float


@dataclass(frozen=True)
class PostprocessingSearchResult:
    baseline_trial: PostprocessingSearchTrial
    default_enabled_trial: PostprocessingSearchTrial
    best_trial: PostprocessingSearchTrial
    history: Tuple[PostprocessingSearchTrial, ...]
    rounds_completed: int


def split_flat_score_payload(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    score_array = np.asarray(scores, dtype=np.float64).ravel()
    label_array = np.asarray(labels).ravel()
    if score_array.shape != label_array.shape:
        raise ValueError(
            "scores and labels must have the same flattened shape for sequence reconstruction; "
            f"received {score_array.shape} vs {label_array.shape}."
        )

    score_sequences = []
    label_sequences = []
    valid_mask = np.isfinite(score_array)
    for start, end in _true_segments(valid_mask):
        score_sequences.append(score_array[start:end].copy())
        label_sequences.append(label_array[start:end].copy())

    if not score_sequences:
        raise ValueError("No finite score sequences were found in the anomaly score payload.")
    return tuple(score_sequences), tuple(label_sequences)


def search_evaluation_postprocessing(
    *,
    score_sequences: Sequence[np.ndarray],
    label_sequences: Sequence[np.ndarray],
    metric_names: Sequence[str],
    target_metric: str,
    dataset_name: str,
    initial_config: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Sequence[Any]]] = None,
    search_order: Optional[Sequence[str]] = None,
    max_rounds: int = 3,
    fix_threshold: bool = False,
) -> PostprocessingSearchResult:
    resolved_initial = resolve_evaluation_postprocessing_config(initial_config)
    current_payload = _config_to_payload(resolved_initial)
    current_payload["enabled"] = True

    candidate_space = {
        name: tuple(values)
        for name, values in (search_space or DEFAULT_SEARCH_SPACE).items()
    }
    ordered_params = tuple(search_order or DEFAULT_SEARCH_ORDER)

    history = []
    cached_trials: Dict[str, PostprocessingSearchTrial] = {}

    def evaluate_payload(payload: Mapping[str, Any]) -> PostprocessingSearchTrial:
        config_payload = dict(payload)
        config = resolve_evaluation_postprocessing_config(config_payload)
        cache_key = json.dumps(_config_to_payload(config), sort_keys=True)
        cached = cached_trials.get(cache_key)
        if cached is not None:
            return cached

        evaluation = evaluate_sequences_with_optional_postprocessing(
            score_sequences,
            label_sequences,
            metric_names=metric_names,
            config=config,
            fix_threshold=fix_threshold,
            dataset_name=dataset_name,
        )
        trial = PostprocessingSearchTrial(
            config=_config_to_payload(config),
            metric_values=dict(evaluation.metric_values),
            threshold=float(evaluation.threshold),
            threshold_on=float(evaluation.threshold_on),
            threshold_off=float(evaluation.threshold_off),
            robust_scale=float(evaluation.robust_scale),
        )
        cached_trials[cache_key] = trial
        history.append(trial)
        return trial

    baseline_trial = evaluate_payload({"enabled": False})
    default_enabled_trial = evaluate_payload(current_payload)
    best_trial = default_enabled_trial
    rounds_completed = 0

    for _round_index in range(max_rounds):
        rounds_completed += 1
        improved = False
        for param_name in ordered_params:
            if param_name not in candidate_space:
                continue

            candidate_values = _candidate_values(
                current_value=current_payload[param_name],
                values=candidate_space[param_name],
            )
            local_best_payload = dict(current_payload)
            local_best_trial = best_trial
            for candidate_value in candidate_values:
                candidate_payload = dict(current_payload)
                candidate_payload[param_name] = candidate_value
                candidate_payload["enabled"] = True
                candidate_trial = evaluate_payload(candidate_payload)
                if _is_trial_better(candidate_trial, local_best_trial, target_metric=target_metric):
                    local_best_trial = candidate_trial
                    local_best_payload = dict(candidate_payload)

            if local_best_payload != current_payload:
                current_payload = local_best_payload
                best_trial = local_best_trial
                improved = True

        if not improved:
            break

    return PostprocessingSearchResult(
        baseline_trial=baseline_trial,
        default_enabled_trial=default_enabled_trial,
        best_trial=best_trial,
        history=tuple(history),
        rounds_completed=rounds_completed,
    )


def _candidate_values(current_value: Any, values: Sequence[Any]) -> Tuple[Any, ...]:
    ordered = [current_value]
    ordered.extend(values)
    deduplicated = []
    for value in ordered:
        if value not in deduplicated:
            deduplicated.append(value)
    return tuple(deduplicated)


def _config_to_payload(config: EvaluationPostprocessingConfig) -> Dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "short_window": int(config.short_window),
        "long_ewma_span": int(config.long_ewma_span),
        "hysteresis_delta_mad": float(config.hysteresis_delta_mad),
        "enter_consecutive": int(config.enter_consecutive),
        "exit_consecutive": int(config.exit_consecutive),
        "merge_gap": int(config.merge_gap),
        "min_event_length": int(config.min_event_length),
    }


def _signed_metric_value(metric_name: str, metric_value: float) -> float:
    if not math.isfinite(metric_value):
        return float("-inf")
    if metric_name in MINIMIZE_METRICS:
        return -metric_value
    return metric_value


def _trial_rank_key(
    trial: PostprocessingSearchTrial,
    *,
    target_metric: str,
) -> Tuple[float, float, float, float, float]:
    metric_values = trial.metric_values
    return (
        _signed_metric_value(target_metric, float(metric_values.get(target_metric, float("nan")))),
        _signed_metric_value("event_recall", float(metric_values.get("event_recall", float("nan")))),
        _signed_metric_value("aaf", float(metric_values.get("aaf", float("nan")))),
        _signed_metric_value("edf", float(metric_values.get("edf", float("nan")))),
        _signed_metric_value("ldf", float(metric_values.get("ldf", float("nan")))),
    )


def _is_trial_better(
    candidate: PostprocessingSearchTrial,
    incumbent: PostprocessingSearchTrial,
    *,
    target_metric: str,
) -> bool:
    return _trial_rank_key(candidate, target_metric=target_metric) > _trial_rank_key(
        incumbent,
        target_metric=target_metric,
    )


def _true_segments(mask: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    mask_array = np.asarray(mask, dtype=bool).ravel()
    if mask_array.size == 0:
        return ()

    padded = np.concatenate(([False], mask_array, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))
