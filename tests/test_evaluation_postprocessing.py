import numpy as np

from noboom_benchmark.noboom_lib.core.metric_utils.alarm_threshold_search import select_threshold
from noboom_benchmark.noboom_lib.core.metric_utils.evaluation_postprocessing import (
    EvaluationPostprocessingConfig,
    aggregate_prediction_events,
    build_hysteresis_predictions,
    causal_ewma,
    causal_rolling_max,
    compute_decision_scores,
    evaluate_sequences_with_optional_postprocessing,
)
from noboom_benchmark.noboom_lib.core.metric_utils.metrics.alarm import alarm_score


def test_disabled_mode_matches_legacy_pointwise_thresholding() -> None:
    score_sequences = [
        np.array([0.1, 0.8, 0.2], dtype=np.float64),
        np.array([0.9, 0.1], dtype=np.float64),
    ]
    label_sequences = [
        np.array([0, 1, 1], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    ]

    result = evaluate_sequences_with_optional_postprocessing(
        score_sequences,
        label_sequences,
        metric_names=("alarm_score",),
        config=EvaluationPostprocessingConfig(enabled=False),
    )

    expected_scores = np.array([-np.inf, 0.1, 0.8, 0.2, -np.inf, 0.9, 0.1], dtype=np.float64)
    expected_labels = np.array([0, 0, 1, 1, 0, 0, 0], dtype=np.int32)
    expected_threshold = select_threshold(expected_scores, expected_labels).best_threshold
    expected_predictions = (expected_scores > expected_threshold).astype(np.int32)

    np.testing.assert_allclose(result.raw_scores, expected_scores)
    np.testing.assert_array_equal(result.raw_labels, expected_labels)
    np.testing.assert_array_equal(result.predictions, expected_predictions)
    assert result.threshold == expected_threshold
    assert result.metric_values["alarm_score"] == alarm_score(expected_predictions, expected_labels)


def test_causal_filters_behave_as_expected() -> None:
    values = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float64)

    np.testing.assert_allclose(causal_rolling_max(values, 2), np.array([1.0, 3.0, 3.0, 5.0]))
    np.testing.assert_allclose(
        causal_ewma(values, 3),
        np.array([1.0, 2.0, 2.0, 3.5]),
    )


def test_compute_decision_scores_resets_across_nonfinite_boundaries() -> None:
    config = EvaluationPostprocessingConfig(
        enabled=True,
        short_window=2,
        long_ewma_span=2,
    )
    scores = np.array([-np.inf, 1.0, 3.0, -np.inf, 2.0, 1.0], dtype=np.float64)

    decision_scores = compute_decision_scores(scores, config)

    assert not np.isfinite(decision_scores[0])
    assert not np.isfinite(decision_scores[3])
    np.testing.assert_allclose(decision_scores[1:3], np.array([1.0, 3.0]))
    np.testing.assert_allclose(decision_scores[4:], np.array([2.0, 2.0]))


def test_hysteresis_requires_consecutive_enter_and_exit_points() -> None:
    config = EvaluationPostprocessingConfig(
        enabled=True,
        enter_consecutive=2,
        exit_consecutive=2,
    )
    decision_scores = np.array([0.4, 1.1, 1.2, 0.3, 0.2, 0.1], dtype=np.float64)

    predictions = build_hysteresis_predictions(
        decision_scores,
        threshold_on=1.0,
        threshold_off=0.5,
        config=config,
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1, 1, 1, 1, 0], dtype=np.int32))


def test_event_aggregation_fills_short_gaps_and_drops_short_events() -> None:
    config = EvaluationPostprocessingConfig(
        enabled=True,
        merge_gap=1,
        min_event_length=2,
    )
    decision_scores = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    merged = aggregate_prediction_events(
        np.array([1, 0, 1, 0, 1, 1, 0], dtype=np.int32),
        decision_scores,
        config,
    )
    np.testing.assert_array_equal(merged, np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.int32))

    filtered = aggregate_prediction_events(
        np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.int32),
        decision_scores,
        config,
    )
    np.testing.assert_array_equal(filtered, np.array([0, 0, 0, 0, 1, 1, 0], dtype=np.int32))


def test_enabled_mode_keeps_invalid_positions_suppressed() -> None:
    config = EvaluationPostprocessingConfig(enabled=True)
    score_sequences = [
        np.array([-np.inf, 0.4, 0.8, -np.inf], dtype=np.float64),
    ]
    label_sequences = [
        np.array([0, 0, 1, 0], dtype=np.int32),
    ]

    result = evaluate_sequences_with_optional_postprocessing(
        score_sequences,
        label_sequences,
        metric_names=("alarm_score",),
        config=config,
    )

    assert not np.isfinite(result.processed_scores[0])
    assert not np.isfinite(result.processed_scores[-1])
    assert result.predictions[0] == 0
    assert result.predictions[-1] == 0
