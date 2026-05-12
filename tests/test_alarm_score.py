from pathlib import Path

import numpy as np

from noboom_benchmark.noboom_lib.core.metric_utils.alarm_threshold_search import select_threshold
from noboom_benchmark.noboom_lib.core.metric_utils.metrics.alarm import alarm_score


def _score_from_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    result = select_threshold(scores, targets)
    predictions = (scores >= result.best_threshold).astype(np.int32)
    return result.best_score, alarm_score(predictions, targets)


def test_alarm_score_matches_sweep_simple_segments() -> None:
    scores = np.array([0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3], dtype=np.float64)
    targets = np.array([0, 1, 1, 0, 0, 1, 1], dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"


def test_alarm_score_matches_sweep_with_ties() -> None:
    scores = np.array([0.5, 0.5, 0.2, 0.9, 0.9, 0.1, 0.3, 0.5], dtype=np.float64)
    targets = np.array([0, 1, 1, 0, 1, 1, 0, 0], dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"


def test_alarm_score_matches_sweep_without_targets() -> None:
    scores = np.array([0.2, 0.8, 0.3, 0.4], dtype=np.float64)
    targets = np.zeros_like(scores, dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"


def test_alarm_score_matches_sweep_all_scores_equal() -> None:
    scores = np.array([0.4, 0.4, 0.4, 0.4, 0.4], dtype=np.float64)
    targets = np.array([0, 1, 0, 1, 0], dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"


def test_alarm_score_matches_sweep_sparse_targets() -> None:
    scores = np.array([0.6, 0.1, 0.7, 0.2, 0.8, 0.05, 0.3, 0.9], dtype=np.float64)
    targets = np.array([0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"


def test_alarm_score_zero_when_best_threshold_disables_all() -> None:
    scores = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    targets = np.zeros_like(scores, dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"
    assert alarm_score_value == 0.0


def test_alarm_score_zero_for_empty_arrays() -> None:
    scores = np.array([], dtype=np.float64)
    targets = np.array([], dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"
    assert alarm_score_value == 0.0


def test_sweep_min_score_zero_due_to_inf_threshold() -> None:
    scores = np.array([0.2, 0.8, 0.3, 0.4], dtype=np.float64)
    targets = np.zeros_like(scores, dtype=np.int32)
    result = select_threshold(scores, targets)
    assert result.best_score == 0.0
    assert result.best_threshold == np.inf


def test_alarm_score_matches_sweep_single_element() -> None:
    scores = np.array([0.25], dtype=np.float64)
    targets = np.array([1], dtype=np.int32)
    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"


def test_alarm_score_matches_sweep_all_targets_zero_never_positive() -> None:
    # With no ground-truth anomalies, any non-empty prediction set incurs penalties.
    # The sweep should be able to choose a threshold that yields zero active predictions,
    # which forces alarm_score == 0.0 (and thus never positive).
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.7], dtype=np.float64)
    targets = np.array([0, 0, 0, 0, 0], dtype=np.int32)

    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"
    assert best_score <= 0.0, f"Expected non-positive best_score, got {best_score}"


def test_alarm_score_matches_sweep_sparse_scores_all_targets_zero_never_positive() -> None:
    # Same logic as above, but with a wider score spread to exercise thresholding behavior.
    scores = np.array([10.0, -3.0, 2.5, 0.0, 7.2, -1.1], dtype=np.float64)
    targets = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)

    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"
    assert best_score <= 0.0, f"Expected non-positive best_score, got {best_score}"


def test_alarm_score_matches_sweep_targets_present_but_scores_force_missed_detections_never_positive() -> None:
    # Construct scores so that the highest scores are outside the anomaly regions.
    # The sweep can either predict nothing (score 0) or predict false alarms / miss anomalies,
    # which should not yield a positive alarm score overall.
    #
    # targets: anomalies at [2,4) and [7,9)
    targets = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int32)

    # High scores concentrated in non-anomalous positions (0, 5, 6, 9),
    # low scores inside anomalies (2,3,7,8).
    scores = np.array([0.99, 0.10, 0.01, 0.02, 0.05, 0.98, 0.97, 0.03, 0.04, 0.96], dtype=np.float64)

    best_score, alarm_score_value = _score_from_threshold(scores, targets)
    assert np.isclose(best_score, alarm_score_value), f"{best_score} != {alarm_score_value}"
    assert best_score <= 0.0, f"Expected non-positive best_score, got {best_score}"


def test_alarm_score_matches_sweep_large_score_array() -> None:
    tests_root = Path(__file__).parent
    data = np.load(tests_root / f'data/{test_alarm_score_matches_sweep_large_score_array.__name__[5:]}.npz')
    scores, targets = data['scores'], data['targets']
    assert targets.size == scores.size

    best_th = np.inf
    best_sc = 0.0
    for t in scores:
        preds = (scores > t).astype(np.int32)
        score = alarm_score(preds, targets)
        if score > best_sc:
            best_sc = score
            best_th = t

    result = select_threshold(scores, targets)
    assert np.isclose(best_sc, result.best_score), f"{best_sc} != {result.best_score}"
    assert np.isclose(best_th, result.best_threshold), f"{best_th} != {result.best_threshold}"
