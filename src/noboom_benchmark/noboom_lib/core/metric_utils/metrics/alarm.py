"""Alarm score utilities for evaluating anomaly detection predictions."""

import numpy as np
import numpy.typing as npt

from noboom.tsad.metrics import continuous_segments

FALSE_ALARM_TOLERANCE = 2


def _false_positive_weights(num_predictions: int) -> float:
    """Compute the false-positive penalty weight.

    Args:
        num_predictions (int): Number of predicted positives.

    Returns:
        float: False-positive weight.
    """
    return 1.0 - 1.0 / num_predictions if num_predictions else 0.0


def alarm_score(predictions: npt.NDArray[np.integer], targets: npt.NDArray[np.integer]) -> float:
    """Compute the alarm score metric for discrete anomaly predictions.

    The score rewards detected ground-truth alarm segments, gives extra credit
    to earlier hits within a detected segment, and penalizes false alarm
    segments plus alarm overflow across anomaly boundaries.

    Args:
        predictions (npt.NDArray[np.integer]): Predicted binary anomaly array.
        targets (npt.NDArray[np.integer]): Ground-truth binary anomaly array.

    Returns:
        float: Alarm score value.
    """

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    preds_b = (predictions != 0)
    tgts_b = (targets != 0)

    active_count = int(np.count_nonzero(preds_b))
    if active_count == 0:
        # Match sweep behavior: score is forced to 0 when no predictions are active.
        return 0.0

    ground_truth_alarms = continuous_segments(tgts_b, True)
    predicted_alarms = continuous_segments(preds_b, True)

    # For segment-level detection accounting, keep only the first hit inside each
    # predicted alarm segment that overlaps a ground-truth anomaly.
    first_predictions = preds_b.copy()

    num_points = first_predictions.size
    num_predicted_alarms = predicted_alarms.shape[0]

    hit_positions = np.nonzero(tgts_b)[0]

    if num_predicted_alarms and hit_positions.size:
        starts = predicted_alarms[:, 0]
        ends = predicted_alarms[:, 1]
        first_hit_index = np.searchsorted(hit_positions, starts, side="left")
        clipped_hit_index = np.minimum(first_hit_index, hit_positions.size - 1)
        first_hit_positions = hit_positions[clipped_hit_index]
        has_overlap = (first_hit_index < hit_positions.size) & (first_hit_positions < ends)

        if np.any(has_overlap):
            # Zero everything after the first hit in each overlapping predicted
            # segment so the segment contributes only one detection event.
            left = (first_hit_positions[has_overlap] + 1).astype(np.int64, copy=False)
            right = ends[has_overlap]

            diff = np.zeros(num_points + 1, dtype=np.int32)
            np.add.at(diff, left, 1)
            np.add.at(diff, right, -1)
            first_predictions[np.cumsum(diff[:-1]) > 0] = False

    # Reuse one prefix-sum buffer to test whether segments contain any retained
    # prediction or any ground-truth positive.
    p = np.empty(first_predictions.size + 1, dtype=np.int32)
    p[0] = 0
    np.cumsum(first_predictions, out=p[1:])
    detected_mask = (p[ground_truth_alarms[:, 1]] - p[ground_truth_alarms[:, 0]]) != 0
    detected_anomalies = ground_truth_alarms[detected_mask]

    # A predicted alarm segment is false if it contains no ground-truth points.
    np.cumsum(tgts_b, out=p[1:])
    true_false_alarms = np.count_nonzero((p[predicted_alarms[:, 1]] - p[predicted_alarms[:, 0]]) == 0)
    num_fp = int(np.count_nonzero(preds_b & ~tgts_b))
    fpw = float(_false_positive_weights(num_fp))

    score = float(len(detected_anomalies))
    score -= float(true_false_alarms) / FALSE_ALARM_TOLERANCE
    score -= fpw

    for anomaly in detected_anomalies:
        start, end = anomaly
        anomaly_prediction = preds_b[start:end]

        if anomaly_prediction.size == 0:
            continue

        num_alarms = int(anomaly_prediction[0]) + np.count_nonzero(anomaly_prediction[1:] & ~anomaly_prediction[:-1])

        # Earlier hits within the anomaly segment contribute more reward.
        idx = np.nonzero(anomaly_prediction)[0]
        nom = np.ldexp(1 + np.exp2(-(idx + 1)).sum(), -num_alarms)
        score += nom / len(detected_anomalies)

    false_alarms_early_overflow = 0
    if len(ground_truth_alarms) > 0:
        # Penalize predicted alarms that were already active immediately before an
        # anomaly starts.
        idx = ground_truth_alarms[:, 0]
        if idx[0] > 0 or len(idx) > 1:
            if idx[0] == 0:
                idx = idx[1:]
            false_alarms_early_overflow = np.count_nonzero(preds_b[idx - 1] & preds_b[idx])

    score -= 1.5 * false_alarms_early_overflow / FALSE_ALARM_TOLERANCE

    false_alarms_late_overflow = 0
    if len(ground_truth_alarms) > 0:
        # Penalize predicted alarms that remain active immediately after an
        # anomaly ends.
        idx = ground_truth_alarms[:, 1]
        if idx[-1] < len(preds_b) or len(idx) > 1:
            if idx[-1] == len(preds_b):
                idx = idx[:-1]
            false_alarms_late_overflow = np.count_nonzero(preds_b[idx - 1] & preds_b[idx])

    score -= 0.5 * false_alarms_late_overflow / FALSE_ALARM_TOLERANCE
    return float(score)
