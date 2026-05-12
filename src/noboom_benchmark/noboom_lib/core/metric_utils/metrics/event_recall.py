import numpy as np

from noboom.tsad.metrics import continuous_segments


def event_recall(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute event-level recall for anomaly predictions.

    Args:
        predictions (np.ndarray): Predicted binary anomaly array.
        targets (np.ndarray): Ground-truth binary anomaly array.

    Returns:
        float: Event recall score.
    """
    preds = np.asarray(predictions)
    t = np.asarray(targets)
    if preds.ndim != 1:
        preds = preds.ravel()
    if t.ndim != 1:
        t = t.ravel()

    anomalies = continuous_segments(t != 0, True)
    if anomalies.shape[0] == 0:
        return 0.0

    preds_b = (preds != 0)

    p = np.empty(preds_b.size + 1, dtype=np.int32)
    p[0] = 0
    np.cumsum(preds_b, out=p[1:])

    s = anomalies[:, 0]
    e = anomalies[:, 1]
    detected = (p[e] - p[s]) != 0

    return float(np.count_nonzero(detected) / anomalies.shape[0])
