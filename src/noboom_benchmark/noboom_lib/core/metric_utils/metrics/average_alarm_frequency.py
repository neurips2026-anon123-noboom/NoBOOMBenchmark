import numpy as np

from noboom.tsad.metrics import continuous_segments


def aaf(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute average alarm frequency (AAF).

    Args:
        predictions (np.ndarray): Predicted binary anomaly array.
        targets (np.ndarray): Ground-truth binary anomaly array.

    Returns:
        float: Average alarm frequency across detected anomalies.
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

    # --- Detect which anomalies are detected (prefix sum, O(n)) ---
    p = np.empty(preds_b.size + 1, dtype=np.int32)
    p[0] = 0
    np.cumsum(preds_b, out=p[1:])

    s_all = anomalies[:, 0]
    e_all = anomalies[:, 1]
    detected_mask = (p[e_all] - p[s_all]) != 0
    if not np.any(detected_mask):
        return 0.0

    segs = anomalies[detected_mask]
    s = segs[:, 0]
    e = segs[:, 1]

    # Heuristic switch: if few detected anomalies or low covered length, count alarms by slicing.
    # Tune these constants based on your benchmark output:
    M = segs.shape[0]
    coverage = int(np.sum(e - s))
    FEW_M = 64
    FEW_COVERAGE = 5000

    if M <= FEW_M or coverage <= FEW_COVERAGE:
        # O(coverage) path (wins for few anomalies)
        alarms = np.empty(M, dtype=np.int32)
        for i, (si, ei) in enumerate(segs):
            x = preds_b[si:ei]
            if x.size == 0:
                alarms[i] = 0
            else:
                # alarms in slice: x[0] + count of 0->1 transitions
                alarms[i] = int(x[0]) + int(np.count_nonzero(x[1:] & ~x[:-1]))
        return float(np.mean(alarms))

    # --- Many anomalies path: fully vectorized (your previous approach, O(n)) ---
    run_start = preds_b.astype(np.uint8, copy=True)
    run_start[1:] &= ~preds_b[:-1]

    rs = np.empty(run_start.size + 1, dtype=np.int32)
    rs[0] = 0
    np.cumsum(run_start, out=rs[1:])

    alarms = (rs[e] - rs[s]).astype(np.int32, copy=False)

    # boundary correction to match slice semantics
    s_gt0 = (s != 0)
    alarms += (preds_b[s] & s_gt0 & preds_b[s - 1]).astype(np.int32, copy=False)

    return float(np.mean(alarms))
