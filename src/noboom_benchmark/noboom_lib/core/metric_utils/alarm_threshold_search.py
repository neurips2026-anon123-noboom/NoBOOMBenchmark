from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Used for debugging
DATASET_FRACTIONS = {
    "cont_single_component_water": 9183 / 37250,
    "cont_binary_component_n_butanol": 1667 / 4082,
    "cont_reactive_ome": 353 / 986,
    "cont_reactive_ome_ext": 353 / 986,
    "cont_reactive_ome_red": 353 / 986,
    "batch_dist_ternary_1_butanol_2_propanol_water": 83608 / 395712,
    "batch_dist_ternary_acetone_1_butanol_methanol": 16706 / 69558,
    "industry_process": 309478 / 1842436,
}

FALSE_ALARM_TOLERANCE = 2.0


# -----------------------------
# Pure-Python helpers (wrapper level)
# -----------------------------

def _false_positive_weights(num_predictions: int) -> float:
    """Compute the false-positive weight for an alarm score.

    Args:
        num_predictions (int): Number of predicted positives.

    Returns:
        float: False-positive weight.
    """
    weight = 1.0 - 1.0 / num_predictions if num_predictions else 0.0
    logger.debug(
        "Computed false positive weight %.6f for %d predictions.",
        weight,
        num_predictions,
    )
    return weight


@dataclass
class SweepExactResult:
    best_threshold: float
    best_score: float
    num_steps: int


# ============================================================
# Numba utilities: GT segments for binary targets
# ============================================================

def _count_true_segments(b: np.ndarray) -> int:
    """Count contiguous true segments in a binary array.

    Args:
        b (np.ndarray): Binary array of targets.

    Returns:
        int: Number of true segments.
    """
    n = b.size
    logger.debug("Counting true segments for array size %d.", n)
    if n == 0:
        return 0
    b = b != 0
    starts = np.flatnonzero(b & ~np.r_[False, b[:-1]])
    logger.debug("Counted %d true segments.", int(starts.size))
    return int(starts.size)


def _true_segments(b: np.ndarray) -> np.ndarray:
    """Return start/end indices for true segments in a binary array.

    Args:
        b (np.ndarray): 1D uint8/bool array.

    Returns:
        np.ndarray: (K, 2) int64 array of [start, end) segments.
    """
    n = b.size
    logger.debug("Building true segments for array size %d.", n)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)
    b = b != 0
    starts = np.flatnonzero(b & ~np.r_[False, b[:-1]])
    ends = np.flatnonzero(b & ~np.r_[b[1:], False]) + 1
    logger.debug("Detected %d true segments.", int(starts.size))
    return np.column_stack([starts, ends]).astype(np.int64, copy=False)


# ============================================================
# Numba Fenwick (BIT) over alive indices
# ============================================================

def fenwick_create(n: int) -> np.ndarray:
    """Create a Fenwick tree array.

    Args:
        n (int): Number of elements.

    Returns:
        np.ndarray: Fenwick tree array of length n + 1.
    """
    logger.debug("Creating Fenwick tree for size %d.", n)
    return np.zeros(n + 1, dtype=np.int32)  # 1-indexed


def fenwick_add(bit: np.ndarray, idx0: int, delta: int) -> None:
    """Add a delta to a Fenwick tree index.

    Args:
        bit (np.ndarray): Fenwick tree array.
        idx0 (int): Zero-based index to update.
        delta (int): Delta to add.

    Returns:
        None: Tree is updated in place.
    """
    n = bit.size - 1
    #logger.debug("Fenwick add delta %d at index %d (size %d).", delta, idx0, n)
    # Move to 1-based index for Fenwick operations.
    i = idx0 + 1
    while i <= n:
        # Apply delta to current node.
        bit[i] += delta
        # Jump to next responsible node by adding lowbit.
        i += i & -i


def fenwick_sum_prefix(bit: np.ndarray, r0: int) -> int:
    """Compute prefix sum up to r0 in a Fenwick tree.

    Args:
        bit (np.ndarray): Fenwick tree array.
        r0 (int): Zero-based exclusive end index.

    Returns:
        int: Prefix sum.
    """
    s = 0
    i = r0
    while i > 0:
        # Accumulate sums while stripping lowbit.
        s += bit[i]
        i -= i & -i
    return int(s)


def _highest_power_of_two_leq(n: int) -> int:
    """Return the highest power of two less than or equal to n.

    Args:
        n (int): Upper bound.

    Returns:
        int: Highest power of two <= n.
    """
    p = 1
    # logger.debug("Computing highest power of two <= %d.", n)
    while (p << 1) <= n:
        # Exponentially increase until the next shift would exceed n.
        p <<= 1
    return p


def fenwick_find_by_order(bit: np.ndarray, k: int) -> int:
    """Find the smallest index with prefix sum >= k.

    Args:
        bit (np.ndarray): Fenwick tree array.
        k (int): 1-based order statistic (1 <= k <= total).

    Returns:
        int: Zero-based index matching the order statistic.
    """
    n = bit.size - 1
    # logger.debug("Finding Fenwick order %d within size %d.", k, n)
    idx = 0
    # Start from the highest power of two to binary-search the prefix sum space.
    bitmask = _highest_power_of_two_leq(n) if n > 0 else 0
    while bitmask:
        nxt = idx + bitmask
        # If stepping to nxt keeps prefix sum < k, advance and subtract.
        if nxt <= n and bit[nxt] < k:
            k -= int(bit[nxt])
            idx = nxt
        bitmask >>= 1
    # idx is the last prefix with sum < k; return zero-based index.
    return int(idx)


def fenwick_next_one_at_or_after(bit: np.ndarray, l0: int) -> int:
    """Return the next index with a value of 1 at or after l0.

    Args:
        bit (np.ndarray): Fenwick tree array.
        l0 (int): Starting index.

    Returns:
        int: Next index with value 1, or n as sentinel if none.
    """
    n = bit.size - 1
    # logger.debug("Finding next active index at/after %d (size %d).", l0, n)
    if l0 < 0:
        # Clamp to lower bound.
        l0 = 0
    if l0 >= n:
        # No valid indices at/after n.
        return n
    before = fenwick_sum_prefix(bit, l0)
    tot = fenwick_sum_prefix(bit, n)
    if before == tot:
        # No ones after l0.
        return n
    # Find the first one after l0 using order-statistic search.
    return fenwick_find_by_order(bit, before + 1)


def _fenwick_self_test(seed: int = 0, trials: int = 200) -> None:
    """Self-test for Fenwick order-statistic helpers."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        n = int(rng.integers(1, 129))
        arr = rng.integers(0, 2, size=n, dtype=np.int32)
        bit = fenwick_create(n)
        for idx0, val in enumerate(arr):
            if val:
                fenwick_add(bit, idx0, int(val))

        total = int(arr.sum())
        for k in range(1, total + 1):
            naive = int(np.flatnonzero(np.cumsum(arr) >= k)[0])
            got = fenwick_find_by_order(bit, k)
            assert got == naive, (arr, k, got, naive)

        for l0 in range(n):
            expected_candidates = np.flatnonzero(arr[l0:] != 0)
            expected = n if expected_candidates.size == 0 else int(l0 + expected_candidates[0])
            got = fenwick_next_one_at_or_after(bit, l0)
            assert got == expected, (arr, l0, got, expected)


# ============================================================
# Numba DSU with first_hit + false alarm segment counting
# ============================================================

def dsu_find(parent: np.ndarray, x: int) -> int:
    """Find the root of an element in a DSU structure.

    Args:
        parent (np.ndarray): Parent array.
        x (int): Element index.

    Returns:
        int: Root index.
    """
    p = parent[x]
    while p != x:
        # Path compression: point node directly to grandparent.
        parent[x] = parent[p]
        x = p
        p = parent[x]
    # x is the root representative.
    return x


def dsu_union(parent, size, left, right, has_tgt, first_hit, n_false_alarm_segments, a, b):
    """Union two DSU sets and update false-alarm counters.

    Args:
        parent (np.ndarray): Parent array.
        size (np.ndarray): Size array.
        left (np.ndarray): Left boundary array.
        right (np.ndarray): Right boundary array.
        has_tgt (np.ndarray): Target-presence flags.
        first_hit (np.ndarray): First-hit index array.
        n_false_alarm_segments (int): Current false alarm segment count.
        a (int): First element index.
        b (int): Second element index.

    Returns:
        tuple: (root_index, updated_false_alarm_segments)
    """
    # Find current roots to avoid merging within the same set.
    ra = dsu_find(parent, a)
    rb = dsu_find(parent, b)
    if ra == rb:
        return ra, n_false_alarm_segments

    # Union by size to keep the tree shallow.
    if size[ra] < size[rb]:
        ra, rb = rb, ra

    # Track whether each component was a false-alarm-only segment before merging.
    fa = (has_tgt[ra] == 0)
    fb = (has_tgt[rb] == 0)

    # Attach rb under ra.
    parent[rb] = ra
    size[ra] += size[rb]

    # Update segment boundaries for the merged component.
    if left[rb] < left[ra]:
        left[ra] = left[rb]
    if right[rb] > right[ra]:
        right[ra] = right[rb]

    # Merge target presence and earliest hit across components.
    has_tgt[ra] = has_tgt[ra] | has_tgt[rb]
    if first_hit[rb] < first_hit[ra]:
        first_hit[ra] = first_hit[rb]

    # Update false-alarm segment count: two FA-only segments merged into one.
    fnew = (has_tgt[ra] == 0)
    n_false_alarm_segments += int(fnew) - (int(fa) + int(fb))
    return ra, n_false_alarm_segments


def dsu_activate(parent, size, left, right, active, has_tgt, first_hit, n_false_alarm_segments, i, tgts_b):
    """Activate an index in the DSU and update counters.

    Args:
        parent (np.ndarray): Parent array.
        size (np.ndarray): Size array.
        left (np.ndarray): Left boundary array.
        right (np.ndarray): Right boundary array.
        active (np.ndarray): Active flags array.
        has_tgt (np.ndarray): Target-presence flags.
        first_hit (np.ndarray): First-hit index array.
        n_false_alarm_segments (int): Current false alarm segment count.
        i (int): Index to activate.
        tgts_b (np.ndarray): Binary targets array.

    Returns:
        tuple: (root_index, updated_false_alarm_segments).
    """
    n = active.size
    logger.debug("Activating DSU index %d (size %d).", i, n)

    # Initialize the new singleton component.
    active[i] = 1
    parent[i] = i
    size[i] = 1
    left[i] = i
    right[i] = i
    # Mark target presence and earliest hit for this point.
    if tgts_b[i] != 0:
        has_tgt[i] = 1
        first_hit[i] = i
    else:
        has_tgt[i] = 0
        # first_hit[i] already initialized to INF outside

    # A new component without a target creates a false-alarm segment.
    if has_tgt[i] == 0:
        n_false_alarm_segments += 1

    # Merge with immediate neighbors if they are active.
    root = i
    if i > 0 and active[i - 1] != 0:
        root, n_false_alarm_segments = dsu_union(parent, size, left, right, has_tgt, first_hit,
                                                 n_false_alarm_segments, root, i - 1)
    if i + 1 < n and active[i + 1] != 0:
        root, n_false_alarm_segments = dsu_union(parent, size, left, right, has_tgt, first_hit,
                                                 n_false_alarm_segments, root, i + 1)

    # Return the canonical root after unions.
    root = dsu_find(parent, root)
    return root, n_false_alarm_segments


# ============================================================
# Full Numba sweep kernel
# ============================================================

def _sweep_alarm_score_exact(scores: np.ndarray, targets_b: np.ndarray, order: np.ndarray):
    """Numba-accelerated sweep to compute best alarm score threshold.

    Algorithm overview:
        - We sweep thresholds in descending score order (grouping equal scores),
          progressively activating predictions for indices whose score exceeds
          the current threshold. This matches the standard "prefix" sweep used
          for threshold selection.
        - For each threshold group we update three interacting structures:
            1) A DSU (disjoint-set union) that tracks contiguous predicted
               segments, their left/right boundaries, and whether each segment
               contains any target. This lets us count false-alarm segments and
               compute first-hit suppression boundaries.
            2) A Fenwick tree and "alive" mask that track which predictions are
               still active after first-hit suppression (only the earliest
               point in a predicted segment is kept alive for scoring).
            3) Per-anomaly counters that maintain the "nominal" term, which
               depends on how many raw predicted segments overlap a ground-truth
               anomaly and the decay-weighted distance of hits within it.
        - At each threshold step we update: (a) false positives count, (b) early
          and late overflow penalties for predictions crossing anomaly borders,
          (c) false-alarm segment count from DSU, (d) detected anomalies based
          on alive predictions, and (e) the nominal term for detected anomalies.
        - The alarm score is computed from these statistics; the best score and
          threshold are tracked over the entire sweep. We also evaluate the
          final "all active" threshold at -inf.

    NaN scores are rejected because thresholds require a total ordering; callers
    must sanitize or impute scores before sweeping.

    Args:
        scores (np.ndarray): Score array (1D float64).
        targets_b (np.ndarray): Binary targets array (1D uint8).
        order (np.ndarray): Indices sorted by descending score.

    Returns:
        tuple: (best_threshold, best_score, num_steps).
    """
    logger.debug(
        "Starting alarm score sweep with %d scores and %d targets.",
        scores.size,
        targets_b.size,
    )
    # Use shorter aliases to avoid repeated attribute lookups in the hot loop.
    s = scores
    tgts_b = targets_b
    n = s.size

    if np.isnan(s).any():
        # Warn and sanitize NaNs; the ordering is undefined otherwise.
        nan_mask = np.isnan(s)
        nan_count = int(nan_mask.sum())
        nan_indices = np.flatnonzero(nan_mask)
        logger.warning(
            "NaN scores detected in sweep: count=%d, sample_indices=%s. "
            "NaNs are treated as +inf; order should be computed after sanitization.",
            nan_count,
            nan_indices[:10],
        )
        s = np.nan_to_num(s, nan=np.inf)

    # ---- GT segments + anomaly_id mapping
    # Compute contiguous anomaly segments and a per-index anomaly id lookup.
    gt = _true_segments(tgts_b)  # (K,2)
    K = gt.shape[0]
    starts = gt[:, 0]
    ends = gt[:, 1]
    lens = ends - starts
    logger.debug("Prepared %d ground truth segments.", K)

    # anomaly_id[i] = k gives the anomaly segment containing i, or -1.
    anomaly_id = np.full(n, -1, dtype=np.int32)
    anom_start = starts.astype(np.int32, copy=False)
    anom_end = ends.astype(np.int32, copy=False)
    if K > 0:
        # Build a flat mapping by enumerating indices inside each segment.
        idx = np.concatenate([np.arange(s, e, dtype=np.int32) for s, e in zip(starts, ends)])
        anomaly_id[idx] = np.repeat(np.arange(K, dtype=np.int32), lens)
        logger.debug(
            "Assigned anomaly ids for %d points across %d segments.",
            int(np.sum(anomaly_id >= 0)),
            K
        )

    # ---- overflow boundary masks
    # Precompute boundaries where predictions can overflow into anomalies.
    is_early_right = np.zeros(n, dtype=np.uint8)  # marks s where pair (s-1,s)
    if K > 0:
        is_early_right[starts[starts > 0]] = 1

    is_late_left = np.zeros(n, dtype=np.uint8)    # marks e-1 where pair (e-1,e)
    if K > 0:
        is_late_left[(ends - 1)[ends < n]] = 1
    # Track whether each boundary has already been counted for overflow penalties.
    early_overflow_counted = np.zeros(n, dtype=np.uint8)
    late_overflow_counted = np.zeros(n, dtype=np.uint8)
    if K > 0:
        logger.debug(
            "Prepared overflow masks: early_right=%d, late_left=%d.",
            int(is_early_right.sum()),
            int(is_late_left.sum()),
        )

    # ---- per-anomaly nom stats (raw preds, not suppression)
    # alarms: number of raw predicted segments intersecting each anomaly.
    alarms = np.zeros(K, dtype=np.int32)
    # sum_w: decay-weighted count of hits within each anomaly.
    sum_w = np.zeros(K, dtype=np.float64)
    # nom: per-anomaly nominal term derived from sum_w and alarms.
    nom = np.zeros(K, dtype=np.float64)

    # ---- detected defined by alive (suppressed preds)
    # alive_in_anom: count of alive predictions per anomaly.
    alive_in_anom = np.zeros(K, dtype=np.int32)
    # detected flags if an anomaly has at least one alive prediction.
    detected = np.zeros(K, dtype=np.uint8)
    num_detected = 0
    nom_sum_detected = 0.0

    # ---- alive structures (first-hit suppression)
    # alive marks predictions still contributing after suppression.
    alive = np.zeros(n, dtype=np.uint8)
    # Fenwick tree supports efficient next-alive queries during suppression.
    fw = fenwick_create(n)

    # ---- DSU arrays
    # DSU tracks contiguous predicted segments and their metadata.
    parent = np.full(n, -1, dtype=np.int32)
    size = np.zeros(n, dtype=np.int32)
    left = np.zeros(n, dtype=np.int32)
    right = np.zeros(n, dtype=np.int32)
    preds = np.zeros(n, dtype=np.uint8)          # active flags
    batch_mark = np.zeros(n, dtype=np.int32)
    batch_id = 0
    has_tgt = np.zeros(n, dtype=np.uint8)
    first_hit = np.full(n, n, dtype=np.int32)    # INF=n

    # Counter for predicted segments that contain no target points.
    n_false_alarm_segments = 0
    # Count of active predictions (raw, before suppression).
    active_count = 0

    # Count of false-positive points (raw).
    num_fp = 0
    # Overflow counters at anomaly boundaries.
    early_overflow = 0
    late_overflow = 0

    # initial score at thr=+inf (all preds off)
    # baseline behavior: when no predictions are active, score is forced to 0.0
    # (skip fpw/segments/overflow penalties and nom term).
    best_thr = np.inf
    best_sc = 0.0  # current_score with all off

    # If K==0 and no active -> score = -fpw = 0, matches above.

    steps = 0

    i0 = 0
    while i0 < n:
        # Determine the current threshold value and the range of tied scores.
        v = s[order[i0]]
        j0 = i0
        # activate all indices with this score v (ties batch)
        while j0 < n and s[order[j0]] == v:
            j0 += 1

        # Slice of indices entering at this threshold.
        idx = order[i0:j0]
        # Count false positives among the newly activated indices.
        tgts_slice = tgts_b[idx]
        num_fp += int(np.sum(tgts_slice == 0))
        # Assign a unique batch id to detect cross-batch adjacency.
        batch_id += 1
        batch_mark[idx] = batch_id
        # Mark raw predictions as active and seed DSU metadata for each.
        preds[idx] = 1
        active_count += idx.size
        parent[idx] = idx
        size[idx] = 1
        left[idx] = idx
        right[idx] = idx
        # Initialize target presence and first-hit for new nodes.
        has_mask = tgts_slice != 0
        has_tgt[idx] = has_mask.astype(np.uint8)
        first_hit[idx] = n
        if has_mask.any():
            idx_has = idx[has_mask]
            first_hit[idx_has] = idx_has

        # Scalar per-index updates remain for neighbor-dependent DSU merges,
        # overflow checks, and suppression via Fenwick queries.
        for i in idx:
            i = int(i)

            # ---- FP
            # ---- overflow updates (pairs touching i)
            # Count each boundary once using the post-batch active set at threshold v.
            # early: (s-1, s) for s where is_early_right[s]=1
            if (
                i > 0
                and is_early_right[i] != 0
                and preds[i - 1] != 0
                and early_overflow_counted[i] == 0
            ):
                # Predicted segment crosses into an anomaly from the left.
                early_overflow += 1
                early_overflow_counted[i] = 1
            if (
                i + 1 < n
                and is_early_right[i + 1] != 0
                and preds[i + 1] != 0
                and early_overflow_counted[i + 1] == 0
            ):
                # Predicted segment crosses into an anomaly from the left (neighbor activation).
                early_overflow += 1
                early_overflow_counted[i + 1] = 1

            # late: (e-1,e) where is_late_left[e-1]=1
            if (
                i < n - 1
                and is_late_left[i] != 0
                and preds[i + 1] != 0
                and late_overflow_counted[i] == 0
            ):
                # Predicted segment crosses out of an anomaly on the right.
                late_overflow += 1
                late_overflow_counted[i] = 1
            if (
                i > 0
                and is_late_left[i - 1] != 0
                and preds[i - 1] != 0
                and late_overflow_counted[i - 1] == 0
            ):
                # Predicted segment crosses out of an anomaly on the right (neighbor activation).
                late_overflow += 1
                late_overflow_counted[i - 1] = 1

            # ---- DSU activate
            # Each newly active point starts as its own segment.
            if has_tgt[i] == 0:
                n_false_alarm_segments += 1

            root = i
            if i > 0 and preds[i - 1] != 0:
                # Merge with active left neighbor.
                root, n_false_alarm_segments = dsu_union(parent, size, left, right, has_tgt, first_hit,
                                                         n_false_alarm_segments, root, i - 1)
            if i + 1 < n and preds[i + 1] != 0:
                # Merge with active right neighbor.
                root, n_false_alarm_segments = dsu_union(parent, size, left, right, has_tgt, first_hit,
                                                         n_false_alarm_segments, root, i + 1)

            # Resolve the root and current segment metadata.
            root = dsu_find(parent, root)
            R = int(right[root])
            h = int(first_hit[root])

            # ---- raw preds anomaly stats (nom term)
            k = anomaly_id[i]
            if k >= 0:
                a = int(anom_start[k])
                b = int(anom_end[k])

                # Update decay-weighted hit count based on relative offset.
                rel = i - a
                sum_w[k] += np.ldexp(1.0, -(rel + 1))

                # Determine whether this index starts/ends a new raw alarm segment.
                prev_left = (
                    1 if i > 0 and preds[i - 1] != 0 and batch_mark[i - 1] != batch_id else 0
                )
                prev_right = (
                    1 if i + 1 < n and preds[i + 1] != 0 and batch_mark[i + 1] != batch_id else 0
                )
                if i == a or prev_left == 0:
                    # New raw predicted segment starts within the anomaly.
                    alarms[k] += 1
                if i + 1 < b and prev_right != 0:
                    # Segment continuation past this index reduces the count.
                    alarms[k] -= 1
                if alarms[k] < 0:
                    logger.warning(
                        "Negative alarms[%d] detected (a=%d, b=%d, i=%d, batch_id=%d, "
                        "prev_left=%d, prev_right=%d). Clamping to 0.",
                        int(k),
                        int(a),
                        int(b),
                        int(i),
                        int(batch_id),
                        prev_left,
                        prev_right,
                    )
                    alarms[k] = 0

                # Update nominal term; if already detected, adjust running sum.
                old_nom = nom[k]
                nom[k] = np.ldexp(1.0 + sum_w[k], -int(alarms[k]))
                if detected[k] != 0:
                    nom_sum_detected += (nom[k] - old_nom)

            # ---- first-hit suppression: alive updates
            # alive if no hit OR i <= h
            if h >= n or i <= h:
                if alive[i] == 0:
                    # Activate this point in the alive set and Fenwick tree.
                    alive[i] = 1
                    fenwick_add(fw, i, +1)
                    k2 = anomaly_id[i]
                    if k2 >= 0:
                        prev = alive_in_anom[k2]
                        alive_in_anom[k2] = prev + 1
                        if prev == 0:
                            # First alive hit for this anomaly.
                            detected[k2] = 1
                            num_detected += 1
                            nom_sum_detected += nom[k2]

            # If hit exists, suppress everything after h inside predicted segment: [h+1, R]
            if h < n:
                suppress_left = h + 1
                suppress_right = R
                if suppress_left < 0:
                    suppress_left = 0
                if suppress_right >= n:
                    suppress_right = n - 1
                if suppress_left <= suppress_right:
                    # Walk alive points in [l, r] and deactivate them.
                    cur = fenwick_next_one_at_or_after(fw, suppress_left)
                    while cur <= suppress_right:
                        if alive[cur] != 0:
                            # Remove alive points to enforce first-hit suppression.
                            alive[cur] = 0
                            fenwick_add(fw, cur, -1)
                            kk = anomaly_id[cur]
                            if kk >= 0:
                                prev = alive_in_anom[kk]
                                alive_in_anom[kk] = prev - 1
                                if prev == 1:
                                    # Lost the last alive hit for this anomaly.
                                    detected[kk] = 0
                                    num_detected -= 1
                                    nom_sum_detected -= nom[kk]
                        cur = fenwick_next_one_at_or_after(fw, cur + 1)

        i0 = j0
        steps += 1

        # ---- compute current score at thr = v (after activating all with score==v)
        # fpw
        # TODO: Confirm whether fpw should use active_count vs num_fp (see _false_positive_weights).
        fpw = 0.0
        if num_fp != 0:
            # False positive penalty grows as more false positives accumulate.
            fpw = 1.0 - 1.0 / float(num_fp)

        if active_count == 0:
            # Score is defined as 0 when no predictions are active.
            sc = 0.0
        else:
            # Base score is the number of detected anomalies.
            sc = float(num_detected)
            # Penalize false alarm segments.
            sc -= float(n_false_alarm_segments) / FALSE_ALARM_TOLERANCE
            # Penalize false positive volume.
            sc -= fpw
            # Penalize boundary overflow.
            sc -= 1.5 * float(early_overflow) / FALSE_ALARM_TOLERANCE
            sc -= 0.5 * float(late_overflow) / FALSE_ALARM_TOLERANCE
            if num_detected > 0:
                # Add nominal term averaged across detected anomalies.
                sc += float(nom_sum_detected) / float(num_detected)

        if sc > best_sc:
            # Track new best score and the corresponding threshold.
            best_sc = sc
            best_thr = float(v)
            logger.debug(
                "New best score %.6f at threshold %.6f after %d steps.",
                best_sc,
                best_thr,
                steps,
            )

    # Evaluate at thr=-inf (all active)
    # TODO: Confirm whether fpw should use active_count vs num_fp (see _false_positive_weights).
    fpw = 0.0
    if num_fp != 0:
        # Final false positive penalty with all predictions active.
        fpw = 1.0 - 1.0 / float(num_fp)

    if active_count == 0:
        sc = 0.0
    else:
        # Final score uses the same components as each sweep step.
        sc = float(num_detected)
        sc -= float(n_false_alarm_segments) / FALSE_ALARM_TOLERANCE
        sc -= fpw
        sc -= 1.5 * float(early_overflow) / FALSE_ALARM_TOLERANCE
        sc -= 0.5 * float(late_overflow) / FALSE_ALARM_TOLERANCE
        if num_detected > 0:
            sc += float(nom_sum_detected) / float(num_detected)

    if sc > best_sc:
        # Update best if the fully active threshold is superior.
        best_sc = sc
        best_thr = -np.inf
        logger.debug(
            "New best score %.6f at threshold -inf after sweep completion.",
            best_sc,
        )

    logger.debug(
        "Sweep complete with best threshold %.6f, score %.6f, steps %d.",
        best_thr,
        best_sc,
        steps,
    )
    logger.debug(
        "Final sweep stats: num_fp=%d, num_detected=%d, false_segments=%d, "
        "early_overflow=%d, late_overflow=%d.",
        num_fp,
        num_detected,
        n_false_alarm_segments,
        early_overflow,
        late_overflow,
    )
    return best_thr, best_sc, steps


# ============================================================
# Public wrapper (stable order preserved)
# ============================================================

def select_threshold(scores: np.ndarray, targets: np.ndarray) -> SweepExactResult:
    """Select the best threshold for alarm score based on targets.

    Args:
        scores (np.ndarray): Score array.
        targets (np.ndarray): Target array.

    Returns:
        SweepExactResult: Best threshold, score, and step count.
    """
    logger.debug(
        "Selecting threshold for scores shape %s and targets shape %s.",
        np.shape(scores),
        np.shape(targets),
    )
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets)
    if s.ndim != 1:
        s = s.ravel()
    if t.ndim != 1:
        t = t.ravel()
    logger.debug("Flattened scores to %s, targets to %s.", s.shape, t.shape)

    # targets strictly binary (as you stated)
    tgts_b = (t != 0).astype(np.uint8, copy=False)
    logger.debug("Converted targets to binary with %d positives.", int(tgts_b.sum()))

    # Stable descending order to preserve score order
    if np.isnan(s).any():
        nan_mask = np.isnan(s)
        nan_count = int(nan_mask.sum())
        nan_indices = np.flatnonzero(nan_mask)
        logger.warning(
            "NaN scores detected before sorting: count=%d, sample_indices=%s. "
            "Sanitizing to +inf for stable ordering.",
            nan_count,
            nan_indices[:10],
        )
        s = np.nan_to_num(s, nan=np.inf)
    order = np.argsort(s, kind="stable")[::-1].astype(np.int64, copy=False)
    logger.debug("Computed stable order for %d scores.", order.size)

    best_thr, best_sc, steps = _sweep_alarm_score_exact(s, tgts_b, order)
    logger.debug(
        "Selected threshold %.6f with score %.6f in %d steps.",
        best_thr,
        best_sc,
        steps,
    )
    return SweepExactResult(float(best_thr), float(best_sc), int(steps))


# ============================================================
# Warm-up compile (optional, recommended in notebooks)
# ============================================================

if __name__ == "__main__":
    _fenwick_self_test()
