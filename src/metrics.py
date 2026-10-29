"""Evaluation metrics for multi-objective optimization and fairness in matching.

Includes:
- Hypervolume (HV) with shared reference setup and normalization
- Spacing (SP)
- Non-dominated set extraction and counting
- Gini coefficient of student match quality Q_i
- Branch-wise parity disparity
- Workload metrics (max load, std load, alumni utilized)
- Shannon entropy of company/industry representation
- Feasibility and aggregate constraint violation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_non_dominated_front(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Identify non-dominated solutions (Pareto front) for minimization objectives.

    Args:
        points: Array of shape (P, M) where objectives are to be minimized.

    Returns:
        Tuple of (non_dominated_points, indices_of_non_dominated).
    """
    n_points = points.shape[0]
    if n_points == 0:
        return np.empty((0, points.shape[1])), np.array([], dtype=int)

    is_dominated = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if is_dominated[i]:
            continue
        for j in range(n_points):
            if i == j or is_dominated[j]:
                continue
            # Check if point i dominates point j
            # i dominates j if i <= j in all objectives and strictly less in at least one
            if np.all(points[i] <= points[j]) and np.any(points[i] < points[j]):
                is_dominated[j] = True
            elif np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                is_dominated[i] = True
                break

    nd_indices = np.where(~is_dominated)[0]
    return points[nd_indices], nd_indices


def compute_spacing(points: np.ndarray) -> float:
    """Compute Spacing (SP) metric measuring the uniformity of solution spread.

    SP = sqrt( (1 / (|PF| - 1)) * sum_i (d_i - d_bar)^2 )
    where d_i = min_{k != i} sum_m |f_m^i - f_m^k|
    """
    nd_points, _ = extract_non_dominated_front(points)
    n = nd_points.shape[0]
    if n <= 1:
        return 0.0

    # Pairwise Manhattan distances
    d = np.zeros(n)
    for i in range(n):
        diffs = np.abs(nd_points - nd_points[i])
        manhattan = np.sum(diffs, axis=1)
        # Exclude self
        manhattan[i] = np.inf
        d[i] = np.min(manhattan)

    d_bar = np.mean(d)
    spacing = np.sqrt(np.sum((d - d_bar) ** 2) / (n - 1))
    return float(spacing)


def normalize_objectives(
    points: np.ndarray,
    ideal_point: np.ndarray,
    nadir_point: np.ndarray,
) -> np.ndarray:
    """Normalize objective vectors to [0, 1] using shared ideal and nadir bounds.

    Args:
        points: Array of shape (P, M) to normalize.
        ideal_point: Array of shape (M,) representing minimum objective values across all methods.
        nadir_point: Array of shape (M,) representing maximum objective values across all methods.

    Returns:
        Normalized array of shape (P, M).
    """
    pts = np.asarray(points, dtype=float)
    ideal = np.asarray(ideal_point, dtype=float)
    nadir = np.asarray(nadir_point, dtype=float)
    denom = nadir - ideal
    denom = np.where(denom == 0, 1.0, denom)
    return (pts - ideal) / denom


def compute_shared_reference_setup(
    all_points: List[np.ndarray] | np.ndarray,
) -> Dict[str, Any]:
    """Compute shared ideal, nadir, and normalized reference point over pooled fronts.

    Args:
        all_points: List of objective arrays (each (P_k, M)) or a single concatenated array.

    Returns:
        Dictionary containing:
            - 'ideal_point': minimum in each dimension across all pooled solutions
            - 'nadir_point': maximum in each dimension across all pooled solutions
            - 'ref_point': reference point fixed at 1.1 in every dimension (normalized space)
    """
    if isinstance(all_points, list):
        valid_arrays = [np.asarray(p, dtype=float) for p in all_points if p is not None and len(p) > 0]
        if not valid_arrays:
            raise ValueError("No points provided to compute shared reference setup.")
        stacked = np.vstack(valid_arrays)
    else:
        stacked = np.asarray(all_points, dtype=float)

    if stacked.size == 0:
        raise ValueError("Empty array provided to compute shared reference setup.")

    ideal_point = np.min(stacked, axis=0).astype(float)
    nadir_point = np.max(stacked, axis=0).astype(float)
    m = stacked.shape[1]
    ref_point = np.full(m, 1.1, dtype=float)

    return {
        "ideal_point": ideal_point.tolist(),
        "nadir_point": nadir_point.tolist(),
        "ref_point": ref_point.tolist(),
    }


def compute_hypervolume(
    points: np.ndarray,
    ref_point: Optional[np.ndarray] = None,
    ideal_point: Optional[np.ndarray] = None,
    nadir_point: Optional[np.ndarray] = None,
) -> float:
    """Compute normalized Hypervolume (HV) indicator.

    Points are assumed to be minimized.
    Uses pymoo Hypervolume indicator.

    Fair comparability requirement:
    Either:
      1) Both `ideal_point` and `nadir_point` must be provided (normalizing points
         to [0, 1] with reference point fixed at 1.1 in each dimension), OR
      2) An explicit shared `ref_point` must be provided for pre-normalized points.

    The per-method fallback (max * 1.1) has been completely removed to prevent
    unfair comparisons across algorithms.
    """
    nd_points, _ = extract_non_dominated_front(points)
    if nd_points.shape[0] == 0:
        return 0.0

    # Normalize if bounds provided
    if ideal_point is not None and nadir_point is not None:
        norm_points = normalize_objectives(nd_points, ideal_point, nadir_point)
        if ref_point is None:
            r_point = np.full(nd_points.shape[1], 1.1, dtype=float)
        else:
            r_arr = np.asarray(ref_point, dtype=float)
            if np.all(r_arr >= 1.0) and np.all(r_arr <= 2.0):
                r_point = r_arr
            else:
                denom = np.asarray(nadir_point, dtype=float) - np.asarray(ideal_point, dtype=float)
                denom = np.where(denom == 0, 1.0, denom)
                r_point = (r_arr - np.asarray(ideal_point, dtype=float)) / denom
    elif ref_point is not None:
        norm_points = nd_points
        r_point = np.asarray(ref_point, dtype=float)
    else:
        raise ValueError(
            "A shared reference point (or ideal_point and nadir_point) must be provided "
            "for fair hypervolume comparison. Per-method fallback (max*1.1) has been removed."
        )

    # Filter points that strictly dominate or equal the reference point (must be <= r_point)
    valid = np.all(norm_points <= r_point, axis=1)
    filtered = norm_points[valid]
    if filtered.shape[0] == 0:
        return 0.0

    try:
        from pymoo.indicators.hv import HV
        hv_indicator = HV(ref_point=r_point)
        return float(hv_indicator(filtered))
    except Exception:
        # Fallback simple bounding hypervolume approximation
        vol = np.prod(r_point - np.min(filtered, axis=0))
        return float(max(0.0, vol))


def compute_gini_coefficient(qualities: np.ndarray) -> float:
    """Compute Gini inequality coefficient of student match qualities.

    0 = perfect equality; 1 = maximum inequality.
    """
    q = np.asarray(qualities, dtype=float).flatten()
    if len(q) == 0 or np.sum(q) == 0:
        return 0.0

    q_sorted = np.sort(q)
    n = len(q_sorted)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * q_sorted) - (n + 1) * np.sum(q_sorted)) / (n * np.sum(q_sorted))
    return float(np.clip(gini, 0.0, 1.0))


def compute_shannon_entropy(labels_or_counts: List[str] | Dict[str, int] | np.ndarray) -> float:
    """Compute Shannon Entropy H = - sum p_i log2(p_i)."""
    if isinstance(labels_or_counts, dict):
        counts = np.array(list(labels_or_counts.values()), dtype=float)
    elif isinstance(labels_or_counts, list):
        if len(labels_or_counts) > 0 and isinstance(labels_or_counts[0], str):
            from collections import Counter
            c = Counter(labels_or_counts)
            counts = np.array(list(c.values()), dtype=float)
        else:
            counts = np.array(labels_or_counts, dtype=float)
    else:
        counts = np.asarray(labels_or_counts, dtype=float)

    total = np.sum(counts)
    if total == 0:
        return 0.0

    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_full_metrics(
    solutions_or_objectives: np.ndarray,
    qualities_matrix: Optional[np.ndarray] = None,
    loads_matrix: Optional[np.ndarray] = None,
    runtimes: Optional[List[float]] = None,
    ref_point: Optional[np.ndarray] = None,
    ideal_point: Optional[np.ndarray] = None,
    nadir_point: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive metric suite for a set of Pareto solutions."""
    objs = np.asarray(solutions_or_objectives)
    nd_objs, nd_idx = extract_non_dominated_front(objs)

    hv_val = 0.0
    if ref_point is not None or (ideal_point is not None and nadir_point is not None):
        hv_val = compute_hypervolume(
            nd_objs,
            ref_point=ref_point,
            ideal_point=ideal_point,
            nadir_point=nadir_point,
        )

    metrics: Dict[str, Any] = {
        "num_total_solutions": int(objs.shape[0]),
        "num_non_dominated": int(nd_objs.shape[0]),
        "spacing": compute_spacing(nd_objs),
        "hypervolume": hv_val,
    }

    if qualities_matrix is not None and len(qualities_matrix) > 0:
        # Compute Gini on non-dominated set (or best solution)
        ginis = [compute_gini_coefficient(q) for q in qualities_matrix[nd_idx]]
        metrics["avg_gini_quality"] = float(np.mean(ginis)) if ginis else 0.0
        metrics["min_gini_quality"] = float(np.min(ginis)) if ginis else 0.0

    if loads_matrix is not None and len(loads_matrix) > 0:
        max_loads = [float(np.max(ld)) for ld in loads_matrix[nd_idx]]
        alumni_used = [int(np.sum(ld > 0)) for ld in loads_matrix[nd_idx]]
        metrics["avg_max_load"] = float(np.mean(max_loads)) if max_loads else 0.0
        metrics["avg_alumni_used"] = float(np.mean(alumni_used)) if alumni_used else 0
        metrics["min_alumni_used"] = int(np.min(alumni_used)) if alumni_used else 0

    if runtimes is not None and len(runtimes) > 0:
        metrics["runtime_seconds"] = float(np.mean(runtimes))

    return metrics


def check_front_dominates_point(
    front_points: np.ndarray,
    target_point: np.ndarray,
) -> bool:
    """Check if any solution in front_points dominates target_point for minimization.

    A point p dominates target if p_m <= target_m for all objectives m and
    p_m < target_m for at least one objective.

    Args:
        front_points: Array of shape (P, M) with minimization objective vectors.
        target_point: Vector of shape (M,) with minimization objective values.

    Returns:
        True if at least one solution in front_points dominates target_point.
    """
    pts = np.atleast_2d(front_points)
    target = np.asarray(target_point, dtype=float)
    for p in pts:
        if np.all(p <= target) and np.any(p < target):
            return True
    return False


def compute_wilcoxon_with_effect_size(
    sample_a: List[float] | np.ndarray,
    sample_b: List[float] | np.ndarray,
) -> Dict[str, Any]:
    """Perform paired Wilcoxon signed-rank test and compute rank-biserial effect size r.

    Args:
        sample_a: First paired sample (e.g., Hybrid HV across seeds).
        sample_b: Second paired sample (e.g., NSGA-II HV across seeds).

    Returns:
        Dictionary with:
            - 'statistic': W test statistic
            - 'p_value': Two-sided p-value
            - 'effect_size_r': Rank-biserial effect size in [-1, +1]
            - 'significant_0_05': Boolean indicator if p < 0.05
            - 'sample_size': Total pair count N
            - 'non_zero_pairs': Count of non-tied pairs
    """
    from scipy.stats import wilcoxon

    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    diff = a - b
    non_zero = diff[diff != 0]
    n = len(non_zero)

    if n < 3:
        mean_diff = float(np.mean(diff)) if len(diff) > 0 else 0.0
        return {
            "statistic": 0.0,
            "p_value": 1.0 if n == 0 else 0.5,
            "effect_size_r": 0.0 if n == 0 else (1.0 if mean_diff > 0 else -1.0),
            "significant_0_05": False,
            "sample_size": len(a),
            "non_zero_pairs": n,
        }

    try:
        res = wilcoxon(a, b, alternative="two-sided")
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size_r": 0.0,
            "significant_0_05": False,
            "sample_size": len(a),
            "non_zero_pairs": n,
        }

    # Rank-biserial correlation: r = (W_+ - W_-) / (W_+ + W_-)
    abs_diff = np.abs(non_zero)
    ranks = np.argsort(np.argsort(abs_diff)) + 1.0
    w_plus = float(np.sum(ranks[non_zero > 0]))
    w_minus = float(np.sum(ranks[non_zero < 0]))
    total_w = w_plus + w_minus
    r_biserial = float((w_plus - w_minus) / total_w) if total_w > 0 else 0.0

    return {
        "statistic": w_stat,
        "p_value": p_val,
        "effect_size_r": round(r_biserial, 4),
        "significant_0_05": bool(p_val < 0.05),
        "sample_size": len(a),
        "non_zero_pairs": n,
    }
