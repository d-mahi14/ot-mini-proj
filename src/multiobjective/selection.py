"""Selection and comparative analysis operators for multi-objective Pareto frontiers.

Provides:
1. Identification of representative Pareto trade-off solutions:
   - Highest Quality (maximum utilitarian Z1)
   - Fairest (maximum Rawlsian egalitarian Z2)
   - Lightest Load (minimum peak alumni load Z3)
   - Knee Point (minimum Euclidean distance to ideal point in normalized objective space)
2. Quantitative front comparison against ground-truth Exact MILP:
   - Coverage / Dominance metric C(A, B)
   - Hypervolume gap and percentage difference
   - Generational Distance (GD) and Inverted Generational Distance (IGD)
3. Pseudonymous matching list generation and serialization.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_generator import MatchingInstance
from src.metrics import (
    compute_gini_coefficient,
    compute_hypervolume,
    compute_shared_reference_setup,
    extract_non_dominated_front,
    normalize_objectives,
)
from src.problem import MatchingSolution, StudentAlumniMatchingProblem


def select_representative_solutions(
    solutions: List[MatchingSolution],
) -> Dict[str, MatchingSolution]:
    """Identify the 4 canonical representative solutions from a Pareto front.

    Representative profiles:
    - 'highest_quality': Max utilitarian total compatibility Z1 = sum q_ij x_ij
    - 'fairest': Max Rawlsian egalitarian minimum student quality Z2 = min_i Q_i
    - 'lightest_load': Min maximum alumni workload Z3 = max_j L_j
    - 'knee_point': Best compromise point closest to ideal point in normalized objective space

    Args:
        solutions: List of MatchingSolution candidates.

    Returns:
        Dictionary mapping profile keys to selected MatchingSolution objects.

    Raises:
        ValueError: If solutions list is empty.
    """
    if not solutions:
        raise ValueError("Cannot select representative solutions from an empty solution list.")

    # Extract non-dominated subset for fair Pareto analysis
    objs = np.array([s.objectives for s in solutions])
    _, nd_indices = extract_non_dominated_front(objs)
    pareto_sols = [solutions[i] for i in nd_indices] if len(nd_indices) > 0 else solutions

    # 1. Highest Quality: Maximize Z1 (Total Match Quality)
    # Tie-breakers: higher Z2, then lower Z3
    sol_quality = max(
        pareto_sols,
        key=lambda s: (s.z1_total_quality, s.z2_worst_quality, -s.z3_max_load),
    )

    # 2. Fairest: Maximize Z2 (Worst Student Quality)
    # Tie-breakers: lower Gini coefficient, then higher Z1
    sol_fairest = max(
        pareto_sols,
        key=lambda s: (
            s.z2_worst_quality,
            -compute_gini_coefficient(s.student_qualities),
            s.z1_total_quality,
        ),
    )

    # 3. Lightest Load: Minimize Z3 (Peak Alumni Load)
    # Tie-breakers: higher Z1, then higher company entropy
    sol_lightest = min(
        pareto_sols,
        key=lambda s: (s.z3_max_load, -s.z1_total_quality, -s.company_entropy),
    )

    # 4. Knee Point: Closest to normalized ideal point (0, 0, 0)
    nd_objs = np.array([s.objectives for s in pareto_sols])  # [-Z1, -Z2, +Z3]
    f_min = np.min(nd_objs, axis=0)
    f_max = np.max(nd_objs, axis=0)
    denom = np.where((f_max - f_min) == 0, 1.0, f_max - f_min)

    # Normalized objectives in [0, 1] where 0 represents the ideal best
    norm_objs = (nd_objs - f_min) / denom
    distances = np.linalg.norm(norm_objs, axis=1)

    knee_idx = int(np.argmin(distances))
    sol_knee = pareto_sols[knee_idx]

    return {
        "highest_quality": sol_quality,
        "fairest": sol_fairest,
        "lightest_load": sol_lightest,
        "knee_point": sol_knee,
    }


def compare_fronts(
    hybrid_sols: List[MatchingSolution],
    exact_sols: List[MatchingSolution],
    ref_setup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare a heuristic/hybrid Pareto front against an exact MILP benchmark front.

    Computes:
    - Coverage metric C(Exact, Hybrid): Fraction of Hybrid points dominated by Exact MILP
    - Coverage metric C(Hybrid, Exact): Fraction of Exact points dominated by Hybrid
    - Hypervolume gap: Absolute and percentage deficit relative to Exact MILP
    - Generational Distance (GD): Average distance from Hybrid front to true MILP front
    - Inverted Generational Distance (IGD): Average distance from true MILP front to Hybrid front

    Args:
        hybrid_sols: Solutions produced by Hybrid Memetic NSGA-II.
        exact_sols: True Pareto solutions from HiGHS Exact MILP.
        ref_setup: Optional shared ideal/nadir reference frame.

    Returns:
        Dictionary of comparison metrics.
    """
    if not hybrid_sols or not exact_sols:
        return {
            "error": "Both hybrid_sols and exact_sols must be non-empty for comparison.",
            "coverage_exact_over_hybrid": 0.0,
            "coverage_hybrid_over_exact": 0.0,
            "hv_gap": 0.0,
            "hv_gap_pct": 0.0,
            "gd": 0.0,
            "igd": 0.0,
        }

    objs_h = np.array([s.objectives for s in hybrid_sols])
    objs_e = np.array([s.objectives for s in exact_sols])

    nd_h, _ = extract_non_dominated_front(objs_h)
    nd_e, _ = extract_non_dominated_front(objs_e)

    # 1. Coverage Metric C(A, B) = fraction of points in B dominated by at least one point in A
    def _compute_coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
        if front_b.shape[0] == 0:
            return 0.0
        dominated_count = 0
        for b_pt in front_b:
            is_dom = False
            for a_pt in front_a:
                if np.all(a_pt <= b_pt) and np.any(a_pt < b_pt):
                    is_dom = True
                    break
            if is_dom:
                dominated_count += 1
        return float(dominated_count / front_b.shape[0])

    cov_e_over_h = _compute_coverage(nd_e, nd_h)
    cov_h_over_e = _compute_coverage(nd_h, nd_e)

    # 2. Hypervolume Comparison
    if ref_setup is None:
        pooled = np.vstack([nd_h, nd_e])
        ref_setup = compute_shared_reference_setup(pooled)

    ideal_pt = np.array(ref_setup["ideal_point"])
    nadir_pt = np.array(ref_setup["nadir_point"])
    ref_pt = np.array(ref_setup["ref_point"])

    hv_exact = compute_hypervolume(nd_e, ideal_point=ideal_pt, nadir_point=nadir_pt, ref_point=ref_pt)
    hv_hybrid = compute_hypervolume(nd_h, ideal_point=ideal_pt, nadir_point=nadir_pt, ref_point=ref_pt)

    hv_gap = max(0.0, hv_exact - hv_hybrid)
    hv_gap_pct = round((hv_gap / max(1e-6, hv_exact)) * 100.0, 2)

    # 3. Normalized Generational Distance (GD) & Inverted Generational Distance (IGD)
    norm_h = normalize_objectives(nd_h, ideal_pt, nadir_pt)
    norm_e = normalize_objectives(nd_e, ideal_pt, nadir_pt)

    # GD: from Hybrid to closest Exact point
    gd_dists = [min(np.linalg.norm(h - norm_e, axis=1)) for h in norm_h]
    gd = float(np.mean(gd_dists)) if gd_dists else 0.0

    # IGD: from Exact to closest Hybrid point
    igd_dists = [min(np.linalg.norm(e - norm_h, axis=1)) for e in norm_e]
    igd = float(np.mean(igd_dists)) if igd_dists else 0.0

    return {
        "num_hybrid_points": int(nd_h.shape[0]),
        "num_exact_points": int(nd_e.shape[0]),
        "hv_exact": round(hv_exact, 6),
        "hv_hybrid": round(hv_hybrid, 6),
        "hv_gap": round(hv_gap, 6),
        "hv_gap_pct": hv_gap_pct,
        "coverage_exact_over_hybrid": round(cov_e_over_h, 4),
        "coverage_hybrid_over_exact": round(cov_h_over_e, 4),
        "generational_distance_gd": round(gd, 5),
        "inverted_generational_distance_igd": round(igd, 5),
    }


def export_pseudonymous_matching_table(
    instance: MatchingInstance,
    solution: MatchingSolution,
    output_path: str | Path,
) -> pd.DataFrame:
    """Export the student-alumni matching roster using pseudonymous IDs only.

    Includes student branch, domain, GPA, paired alumni IDs, alumni branches,
    alumni companies, and realized bilateral match qualities.
    """
    rows: List[Dict[str, Any]] = []
    x = solution.x  # Shape (N, M)
    q_mat = instance.q_matrix

    for i, stu in enumerate(instance.students):
        assigned_alumni_idx = np.where(x[i] > 0)[0]
        row: Dict[str, Any] = {
            "student_id": stu.id,
            "student_branch": stu.branch,
            "student_target_domain": stu.target_domain,
            "student_gpa": stu.gpa,
            "total_match_quality_Qi": round(float(solution.student_qualities[i]), 4),
            "meets_quality_floor": bool(solution.student_qualities[i] >= instance.theta_min_quality),
        }

        # List paired mentors
        for m_slot in range(instance.k_matches):
            if m_slot < len(assigned_alumni_idx):
                alm_idx = assigned_alumni_idx[m_slot]
                alm = instance.alumni[alm_idx]
                q_val = round(float(q_mat[i, alm_idx]), 4)
                row[f"mentor_{m_slot+1}_id"] = alm.id
                row[f"mentor_{m_slot+1}_branch"] = alm.branch
                row[f"mentor_{m_slot+1}_company"] = alm.company
                row[f"mentor_{m_slot+1}_quality"] = q_val
            else:
                row[f"mentor_{m_slot+1}_id"] = "UNASSIGNED"
                row[f"mentor_{m_slot+1}_branch"] = "N/A"
                row[f"mentor_{m_slot+1}_company"] = "N/A"
                row[f"mentor_{m_slot+1}_quality"] = 0.0

        # Verify employer diversity compliance
        matched_companies = [instance.alumni[j].company for j in assigned_alumni_idx]
        row["employer_diversity_valid"] = bool(len(matched_companies) == len(set(matched_companies)))

        rows.append(row)

    df = pd.DataFrame(rows)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[Selection] Saved pseudonymous matching list to: {out_file}")
    return df
