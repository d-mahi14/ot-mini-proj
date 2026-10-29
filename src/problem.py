"""Problem definition and evaluation for Multi-Objective Student-Alumni Matching.

Formulates the assignment problem for pymoo and mathematical solvers:
- Objectives:
    min f1 = -Z1 = - sum_{i,j} q_{ij} x_{ij} (Utilitarian Quality)
    min f2 = -Z2 = - min_i Q_i (Rawlsian Fairness / Worst-off student)
    min f3 = +Z3 = + max_j L_j (Workload Balance / Maximum Alumni Load)
- Constraints:
    (C1) sum_j x_{ij} = K for all i
    (C2) sum_i x_{ij} <= C_j y_j for all j
    (C3) x_{ij} <= c_j for all i,j
    (C4) x_{ij} = 0 if q_{ij} < q_min
    (C5) Q_i >= theta for all i
    (C6) |avgQ_b - avgQ_b'| <= delta for all branch pairs (b, b')
    (C7) Diversity cap on matches per company
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.core.problem import Problem

from src.data_generator import MatchingInstance


@dataclass
class MatchingSolution:
    """Represents a candidate or finalized matching assignment."""

    x: np.ndarray  # Binary assignment matrix (N, M)
    y: np.ndarray  # Binary alumni activation vector (M,)
    student_qualities: np.ndarray  # Vector of length N: Q_i = sum_j q_ij x_ij
    alumni_loads: np.ndarray  # Vector of length M: L_j = sum_i x_ij
    z1_total_quality: float  # sum_i Q_i
    z2_worst_quality: float  # min_i Q_i
    z3_max_load: int  # max_j L_j
    objectives: np.ndarray  # [-Z1, -Z2, Z3] for minimization
    capacity_violations: float
    consent_violations: int
    quality_threshold_violations: int
    quality_floor_violations: float
    branch_disparity: float
    branch_disparity_violation: float
    diversity_violations: int
    degree_violations: int
    total_constraint_violation: float
    is_feasible: bool
    branch_avg_qualities: Dict[str, float] = field(default_factory=dict)
    active_alumni_count: int = 0
    company_entropy: float = 0.0

    def to_summary(self) -> Dict[str, Any]:
        """Return a readable dictionary of performance and fairness metrics."""
        return {
            "Z1_total_quality": round(self.z1_total_quality, 4),
            "avg_student_quality": round(float(np.mean(self.student_qualities)), 4),
            "Z2_worst_quality": round(self.z2_worst_quality, 4),
            "Z3_max_load": int(self.z3_max_load),
            "active_alumni": int(self.active_alumni_count),
            "is_feasible": bool(self.is_feasible),
            "total_constraint_violation": round(float(self.total_constraint_violation), 5),
            "branch_disparity": round(float(self.branch_disparity), 4),
            "company_entropy": round(float(self.company_entropy), 4),
        }


class StudentAlumniMatchingProblem(Problem):
    """Pymoo Problem for Fair and Load-Balanced Multi-Objective Matching.

    Representation:
        An integer decision vector of length N * K.
        Reshaped to (N, K), the entry at (i, k) is the alumni index j in {0, ..., M-1}
        assigned to student i.
    """

    def __init__(
        self,
        instance: MatchingInstance,
        penalty_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize problem from MatchingInstance.

        Args:
            instance: Synthetic or benchmark MatchingInstance.
            penalty_weights: Optional weighting for individual constraint violations.
        """
        self.instance = instance
        self.n_students = instance.n_students
        self.n_alumni = instance.n_alumni
        self.k_matches = instance.k_matches
        self.q_matrix = instance.q_matrix
        self.q_min = instance.q_min
        self.theta = instance.theta_min_quality
        self.delta = instance.delta_branch_parity
        self.max_company_share = instance.max_company_share

        # Cache alumni metadata vectors
        self.capacities = np.array([a.capacity for a in instance.alumni], dtype=int)
        self.consents = np.array([a.consent for a in instance.alumni], dtype=int)
        self.alumni_companies = [a.company for a in instance.alumni]
        self.student_branches = [s.branch for s in instance.students]
        self.unique_branches = sorted(list(set(self.student_branches)))

        # Precompute student indices per branch
        self.branch_to_student_indices: Dict[str, np.ndarray] = {
            b: np.array([i for i, br in enumerate(self.student_branches) if br == b], dtype=int)
            for b in self.unique_branches
        }

        # Chromosome shape: length N * K
        n_var = self.n_students * self.k_matches
        # Objectives: 3 (min -Z1, min -Z2, min +Z3)
        # Constraints: 4 aggregated inequality constraints (G <= 0):
        #   g1: Capacity overflows
        #   g2: Consent and quality threshold violations
        #   g3: Quality floor shortfall (theta - Q_i)
        #   g4: Branch parity disparity (disparity - delta)
        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=4,
            xl=np.zeros(n_var, dtype=int),
            xu=np.full(n_var, self.n_alumni - 1, dtype=int),
            vtype=int,
        )

        self.penalty_weights = penalty_weights or {
            "capacity": 10.0,
            "consent": 20.0,
            "q_min": 15.0,
            "theta": 8.0,
            "branch_parity": 12.0,
            "duplicate": 25.0,
            "diversity": 5.0,
        }

    def decode_chromosome(self, x_row: np.ndarray) -> np.ndarray:
        """Convert a 1D integer chromosome of length N*K to an (N, M) binary matrix."""
        assigned = x_row.reshape((self.n_students, self.k_matches))
        x_bin = np.zeros((self.n_students, self.n_alumni), dtype=int)
        for i in range(self.n_students):
            for j in assigned[i]:
                x_bin[i, int(j)] += 1
        return x_bin

    def encode_solution(self, x_bin: np.ndarray) -> np.ndarray:
        """Convert an (N, M) binary matrix to a 1D integer chromosome of length N*K."""
        arr = np.zeros((self.n_students, self.k_matches), dtype=int)
        for i in range(self.n_students):
            matched = np.where(x_bin[i] > 0)[0]
            if len(matched) >= self.k_matches:
                arr[i] = matched[:self.k_matches]
            elif len(matched) > 0:
                # Pad if partial
                arr[i, :len(matched)] = matched
                arr[i, len(matched):] = matched[-1]
            else:
                arr[i] = np.random.choice(self.n_alumni, size=self.k_matches, replace=False)
        return arr.flatten()

    def evaluate_solution(self, x_bin: np.ndarray) -> MatchingSolution:
        """Compute full solution metrics, objectives, and constraint violations for an assignment."""
        # Active alumni vector y_j
        loads = np.sum(x_bin, axis=0)
        y = (loads > 0).astype(int)

        # Student total match quality Q_i
        student_qualities = np.sum(self.q_matrix * x_bin, axis=1)

        # Objectives
        z1 = float(np.sum(student_qualities))
        z2 = float(np.min(student_qualities))
        z3 = int(np.max(loads)) if len(loads) > 0 else 0
        objectives = np.array([-z1, -z2, float(z3)])

        # (C1) Degree violations (each student must have exactly K unique matches)
        student_degrees = np.sum(x_bin, axis=1)
        degree_violations = int(np.sum(np.abs(student_degrees - self.k_matches)))
        # Duplicate matches count if x_bin has entries > 1
        duplicates = int(np.sum(np.maximum(0, x_bin - 1)))

        # (C2) Capacity violations
        cap_overflow = np.maximum(0, loads - self.capacities)
        cap_violation_sum = float(np.sum(cap_overflow))

        # (C3) Consent violations
        # Matches assigned to non-consenting alumni
        non_consenting = np.where(self.consents == 0)[0]
        consent_violations = int(np.sum(x_bin[:, non_consenting]))

        # (C4) Quality threshold violations (q_ij < q_min)
        invalid_mask = (self.q_matrix < self.q_min) & (x_bin > 0)
        q_min_violations = int(np.sum(invalid_mask))

        # (C5) Quality floor violations: theta - Q_i
        floor_shortfalls = np.maximum(0.0, self.theta - student_qualities)
        theta_violation_sum = float(np.sum(floor_shortfalls))

        # (C6) Branch parity disparity: max_b avg(Q_b) - min_b avg(Q_b)
        branch_avgs: Dict[str, float] = {}
        for b, indices in self.branch_to_student_indices.items():
            if len(indices) > 0:
                branch_avgs[b] = float(np.mean(student_qualities[indices]))
            else:
                branch_avgs[b] = 0.0

        if len(branch_avgs) > 1:
            vals = list(branch_avgs.values())
            branch_disparity = float(max(vals) - min(vals))
        else:
            branch_disparity = 0.0
        disparity_violation = max(0.0, branch_disparity - self.delta)

        # (C7) Diversity cap per company
        diversity_violations = 0
        for i in range(self.n_students):
            assigned_alumni = np.where(x_bin[i] > 0)[0]
            if len(assigned_alumni) > 1:
                comp_counts: Dict[str, int] = {}
                for j in assigned_alumni:
                    c = self.alumni_companies[j]
                    comp_counts[c] = comp_counts.get(c, 0) + int(x_bin[i, j])
                for cnt in comp_counts.values():
                    if cnt > self.max_company_share:
                        diversity_violations += (cnt - self.max_company_share)

        # Aggregate total constraint violation
        total_cv = (
            cap_violation_sum
            + consent_violations * 2.0
            + q_min_violations * 1.5
            + theta_violation_sum
            + disparity_violation * 2.0
            + degree_violations * 3.0
            + duplicates * 5.0
            + diversity_violations * 1.0
        )
        is_feasible = (
            cap_violation_sum == 0.0
            and consent_violations == 0
            and q_min_violations == 0
            and theta_violation_sum == 0.0
            and disparity_violation == 0.0
            and degree_violations == 0
            and duplicates == 0
            and diversity_violations == 0
        )

        # Shannon Entropy of Company Representation across active assignments
        company_counts: Dict[str, int] = {}
        total_assignments = int(np.sum(x_bin))
        for j in range(self.n_alumni):
            if loads[j] > 0:
                comp = self.alumni_companies[j]
                company_counts[comp] = company_counts.get(comp, 0) + int(loads[j])

        if total_assignments > 0 and len(company_counts) > 0:
            probs = np.array(list(company_counts.values())) / float(total_assignments)
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        return MatchingSolution(
            x=x_bin,
            y=y,
            student_qualities=student_qualities,
            alumni_loads=loads,
            z1_total_quality=z1,
            z2_worst_quality=z2,
            z3_max_load=z3,
            objectives=objectives,
            capacity_violations=cap_violation_sum,
            consent_violations=consent_violations,
            quality_threshold_violations=q_min_violations,
            quality_floor_violations=theta_violation_sum,
            branch_disparity=branch_disparity,
            branch_disparity_violation=disparity_violation,
            diversity_violations=diversity_violations,
            degree_violations=degree_violations + duplicates,
            total_constraint_violation=float(total_cv),
            is_feasible=is_feasible,
            branch_avg_qualities=branch_avgs,
            active_alumni_count=int(np.sum(y)),
            company_entropy=entropy,
        )

    def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Pymoo interface to evaluate a population of candidate chromosomes."""
        n_candidates = X.shape[0]
        F = np.zeros((n_candidates, self.n_obj))
        G = np.zeros((n_candidates, self.n_ieq_constr))

        for idx in range(n_candidates):
            x_bin = self.decode_chromosome(X[idx])
            sol = self.evaluate_solution(x_bin)
            F[idx, :] = sol.objectives

            # Pymoo treats g <= 0 as satisfied constraint
            # g1: Capacity violations
            g1 = sol.capacity_violations
            # g2: Consent and quality threshold violations + duplicates
            g2 = float(sol.consent_violations + sol.quality_threshold_violations + sol.degree_violations)
            # g3: Minimum quality floor shortfall
            g3 = sol.quality_floor_violations
            # g4: Branch disparity violation
            g4 = sol.branch_disparity_violation

            G[idx, 0] = g1
            G[idx, 1] = g2
            G[idx, 2] = g3
            G[idx, 3] = g4

        out["F"] = F
        out["G"] = G

def calibrate_thresholds(
    instance,
    theta_fraction: float = 0.85,
    delta_fraction: float = 1.2,
    output_path=None,
):
    """Calibrate theta (quality floor) and delta (branch parity cap) from data.

    Strategy
    --------
    theta = theta_fraction * Z2_greedy
        Z2_greedy is the worst-student quality produced by the greedy baseline
        on an *unconstrained* copy of the instance (theta=0, delta=inf).
        Using fraction < 1 guarantees the greedy solution is still feasible
        under the calibrated theta.

    delta = delta_fraction * spread_greedy
        spread_greedy is max_b(avgQ_b) - min_b(avgQ_b) from the same
        unconstrained greedy solution.  Using fraction > 1 adds slack so the
        greedy solution itself remains feasible, while still bounding worse
        solutions.

    Parameters
    ----------
    instance        : MatchingInstance to calibrate against.
    theta_fraction  : Fraction of greedy Z2 to use as theta (default 0.85).
    delta_fraction  : Multiplier on greedy branch spread to use as delta
                      (default 1.2, i.e. 20 pct slack above greedy baseline).
    output_path     : If given, save calibration report as JSON to this path.

    Returns
    -------
    dict with keys: theta, delta, greedy_z2, greedy_spread,
    theta_fraction, delta_fraction, instance_name.

    Raises
    ------
    ValueError if fractions are non-positive or greedy produces zero Z2.
    """
    import json
    from pathlib import Path as _Path
    from src.data_generator import MatchingInstance as _MI
    from src.problem import StudentAlumniMatchingProblem
    from src.baseline.greedy_baseline import GreedyBaseline
    from typing import Optional, Dict

    if theta_fraction <= 0 or delta_fraction <= 0:
        raise ValueError(
            f"theta_fraction and delta_fraction must be positive; "
            f"got {theta_fraction}, {delta_fraction}"
        )

    # Build unconstrained variant (no floor, no parity cap)
    unconstrained = _MI(
        name=instance.name,
        n_students=instance.n_students,
        n_alumni=instance.n_alumni,
        k_matches=instance.k_matches,
        q_matrix=instance.q_matrix,
        q_min=instance.q_min,
        theta_min_quality=0.0,
        delta_branch_parity=1e9,
        max_company_share=instance.max_company_share,
        students=instance.students,
        alumni=instance.alumni,
    )

    prob_unconstrained = StudentAlumniMatchingProblem(unconstrained)
    greedy_sol = GreedyBaseline(prob_unconstrained).solve()

    if greedy_sol.z2_worst_quality <= 0.0:
        raise ValueError(
            "Greedy baseline produced zero worst-student quality on the "
            "unconstrained instance -- cannot calibrate theta safely."
        )

    # Calibrate theta
    theta = theta_fraction * greedy_sol.z2_worst_quality

    # Calibrate delta from branch spread
    branch_avgs = greedy_sol.branch_avg_qualities
    if len(branch_avgs) >= 2:
        vals = list(branch_avgs.values())
        greedy_spread = float(max(vals) - min(vals))
    else:
        greedy_spread = 0.0

    delta = delta_fraction * greedy_spread if greedy_spread > 0.0 else 1e9

    result = {
        "instance_name": instance.name,
        "theta": round(theta, 6),
        "delta": round(delta, 6),
        "greedy_z2": round(greedy_sol.z2_worst_quality, 6),
        "greedy_spread": round(greedy_spread, 6),
        "theta_fraction": theta_fraction,
        "delta_fraction": delta_fraction,
    }

    if output_path is not None:
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)

    return result
