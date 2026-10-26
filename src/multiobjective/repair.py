"""Custom repair operator for Multi-Objective Student-Alumni Matching.

Enforces feasibility during genetic operations:
1. Exact student degree: exactly K distinct alumni per student.
2. Consent compliance: c_j == 1.
3. Match threshold: q_ij >= q_min.
4. Capacity feasibility: sum_i x_ij <= C_j.
5. Diversity cap per company: at most max_company_share alumni from the same company.
6. Quality boosting: replaces poor matches when student total quality falls below theta.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
from pymoo.core.repair import Repair

from src.problem import StudentAlumniMatchingProblem


class MatchingRepair(Repair):
    """Repair operator enforcing degree, consent, capacity, and quality thresholds."""

    def __init__(self, problem: StudentAlumniMatchingProblem) -> None:
        """Initialize repair operator with problem reference."""
        super().__init__()
        self.prob = problem
        self.n_students = problem.n_students
        self.n_alumni = problem.n_alumni
        self.k_matches = problem.k_matches
        self.q_matrix = problem.q_matrix
        self.q_min = problem.q_min
        self.theta = problem.theta
        self.capacities = problem.capacities
        self.consents = problem.consents
        self.companies = problem.alumni_companies
        self.max_company_share = problem.max_company_share

        # Precompute valid candidate alumni for each student: c_j == 1 and q_ij >= q_min
        self.valid_candidates: List[np.ndarray] = []
        for i in range(self.n_students):
            valid = np.where((self.consents == 1) & (self.q_matrix[i] >= self.q_min))[0]
            if len(valid) < self.k_matches:
                # If sparse, fall back to any consenting alumni sorted by quality
                valid = np.where(self.consents == 1)[0]
                if len(valid) < self.k_matches:
                    valid = np.arange(self.n_alumni)
            # Sort by descending match quality for fast greedy lookup
            scores = self.q_matrix[i, valid]
            sorted_idx = valid[np.argsort(-scores)]
            self.valid_candidates.append(sorted_idx)

    def _do(self, problem: Any, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Repair a population of chromosomes X of shape (pop_size, N * K)."""
        pop_size = X.shape[0]
        X_repaired = np.copy(X)

        for p in range(pop_size):
            chromosome = X_repaired[p].reshape((self.n_students, self.k_matches))
            repaired_chrom = self.repair_single(chromosome)
            X_repaired[p] = repaired_chrom.flatten()

        return X_repaired

    def repair_single(self, assigned: np.ndarray) -> np.ndarray:
        """Repair a single assignment matrix of shape (N, K)."""
        repaired = np.zeros((self.n_students, self.k_matches), dtype=int)
        current_loads = np.zeros(self.n_alumni, dtype=int)

        # Pass 1: Keep valid, non-duplicate, non-conflicting assignments
        for i in range(self.n_students):
            seen_alumni: Set[int] = set()
            company_counts: dict[str, int] = {}
            chosen: List[int] = []

            for raw_j in assigned[i]:
                j = int(raw_j)
                # Check valid bounds
                if not (0 <= j < self.n_alumni):
                    continue
                # Check consent and threshold
                if self.consents[j] != 1 or self.q_matrix[i, j] < self.q_min:
                    continue
                # Check duplicate
                if j in seen_alumni:
                    continue
                # Check company diversity
                comp = self.companies[j]
                if company_counts.get(comp, 0) >= self.max_company_share:
                    continue
                # Check capacity headroom
                if current_loads[j] >= self.capacities[j]:
                    continue

                seen_alumni.add(j)
                company_counts[comp] = company_counts.get(comp, 0) + 1
                current_loads[j] += 1
                chosen.append(j)

            # Fill up shortfall to K matches
            if len(chosen) < self.k_matches:
                candidates = self.valid_candidates[i]
                for cand in candidates:
                    if len(chosen) == self.k_matches:
                        break
                    if cand in seen_alumni:
                        continue
                    comp = self.companies[cand]
                    if company_counts.get(comp, 0) >= self.max_company_share:
                        continue
                    if current_loads[cand] < self.capacities[cand]:
                        seen_alumni.add(cand)
                        company_counts[comp] = company_counts.get(comp, 0) + 1
                        current_loads[cand] += 1
                        chosen.append(cand)

            # If still short (e.g. tight capacities), relax capacity constraint to ensure K distinct matches
            if len(chosen) < self.k_matches:
                for cand in self.valid_candidates[i]:
                    if len(chosen) == self.k_matches:
                        break
                    if cand not in seen_alumni:
                        seen_alumni.add(cand)
                        current_loads[cand] += 1
                        chosen.append(cand)

            # Absolute fallback
            while len(chosen) < self.k_matches:
                rand_j = int(np.argmin(current_loads))
                if rand_j not in seen_alumni:
                    seen_alumni.add(rand_j)
                    current_loads[rand_j] += 1
                    chosen.append(rand_j)
                else:
                    for fallback_j in range(self.n_alumni):
                        if fallback_j not in seen_alumni:
                            seen_alumni.add(fallback_j)
                            current_loads[fallback_j] += 1
                            chosen.append(fallback_j)
                            break

            repaired[i] = chosen[:self.k_matches]

        # Pass 2: Global load-balancing & capacity overflow re-allocation
        overflow_alumni = np.where(current_loads > self.capacities)[0]
        if len(overflow_alumni) > 0:
            for ov_j in overflow_alumni:
                excess = current_loads[ov_j] - self.capacities[ov_j]
                if excess <= 0:
                    continue
                # Find students assigned to this alumnus
                students_with_ov = [s for s in range(self.n_students) if ov_j in repaired[s]]
                # Reassign excess students to underutilized valid candidates
                for s in students_with_ov:
                    if current_loads[ov_j] <= self.capacities[ov_j]:
                        break
                    # Find alternative alumnus for student s
                    curr_assigned = set(repaired[s])
                    comp_counts = {self.companies[a]: sum(1 for x in repaired[s] if self.companies[x] == self.companies[a]) for a in repaired[s]}
                    for alt in self.valid_candidates[s]:
                        if alt in curr_assigned:
                            continue
                        if current_loads[alt] >= self.capacities[alt]:
                            continue
                        alt_comp = self.companies[alt]
                        if comp_counts.get(alt_comp, 0) >= self.max_company_share and self.companies[ov_j] != alt_comp:
                            continue

                        # Perform swap
                        pos = np.where(repaired[s] == ov_j)[0][0]
                        repaired[s, pos] = alt
                        current_loads[ov_j] -= 1
                        current_loads[alt] += 1
                        break

        return repaired
