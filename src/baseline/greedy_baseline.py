"""Greedy Score-First Assignment Baseline for Student-Alumni Matching.

Prioritizes highest match scores q_ij while respecting capacity, consent,
and diversity constraints.
"""

from __future__ import annotations

import time
from typing import Dict, List, Set

import numpy as np

from src.problem import MatchingSolution, StudentAlumniMatchingProblem


class GreedyBaseline:
    """Greedy priority assignment based on descending match quality."""

    def __init__(self, problem: StudentAlumniMatchingProblem) -> None:
        """Initialize greedy baseline solver."""
        self.problem = problem

    def solve(self) -> MatchingSolution:
        """Execute greedy score-first matching algorithm."""
        start_time = time.perf_counter()
        n_students = self.problem.n_students
        n_alumni = self.problem.n_alumni
        k_matches = self.problem.k_matches

        x_bin = np.zeros((n_students, n_alumni), dtype=int)
        loads = np.zeros(n_alumni, dtype=int)
        student_degrees = np.zeros(n_students, dtype=int)
        student_companies: Dict[int, Dict[str, int]] = {i: {} for i in range(n_students)}

        # Create all candidate pairs (i, j) where c_j == 1 and q_ij >= q_min
        pairs: List[tuple[float, int, int]] = []
        for i in range(n_students):
            for j in range(n_alumni):
                if self.problem.consents[j] == 1 and self.problem.q_matrix[i, j] >= self.problem.q_min:
                    pairs.append((float(self.problem.q_matrix[i, j]), i, j))

        # Sort pairs by descending match quality q_ij
        pairs.sort(key=lambda item: item[0], reverse=True)

        # Primary greedy pass
        for score, i, j in pairs:
            if student_degrees[i] >= k_matches:
                continue
            if loads[j] >= self.problem.capacities[j]:
                continue
            comp = self.problem.alumni_companies[j]
            if student_companies[i].get(comp, 0) >= self.problem.max_company_share:
                continue

            # Assign match
            x_bin[i, j] = 1
            loads[j] += 1
            student_degrees[i] += 1
            student_companies[i][comp] = student_companies[i].get(comp, 0) + 1

        # Second pass: fill any student who didn't reach K matches
        unfilled_students = np.where(student_degrees < k_matches)[0]
        for i in unfilled_students:
            # Sort all consenting alumni by quality for student i
            candidate_indices = np.argsort(-self.problem.q_matrix[i])
            for j in candidate_indices:
                if student_degrees[i] >= k_matches:
                    break
                if x_bin[i, j] == 1:
                    continue
                if self.problem.consents[j] != 1:
                    continue
                if loads[j] >= self.problem.capacities[j]:
                    continue
                comp = self.problem.alumni_companies[j]
                if student_companies[i].get(comp, 0) >= self.problem.max_company_share:
                    continue

                x_bin[i, j] = 1
                loads[j] += 1
                student_degrees[i] += 1
                student_companies[i][comp] = student_companies[i].get(comp, 0) + 1

        # Final pass: safety guarantee each student has K matches
        for i in range(n_students):
            while student_degrees[i] < k_matches:
                # Pick alumnus with lowest load
                j = int(np.argmin(loads))
                if x_bin[i, j] == 0:
                    x_bin[i, j] = 1
                    loads[j] += 1
                    student_degrees[i] += 1
                else:
                    for fallback in range(n_alumni):
                        if x_bin[i, fallback] == 0:
                            x_bin[i, fallback] = 1
                            loads[fallback] += 1
                            student_degrees[i] += 1
                            break

        solution = self.problem.evaluate_solution(x_bin)
        return solution
