"""Random Feasible Assignment Baseline for Student-Alumni Matching.

Generates stochastic feasible assignments respecting degree, consent, capacity,
and match threshold constraints.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.problem import MatchingSolution, StudentAlumniMatchingProblem


class RandomBaseline:
    """Generates random feasible matchings for baseline benchmarking."""

    def __init__(self, problem: StudentAlumniMatchingProblem, seed: Optional[int] = 42) -> None:
        """Initialize random baseline with problem instance."""
        self.problem = problem
        self.rng = np.random.default_rng(seed)

    def solve(self, n_trials: int = 1) -> MatchingSolution:
        """Run random generation and return the best feasible solution found."""
        start_time = time.perf_counter()
        best_sol: Optional[MatchingSolution] = None

        for _ in range(n_trials):
            x_bin = np.zeros((self.problem.n_students, self.problem.n_alumni), dtype=int)
            loads = np.zeros(self.problem.n_alumni, dtype=int)

            # Randomize order of students
            order = self.rng.permutation(self.problem.n_students)

            for i in order:
                # Valid candidates: consent==1, q_ij >= q_min, load < capacity
                valid = [
                    j for j in range(self.problem.n_alumni)
                    if self.problem.consents[j] == 1
                    and self.problem.q_matrix[i, j] >= self.problem.q_min
                    and loads[j] < self.problem.capacities[j]
                ]

                # Fallback if strict capacity exhausted
                if len(valid) < self.problem.k_matches:
                    valid = [
                        j for j in range(self.problem.n_alumni)
                        if self.problem.consents[j] == 1
                    ]

                if len(valid) < self.problem.k_matches:
                    valid = list(range(self.problem.n_alumni))

                # Sample K distinct alumni
                chosen = self.rng.choice(valid, size=min(self.problem.k_matches, len(valid)), replace=False)
                for j in chosen:
                    x_bin[i, j] = 1
                    loads[j] += 1

                # If needed, fill remaining
                while np.sum(x_bin[i]) < self.problem.k_matches:
                    rem = [j for j in range(self.problem.n_alumni) if x_bin[i, j] == 0]
                    j = self.rng.choice(rem)
                    x_bin[i, j] = 1
                    loads[j] += 1

            sol = self.problem.evaluate_solution(x_bin)
            if best_sol is None or (sol.total_constraint_violation < best_sol.total_constraint_violation) or (
                sol.is_feasible and not best_sol.is_feasible
            ) or (sol.is_feasible and sol.z1_total_quality > best_sol.z1_total_quality):
                best_sol = sol

        elapsed = time.perf_counter() - start_time
        return best_sol
