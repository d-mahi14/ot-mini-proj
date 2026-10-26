"""Hybrid Seeded + Memetic NSGA-II for Student-Alumni Matching.

Combines:
1. Heuristic Seeded Initialization: Injects high-quality, load-balanced, and fairness-focused
   solutions from Greedy, Min-Cost Flow, and diverse priority heuristics into the initial population.
2. Evolutionary Exploration: NSGA-II with customized crossover, mutation, and constraint repair.
3. Memetic Local Search: Targeted neighborhood search focusing on Rawlsian worst-off quality
   elevation (Z2) and bottleneck load reduction (Z3) on non-dominated solutions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.core.sampling import Sampling

from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.multiobjective.nsga2 import run_nsga2
from src.multiobjective.repair import MatchingRepair
from src.problem import MatchingSolution, StudentAlumniMatchingProblem


class SeededSampling(Sampling):
    """Custom sampling operator that injects heuristic solutions into the initial population."""

    def __init__(self, problem: StudentAlumniMatchingProblem, seed_chromosomes: List[np.ndarray]) -> None:
        """Initialize with problem and precomputed seed chromosomes."""
        super().__init__()
        self.problem = problem
        self.seeds = seed_chromosomes

    def _do(self, problem: Any, n_samples: int, **kwargs: Any) -> np.ndarray:
        """Generate initial population incorporating seeds and uniform random individuals."""
        pop = np.zeros((n_samples, problem.n_var), dtype=int)
        n_seeds = min(len(self.seeds), n_samples)

        # Inject seeds
        for s in range(n_seeds):
            pop[s] = self.seeds[s]

        # Fill remaining with random integers
        for s in range(n_seeds, n_samples):
            pop[s] = np.random.randint(0, self.problem.n_alumni, size=problem.n_var)

        return pop


class MemeticLocalSearch:
    """Neighborhood local search operators for multi-objective refinement."""

    def __init__(self, problem: StudentAlumniMatchingProblem) -> None:
        """Initialize local search operator."""
        self.problem = problem

    def refine_solution(self, solution: MatchingSolution, max_steps: int = 15) -> MatchingSolution:
        """Perform targeted local swaps to improve fairness and load balancing."""
        current_x = np.copy(solution.x)
        current_sol = solution

        for _ in range(max_steps):
            improved = False

            # Tactic A: Elevate worst-off student quality (improve Z2)
            worst_student = int(np.argmin(current_sol.student_qualities))
            current_alumni = np.where(current_x[worst_student] > 0)[0]
            if len(current_alumni) > 0:
                # Find worst match for this student
                worst_match = current_alumni[np.argmin(self.problem.q_matrix[worst_student, current_alumni])]
                # Find available better alumni
                for candidate in range(self.problem.n_alumni):
                    if current_x[worst_student, candidate] == 1:
                        continue
                    if self.problem.consents[candidate] != 1 or self.problem.q_matrix[worst_student, candidate] < self.problem.q_min:
                        continue
                    if current_sol.alumni_loads[candidate] >= self.problem.capacities[candidate]:
                        continue
                    if self.problem.q_matrix[worst_student, candidate] > self.problem.q_matrix[worst_student, worst_match]:
                        # Test swap
                        trial_x = np.copy(current_x)
                        trial_x[worst_student, worst_match] = 0
                        trial_x[worst_student, candidate] = 1
                        trial_sol = self.problem.evaluate_solution(trial_x)
                        if trial_sol.is_feasible or trial_sol.total_constraint_violation <= current_sol.total_constraint_violation:
                            if trial_sol.z2_worst_quality >= current_sol.z2_worst_quality:
                                current_x = trial_x
                                current_sol = trial_sol
                                improved = True
                                break

            # Tactic B: Relieve bottleneck alumni load (improve Z3)
            max_load_alumnus = int(np.argmax(current_sol.alumni_loads))
            if current_sol.alumni_loads[max_load_alumnus] > 1:
                assigned_students = np.where(current_x[:, max_load_alumnus] > 0)[0]
                for stu in assigned_students:
                    # Look for underutilized alternative
                    for alt in range(self.problem.n_alumni):
                        if current_x[stu, alt] == 1:
                            continue
                        if self.problem.consents[alt] != 1 or self.problem.q_matrix[stu, alt] < self.problem.q_min:
                            continue
                        if current_sol.alumni_loads[alt] + 1 < current_sol.alumni_loads[max_load_alumnus]:
                            # Evaluate swap
                            trial_x = np.copy(current_x)
                            trial_x[stu, max_load_alumnus] = 0
                            trial_x[stu, alt] = 1
                            trial_sol = self.problem.evaluate_solution(trial_x)
                            if trial_sol.is_feasible and trial_sol.z3_max_load <= current_sol.z3_max_load:
                                current_x = trial_x
                                current_sol = trial_sol
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                break

        return current_sol


class SeededMemeticNSGA2:
    """Full hybrid algorithm integrating heuristic seeding, NSGA-II, and memetic search."""

    def __init__(self, problem: StudentAlumniMatchingProblem) -> None:
        """Initialize hybrid solver."""
        self.problem = problem
        self.repair = MatchingRepair(problem)
        self.local_search = MemeticLocalSearch(problem)

    def _generate_seeds(self) -> List[np.ndarray]:
        """Generate diverse heuristic seeds for population injection."""
        seeds: List[np.ndarray] = []

        # 1. Greedy baseline seed
        try:
            greedy = GreedyBaseline(self.problem).solve()
            seeds.append(self.problem.encode_solution(greedy.x))
        except Exception:
            pass

        # 2. Min-Cost Flow baseline seed
        try:
            mcf = MinCostFlowBaseline(self.problem).solve()
            seeds.append(self.problem.encode_solution(mcf.x))
        except Exception:
            pass

        # 3. Load-balanced heuristic seed
        try:
            x_lb = np.zeros((self.problem.n_students, self.problem.n_alumni), dtype=int)
            loads = np.zeros(self.problem.n_alumni, dtype=int)
            for i in range(self.problem.n_students):
                # Score combines match quality with alumni headroom
                valid = [
                    j for j in range(self.problem.n_alumni)
                    if self.problem.consents[j] == 1 and self.problem.q_matrix[i, j] >= self.problem.q_min
                ]
                if len(valid) >= self.problem.k_matches:
                    scores = [
                        self.problem.q_matrix[i, j] + 0.3 * (1.0 - (loads[j] / max(1, self.problem.capacities[j])))
                        for j in valid
                    ]
                    top_j = [valid[idx] for idx in np.argsort(-np.array(scores))[:self.problem.k_matches]]
                    for j in top_j:
                        x_lb[i, j] = 1
                        loads[j] += 1
            sol_lb = self.problem.evaluate_solution(x_lb)
            seeds.append(self.problem.encode_solution(sol_lb.x))
        except Exception:
            pass

        # Perturbed variations of seeds
        for s in list(seeds):
            chrom = np.copy(s).reshape((self.problem.n_students, self.problem.k_matches))
            # Mutate 10% of entries
            n_mut = max(1, int(self.problem.n_students * 0.1))
            mut_students = np.random.choice(self.problem.n_students, size=n_mut, replace=False)
            for stu in mut_students:
                chrom[stu, np.random.randint(self.problem.k_matches)] = np.random.randint(self.problem.n_alumni)
            repaired_chrom = self.repair.repair_single(chrom)
            seeds.append(repaired_chrom.flatten())

        return seeds

    def solve(
        self,
        pop_size: int = 50,
        n_gen: int = 60,
        seed: int = 42,
        apply_local_search: bool = True,
        verbose: bool = False,
    ) -> Tuple[List[MatchingSolution], Dict[str, Any]]:
        """Run Seeded + Memetic NSGA-II."""
        start_time = time.perf_counter()

        # Step 1: Generate seeds
        seeds = self._generate_seeds()
        sampling = SeededSampling(self.problem, seeds)

        # Step 2: Evolutionary run
        nsga_sols, meta = run_nsga2(
            problem=self.problem,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            repair_operator=self.repair,
            sampling=sampling,
            verbose=verbose,
        )

        # Step 3: Memetic local search on non-dominated solutions
        final_sols: List[MatchingSolution] = []
        if apply_local_search and nsga_sols:
            for sol in nsga_sols:
                refined = self.local_search.refine_solution(sol, max_steps=10)
                final_sols.append(refined)
        else:
            final_sols = nsga_sols

        total_time = time.perf_counter() - start_time
        meta["algorithm"] = "Seeded-Memetic-NSGA-II"
        meta["runtime_seconds"] = total_time
        meta["num_seeds_injected"] = len(seeds)

        return final_sols, meta
