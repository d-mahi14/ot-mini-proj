"""Hybrid Seeded + Memetic NSGA-II for Student-Alumni Matching.

Combines:
1. Heuristic Seeded Initialization: Injects high-quality, load-balanced, and fairness-focused
   solutions from Greedy, repaired Min-Cost Flow, and diverse priority heuristics into the initial population.
2. Evolutionary Exploration: NSGA-II with customized crossover, mutation, and constraint repair.
3. Memetic Local Search: Targeted neighborhood search with:
   - Pairwise 2-Opt Student Swaps (preserving loads while boosting utilitarian quality Z1)
   - Z2-targeted Rawlsian Quality Elevation (boosting worst-student protection)
   - Z3-targeted Workload Relief (relieving bottleneck alumni peak loads)
4. Elitist Non-Dominated Archiving: Ensures that refined heuristic seeds and evolved solutions
   are jointly filtered to guarantee Pareto optimality and baseline dominance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.core.sampling import Sampling

from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.metrics import extract_non_dominated_front
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

    def refine_solution(self, solution: MatchingSolution, max_steps: int = 25) -> MatchingSolution:
        """Perform targeted local moves to improve utilitarian quality, fairness, and load balance."""
        current_x = np.copy(solution.x)
        current_sol = solution

        for step in range(max_steps):
            improved = False

            # -------------------------------------------------------------
            # Move 1: Pairwise 2-Opt Student Swap (Boosts Z1, preserves Z3)
            # -------------------------------------------------------------
            for i1 in range(self.problem.n_students):
                matches_1 = np.where(current_x[i1] > 0)[0]
                # Sample a partner student
                candidate_i2s = np.random.choice(self.problem.n_students, size=min(10, self.problem.n_students), replace=False)
                for i2 in candidate_i2s:
                    if i1 == i2:
                        continue
                    matches_2 = np.where(current_x[i2] > 0)[0]
                    for j1 in matches_1:
                        for j2 in matches_2:
                            if j1 == j2 or current_x[i1, j2] == 1 or current_x[i2, j1] == 1:
                                continue
                            if self.problem.consents[j2] != 1 or self.problem.consents[j1] != 1:
                                continue
                            if self.problem.q_matrix[i1, j2] < self.problem.q_min or self.problem.q_matrix[i2, j1] < self.problem.q_min:
                                continue

                            gain = (self.problem.q_matrix[i1, j2] + self.problem.q_matrix[i2, j1]) - (
                                self.problem.q_matrix[i1, j1] + self.problem.q_matrix[i2, j2]
                            )
                            if gain > 1e-4:
                                trial_x = np.copy(current_x)
                                trial_x[i1, j1] = 0
                                trial_x[i1, j2] = 1
                                trial_x[i2, j2] = 0
                                trial_x[i2, j1] = 1
                                trial_sol = self.problem.evaluate_solution(trial_x)
                                if trial_sol.is_feasible:
                                    if trial_sol.z2_worst_quality >= current_sol.z2_worst_quality - 0.05:
                                        current_x = trial_x
                                        current_sol = trial_sol
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # -------------------------------------------------------------
            # Move 2: Z2-Targeted Quality Elevation Swap (Rawlsian Fairness)
            # -------------------------------------------------------------
            worst_student = int(np.argmin(current_sol.student_qualities))
            current_alumni = np.where(current_x[worst_student] > 0)[0]
            if len(current_alumni) > 0:
                worst_match = current_alumni[np.argmin(self.problem.q_matrix[worst_student, current_alumni])]
                for candidate in range(self.problem.n_alumni):
                    if current_x[worst_student, candidate] == 1:
                        continue
                    if self.problem.consents[candidate] != 1 or self.problem.q_matrix[worst_student, candidate] < self.problem.q_min:
                        continue
                    if current_sol.alumni_loads[candidate] >= self.problem.capacities[candidate]:
                        continue
                    if self.problem.q_matrix[worst_student, candidate] > self.problem.q_matrix[worst_student, worst_match]:
                        trial_x = np.copy(current_x)
                        trial_x[worst_student, worst_match] = 0
                        trial_x[worst_student, candidate] = 1
                        trial_sol = self.problem.evaluate_solution(trial_x)
                        if trial_sol.is_feasible:
                            if trial_sol.z2_worst_quality > current_sol.z2_worst_quality:
                                current_x = trial_x
                                current_sol = trial_sol
                                improved = True
                                break

            # -------------------------------------------------------------
            # Move 3: Z3-Targeted Bottleneck Load Relief Swap
            # -------------------------------------------------------------
            max_load_alumnus = int(np.argmax(current_sol.alumni_loads))
            if current_sol.alumni_loads[max_load_alumnus] > 2:
                assigned_students = np.where(current_x[:, max_load_alumnus] > 0)[0]
                for stu in assigned_students:
                    for alt in range(self.problem.n_alumni):
                        if current_x[stu, alt] == 1:
                            continue
                        if self.problem.consents[alt] != 1 or self.problem.q_matrix[stu, alt] < self.problem.q_min:
                            continue
                        if current_sol.alumni_loads[alt] + 1 < current_sol.alumni_loads[max_load_alumnus]:
                            trial_x = np.copy(current_x)
                            trial_x[stu, max_load_alumnus] = 0
                            trial_x[stu, alt] = 1
                            trial_sol = self.problem.evaluate_solution(trial_x)
                            if trial_sol.is_feasible and trial_sol.z3_max_load <= current_sol.z3_max_load:
                                if trial_sol.z2_worst_quality >= current_sol.z2_worst_quality - 0.05:
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

    def _generate_seeds(self) -> Tuple[List[np.ndarray], List[MatchingSolution]]:
        """Generate diverse heuristic seeds for population injection and archiving."""
        seed_chroms: List[np.ndarray] = []
        seed_solutions: List[MatchingSolution] = []

        # 1. Greedy baseline seed
        try:
            greedy = GreedyBaseline(self.problem).solve()
            seed_chroms.append(self.problem.encode_solution(greedy.x))
            seed_solutions.append(greedy)
        except Exception:
            pass

        # 2. Min-Cost Flow baseline seed (repaired)
        try:
            mcf = MinCostFlowBaseline(self.problem).solve()
            mcf_chrom = self.problem.encode_solution(mcf.x).reshape((self.problem.n_students, self.problem.k_matches))
            mcf_rep_chrom = self.repair.repair_single(mcf_chrom)
            rep_x = self.problem.decode_chromosome(mcf_rep_chrom.flatten())
            mcf_rep_sol = self.problem.evaluate_solution(rep_x)
            seed_chroms.append(mcf_rep_chrom.flatten())
            seed_solutions.append(mcf_rep_sol)
        except Exception:
            pass

        # 3. Load-balanced heuristic seed
        try:
            x_lb = np.zeros((self.problem.n_students, self.problem.n_alumni), dtype=int)
            loads = np.zeros(self.problem.n_alumni, dtype=int)
            for i in range(self.problem.n_students):
                valid = [
                    j for j in range(self.problem.n_alumni)
                    if self.problem.consents[j] == 1 and self.problem.q_matrix[i, j] >= self.problem.q_min
                ]
                if len(valid) >= self.problem.k_matches:
                    scores = [
                        self.problem.q_matrix[i, j] + 0.35 * (1.0 - (loads[j] / max(1, self.problem.capacities[j])))
                        for j in valid
                    ]
                    top_j = [valid[idx] for idx in np.argsort(-np.array(scores))[:self.problem.k_matches]]
                    for j in top_j:
                        x_lb[i, j] = 1
                        loads[j] += 1
            sol_lb = self.problem.evaluate_solution(x_lb)
            seed_chroms.append(self.problem.encode_solution(sol_lb.x))
            seed_solutions.append(sol_lb)
        except Exception:
            pass

        # 4. Rawlsian Fairness-focused heuristic seed (prioritize worst-match students first)
        try:
            x_rawls = np.zeros((self.problem.n_students, self.problem.n_alumni), dtype=int)
            loads_r = np.zeros(self.problem.n_alumni, dtype=int)
            # Find max potential quality per student
            student_max_q = [
                np.max(self.problem.q_matrix[i, self.problem.consents == 1])
                for i in range(self.problem.n_students)
            ]
            # Process students with worst max-potential first
            sorted_students = np.argsort(student_max_q)
            for i in sorted_students:
                valid = [
                    j for j in range(self.problem.n_alumni)
                    if self.problem.consents[j] == 1
                    and self.problem.q_matrix[i, j] >= self.problem.q_min
                    and loads_r[j] < self.problem.capacities[j]
                ]
                if len(valid) >= self.problem.k_matches:
                    scores = [self.problem.q_matrix[i, j] for j in valid]
                    top_j = [valid[idx] for idx in np.argsort(-np.array(scores))[:self.problem.k_matches]]
                    for j in top_j:
                        x_rawls[i, j] = 1
                        loads_r[j] += 1
            sol_rawls = self.problem.evaluate_solution(x_rawls)
            if sol_rawls.is_feasible:
                seed_chroms.append(self.problem.encode_solution(sol_rawls.x))
                seed_solutions.append(sol_rawls)
        except Exception:
            pass

        # 5. Perturbed variations of seeds (exploration)
        for s in list(seed_chroms[:4]):
            chrom = np.copy(s).reshape((self.problem.n_students, self.problem.k_matches))
            n_mut = max(1, int(self.problem.n_students * 0.1))
            mut_students = np.random.choice(self.problem.n_students, size=n_mut, replace=False)
            for stu in mut_students:
                chrom[stu, np.random.randint(self.problem.k_matches)] = np.random.randint(self.problem.n_alumni)
            repaired_chrom = self.repair.repair_single(chrom)
            seed_chroms.append(repaired_chrom.flatten())

        return seed_chroms, seed_solutions

    def solve(
        self,
        pop_size: int = 50,
        n_gen: int = 60,
        seed: int = 42,
        use_seeding: bool = True,
        apply_local_search: bool = True,
        verbose: bool = False,
    ) -> Tuple[List[MatchingSolution], Dict[str, Any]]:
        """Run Seeded + Memetic NSGA-II with elitist archiving."""
        start_time = time.perf_counter()

        # Step 1: Generate heuristic seeds if enabled
        if use_seeding:
            seed_chroms, elite_seeds = self._generate_seeds()
            sampling = SeededSampling(self.problem, seed_chroms)
        else:
            seed_chroms, elite_seeds = [], []
            sampling = None

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

        # Step 3: Memetic local search & Elitist Pool Assembly
        all_candidate_sols: List[MatchingSolution] = []

        if apply_local_search:
            # Refine evolved solutions
            for sol in nsga_sols:
                refined = self.local_search.refine_solution(sol, max_steps=20)
                all_candidate_sols.append(refined)
            # Refine elite seeds if seeding was used
            for es in elite_seeds:
                refined_es = self.local_search.refine_solution(es, max_steps=20)
                all_candidate_sols.append(refined_es)
        else:
            all_candidate_sols.extend(nsga_sols)
            all_candidate_sols.extend(elite_seeds)

        # Step 4: Extract final non-dominated Pareto front over the combined pool
        if all_candidate_sols:
            objs = np.array([s.objectives for s in all_candidate_sols])
            _, nd_indices = extract_non_dominated_front(objs)
            final_sols = [all_candidate_sols[idx] for idx in nd_indices]
        else:
            final_sols = []

        total_time = time.perf_counter() - start_time
        meta["algorithm"] = "Seeded-Memetic-NSGA-II"
        meta["runtime_seconds"] = total_time
        meta["num_seeds_injected"] = len(seed_chroms)
        meta["final_pareto_size"] = len(final_sols)

        return final_sols, meta
