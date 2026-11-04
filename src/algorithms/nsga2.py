"""NSGA-II multi-objective evolutionary algorithm with custom repair for matching.

Uses Non-Dominated Sorting Genetic Algorithm II (NSGA-II) equipped with:
- Integer uniform / random sampling
- Simulated binary crossover (SBX) or Point Crossover adapted for discrete assignment
- Polynomial / uniform mutation
- Domain-specific MatchingRepair operator enforcing hard feasibility
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.multiobjective.repair import MatchingRepair
from src.problem import MatchingSolution, StudentAlumniMatchingProblem


def run_nsga2(
    problem: StudentAlumniMatchingProblem,
    pop_size: int = 50,
    n_gen: int = 60,
    seed: int = 42,
    repair_operator: Optional[MatchingRepair] = None,
    sampling: Any = None,
    verbose: bool = False,
) -> Tuple[List[MatchingSolution], Dict[str, Any]]:
    """Execute NSGA-II optimization run.

    Args:
        problem: The matching problem instance.
        pop_size: Population size.
        n_gen: Number of generations.
        seed: Random seed.
        repair_operator: Optional custom repair instance.
        sampling: Custom sampling (e.g., seeded population).
        verbose: Verbose progress printing.

    Returns:
        Tuple of (list of non-dominated MatchingSolutions, execution metadata dictionary).
    """
    repair = repair_operator or MatchingRepair(problem)
    sampling_op = sampling if sampling is not None else IntegerRandomSampling()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling_op,
        crossover=SBX(prob=0.9, eta=15, vtype=int, repair=repair),
        mutation=PM(prob=0.15, eta=20, vtype=int, repair=repair),
        repair=repair,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    start_time = time.perf_counter()
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=False,
        verbose=verbose,
    )
    runtime = time.perf_counter() - start_time

    # Decode and evaluate final population / Pareto front
    solutions: List[MatchingSolution] = []
    X_arr = None
    if res.X is not None:
        X_arr = np.atleast_2d(res.X)
    elif hasattr(res, "pop") and res.pop is not None and len(res.pop) > 0:
        pop_X = res.pop.get("X")
        if pop_X is not None and len(pop_X) > 0:
            X_arr = np.atleast_2d(pop_X)

    if X_arr is not None:
        for i in range(X_arr.shape[0]):
            x_bin = problem.decode_chromosome(X_arr[i])
            sol = problem.evaluate_solution(x_bin)
            solutions.append(sol)

    meta = {
        "algorithm": "NSGA-II",
        "pop_size": pop_size,
        "n_gen": n_gen,
        "runtime_seconds": runtime,
        "n_evals": getattr(res, "algorithm", None).evaluator.n_eval if hasattr(res, "algorithm") else pop_size * n_gen,
        "success": bool(len(solutions) > 0),
    }

    return solutions, meta
