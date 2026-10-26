"""Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).

Decomposes the 3-objective matching problem into a set of scalar subproblems
using Das-Dennis uniform reference directions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from src.multiobjective.repair import MatchingRepair
from src.problem import MatchingSolution, StudentAlumniMatchingProblem


def run_moead(
    problem: StudentAlumniMatchingProblem,
    n_partitions: int = 8,
    n_gen: int = 60,
    seed: int = 42,
    repair_operator: Optional[MatchingRepair] = None,
    verbose: bool = False,
) -> Tuple[List[MatchingSolution], Dict[str, Any]]:
    """Execute MOEA/D optimization run.

    Args:
        problem: The matching problem instance.
        n_partitions: Das-Dennis partition count (controls number of reference vectors).
        n_gen: Number of generations.
        seed: Random seed.
        repair_operator: Optional custom repair instance.
        verbose: Verbose progress printing.

    Returns:
        Tuple of (list of non-dominated MatchingSolutions, execution metadata dictionary).
    """
    repair = repair_operator or MatchingRepair(problem)

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    n_neighbors = min(15, len(ref_dirs))

    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=n_neighbors,
        decomposition="tchebi",
        prob_neighbor_mating=0.7,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=int, repair=repair),
        mutation=PM(prob=0.15, eta=20, vtype=int, repair=repair),
        repair=repair,
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
        "algorithm": "MOEA/D",
        "n_ref_dirs": len(ref_dirs),
        "n_gen": n_gen,
        "runtime_seconds": runtime,
        "success": bool(len(solutions) > 0),
    }

    return solutions, meta
