"""Baselines execution script for Student-Alumni Matching optimization.

Runs deterministic and stochastic baselines:
- Random Baseline (stochastic average across trials)
- Greedy Priority Baseline (score-first)
- Single-Objective Min-Cost Flow (raw max-Z1, unconstrained diversity)
- Min-Cost Flow Repaired (feasible with MatchingRepair)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.baseline.random_baseline import RandomBaseline
from src.data_generator import MatchingInstance, generate_all_datasets
from src.problem import StudentAlumniMatchingProblem


def run_baselines_evaluation(config_path: str | Path = "experiments/configs/small_50x30.yaml") -> Dict[str, Any]:
    """Execute all baselines on configured instance and display results."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"Config {cfg_file} not found, generating datasets and using default small_50x30.yaml")
        generate_all_datasets()
        cfg_file = PROJECT_ROOT / "experiments" / "configs" / "small_50x30.yaml"

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    inst_file = Path(cfg["instance"]["file_path"])
    if not inst_file.exists():
        generate_all_datasets()

    instance = MatchingInstance.load_json(inst_file)
    problem = StudentAlumniMatchingProblem(instance)

    print("=" * 90)
    print(f"       BASELINE ALGORITHMS BENCHMARK EVALUATION: {instance.name.upper()}")
    print(f"       Dimensions: {instance.n_students} Students x {instance.n_alumni} Alumni | Demand: {instance.n_students * instance.k_matches} matches")
    print("=" * 90)

    results = {}

    # 1. Random Baseline
    t0 = time.perf_counter()
    rand_sol = RandomBaseline(problem, seed=42).solve(n_trials=5)
    t_rand = time.perf_counter() - t0
    results["Random"] = (rand_sol, t_rand)

    # 2. Greedy Baseline
    t0 = time.perf_counter()
    greedy_sol = GreedyBaseline(problem).solve()
    t_greedy = time.perf_counter() - t0
    results["Greedy"] = (greedy_sol, t_greedy)

    # 3. MCF Raw
    t0 = time.perf_counter()
    mcf_sol = MinCostFlowBaseline(problem).solve()
    t_mcf = time.perf_counter() - t0
    results["Min-Cost Flow (raw)"] = (mcf_sol, t_mcf)

    # 4. MCF Repaired
    t0 = time.perf_counter()
    mcf_rep_sol = MinCostFlowBaseline(problem).solve_repaired()
    t_rep = time.perf_counter() - t0
    results["MCF Repaired (feasible)"] = (mcf_rep_sol, t_rep)

    print(f"\n{'Algorithm':<28} | {'Z1 (Total)':<10} | {'Z2 (Worst)':<10} | {'Z3 (MaxLoad)':<12} | {'Feasible':<9} | {'CV':<8} | {'Runtime':<8}")
    print("-" * 90)
    for name, (sol, rtime) in results.items():
        feas_str = "YES" if sol.is_feasible else "NO"
        print(f"{name:<28} | {sol.z1_total_quality:<10.2f} | {sol.z2_worst_quality:<10.2f} | {sol.z3_max_load:<12} | {feas_str:<9} | {sol.total_constraint_violation:<8.2f} | {rtime:<8.4f}s")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline Algorithms")
    parser.add_argument("--config", type=str, default="experiments/configs/small_50x30.yaml")
    args = parser.parse_args()
    run_baselines_evaluation(args.config)
