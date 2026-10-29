"""Master experiment runner for Student-Alumni Matching optimization.

Reads configuration from YAML, executes baselines, exact methods, MOEAs, and hybrid
algorithms, calculates comprehensive metrics with shared reference hypervolume, and plots publication figures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from src.algorithms.exact_milp import ExactMILP
from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.baseline.random_baseline import RandomBaseline
from src.data_generator import MatchingInstance, SyntheticDataGenerator
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.metrics import (
    compute_full_metrics,
    compute_hypervolume,
    compute_shared_reference_setup,
    extract_non_dominated_front,
)
from src.multiobjective.moead import run_moead
from src.multiobjective.nsga2 import run_nsga2
from src.multiobjective.spea2 import run_spea2
from src.problem import MatchingSolution, StudentAlumniMatchingProblem
from src.visualization import (
    plot_branch_parity,
    plot_metrics_comparison,
    plot_pareto_front_3d,
    plot_tradeoff_2d,
    plot_workload_distribution,
)


def load_or_create_instance(inst_cfg: Dict[str, Any]) -> MatchingInstance:
    """Load matching instance from file or generate synthetically if missing."""
    file_path = Path(inst_cfg["file_path"])
    if file_path.exists():
        print(f"Loading cached instance from {file_path}...")
        return MatchingInstance.load_json(file_path)

    print(f"Generating synthetic instance: {inst_cfg['name']} ({inst_cfg['n_students']}x{inst_cfg['n_alumni']})...")
    gen = SyntheticDataGenerator(
        seed=inst_cfg.get("seed", 42),
        k_matches=inst_cfg.get("k_matches", 2),
        q_min=inst_cfg.get("q_min", 0.25),
        theta_min_quality=inst_cfg.get("theta_min_quality", 0.70),
        delta_branch_parity=inst_cfg.get("delta_branch_parity", 0.35),
        max_company_share=inst_cfg.get("max_company_share", 1),
    )
    instance = gen.generate_instance(
        n_students=inst_cfg["n_students"],
        n_alumni=inst_cfg["n_alumni"],
        name=inst_cfg["name"],
        consent_rate=inst_cfg.get("consent_rate", 0.95),
        avg_capacity=inst_cfg.get("avg_capacity", 4.0),
    )
    instance.save_json(file_path)
    print(f"Saved instance to {file_path}")
    return instance


def run_experiment_on_instance(config_path: str | Path) -> Dict[str, Any]:
    """Execute full algorithmic benchmark for a specific YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inst_cfg = cfg["instance"]
    algo_cfg = cfg.get("algorithms", {})
    instance = load_or_create_instance(inst_cfg)
    problem = StudentAlumniMatchingProblem(instance)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inst_name = instance.name
    print(f"\n=======================================================")
    print(f"Running Benchmark on: {inst_name} ({instance.n_students} students x {instance.n_alumni} alumni)")
    print(f"=======================================================\n")

    solutions_by_algo: Dict[str, List[MatchingSolution]] = {}
    runtimes_by_algo: Dict[str, float] = {}

    # 1. Baseline: Random Feasible
    print("[1/8] Running Random Feasible Baseline...")
    t0 = time.perf_counter()
    rand_sol = RandomBaseline(problem, seed=inst_cfg.get("seed", 42)).solve(n_trials=5)
    runtimes_by_algo["Random"] = time.perf_counter() - t0
    solutions_by_algo["Random"] = [rand_sol]

    # 2. Baseline: Greedy Score-First
    print("[2/8] Running Greedy Score-First Baseline...")
    t0 = time.perf_counter()
    greedy_sol = GreedyBaseline(problem).solve()
    runtimes_by_algo["Greedy"] = time.perf_counter() - t0
    solutions_by_algo["Greedy"] = [greedy_sol]

    # 3. Baseline: Single-Objective Min-Cost Flow
    print("[3/8] Running Single-Objective Min-Cost Flow (Max Z1)...")
    t0 = time.perf_counter()
    mcf_sol = MinCostFlowBaseline(problem).solve()
    runtimes_by_algo["Min-Cost Flow"] = time.perf_counter() - t0
    solutions_by_algo["Min-Cost Flow"] = [mcf_sol]

    # 4. Exact MILP (Enabled on small instances)
    if "milp" in algo_cfg and instance.n_students <= 100:
        print("[4/8] Running Exact MILP Pareto Sweep (Weighted-Sum & Epsilon-Constraint)...")
        m_cfg = algo_cfg["milp"]
        milp_solver = ExactMILP(problem)
        t0 = time.perf_counter()
        milp_sols = milp_solver.trace_pareto_front(
            grid_steps=m_cfg.get("grid_steps", 4),
            time_limit_per_point=m_cfg.get("time_limit_sec", 30.0),
        )
        runtimes_by_algo["Exact-MILP"] = time.perf_counter() - t0
        if milp_sols:
            solutions_by_algo["Exact-MILP"] = milp_sols
    else:
        print("[4/8] Skipping Exact MILP (Only configured for small instance).")

    # 5. NSGA-II with Custom Repair
    print("[5/8] Running NSGA-II with Custom Repair...")
    n_cfg = algo_cfg.get("nsga2", {})
    nsga_sols, nsga_meta = run_nsga2(
        problem,
        pop_size=n_cfg.get("pop_size", 40),
        n_gen=n_cfg.get("n_gen", 40),
        seed=n_cfg.get("seed", 42),
    )
    runtimes_by_algo["NSGA-II"] = nsga_meta["runtime_seconds"]
    solutions_by_algo["NSGA-II"] = nsga_sols

    # 6. Comparison MOEA: SPEA2
    print("[6/8] Running SPEA2 Comparison...")
    sp_cfg = algo_cfg.get("spea2", {})
    spea_sols, spea_meta = run_spea2(
        problem,
        pop_size=sp_cfg.get("pop_size", 40),
        n_gen=sp_cfg.get("n_gen", 40),
        seed=sp_cfg.get("seed", 42),
    )
    runtimes_by_algo["SPEA2"] = spea_meta["runtime_seconds"]
    solutions_by_algo["SPEA2"] = spea_sols

    # 7. Comparison MOEA: MOEA/D
    if "moead" in algo_cfg:
        print("[7/8] Running MOEA/D Comparison...")
        mo_cfg = algo_cfg.get("moead", {})
        moead_sols, moead_meta = run_moead(
            problem,
            n_partitions=mo_cfg.get("n_partitions", 6),
            n_gen=mo_cfg.get("n_gen", 40),
            seed=mo_cfg.get("seed", 42),
        )
        runtimes_by_algo["MOEA/D"] = moead_meta["runtime_seconds"]
        solutions_by_algo["MOEA/D"] = moead_sols

    # 8. Hybrid: Seeded + Memetic NSGA-II
    print("[8/8] Running Hybrid Seeded + Memetic NSGA-II...")
    h_cfg = algo_cfg.get("hybrid_memetic", {})
    hybrid_solver = SeededMemeticNSGA2(problem)
    hybrid_sols, hybrid_meta = hybrid_solver.solve(
        pop_size=h_cfg.get("pop_size", 40),
        n_gen=h_cfg.get("n_gen", 40),
        seed=h_cfg.get("seed", 42),
        apply_local_search=h_cfg.get("apply_local_search", True),
    )
    runtimes_by_algo["Hybrid-Memetic-NSGA-II"] = hybrid_meta["runtime_seconds"]
    solutions_by_algo["Hybrid-Memetic-NSGA-II"] = hybrid_sols

    # ---------------------------------------------------------
    # Shared Hypervolume Reference Setup Across ALL Methods
    # ---------------------------------------------------------
    all_objs_list: List[np.ndarray] = []
    for algo, sols in solutions_by_algo.items():
        if sols:
            all_objs_list.append(np.array([s.objectives for s in sols]))

    ref_setup = compute_shared_reference_setup(all_objs_list)
    ideal_pt = np.array(ref_setup["ideal_point"])
    nadir_pt = np.array(ref_setup["nadir_point"])
    ref_pt = np.array(ref_setup["ref_point"])  # Fixed at 1.1 in normalized space

    # Store ideal, nadir and ref point in results/<instance>_hv_reference.json
    hv_ref_file = results_dir / f"{inst_name}_hv_reference.json"
    with open(hv_ref_file, "w", encoding="utf-8") as f:
        json.dump({
            "instance": inst_name,
            "ideal_point": ref_setup["ideal_point"],
            "nadir_point": ref_setup["nadir_point"],
            "ref_point": ref_setup["ref_point"],
        }, f, indent=2)
    print(f"\nStored shared HV reference setup in: {hv_ref_file}")

    # Compute standardized metrics using shared reference bounds
    metrics_by_algo: Dict[str, Dict[str, Any]] = {}
    for algo, sols in solutions_by_algo.items():
        if not sols:
            continue
        objs = np.array([s.objectives for s in sols])
        q_mat = np.array([s.student_qualities for s in sols])
        ld_mat = np.array([s.alumni_loads for s in sols])
        t_run = runtimes_by_algo.get(algo, 0.0)

        m = compute_full_metrics(
            objs,
            qualities_matrix=q_mat,
            loads_matrix=ld_mat,
            runtimes=[t_run],
            ref_point=ref_pt,
            ideal_point=ideal_pt,
            nadir_point=nadir_pt,
        )
        if len(sols) == 1:
            m.update(sols[0].to_summary())
        metrics_by_algo[algo] = m

    # Save summary metrics to JSON
    metrics_file = results_dir / f"{inst_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_by_algo, f, indent=2)
    print(f"Saved metrics summary to: {metrics_file}")

    # Generate Publication Plots
    print("Generating visualization figures...")
    # 3D Pareto Front
    plot_pareto_front_3d(
        solutions_by_algo,
        figures_dir / f"{inst_name}_pareto_3d.png",
        title=f"3D Pareto Frontier ({inst_name})",
    )

    # 2D Trade-off Projections
    plot_tradeoff_2d(
        solutions_by_algo,
        figures_dir / f"{inst_name}_tradeoffs_2d.png",
    )

    # Pick representative balanced solution from Hybrid or NSGA-II for detailed analysis
    rep_sols = solutions_by_algo.get("Hybrid-Memetic-NSGA-II") or solutions_by_algo.get("NSGA-II") or [greedy_sol]
    best_rep = rep_sols[0]
    if len(rep_sols) > 1:
        z1_vals = [s.z1_total_quality for s in rep_sols]
        median_idx = int(np.argsort(z1_vals)[len(z1_vals) // 2])
        best_rep = rep_sols[median_idx]

    # Branch Parity
    plot_branch_parity(
        best_rep,
        figures_dir / f"{inst_name}_branch_parity.png",
        title=f"VJTI Branch Quality Parity ({inst_name})",
    )

    # Alumni Workload Distribution
    plot_workload_distribution(
        best_rep,
        problem.capacities,
        figures_dir / f"{inst_name}_workload_distribution.png",
    )

    # Multi-metric Algorithm Comparison
    moea_metrics = {
        k: v for k, v in metrics_by_algo.items()
        if "hypervolume" in v
    }
    if moea_metrics:
        plot_metrics_comparison(
            moea_metrics,
            figures_dir / f"{inst_name}_metrics_comparison.png",
        )

    print(f"All figures saved to {figures_dir.resolve()}/")
    return metrics_by_algo


def main() -> None:
    """CLI driver for experiments."""
    parser = argparse.ArgumentParser(description="Run VJTI Student-Alumni Matching Experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/small_50x30.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    run_experiment_on_instance(args.config)


if __name__ == "__main__":
    main()
