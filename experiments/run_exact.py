"""Exact MILP benchmark and secondary epsilon-constraint exploration.

Focuses on:
1. Exact Weighted-Sum & Epsilon-Constraint Pareto frontier on small instance (50x30).
2. Secondary objective scenario: Minimizing active alumni count sum_j y_j under epsilon constraints.
3. Comparative overlay of exact frontier against metaheuristic approximations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.algorithms.exact_milp import ExactMILP
from src.data_generator import MatchingInstance
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.multiobjective.nsga2 import run_nsga2
from src.problem import StudentAlumniMatchingProblem


def run_exact_analysis(config_path: str = "experiments/configs/small_50x30.yaml") -> None:
    """Run exact MILP analysis and secondary epsilon-constraint scenario."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inst_cfg = cfg["instance"]
    inst_path = Path(inst_cfg["file_path"])
    if not inst_path.exists():
        from experiments.run_experiments import load_or_create_instance
        load_or_create_instance(inst_cfg)

    instance = MatchingInstance.load_json(inst_path)
    problem = StudentAlumniMatchingProblem(instance)
    solver = ExactMILP(problem)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("--- [Exact Analysis] Tracing True Pareto Frontier via MILP ---")
    exact_sols = solver.trace_pareto_front(grid_steps=5, time_limit_per_point=30.0)
    print(f"Discovered {len(exact_sols)} unique exact Pareto-optimal solutions.")

    print("\n--- [Exact Analysis] Running Secondary Scenario: Minimizing Alumni Count (sum y_j) ---")
    secondary_results = []
    for ratio in [0.80, 0.85, 0.90, 0.95]:
        sol_alumni = solver.solve_minimize_alumni_usage(min_quality_ratio=ratio, time_limit_sec=45.0)
        if sol_alumni is not None:
            secondary_results.append({
                "min_quality_ratio": ratio,
                "active_alumni": sol_alumni.active_alumni_count,
                "total_quality_z1": round(sol_alumni.z1_total_quality, 3),
                "worst_quality_z2": round(sol_alumni.z2_worst_quality, 3),
                "max_load_z3": sol_alumni.z3_max_load,
                "company_entropy": round(sol_alumni.company_entropy, 3),
            })
            print(f"Target Quality Ratio >= {ratio*100:.0f}% -> Min Active Alumni Required: {sol_alumni.active_alumni_count} (Entropy: {sol_alumni.company_entropy:.2f})")

    # Run NSGA-II and Hybrid for overlay
    print("\nRunning NSGA-II and Hybrid for comparison overlay...")
    nsga_sols, _ = run_nsga2(problem, pop_size=40, n_gen=35, seed=42)
    hybrid_sols, _ = SeededMemeticNSGA2(problem).solve(pop_size=40, n_gen=35, seed=42)

    # Save exact results
    exact_summary = {
        "num_exact_pareto_points": len(exact_sols),
        "exact_points": [s.to_summary() for s in exact_sols],
        "secondary_alumni_minimization": secondary_results,
    }
    with open(results_dir / "exact_milp_results.json", "w", encoding="utf-8") as f:
        json.dump(exact_summary, f, indent=2)

    # Plot Comparison Overlay
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Z1 vs Z2
    if exact_sols:
        axes[0].scatter(
            [s.z1_total_quality for s in exact_sols],
            [s.z2_worst_quality for s in exact_sols],
            color="black", marker="s", s=60, label="Exact MILP (True Frontier)", zorder=5
        )
    if nsga_sols:
        axes[0].scatter(
            [s.z1_total_quality for s in nsga_sols],
            [s.z2_worst_quality for s in nsga_sols],
            color="#1f77b4", marker="o", s=35, alpha=0.7, label="NSGA-II"
        )
    if hybrid_sols:
        axes[0].scatter(
            [s.z1_total_quality for s in hybrid_sols],
            [s.z2_worst_quality for s in hybrid_sols],
            color="#2ca02c", marker="^", s=45, alpha=0.85, label="Hybrid Memetic NSGA-II"
        )
    axes[0].set_xlabel("Total Quality ($Z_1$)")
    axes[0].set_ylabel("Rawlsian Worst-off Quality ($Z_2$)")
    axes[0].set_title("Exact vs Metaheuristic: Efficiency vs Fairness")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Secondary Scenario Trade-off (Active Alumni vs Target Quality)
    if secondary_results:
        ratios = [r["min_quality_ratio"] * 100 for r in secondary_results]
        alumni_counts = [r["active_alumni"] for r in secondary_results]
        axes[1].plot(ratios, alumni_counts, marker="o", color="#d62728", linewidth=2.0, markersize=8)
        axes[1].set_xlabel("Quality Bound (% of Max Quality $Z_1$)")
        axes[1].set_ylabel("Minimum Active Alumni Needed ($\\sum y_j$)")
        axes[1].set_title("Secondary Epsilon-Constraint: Alumni Utilization")
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(figures_dir / "exact_vs_metaheuristics.png", dpi=300)
    plt.close()
    print("Exact analysis completed and figures saved.")


if __name__ == "__main__":
    run_exact_analysis()
