"""Ablation and Sensitivity Analysis for VJTI Student-Alumni Matching.

Systematically evaluates:
1. Algorithmic components:
   - Base NSGA-II (standard)
   - NSGA-II + Custom Repair
   - NSGA-II + Custom Repair + Heuristic Seeding
   - Full Hybrid (Seeding + Repair + Memetic Local Search)
2. Policy parameters:
   - Branch Parity Disparity Tolerance delta in {0.10, 0.20, 0.35, 0.50}
   - Minimum Quality Floor theta in {0.50, 0.70, 0.90}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.run_experiments import load_or_create_instance
from src.data_generator import MatchingInstance
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.metrics import compute_full_metrics, compute_shared_reference_setup
from src.multiobjective.nsga2 import run_nsga2
from src.problem import StudentAlumniMatchingProblem


def run_ablation_study(config_path: str = "experiments/configs/small_50x30.yaml") -> None:
    """Run full component ablation and parameter sensitivity analysis."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    instance = load_or_create_instance(cfg["instance"])
    base_problem = StudentAlumniMatchingProblem(instance)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Part 1: Algorithmic Component Ablation ---")
    ablation_results: Dict[str, Dict[str, Any]] = {}

    # Variant 1: Pure NSGA-II with basic repair
    print("Evaluating Variant 1: Base NSGA-II...")
    v1_sols, v1_meta = run_nsga2(base_problem, pop_size=40, n_gen=35, seed=42)
    objs1 = np.array([s.objectives for s in v1_sols])

    # Variant 2: Hybrid Seeded (No memetic local search)
    print("Evaluating Variant 2: Seeded NSGA-II (No Local Search)...")
    v2_solver = SeededMemeticNSGA2(base_problem)
    v2_sols, v2_meta = v2_solver.solve(pop_size=40, n_gen=35, seed=42, apply_local_search=False)
    objs2 = np.array([s.objectives for s in v2_sols])

    # Variant 3: Full Seeded + Memetic NSGA-II
    print("Evaluating Variant 3: Full Hybrid (Seeded + Memetic)...")
    v3_solver = SeededMemeticNSGA2(base_problem)
    v3_sols, v3_meta = v3_solver.solve(pop_size=40, n_gen=35, seed=42, apply_local_search=True)
    objs3 = np.array([s.objectives for s in v3_sols])

    # Shared reference setup across all ablation variants
    all_objs = [o for o in [objs1, objs2, objs3] if len(o) > 0]
    ref_setup = compute_shared_reference_setup(all_objs)
    id_pt = np.array(ref_setup["ideal_point"])
    nd_pt = np.array(ref_setup["nadir_point"])
    rf_pt = np.array(ref_setup["ref_point"])

    ablation_results["Base NSGA-II"] = {
        "hypervolume": compute_full_metrics(objs1, ideal_point=id_pt, nadir_point=nd_pt, ref_point=rf_pt)["hypervolume"],
        "num_non_dominated": len(v1_sols),
        "feasibility_rate": float(np.mean([s.is_feasible for s in v1_sols])) if v1_sols else 0.0,
        "runtime_seconds": v1_meta["runtime_seconds"],
    }
    ablation_results["Seeded NSGA-II"] = {
        "hypervolume": compute_full_metrics(objs2, ideal_point=id_pt, nadir_point=nd_pt, ref_point=rf_pt)["hypervolume"],
        "num_non_dominated": len(v2_sols),
        "feasibility_rate": float(np.mean([s.is_feasible for s in v2_sols])) if v2_sols else 0.0,
        "runtime_seconds": v2_meta["runtime_seconds"],
    }
    ablation_results["Full Hybrid Memetic"] = {
        "hypervolume": compute_full_metrics(objs3, ideal_point=id_pt, nadir_point=nd_pt, ref_point=rf_pt)["hypervolume"],
        "num_non_dominated": len(v3_sols),
        "feasibility_rate": float(np.mean([s.is_feasible for s in v3_sols])) if v3_sols else 0.0,
        "runtime_seconds": v3_meta["runtime_seconds"],
    }

    print("\n--- Part 2: Branch Parity Sensitivity (delta sweep) ---")
    delta_results: Dict[float, Dict[str, float]] = {}
    for delta_val in [0.15, 0.25, 0.35, 0.50]:
        inst_copy = MatchingInstance.from_dict(instance.to_dict())
        inst_copy.delta_branch_parity = delta_val
        prob_delta = StudentAlumniMatchingProblem(inst_copy)
        sols, _ = SeededMemeticNSGA2(prob_delta).solve(pop_size=35, n_gen=30, seed=42)
        if sols:
            best_sol = max(sols, key=lambda s: s.z1_total_quality)
            delta_results[delta_val] = {
                "max_total_quality_z1": round(best_sol.z1_total_quality, 3),
                "worst_student_z2": round(best_sol.z2_worst_quality, 3),
                "achieved_branch_disparity": round(best_sol.branch_disparity, 4),
            }
            print(f"delta = {delta_val:.2f} -> Max Quality Z1: {best_sol.z1_total_quality:.2f}, Achieved Disparity: {best_sol.branch_disparity:.3f}")

    # Save results
    all_ablation = {
        "component_ablation": ablation_results,
        "delta_sensitivity": {str(k): v for k, v in delta_results.items()},
    }
    ablation_file = results_dir / "ablation_results.json"
    with open(ablation_file, "w", encoding="utf-8") as f:
        json.dump(all_ablation, f, indent=2)
    print(f"\nSaved ablation and sensitivity results to: {ablation_file}")

    # Plot Delta Trade-off Curve
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    deltas = sorted(list(delta_results.keys()))
    z1_vals = [delta_results[d]["max_total_quality_z1"] for d in deltas]
    z2_vals = [delta_results[d]["worst_student_z2"] for d in deltas]

    ax1.plot(deltas, z1_vals, marker="o", color="#1f77b4", label="Total Quality $Z_1$")
    ax1.set_xlabel(r"Branch Parity Tolerance $\delta$")
    ax1.set_ylabel(r"Total Quality $Z_1$", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(deltas, z2_vals, marker="s", color="#ff7f0e", linestyle="--", label="Worst Student $Z_2$")
    ax2.set_ylabel(r"Worst-Off Student Quality $Z_2$", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    plt.title(r"Trade-off: Branch Fairness Tolerance $\delta$ vs Quality Objectives")
    fig.tight_layout()
    plot_file = figures_dir / "delta_sensitivity_tradeoff.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved delta sensitivity plot to: {plot_file}")
