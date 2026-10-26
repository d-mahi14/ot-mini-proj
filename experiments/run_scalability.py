"""Scalability benchmark evaluating runtime across Small, Medium, and Large instances.

Evaluates computational behavior:
- Small: 50x30
- Medium: 300x150
- Large: 1000x500
Saves results and generates empirical scaling curves.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from experiments.run_experiments import load_or_create_instance
from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.baseline.random_baseline import RandomBaseline
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.multiobjective.nsga2 import run_nsga2
from src.multiobjective.spea2 import run_spea2
from src.problem import StudentAlumniMatchingProblem
from src.visualization import plot_scalability_curve


def run_scalability_benchmark() -> None:
    """Execute scalability experiment across small, medium, and large instances."""
    configs = [
        ("Small (50x30)", "experiments/configs/small_50x30.yaml"),
        ("Medium (300x150)", "experiments/configs/medium_300x150.yaml"),
        ("Large (1000x500)", "experiments/configs/large_1000x500.yaml"),
    ]

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    instance_names: List[str] = []
    algo_runtimes: Dict[str, List[float]] = {
        "Greedy": [],
        "Min-Cost Flow": [],
        "NSGA-II": [],
        "SPEA2": [],
        "Hybrid Memetic": [],
    }

    for label, cfg_path in configs:
        print(f"\n==========================================")
        print(f"Benchmarking Scalability on: {label}")
        print(f"==========================================")
        instance_names.append(label)

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        instance = load_or_create_instance(cfg["instance"])
        problem = StudentAlumniMatchingProblem(instance)

        # 1. Greedy
        t0 = time.perf_counter()
        GreedyBaseline(problem).solve()
        t_greedy = time.perf_counter() - t0
        algo_runtimes["Greedy"].append(round(t_greedy, 4))
        print(f"Greedy: {t_greedy:.4f}s")

        # 2. Min-Cost Flow
        t0 = time.perf_counter()
        MinCostFlowBaseline(problem).solve()
        t_mcf = time.perf_counter() - t0
        algo_runtimes["Min-Cost Flow"].append(round(t_mcf, 4))
        print(f"Min-Cost Flow: {t_mcf:.4f}s")

        # For medium and large, adapt iterations slightly for responsive benchmark
        n_gen = 25 if instance.n_students > 500 else 30
        pop_size = 35 if instance.n_students > 500 else 40

        # 3. NSGA-II
        t0 = time.perf_counter()
        run_nsga2(problem, pop_size=pop_size, n_gen=n_gen, seed=42)
        t_nsga = time.perf_counter() - t0
        algo_runtimes["NSGA-II"].append(round(t_nsga, 4))
        print(f"NSGA-II ({n_gen} gens): {t_nsga:.4f}s")

        # 4. SPEA2
        t0 = time.perf_counter()
        run_spea2(problem, pop_size=pop_size, n_gen=n_gen, seed=42)
        t_spea = time.perf_counter() - t0
        algo_runtimes["SPEA2"].append(round(t_spea, 4))
        print(f"SPEA2 ({n_gen} gens): {t_spea:.4f}s")

        # 5. Hybrid Memetic NSGA-II
        t0 = time.perf_counter()
        SeededMemeticNSGA2(problem).solve(pop_size=pop_size, n_gen=n_gen, seed=42, apply_local_search=True)
        t_hybrid = time.perf_counter() - t0
        algo_runtimes["Hybrid Memetic"].append(round(t_hybrid, 4))
        print(f"Hybrid Memetic: {t_hybrid:.4f}s")

    # Save scalability results
    out_file = results_dir / "scalability_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "instance_sizes": instance_names,
            "runtimes_seconds": algo_runtimes,
        }, f, indent=2)
    print(f"\nSaved scalability results to {out_file}")

    # Plot scalability curve
    plot_scalability_curve(
        instance_names,
        algo_runtimes,
        figures_dir / "scalability_curve.png",
    )
    print(f"Saved scalability curve plot to {figures_dir / 'scalability_curve.png'}")


if __name__ == "__main__":
    run_scalability_benchmark()
