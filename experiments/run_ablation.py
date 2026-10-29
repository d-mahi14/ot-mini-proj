"""Ablation and Sensitivity Analysis for VJTI Student-Alumni Matching.

Systematically evaluates:
1. Algorithmic Component Ablation across 30 seeds:
   - Seeding Only (Heuristic Seeding + NSGA-II, no memetic local search)
   - Local Search Only (Uniform random init + NSGA-II + Memetic Local Search)
   - Both / Full Hybrid (Heuristic Seeding + NSGA-II + Memetic Local Search)
   Saves raw runs to results/ablation_runs.csv with paired Wilcoxon signed-rank tests.
2. Policy parameters:
   - Branch Parity Disparity Tolerance delta in {0.15, 0.25, 0.35, 0.50}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiments import load_or_create_instance
from src.data_generator import MatchingInstance
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.metrics import (
    compute_full_metrics,
    compute_hypervolume,
    compute_shared_reference_setup,
    compute_spacing,
    compute_wilcoxon_with_effect_size,
    extract_non_dominated_front,
)
from src.problem import MatchingSolution, StudentAlumniMatchingProblem


def run_ablation_study(
    config_path: str = "experiments/configs/small_50x30.yaml",
    max_seeds: int | None = None,
) -> Dict[str, Any]:
    """Run full component ablation over 30 seeds and parameter sensitivity analysis."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    instance = load_or_create_instance(cfg["instance"])
    base_problem = StudentAlumniMatchingProblem(instance)

    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_seeds: List[int] = cfg.get("seeds", [42])
    seeds = raw_seeds[:max_seeds] if max_seeds is not None else raw_seeds

    print(f"\n================================================================================")
    print(f" ABLATION STUDY: 3 Components x {len(seeds)} Seeds ({instance.name})")
    print(f" Variants: [1] Seeding Only, [2] Local Search Only, [3] Both (Full Hybrid)")
    print(f"================================================================================\n")

    h_cfg = cfg.get("algorithms", {}).get("hybrid_memetic", {})
    pop_size = h_cfg.get("pop_size", 40)
    n_gen = h_cfg.get("n_gen", 35)

    def _eval_ablation_seed(seed: int) -> List[Dict[str, Any]]:
        records = []
        hybrid_solver = SeededMemeticNSGA2(base_problem)

        # 1. Seeding Only
        t0 = time.perf_counter()
        sols_seed, meta_seed = hybrid_solver.solve(
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            use_seeding=True,
            apply_local_search=False,
        )
        t_seed = time.perf_counter() - t0
        records.append({"seed": seed, "variant": "Seeding Only", "solutions": sols_seed, "runtime": t_seed})

        # 2. Local Search Only
        t0 = time.perf_counter()
        sols_ls, meta_ls = hybrid_solver.solve(
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            use_seeding=False,
            apply_local_search=True,
        )
        t_ls = time.perf_counter() - t0
        records.append({"seed": seed, "variant": "Local Search Only", "solutions": sols_ls, "runtime": t_ls})

        # 3. Both (Full Hybrid)
        t0 = time.perf_counter()
        sols_both, meta_both = hybrid_solver.solve(
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            use_seeding=True,
            apply_local_search=True,
        )
        t_both = time.perf_counter() - t0
        records.append({"seed": seed, "variant": "Both (Full Hybrid)", "solutions": sols_both, "runtime": t_both})

        return records

    n_workers = min(4, os.cpu_count() or 4)
    print(f"[1/3] Running Ablation experiments across {len(seeds)} seeds in parallel ({n_workers} threads)...", flush=True)

    all_raw_runs: List[Dict[str, Any]] = []
    completed_cnt = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_eval_ablation_seed, s): s for s in seeds}
        for fut in as_completed(futures):
            res_list = fut.result()
            all_raw_runs.extend(res_list)
            completed_cnt += 1
            if completed_cnt % 5 == 0 or completed_cnt == len(seeds) or completed_cnt == 1:
                print(f"      -> Completed ablation seed {completed_cnt}/{len(seeds)} (seed={futures[fut]})...", flush=True)

    # -------------------------------------------------------------
    # Pool objective vectors for shared normalization
    # -------------------------------------------------------------
    print("[2/3] Constructing shared reference setup across ablation runs...", flush=True)
    pooled_objs: List[np.ndarray] = []
    for item in all_raw_runs:
        for s in item["solutions"]:
            pooled_objs.append(s.objectives)

    ref_setup = compute_shared_reference_setup(pooled_objs)
    ideal_pt = np.array(ref_setup["ideal_point"])
    nadir_pt = np.array(ref_setup["nadir_point"])
    ref_pt = np.array(ref_setup["ref_point"])

    runs_records: List[Dict[str, Any]] = []
    for item in all_raw_runs:
        seed = item["seed"]
        variant = item["variant"]
        sols = item["solutions"]
        t_sec = item["runtime"]

        objs = np.array([s.objectives for s in sols])
        nd_objs, _ = extract_non_dominated_front(objs)

        hv = compute_hypervolume(nd_objs, ideal_point=ideal_pt, nadir_point=nadir_pt, ref_point=ref_pt)
        sp = compute_spacing(nd_objs)
        num_nd = int(nd_objs.shape[0])
        feasibility_rate = float(np.mean([s.is_feasible for s in sols])) if sols else 0.0

        z1_max = float(max(s.z1_total_quality for s in sols)) if sols else 0.0
        z2_worst = float(max(s.z2_worst_quality for s in sols)) if sols else 0.0
        z3_min = int(min(s.z3_max_load for s in sols)) if sols else 0

        runs_records.append({
            "seed": seed,
            "variant": variant,
            "hypervolume": round(hv, 6),
            "spacing": round(sp, 5),
            "num_non_dominated": num_nd,
            "feasibility_rate": round(feasibility_rate, 4),
            "z1_max": round(z1_max, 4),
            "z2_worst": round(z2_worst, 4),
            "z3_min_load": z3_min,
            "runtime_seconds": round(t_sec, 4),
        })

    ablation_df = pd.DataFrame(runs_records)
    ablation_runs_csv = results_dir / "ablation_runs.csv"
    ablation_df.to_csv(ablation_runs_csv, index=False)
    print(f"      Saved raw ablation runs to: {ablation_runs_csv}")

    # Summary by variant
    summary_rows: List[Dict[str, Any]] = []
    for v in ["Seeding Only", "Local Search Only", "Both (Full Hybrid)"]:
        sub = ablation_df[ablation_df["variant"] == v]
        if sub.empty:
            continue
        summary_rows.append({
            "variant": v,
            "hv_mean": round(float(sub["hypervolume"].mean()), 6),
            "hv_std": round(float(sub["hypervolume"].std(ddof=1) if len(sub) > 1 else 0.0), 6),
            "hv_median": round(float(sub["hypervolume"].median()), 6),
            "feasibility_rate": round(float(sub["feasibility_rate"].mean()), 4),
            "z1_mean": round(float(sub["z1_max"].mean()), 4),
            "z2_mean": round(float(sub["z2_worst"].mean()), 4),
            "z3_mean": round(float(sub["z3_min_load"].mean()), 2),
            "spacing_mean": round(float(sub["spacing"].mean()), 5),
            "runtime_mean": round(float(sub["runtime_seconds"].mean()), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    ablation_summary_csv = results_dir / "ablation_summary.csv"
    summary_df.to_csv(ablation_summary_csv, index=False)
    print(f"      Saved ablation summary to: {ablation_summary_csv}")

    # Wilcoxon Tests on Ablation Variants
    ablation_wilcoxon: Dict[str, Any] = {}
    test_pairs = [
        ("Both (Full Hybrid)", "Seeding Only"),
        ("Both (Full Hybrid)", "Local Search Only"),
        ("Seeding Only", "Local Search Only"),
    ]

    print("[3/3] Performing Wilcoxon hypothesis tests across ablation variants...", flush=True)
    for v1, v2 in test_pairs:
        s1 = ablation_df[ablation_df["variant"] == v1].sort_values("seed")["hypervolume"].values
        s2 = ablation_df[ablation_df["variant"] == v2].sort_values("seed")["hypervolume"].values
        res = compute_wilcoxon_with_effect_size(s1, s2)
        key = f"{v1}_vs_{v2}"
        ablation_wilcoxon[key] = res
        sig_str = "SIGNIFICANT (p < 0.05)" if res["significant_0_05"] else "NOT significant"
        print(f"      Wilcoxon {v1} vs {v2}: p = {res['p_value']:.4e}, effect size r = {res['effect_size_r']:+.4f} -> {sig_str}")

    stat_file = results_dir / "ablation_statistical_tests.json"
    with open(stat_file, "w", encoding="utf-8") as f:
        json.dump(ablation_wilcoxon, f, indent=2)
    print(f"      Saved ablation Wilcoxon test report to: {stat_file}")

    # -------------------------------------------------------------
    # Part 2: Branch Parity Sensitivity (delta sweep)
    # -------------------------------------------------------------
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
            print(f"      delta = {delta_val:.2f} -> Max Z1: {best_sol.z1_total_quality:.2f}, Worst Z2: {best_sol.z2_worst_quality:.3f}, Disparity: {best_sol.branch_disparity:.3f}")

    # Save results
    all_ablation = {
        "component_ablation_summary": summary_rows,
        "ablation_wilcoxon": ablation_wilcoxon,
        "delta_sensitivity": {str(k): v for k, v in delta_results.items()},
    }
    ablation_json = results_dir / "ablation_results.json"
    with open(ablation_json, "w", encoding="utf-8") as f:
        json.dump(all_ablation, f, indent=2)

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
    print(f"      Saved delta sensitivity plot to: {plot_file}")

    return {
        "ablation_df": ablation_df,
        "summary_df": summary_df,
        "wilcoxon": ablation_wilcoxon,
    }


def main() -> None:
    """CLI driver for ablation experiments."""
    parser = argparse.ArgumentParser(description="Run Component Ablation Benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/small_50x30.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Optional override for maximum number of seeds to execute.",
    )
    args = parser.parse_args()
    run_ablation_study(args.config, max_seeds=args.seeds)


if __name__ == "__main__":
    main()
