"""Master experiment runner for Student-Alumni Matching optimization.

Conducts rigorous multi-seed comparative benchmarking across:
- Baselines: Random Feasible, Greedy Score-First, Single-Objective Min-Cost Flow
- Exact Benchmark: HiGHS MILP Pareto sweep (on small instance)
- Metaheuristics: NSGA-II, SPEA2, MOEA/D, Seeded Memetic Hybrid

Evaluates all methods across 30 independent seeds (or 10 seeds for medium/large)
under equal evaluation budgets, using a single shared hypervolume reference setup,
and computes Wilcoxon signed-rank tests with rank-biserial effect sizes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.exact_milp import ExactMILP
from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.baseline.random_baseline import RandomBaseline
from src.data_generator import MatchingInstance, SyntheticDataGenerator
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.metrics import (
    check_front_dominates_point,
    compute_full_metrics,
    compute_gini_coefficient,
    compute_hypervolume,
    compute_shared_reference_setup,
    compute_spacing,
    compute_wilcoxon_with_effect_size,
    extract_non_dominated_front,
)
from src.multiobjective.moead import run_moead
from src.multiobjective.nsga2 import run_nsga2
from src.multiobjective.spea2 import run_spea2
from src.problem import MatchingSolution, StudentAlumniMatchingProblem
from src.visualization import (
    plot_branch_parity,
    plot_hypervolume_boxplots,
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


def run_multi_seed_benchmark(
    config_path: str | Path,
    max_seeds: int | None = None,
) -> Dict[str, Any]:
    """Execute complete multi-seed benchmark according to YAML configuration."""
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
    raw_seeds: List[int] = cfg.get("seeds", [42])
    seeds = raw_seeds[:max_seeds] if max_seeds is not None else raw_seeds

    print(f"\n================================================================================")
    print(f" MULTI-SEED BENCHMARK: {inst_name} ({instance.n_students} students x {instance.n_alumni} alumni)")
    print(f" Independent Seeds ({len(seeds)}): {seeds[:5]}... (Total: {len(seeds)})")
    print(f"================================================================================\n")

    # -------------------------------------------------------------
    # 1. Deterministic Baselines & MILP Ground Truth (Single run)
    # -------------------------------------------------------------
    print("[1/5] Evaluating Deterministic Baselines & Exact Ground Truth...")
    t0 = time.perf_counter()
    greedy_sol = GreedyBaseline(problem).solve()
    t_greedy = time.perf_counter() - t0

    t0 = time.perf_counter()
    mcf_sol = MinCostFlowBaseline(problem).solve()
    t_mcf = time.perf_counter() - t0

    exact_sols: List[MatchingSolution] = []
    t_milp = 0.0
    if "milp" in algo_cfg and instance.n_students <= 100:
        print("      Running Exact MILP Pareto Sweep...")
        m_cfg = algo_cfg["milp"]
        milp_solver = ExactMILP(problem)
        t0 = time.perf_counter()
        exact_sols = milp_solver.trace_pareto_front(
            grid_steps=m_cfg.get("grid_steps", 4),
            time_limit_per_point=m_cfg.get("time_limit_sec", 45.0),
        )
        t_milp = time.perf_counter() - t0
        print(f"      Exact MILP found {len(exact_sols)} Pareto points in {t_milp:.2f}s.")

    # -------------------------------------------------------------
    # 2. Multi-Seed Stochastic Algorithm Execution (Concurrent)
    # -------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    all_runs_raw: List[Dict[str, Any]] = []

    def _eval_single_seed(s_val: int) -> List[Dict[str, Any]]:
        n_cfg = algo_cfg.get("nsga2", {})
        sp_cfg = algo_cfg.get("spea2", {})
        mo_cfg = algo_cfg.get("moead", {})
        h_cfg = algo_cfg.get("hybrid_memetic", {})

        records: List[Dict[str, Any]] = []

        # Baseline: Random (stochastic per seed)
        t0 = time.perf_counter()
        rand_sol = RandomBaseline(problem, seed=s_val).solve(n_trials=5)
        t_rand = time.perf_counter() - t0
        records.append({"seed": s_val, "method": "Random", "solutions": [rand_sol], "runtime": t_rand})

        # Baseline: Greedy (replicated per seed for paired testing)
        records.append({"seed": s_val, "method": "Greedy", "solutions": [greedy_sol], "runtime": t_greedy})

        # Baseline: MCF (replicated per seed for paired testing)
        records.append({"seed": s_val, "method": "Min-Cost Flow", "solutions": [mcf_sol], "runtime": t_mcf})

        # NSGA-II
        nsga_sols, nsga_meta = run_nsga2(
            problem,
            pop_size=n_cfg.get("pop_size", 40),
            n_gen=n_cfg.get("n_gen", 40),
            seed=s_val,
        )
        records.append({"seed": s_val, "method": "NSGA-II", "solutions": nsga_sols, "runtime": nsga_meta["runtime_seconds"]})

        # SPEA2
        spea_sols, spea_meta = run_spea2(
            problem,
            pop_size=sp_cfg.get("pop_size", 40),
            n_gen=sp_cfg.get("n_gen", 40),
            seed=s_val,
        )
        records.append({"seed": s_val, "method": "SPEA2", "solutions": spea_sols, "runtime": spea_meta["runtime_seconds"]})

        # MOEA/D
        moead_sols, moead_meta = run_moead(
            problem,
            n_partitions=mo_cfg.get("n_partitions", 6),
            n_gen=mo_cfg.get("n_gen", 57),
            seed=s_val,
        )
        records.append({"seed": s_val, "method": "MOEA/D", "solutions": moead_sols, "runtime": moead_meta["runtime_seconds"]})

        # Hybrid: Seeded + Memetic NSGA-II
        hybrid_solver_seed = SeededMemeticNSGA2(problem)
        hybrid_sols, hybrid_meta = hybrid_solver_seed.solve(
            pop_size=h_cfg.get("pop_size", 40),
            n_gen=h_cfg.get("n_gen", 35),
            seed=s_val,
            apply_local_search=h_cfg.get("apply_local_search", True),
        )
        records.append({"seed": s_val, "method": "Hybrid", "solutions": hybrid_sols, "runtime": hybrid_meta["runtime_seconds"]})

        return records

    n_workers = min(4, os.cpu_count() or 4)
    print(f"[2/5] Running Stochastic Algorithms across {len(seeds)} seeds in parallel ({n_workers} worker threads)...", flush=True)

    completed_cnt = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_eval_single_seed, s): s for s in seeds}
        for fut in as_completed(futures):
            res_list = fut.result()
            all_runs_raw.extend(res_list)
            completed_cnt += 1
            if completed_cnt % 5 == 0 or completed_cnt == len(seeds) or completed_cnt == 1:
                print(f"      -> Completed seed {completed_cnt}/{len(seeds)} (seed={futures[fut]})...", flush=True)

    # -------------------------------------------------------------
    # 3. Establish Shared Hypervolume Reference Frame
    # -------------------------------------------------------------
    print("[3/5] Pooling all objective fronts to construct Shared Reference Frame...")
    pooled_objs: List[np.ndarray] = []
    for item in all_runs_raw:
        for sol in item["solutions"]:
            pooled_objs.append(sol.objectives)
    if exact_sols:
        for sol in exact_sols:
            pooled_objs.append(sol.objectives)

    ref_setup = compute_shared_reference_setup(pooled_objs)
    ideal_pt = np.array(ref_setup["ideal_point"])
    nadir_pt = np.array(ref_setup["nadir_point"])
    ref_pt = np.array(ref_setup["ref_point"])  # Fixed at 1.1

    # Persist reference frame
    ref_file = results_dir / f"{inst_name}_hv_reference.json"
    with open(ref_file, "w", encoding="utf-8") as f:
        json.dump({
            "instance": inst_name,
            "ideal_point": ref_setup["ideal_point"],
            "nadir_point": ref_setup["nadir_point"],
            "ref_point": ref_setup["ref_point"],
            "total_solutions_pooled": len(pooled_objs),
        }, f, indent=2)
    print(f"      Saved shared reference setup to: {ref_file}")

    # -------------------------------------------------------------
    # 4. Standardized Metrics & Dominance Analysis across Seeds
    # -------------------------------------------------------------
    print("[4/5] Computing normalized metrics and dominance statistics...")
    runs_records: List[Dict[str, Any]] = []

    greedy_obj = greedy_sol.objectives
    mcf_obj = mcf_sol.objectives

    for item in all_runs_raw:
        seed = item["seed"]
        method = item["method"]
        sols = item["solutions"]
        t_sec = item["runtime"]

        objs = np.array([s.objectives for s in sols])
        nd_objs, _ = extract_non_dominated_front(objs)

        hv = compute_hypervolume(nd_objs, ideal_point=ideal_pt, nadir_point=nadir_pt, ref_point=ref_pt)
        sp = compute_spacing(nd_objs)
        num_nd = int(nd_objs.shape[0])
        feasibility_rate = float(np.mean([s.is_feasible for s in sols]))

        z1_max = float(max(s.z1_total_quality for s in sols))
        z2_worst = float(max(s.z2_worst_quality for s in sols))
        z3_min_load = int(min(s.z3_max_load for s in sols))
        avg_gini = float(np.mean([compute_gini_coefficient(s.student_qualities) for s in sols]))

        # Dominance checks against single-point baselines
        rand_sol_for_seed = next(
            (r["solutions"][0] for r in all_runs_raw if r["seed"] == seed and r["method"] == "Random"),
            None,
        )
        rand_obj = rand_sol_for_seed.objectives if rand_sol_for_seed else np.array([0.0, 0.0, 100.0])

        dom_rand = check_front_dominates_point(nd_objs, rand_obj)
        dom_greedy = check_front_dominates_point(nd_objs, greedy_obj)
        dom_mcf = check_front_dominates_point(nd_objs, mcf_obj)

        runs_records.append({
            "seed": seed,
            "method": method,
            "hypervolume": round(hv, 6),
            "spacing": round(sp, 5),
            "num_non_dominated": num_nd,
            "feasibility_rate": round(feasibility_rate, 4),
            "z1_max": round(z1_max, 4),
            "z2_worst": round(z2_worst, 4),
            "z3_min_load": z3_min_load,
            "avg_gini_quality": round(avg_gini, 4),
            "runtime_seconds": round(t_sec, 4),
            "dominates_random": bool(dom_rand),
            "dominates_greedy": bool(dom_greedy),
            "dominates_mcf": bool(dom_mcf),
        })

    runs_df = pd.DataFrame(runs_records)
    runs_csv = results_dir / f"{inst_name}_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"      Saved raw run-level benchmark to: {runs_csv}")

    # Compute Summary Statistics by Method
    summary_rows: List[Dict[str, Any]] = []
    methods_order = ["Random", "Greedy", "Min-Cost Flow", "NSGA-II", "SPEA2", "MOEA/D", "Hybrid"]

    for m in methods_order:
        sub = runs_df[runs_df["method"] == m]
        if sub.empty:
            continue
        summary_rows.append({
            "method": m,
            "hv_mean": round(float(sub["hypervolume"].mean()), 6),
            "hv_std": round(float(sub["hypervolume"].std(ddof=1) if len(sub) > 1 else 0.0), 6),
            "hv_median": round(float(sub["hypervolume"].median()), 6),
            "feasibility_rate": round(float(sub["feasibility_rate"].mean()), 4),
            "z1_mean": round(float(sub["z1_max"].mean()), 4),
            "z1_std": round(float(sub["z1_max"].std(ddof=1) if len(sub) > 1 else 0.0), 4),
            "z2_mean": round(float(sub["z2_worst"].mean()), 4),
            "z2_std": round(float(sub["z2_worst"].std(ddof=1) if len(sub) > 1 else 0.0), 4),
            "z3_mean": round(float(sub["z3_min_load"].mean()), 2),
            "z3_std": round(float(sub["z3_min_load"].std(ddof=1) if len(sub) > 1 else 0.0), 2),
            "spacing_mean": round(float(sub["spacing"].mean()), 5),
            "runtime_mean": round(float(sub["runtime_seconds"].mean()), 4),
            "runtime_std": round(float(sub["runtime_seconds"].std(ddof=1) if len(sub) > 1 else 0.0), 4),
            "pct_dominates_random": round(100.0 * float(sub["dominates_random"].mean()), 1),
            "pct_dominates_greedy": round(100.0 * float(sub["dominates_greedy"].mean()), 1),
            "pct_dominates_mcf": round(100.0 * float(sub["dominates_mcf"].mean()), 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / f"{inst_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"      Saved summary statistics to: {summary_csv}")

    # -------------------------------------------------------------
    # 5. Wilcoxon Signed-Rank Hypothesis Tests & Effect Sizes
    # -------------------------------------------------------------
    print("[5/5] Performing Wilcoxon statistical significance testing...")
    stat_comparisons = [
        ("Hybrid", "NSGA-II"),
        ("Hybrid", "SPEA2"),
        ("Hybrid", "MOEA/D"),
        ("NSGA-II", "Random"),
        ("NSGA-II", "Greedy"),
        ("NSGA-II", "Min-Cost Flow"),
    ]

    wilcoxon_results: Dict[str, Any] = {}
    for a, b in stat_comparisons:
        sub_a = runs_df[runs_df["method"] == a].sort_values("seed")
        sub_b = runs_df[runs_df["method"] == b].sort_values("seed")

        if not sub_a.empty and not sub_b.empty:
            hv_a = sub_a["hypervolume"].values
            hv_b = sub_b["hypervolume"].values
            test_res = compute_wilcoxon_with_effect_size(hv_a, hv_b)
            key = f"{a}_vs_{b}"
            wilcoxon_results[key] = test_res
            sig_flag = "SIGNIFICANT (p < 0.05)" if test_res["significant_0_05"] else "NOT significant"
            print(f"      Wilcoxon {a} vs {b}: p = {test_res['p_value']:.4e}, effect size r = {test_res['effect_size_r']:+.4f} -> {sig_flag}")

    stat_json = results_dir / f"{inst_name}_statistical_tests.json"
    with open(stat_json, "w", encoding="utf-8") as f:
        json.dump(wilcoxon_results, f, indent=2)
    print(f"      Saved statistical test report to: {stat_json}")

    # Generate Figures & Visualizations
    boxplot_png = figures_dir / f"{inst_name}_hv_boxplot.png"
    plot_hypervolume_boxplots(
        runs_df,
        boxplot_png,
        title=f"Hypervolume Distribution ({inst_name}, {len(seeds)} Seeds)",
    )
    print(f"      Saved Hypervolume boxplot figure to: {boxplot_png}")

    # Save Comparison Table in CSV and Markdown formats into results/ and figures/
    csv_results = results_dir / f"{inst_name}_comparison_table.csv"
    csv_figures = figures_dir / f"{inst_name}_comparison_table.csv"
    summary_df.to_csv(csv_results, index=False)
    summary_df.to_csv(csv_figures, index=False)

    table_md_results = results_dir / f"{inst_name}_comparison_table.md"
    table_md_figures = figures_dir / f"{inst_name}_comparison_table.md"
    try:
        md_content = summary_df.to_markdown(index=False)
    except Exception:
        md_content = summary_df.to_csv(sep="|", index=False)

    full_md_text = f"# Benchmark Comparison Table: {inst_name} ({len(seeds)} Seeds)\n\n{md_content}\n\n### Wilcoxon Statistical Hypothesis Tests on Hypervolume\n\n"
    for k, v in wilcoxon_results.items():
        full_md_text += f"- **{k}**: p-value = `{v['p_value']:.4e}`, rank-biserial $r$ = `{v['effect_size_r']}`, significant = `{v['significant_0_05']}`\n"

    with open(table_md_results, "w", encoding="utf-8") as f:
        f.write(full_md_text)
    with open(table_md_figures, "w", encoding="utf-8") as f:
        f.write(full_md_text)
    print(f"      Saved comparison tables to: {csv_results} and {table_md_results}")

    # Return key benchmark dictionary
    return {
        "runs_df": runs_df,
        "summary_df": summary_df,
        "wilcoxon": wilcoxon_results,
        "shared_reference": ref_setup,
    }


def main() -> None:
    """CLI driver for multi-seed experiments."""
    parser = argparse.ArgumentParser(description="Run Multi-Seed VJTI Student-Alumni Matching Benchmark.")
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
    run_multi_seed_benchmark(args.config, max_seeds=args.seeds)


if __name__ == "__main__":
    main()
