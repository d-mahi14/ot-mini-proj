"""Comprehensive Sensitivity Analysis and Coverage-Gap Analysis for VJTI Matching.

Evaluates system sensitivity to:
1. Compatibility Matrix q_ij Weight Vectors (5 combinations: Balanced, Branch-Heavy,
   Domain-Heavy, Experience-Heavy, Equity).
2. Degree Requirement K in {2, 3, 5}.
3. Minimum Student Quality Floor theta in {0.50, 0.60, 0.70, 0.80}.
4. Branch Parity Disparity Tolerance delta in {0.15, 0.25, 0.35, 0.50}.
5. Alumni Mentorship Capacity Scale Factor in {0.8, 1.0, 1.2, 1.5}.

For each setting, evaluates NSGA-II and Hybrid Memetic across 10 independent seeds,
reporting Hypervolume, Worst-Student Quality (Z2), Gini Index, and Max Load (Z3).
Outputs results/sensitivity.csv, results/coverage_gap.csv, and publication figures in figures/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.nsga2 import run_nsga2
from src.data_generator import (
    BRANCH_AFFINITY,
    BRANCHES,
    COMPANIES,
    DOMAIN_BRANCH_AFFINITY,
    DOMAINS,
    AlumniProfile,
    MatchingInstance,
    StudentProfile,
)
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.metrics import (
    compute_gini_coefficient,
    compute_hypervolume,
    compute_shared_reference_setup,
    extract_non_dominated_front,
)
from src.problem import MatchingSolution, StudentAlumniMatchingProblem


# ==============================================================================
# 1. Compatibility Matrix & Variant Builders
# ==============================================================================

def compute_weighted_q_matrix(
    students: List[StudentProfile],
    alumni: List[AlumniProfile],
    w_branch: float = 0.45,
    w_domain: float = 0.35,
    w_exp: float = 0.10,
    seed: int = 42,
) -> np.ndarray:
    """Compute compatibility matrix q_ij using specified component weights."""
    rng = np.random.default_rng(seed)
    n = len(students)
    m = len(alumni)
    q_mat = np.zeros((n, m), dtype=float)

    for i, stu in enumerate(students):
        for j, alm in enumerate(alumni):
            # Branch alignment score
            if stu.branch == alm.branch:
                b_score = 1.0
            elif alm.branch in BRANCH_AFFINITY.get(stu.branch, []):
                b_score = 0.667
            else:
                b_score = 0.267

            # Domain / Industry alignment score
            if stu.target_domain == alm.industry:
                d_score = 1.0
            elif alm.industry in DOMAIN_BRANCH_AFFINITY.get(stu.target_domain, []):
                d_score = 0.571
            else:
                d_score = 0.286

            # Seniority & experience modulation in [0, 1]
            exp_score = min(alm.years_experience / 20.0, 1.0)

            # Stochastic noise in [-0.06, +0.08]
            noise = float(rng.uniform(-0.06, 0.08))

            raw_score = w_branch * b_score + w_domain * d_score + w_exp * exp_score + noise
            q_ij = np.clip(raw_score, 0.05, 0.98)
            q_mat[i, j] = round(float(q_ij), 4)

    return q_mat


def build_variant_instance(
    base: MatchingInstance,
    name_suffix: str = "",
    q_weights: Optional[Tuple[float, float, float]] = None,
    k_matches: Optional[int] = None,
    theta: Optional[float] = None,
    delta: Optional[float] = None,
    capacity_scale: Optional[float] = None,
    seed: int = 42,
) -> MatchingInstance:
    """Construct an adjusted MatchingInstance variant for sensitivity evaluation."""
    new_k = k_matches if k_matches is not None else base.k_matches
    new_theta = theta if theta is not None else base.theta_min_quality
    new_delta = delta if delta is not None else base.delta_branch_parity

    # Deep copy alumni to safely scale capacities
    scaled_alumni: List[AlumniProfile] = []
    for a in base.alumni:
        cap = a.capacity
        # Scale for K if K is varied (ensure proportional capacity scaling)
        if k_matches is not None and k_matches != base.k_matches:
            k_mult = float(k_matches) / float(base.k_matches)
            cap = int(np.ceil(cap * k_mult))

        # Scale by capacity scale factor if specified
        if capacity_scale is not None and capacity_scale != 1.0:
            cap = max(2, int(round(cap * capacity_scale)))

        scaled_alumni.append(
            AlumniProfile(
                id=a.id,
                branch=a.branch,
                industry=a.industry,
                company=a.company,
                capacity=cap,
                consent=a.consent,
                years_experience=a.years_experience,
            )
        )

    # Recompute Q matrix if weights specified
    if q_weights is not None:
        wb, wd, we = q_weights
        new_q = compute_weighted_q_matrix(base.students, scaled_alumni, wb, wd, we, seed=seed)
    else:
        new_q = base.q_matrix.copy()

    # Scale theta proportionally if K changed and theta was not explicitly set
    if k_matches is not None and k_matches != base.k_matches and theta is None:
        new_theta = round(base.theta_min_quality * (float(k_matches) / float(base.k_matches)), 4)

    var_name = f"{base.name}_{name_suffix}" if name_suffix else base.name

    return MatchingInstance(
        name=var_name,
        n_students=base.n_students,
        n_alumni=base.n_alumni,
        k_matches=new_k,
        q_min=base.q_min,
        theta_min_quality=new_theta,
        delta_branch_parity=new_delta,
        max_company_share=base.max_company_share,
        branches=base.branches,
        students=copy.deepcopy(base.students),
        alumni=scaled_alumni,
        q_matrix=new_q,
    )


# ==============================================================================
# 2. Coverage-Gap Analysis Report & Plot
# ==============================================================================

def generate_coverage_gap_report(
    instance: MatchingInstance,
    output_csv: str = "results/coverage_gap.csv",
    figure_path: str = "figures/coverage_gap.png",
) -> pd.DataFrame:
    """Analyze academic branches, career skills, and industries for deficits and unmet demand.

    Identifies structural deficiencies where student demand heavily outstrips
    alumni supply capacity, or where average compatibility scores are severely depressed.
    """
    records: List[Dict[str, Any]] = []
    k = instance.k_matches
    q_mat = instance.q_matrix
    consenting_alumni = [a for a in instance.alumni if a.consent == 1]

    # -------------------------------------------------------------
    # A. Academic Department (Branch) Analysis
    # -------------------------------------------------------------
    for b in instance.branches:
        stu_indices = [i for i, s in enumerate(instance.students) if s.branch == b]
        n_stu = len(stu_indices)
        demand = n_stu * k

        alm_in_b = [a for a in consenting_alumni if a.branch == b]
        m_alm = len(alm_in_b)
        supply_cap = sum(a.capacity for a in alm_in_b)
        unmet_demand = max(0, demand - supply_cap)
        ratio = round(supply_cap / max(1, demand), 3)

        # Average match quality for students of branch b across all consenting alumni
        if n_stu > 0:
            mean_q = float(np.mean(q_mat[stu_indices, :]))
        else:
            mean_q = 0.0

        # Deficit severity score: combines unmet demand and quality penalty
        severity = round(unmet_demand * (1.0 - mean_q) + max(0.0, 1.0 - ratio) * 10.0, 2)

        records.append({
            "category": "branch",
            "name": b,
            "student_count": n_stu,
            "demand_matches": demand,
            "alumni_count": m_alm,
            "supply_capacity": supply_cap,
            "unmet_demand": unmet_demand,
            "supply_demand_ratio": ratio,
            "mean_compatibility_qij": round(mean_q, 4),
            "deficit_severity_score": severity,
        })

    # -------------------------------------------------------------
    # B. Career Skills / Target Domain Analysis
    # -------------------------------------------------------------
    for d in DOMAINS:
        stu_indices = [i for i, s in enumerate(instance.students) if s.target_domain == d]
        n_stu = len(stu_indices)
        demand = n_stu * k

        alm_in_d = [a for a in consenting_alumni if a.industry == d]
        m_alm = len(alm_in_d)
        supply_cap = sum(a.capacity for a in alm_in_d)
        unmet_demand = max(0, demand - supply_cap)
        ratio = round(supply_cap / max(1, demand), 3)

        if n_stu > 0:
            mean_q = float(np.mean(q_mat[stu_indices, :]))
        else:
            mean_q = 0.0

        severity = round(unmet_demand * (1.0 - mean_q) + max(0.0, 1.0 - ratio) * 10.0, 2)

        records.append({
            "category": "skill_domain",
            "name": d,
            "student_count": n_stu,
            "demand_matches": demand,
            "alumni_count": m_alm,
            "supply_capacity": supply_cap,
            "unmet_demand": unmet_demand,
            "supply_demand_ratio": ratio,
            "mean_compatibility_qij": round(mean_q, 4),
            "deficit_severity_score": severity,
        })

    # -------------------------------------------------------------
    # C. Industry Employment Analysis
    # -------------------------------------------------------------
    for comp in COMPANIES:
        alm_in_c = [a for a in consenting_alumni if a.company == comp]
        m_alm = len(alm_in_c)
        supply_cap = sum(a.capacity for a in alm_in_c)
        # Average quality offered by mentors from this company
        alm_indices = [j for j, a in enumerate(instance.alumni) if a.company == comp and a.consent == 1]
        mean_q = float(np.mean(q_mat[:, alm_indices])) if alm_indices else 0.0

        records.append({
            "category": "industry_company",
            "name": comp,
            "student_count": 0,
            "demand_matches": 0,
            "alumni_count": m_alm,
            "supply_capacity": supply_cap,
            "unmet_demand": 0,
            "supply_demand_ratio": 1.0,
            "mean_compatibility_qij": round(mean_q, 4),
            "deficit_severity_score": 0.0,
        })

    df = pd.DataFrame(records)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[Coverage Gap] Saved coverage-gap report to: {out_path}")

    # Generate Figure
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    # Panel 1: Academic Department Coverage Gaps
    df_b = df[df["category"] == "branch"].sort_values("deficit_severity_score", ascending=False)
    x_b = np.arange(len(df_b))
    width = 0.35

    ax0_twin = axes[0].twinx()
    axes[0].bar(x_b - width / 2, df_b["demand_matches"], width, label="Student Demand ($N_b \\times K$)", color="#1f77b4", alpha=0.85)
    axes[0].bar(x_b + width / 2, df_b["supply_capacity"], width, label="Alumni Supply Capacity", color="#2ca02c", alpha=0.85)
    ax0_twin.plot(x_b, df_b["mean_compatibility_qij"], color="#d62728", marker="o", linewidth=2.2, label=r"Mean Match Quality $\bar{q}_{ij}$")

    axes[0].set_title("A. Academic Department Mentorship Coverage Gaps (VJTI Branches)", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x_b)
    axes[0].set_xticklabels(df_b["name"])
    axes[0].set_ylabel("Match Slots (Count)", fontsize=11)
    ax0_twin.set_ylabel(r"Mean Compatibility Score $\bar{q}_{ij}$", color="#d62728", fontsize=11)
    ax0_twin.tick_params(axis="y", labelcolor="#d62728")
    axes[0].legend(loc="upper left")
    ax0_twin.legend(loc="upper right")

    # Panel 2: Career Domain / Skill Gaps
    df_d = df[df["category"] == "skill_domain"].sort_values("unmet_demand", ascending=False)
    x_d = np.arange(len(df_d))

    ax1_twin = axes[1].twinx()
    axes[1].bar(x_d - width / 2, df_d["demand_matches"], width, label="Student Demand", color="#9467bd", alpha=0.85)
    axes[1].bar(x_d + width / 2, df_d["supply_capacity"], width, label="Alumni Supply", color="#8c564b", alpha=0.85)
    ax1_twin.plot(x_d, df_d["mean_compatibility_qij"], color="#e377c2", marker="s", linewidth=2.2, label=r"Mean Quality $\bar{q}_{ij}$")

    axes[1].set_title("B. Career Skills & Domains: Mentorship Demand vs Alumni Supply", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x_d)
    axes[1].set_xticklabels(df_d["name"], rotation=15, ha="right")
    axes[1].set_ylabel("Match Slots (Count)", fontsize=11)
    ax1_twin.set_ylabel(r"Mean Compatibility Score $\bar{q}_{ij}$", color="#e377c2", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="#e377c2")
    axes[1].legend(loc="upper left")
    ax1_twin.legend(loc="upper right")

    # Panel 3: Unmet Demand / Deficit Severity
    combined_deficits = pd.concat([df_b, df_d]).sort_values("deficit_severity_score", ascending=True)
    colors = ["#d62728" if c == "branch" else "#ff7f0e" for c in combined_deficits["category"]]
    axes[2].barh(combined_deficits["name"], combined_deficits["deficit_severity_score"], color=colors, alpha=0.85)
    axes[2].set_title("C. Deficit Severity Ranking (Red: Academic Branch, Orange: Career Domain)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Deficit Severity Score (Unmet Demand $\\times$ Low Compatibility)", fontsize=11)

    fig.tight_layout()
    fig_path = Path(figure_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Coverage Gap] Saved coverage-gap figure to: {fig_path}")

    return df


# ==============================================================================
# 3. Sensitivity Experiment Runner
# ==============================================================================

def run_sensitivity_analysis(
    config_path: str = "experiments/configs/small_50x30.yaml",
    n_seeds: int = 10,
    pop_size: int = 30,
    n_gen: int = 20,
    output_csv: str = "results/sensitivity.csv",
) -> pd.DataFrame:
    """Execute multi-seed sensitivity analysis across all key parameter dimensions."""
    results_dir = Path("results")
    figures_dir = Path("figures")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inst_cfg = cfg["instance"]
    base_file = Path(inst_cfg["file_path"])
    if base_file.exists():
        base_inst = MatchingInstance.load_json(base_file)
    else:
        from experiments.run_experiments import load_or_create_instance
        base_inst = load_or_create_instance(inst_cfg)

    # 1. Generate Coverage Gap Report & Figure first
    generate_coverage_gap_report(
        base_inst,
        output_csv="results/coverage_gap.csv",
        figure_path="figures/coverage_gap.png",
    )

    # 2. Define Parameter Sensitivity Settings
    # Dimension 1: Compatibility q_ij Weight Vectors (5 settings)
    q_weight_settings = [
        ("Balanced (0.45, 0.35, 0.10)", (0.45, 0.35, 0.10)),
        ("Branch-Heavy (0.70, 0.15, 0.05)", (0.70, 0.15, 0.05)),
        ("Domain-Heavy (0.20, 0.70, 0.05)", (0.20, 0.70, 0.05)),
        ("Experience-Heavy (0.30, 0.30, 0.30)", (0.30, 0.30, 0.30)),
        ("Discipline-Career Equity (0.45, 0.45, 0.00)", (0.45, 0.45, 0.00)),
    ]

    # Dimension 2: Degree K in {2, 3, 5}
    k_settings = [2, 3, 5]

    # Dimension 3: Quality Floor theta
    theta_settings = [0.50, 0.60, 0.70, 0.80]

    # Dimension 4: Branch Parity Tolerance delta
    delta_settings = [0.15, 0.25, 0.35, 0.50]

    # Dimension 5: Capacity Scale Factor
    capacity_settings = [0.8, 1.0, 1.2, 1.5]

    seeds = list(range(42, 42 + n_seeds))

    # Collect list of experimental cases to execute
    cases: List[Dict[str, Any]] = []

    for label, weights in q_weight_settings:
        cases.append({
            "param_type": "q_weights",
            "param_value": label,
            "instance": build_variant_instance(base_inst, name_suffix=f"qw_{label[:4]}", q_weights=weights),
        })

    for k in k_settings:
        cases.append({
            "param_type": "k_matches",
            "param_value": f"K={k}",
            "instance": build_variant_instance(base_inst, name_suffix=f"k_{k}", k_matches=k),
        })

    for th in theta_settings:
        cases.append({
            "param_type": "theta",
            "param_value": f"theta={th:.2f}",
            "instance": build_variant_instance(base_inst, name_suffix=f"th_{th}", theta=th),
        })

    for dt in delta_settings:
        cases.append({
            "param_type": "delta",
            "param_value": f"delta={dt:.2f}",
            "instance": build_variant_instance(base_inst, name_suffix=f"dt_{dt}", delta=dt),
        })

    for cs in capacity_settings:
        cases.append({
            "param_type": "capacity_scale",
            "param_value": f"scale={cs:.1f}",
            "instance": build_variant_instance(base_inst, name_suffix=f"cs_{cs}", capacity_scale=cs),
        })

    print(f"\n================================================================================")
    print(f" SENSITIVITY BENCHMARK: {len(cases)} Settings across {n_seeds} Seeds")
    print(f" Algorithms: NSGA-II vs Hybrid Memetic NSGA-II")
    print(f"================================================================================\n")

    runs_data: List[Dict[str, Any]] = []

    # Parallel execution helper per seed
    def _run_single_case_seed(case_dict: Dict[str, Any], s_val: int) -> List[Dict[str, Any]]:
        inst = case_dict["instance"]
        prob = StudentAlumniMatchingProblem(inst)
        ideal_pt = np.array([-float(inst.n_students * inst.k_matches * 1.0), -float(inst.k_matches * 1.0), 1.0])
        nadir_pt = np.array([0.0, 0.0, float(inst.n_alumni)])
        out_records = []

        # 1. NSGA-II
        t0 = time.perf_counter()
        nsga_sols, _ = run_nsga2(prob, pop_size=pop_size, n_gen=n_gen, seed=s_val)
        t_nsga = time.perf_counter() - t0

        if nsga_sols:
            objs = np.array([s.objectives for s in nsga_sols])
            nd_objs, _ = extract_non_dominated_front(objs)
            hv = compute_hypervolume(nd_objs, ideal_point=ideal_pt, nadir_point=nadir_pt)
            worst_q = max(s.z2_worst_quality for s in nsga_sols)
            max_load = min(s.z3_max_load for s in nsga_sols)
            tot_q = max(s.z1_total_quality for s in nsga_sols)
            gini = float(np.mean([compute_gini_coefficient(s.student_qualities) for s in nsga_sols]))
            feas = float(np.mean([s.is_feasible for s in nsga_sols]))
        else:
            hv, worst_q, max_load, tot_q, gini, feas = 0.0, 0.0, 0, 0.0, 0.0, 0.0

        out_records.append({
            "parameter_type": case_dict["param_type"],
            "parameter_value": case_dict["param_value"],
            "algorithm": "NSGA-II",
            "seed": s_val,
            "hypervolume": round(hv, 4),
            "worst_student_quality": round(worst_q, 4),
            "gini_coefficient": round(gini, 4),
            "max_load": max_load,
            "total_quality": round(tot_q, 4),
            "feasibility_rate": round(feas, 3),
            "runtime_seconds": round(t_nsga, 4),
        })

        # 2. Hybrid Memetic NSGA-II
        t0 = time.perf_counter()
        hybrid_solver = SeededMemeticNSGA2(prob)
        hybrid_sols, _ = hybrid_solver.solve(pop_size=pop_size, n_gen=n_gen, seed=s_val)
        t_hybrid = time.perf_counter() - t0

        if hybrid_sols:
            objs = np.array([s.objectives for s in hybrid_sols])
            nd_objs, _ = extract_non_dominated_front(objs)
            hv = compute_hypervolume(nd_objs, ideal_point=ideal_pt, nadir_point=nadir_pt)
            worst_q = max(s.z2_worst_quality for s in hybrid_sols)
            max_load = min(s.z3_max_load for s in hybrid_sols)
            tot_q = max(s.z1_total_quality for s in hybrid_sols)
            gini = float(np.mean([compute_gini_coefficient(s.student_qualities) for s in hybrid_sols]))
            feas = float(np.mean([s.is_feasible for s in hybrid_sols]))
        else:
            hv, worst_q, max_load, tot_q, gini, feas = 0.0, 0.0, 0, 0.0, 0.0, 0.0

        out_records.append({
            "parameter_type": case_dict["param_type"],
            "parameter_value": case_dict["param_value"],
            "algorithm": "Hybrid",
            "seed": s_val,
            "hypervolume": round(hv, 4),
            "worst_student_quality": round(worst_q, 4),
            "gini_coefficient": round(gini, 4),
            "max_load": max_load,
            "total_quality": round(tot_q, 4),
            "feasibility_rate": round(feas, 3),
            "runtime_seconds": round(t_hybrid, 4),
        })

        return out_records

    # Execute all cases with worker threads
    n_workers = min(4, os.cpu_count() or 4)
    total_evals = len(cases) * len(seeds)
    print(f"Running {len(cases)} settings across {len(seeds)} seeds in parallel ({n_workers} threads)...")

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_single_case_seed, c, s)
            for c in cases
            for s in seeds
        ]
        done_cnt = 0
        for f in as_completed(futures):
            res_list = f.result()
            runs_data.extend(res_list)
            done_cnt += 1
            if done_cnt % 20 == 0 or done_cnt == total_evals:
                elapsed = time.perf_counter() - start_time
                print(f"  Progress: {done_cnt}/{total_evals} evaluations completed ({elapsed:.1f}s)...")

    # Save runs dataframe
    df_runs = pd.DataFrame(runs_data)
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_runs.to_csv(out_csv, index=False)
    print(f"[Sensitivity] Saved raw sensitivity runs to: {out_csv}")

    # Generate Summary CSV
    summary_df = df_runs.groupby(["parameter_type", "parameter_value", "algorithm"]).agg({
        "hypervolume": ["mean", "std"],
        "worst_student_quality": ["mean", "std"],
        "gini_coefficient": ["mean", "std"],
        "max_load": ["mean", "std"],
        "total_quality": ["mean", "std"],
        "feasibility_rate": "mean",
        "runtime_seconds": "mean",
    }).reset_index()
    summary_csv = results_dir / "sensitivity_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Sensitivity] Saved summary statistics to: {summary_csv}")

    # 3. Generate Publication Plots
    plot_sensitivity_figures(df_runs, figures_dir)

    return df_runs


# ==============================================================================
# 4. Sensitivity Visualizations
# ==============================================================================

def plot_sensitivity_figures(df_runs: pd.DataFrame, figures_dir: Path) -> None:
    """Generate publication-grade sensitivity plots for all parameter dimensions."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # Plot 1: Comprehensive Multi-Panel Overview
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()

    # Panel 1: Compatibility q_ij Weight Vectors (HV comparison)
    df_qw = df_runs[df_runs["parameter_type"] == "q_weights"]
    if not df_qw.empty:
        sns.barplot(
            data=df_qw,
            x="parameter_value",
            y="hypervolume",
            hue="algorithm",
            ax=axes[0],
            palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
        )
        axes[0].set_title(r"A. Hypervolume vs $q_{ij}$ Weight Vectors", fontsize=12, fontweight="bold")
        axes[0].set_xticklabels(["Balanced", "Branch-Hvy", "Domain-Hvy", "Exp-Hvy", "Equity"], rotation=15)
        axes[0].set_xlabel("Weight Vector Profile", fontsize=10)
        axes[0].set_ylabel("Hypervolume (HV)", fontsize=10)

    # Panel 2: Degree K in {2, 3, 5} vs Total Quality & Worst Quality
    df_k = df_runs[df_runs["parameter_type"] == "k_matches"]
    if not df_k.empty:
        sns.lineplot(
            data=df_k,
            x="parameter_value",
            y="worst_student_quality",
            hue="algorithm",
            marker="o",
            ax=axes[1],
            palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
        )
        axes[1].set_title(r"B. Worst-Student Quality $Z_2$ vs Degree Requirement $K$", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(r"Matches per Student ($K$)", fontsize=10)
        axes[1].set_ylabel(r"Worst Student Quality $Z_2$", fontsize=10)

    # Panel 3: Quality Floor theta vs Worst-Student Quality
    df_th = df_runs[df_runs["parameter_type"] == "theta"]
    if not df_th.empty:
        sns.lineplot(
            data=df_th,
            x="parameter_value",
            y="worst_student_quality",
            hue="algorithm",
            marker="s",
            ax=axes[2],
            palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
        )
        axes[2].set_title(r"C. Realized Worst Student Quality vs Quality Floor $\theta$", fontsize=12, fontweight="bold")
        axes[2].set_xlabel(r"Minimum Quality Floor $\theta$", fontsize=10)
        axes[2].set_ylabel(r"Realized Worst Quality $Z_2$", fontsize=10)

    # Panel 4: Branch Parity Tolerance delta vs Gini Coefficient
    df_dt = df_runs[df_runs["parameter_type"] == "delta"]
    if not df_dt.empty:
        sns.lineplot(
            data=df_dt,
            x="parameter_value",
            y="gini_coefficient",
            hue="algorithm",
            marker="^",
            ax=axes[3],
            palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
        )
        axes[3].set_title(r"D. Equality (Gini Index) vs Branch Parity Tolerance $\delta$", fontsize=12, fontweight="bold")
        axes[3].set_xlabel(r"Parity Disparity Tolerance $\delta$", fontsize=10)
        axes[3].set_ylabel("Gini Coefficient of Quality", fontsize=10)

    # Panel 5: Capacity Scale Factor vs Max Alumni Load Z3
    df_cs = df_runs[df_runs["parameter_type"] == "capacity_scale"]
    if not df_cs.empty:
        sns.barplot(
            data=df_cs,
            x="parameter_value",
            y="max_load",
            hue="algorithm",
            ax=axes[4],
            palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
        )
        axes[4].set_title(r"E. Peak Alumni Mentorship Load $Z_3$ vs Capacity Scale", fontsize=12, fontweight="bold")
        axes[4].set_xlabel("Alumni Capacity Scale Multiplier", fontsize=10)
        axes[4].set_ylabel(r"Peak Mentor Load $Z_3$", fontsize=10)

    # Panel 6: Runtime Comparison Across Parameter Dimensions
    sns.boxplot(
        data=df_runs,
        x="parameter_type",
        y="runtime_seconds",
        hue="algorithm",
        ax=axes[5],
        palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
    )
    axes[5].set_title("F. Algorithm Computational Runtime (Seconds)", fontsize=12, fontweight="bold")
    axes[5].set_xlabel("Sensitivity Parameter Dimension", fontsize=10)
    axes[5].set_ylabel("Runtime (seconds)", fontsize=10)
    axes[5].set_xticklabels(["q_weights", "K", "theta", "delta", "cap_scale"], rotation=15)

    fig.tight_layout()
    analysis_plot = figures_dir / "sensitivity_analysis.png"
    plt.savefig(analysis_plot, dpi=300)
    plt.close()
    print(f"[Sensitivity] Saved master sensitivity plot to: {analysis_plot}")

    # -------------------------------------------------------------
    # Plot 2: Dedicated q_ij Weight Vector Comparison Plot
    # -------------------------------------------------------------
    if not df_qw.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_qw,
            x="parameter_value",
            y="worst_student_quality",
            hue="algorithm",
            palette={"NSGA-II": "#4c72b0", "Hybrid": "#dd8452"},
        )
        plt.title(r"Sensitivity of Rawlsian Worst-Off Student Quality $Z_2$ to $q_{ij}$ Weight Vectors", fontsize=13, fontweight="bold")
        plt.xlabel("Weight Vector Profile", fontsize=11)
        plt.ylabel(r"Worst-Off Student Quality $Z_2$", fontsize=11)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        qw_plot = figures_dir / "sensitivity_q_weights.png"
        plt.savefig(qw_plot, dpi=300)
        plt.close()
        print(f"[Sensitivity] Saved q_weights sensitivity plot to: {qw_plot}")


# ==============================================================================
# 5. CLI Driver
# ==============================================================================

def main() -> None:
    """CLI driver for sensitivity experiments and coverage gap reporting."""
    parser = argparse.ArgumentParser(description="Run Sensitivity and Coverage-Gap Analysis for VJTI Matching.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/small_50x30.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of independent seeds per setting (default: 10).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=30,
        help="Population size for evolutionary solvers.",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=20,
        help="Generations count for evolutionary solvers.",
    )
    args = parser.parse_args()

    run_sensitivity_analysis(
        config_path=args.config,
        n_seeds=args.seeds,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
    )


if __name__ == "__main__":
    main()
