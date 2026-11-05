"""Automated Figure and Visual Deliverables Generator.

Generates all publication-grade figures in figures/ directly from results/ datasets
and compiles final PDFs via generate_pdfs.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.run_sensitivity import plot_sensitivity_figures
from src.visualization import (
    plot_hypervolume_boxplots,
    plot_scalability_curve,
)


def generate_all_figures() -> None:
    """Generate all figures in figures/ from results/ data files."""
    results_dir = Path("results")
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Generating All Publication Figures from results/ Data <<<")

    # 1. Hypervolume Boxplot (Small Instance)
    runs_csv = results_dir / "small_50x30_runs.csv"
    if runs_csv.exists():
        df_runs = pd.read_csv(runs_csv)
        hv_box_path = figures_dir / "small_50x30_hv_boxplot.png"
        plot_hypervolume_boxplots(df_runs, hv_box_path, title="Hypervolume Distribution (small_50x30, 30 Seeds)")
        print(f"  [+] Generated: {hv_box_path}")

    # 2. Scalability Curve
    scale_json = results_dir / "scalability_results.json"
    if scale_json.exists():
        with open(scale_json, "r", encoding="utf-8") as f:
            s_data = json.load(f)
        sizes = s_data["instance_sizes"]
        runtimes = s_data["runtimes_seconds"]
        scale_path = figures_dir / "scalability_curve.png"
        plot_scalability_curve(sizes, runtimes, scale_path)
        print(f"  [+] Generated: {scale_path}")

    # 3. Exact MILP vs Metaheuristics Figure
    exact_json = results_dir / "exact_milp_results.json"
    if exact_json.exists():
        with open(exact_json, "r", encoding="utf-8") as f:
            e_data = json.load(f)
        pts = e_data.get("pareto_points", [])
        if pts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            z1 = [p["z1_total_quality"] for p in pts]
            z2 = [p["z2_worst_quality"] for p in pts]
            z3 = [p["z3_max_load"] for p in pts]

            sc = ax1.scatter(z1, z2, c=z3, cmap="viridis", s=70, edgecolor="k", label="HiGHS MILP Exact")
            cb = plt.colorbar(sc, ax=ax1)
            cb.set_label("Max Load ($Z_3$)")
            ax1.set_xlabel("Total Match Quality ($Z_1$)")
            ax1.set_ylabel("Rawlsian Worst-off Quality ($Z_2$)")
            ax1.set_title("Exact MILP Pareto Frontier (Small 50x30)")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()

            # Secondary scenario bar chart
            sec = e_data.get("secondary_minimization", {})
            if sec:
                bounds = [f"{int(float(k)*100)}%" for k in sec.keys()]
                alumni_req = [v["alumni_needed"] for v in sec.values()]
                ax2.bar(bounds, alumni_req, color="#2b5c8f", edgecolor="black", alpha=0.85)
                ax2.set_xlabel("Quality Bound (% of Max $Z_1$)")
                ax2.set_ylabel("Active Alumni Required ($M^*$)")
                ax2.set_title("Secondary Optimization: Alumni Cohort Sizing")
                ax2.grid(True, linestyle="--", alpha=0.6)
                for i, v in enumerate(alumni_req):
                    ax2.text(i, v + 0.3, str(v), ha="center", fontweight="bold")

            plt.tight_layout()
            exact_fig_path = figures_dir / "exact_vs_metaheuristics.png"
            plt.savefig(exact_fig_path, dpi=300)
            plt.close()
            print(f"  [+] Generated: {exact_fig_path}")

    # 4. Ablation Study Figure
    ablation_csv = results_dir / "ablation_runs.csv"
    if ablation_csv.exists():
        df_ab = pd.read_csv(ablation_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        sns.boxplot(data=df_ab, x="variant", y="hypervolume", hue="variant", legend=False, ax=ax1, palette="Set2")
        ax1.set_title("Ablation: Component Impact on Hypervolume")
        ax1.set_xlabel("Algorithmic Configuration")
        ax1.set_ylabel("Normalized Hypervolume")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Quality Z1
        sns.barplot(data=df_ab, x="variant", y="z1_max", hue="variant", legend=False, ax=ax2, palette="Set2", errorbar="sd", edgecolor="k")
        ax2.set_title("Ablation: Peak Total Quality ($Z_1$)")
        ax2.set_xlabel("Algorithmic Configuration")
        ax2.set_ylabel("Total Quality ($Z_1$)")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        ab_path = figures_dir / "ablation_study.png"
        plt.savefig(ab_path, dpi=300)
        plt.close()
        print(f"  [+] Generated: {ab_path}")

    # 5. Sensitivity & Coverage Gap Figures
    sens_csv = results_dir / "sensitivity.csv"
    if sens_csv.exists():
        df_sens = pd.read_csv(sens_csv)
        plot_sensitivity_figures(df_sens, figures_dir)

    # 6. Delta Sensitivity Trade-off Figure
    ablation_json = results_dir / "ablation_results.json"
    if ablation_json.exists():
        with open(ablation_json, "r", encoding="utf-8") as f:
            ab_data = json.load(f)
        delta_results = ab_data.get("delta_sensitivity", {})
        if delta_results:
            fig, ax1 = plt.subplots(figsize=(7, 4.5))
            deltas = sorted([float(d) for d in delta_results.keys()])
            z1_vals = [delta_results[str(d) if str(d) in delta_results else f"{d:.2f}"]["max_total_quality_z1"] for d in deltas]
            z2_vals = [delta_results[str(d) if str(d) in delta_results else f"{d:.2f}"]["worst_student_z2"] for d in deltas]

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
            print(f"  [+] Generated: {plot_file}")

    # 7. Coverage Gap Figure
    cov_csv = results_dir / "coverage_gap.csv"
    if cov_csv.exists():
        df_cov = pd.read_csv(cov_csv)
        fig, axes = plt.subplots(3, 1, figsize=(11, 13))

        # Panel 1: Department Coverage Gaps
        df_b = df_cov[df_cov["category"] == "branch"].sort_values("deficit_severity_score", ascending=False)
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
        df_d = df_cov[df_cov["category"] == "skill_domain"].sort_values("unmet_demand", ascending=False)
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

        # Panel 3: Industry Supply Capacity
        df_c = df_cov[df_cov["category"] == "industry_company"].sort_values("supply_capacity", ascending=False)
        x_c = np.arange(len(df_c))

        axes[2].bar(x_c, df_c["supply_capacity"], color="#2ca02c", alpha=0.85, edgecolor="black")
        axes[2].set_title("C. Employer Industry Distribution of Consenting Alumni Mentors", fontsize=13, fontweight="bold")
        axes[2].set_xticks(x_c)
        axes[2].set_xticklabels(df_c["name"], rotation=20, ha="right")
        axes[2].set_ylabel("Total Mentorship Capacity (Slots)", fontsize=11)

        fig.tight_layout()
        cov_plot = figures_dir / "coverage_gap.png"
        plt.savefig(cov_plot, dpi=300)
        plt.close()
        print(f"  [+] Generated: {cov_plot}")

    print(">>> All Visual Figures Synchronized in figures/ <<<\n")


def main() -> None:
    """Generate figures and then call PDF compilation."""
    generate_all_figures()
    import generate_pdfs
    generate_pdfs.main()


if __name__ == "__main__":
    main()
