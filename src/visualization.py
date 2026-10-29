"""Visualization module for Student-Alumni Matching optimization results.

Generates publication-quality charts:
1. 3D Pareto Frontiers comparing algorithms (Z1, Z2, Z3).
2. 2D Trade-off projection curves.
3. Branch-wise quality parity distributions.
4. Alumni workload and capacity utilization histograms.
5. Multi-metric algorithm comparison bar charts.
6. Scalability analysis curves (Runtime vs Instance Size).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.problem import MatchingSolution


# Set consistent academic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_pareto_front_3d(
    algo_solutions: Dict[str, List[MatchingSolution]],
    save_path: str | Path,
    title: str = "3D Pareto Frontier Comparison",
) -> None:
    """Plot 3D Pareto frontier comparing multiple algorithms."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = sns.color_palette("tab10", len(algo_solutions))

    for idx, (algo_name, sols) in enumerate(algo_solutions.items()):
        if not sols:
            continue
        z1 = [s.z1_total_quality for s in sols]
        z2 = [s.z2_worst_quality for s in sols]
        z3 = [s.z3_max_load for s in sols]

        ax.scatter(
            z1,
            z2,
            z3,
            color=colors[idx],
            label=algo_name,
            s=45,
            alpha=0.85,
            edgecolors="w",
            linewidth=0.5,
        )

    ax.set_xlabel("Total Match Quality ($Z_1$)", labelpad=10)
    ax.set_ylabel("Rawlsian Fairness / Worst-off ($Z_2$)", labelpad=10)
    ax.set_zlabel("Max Mentor Load ($Z_3$)", labelpad=10)
    ax.set_title(title, pad=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tradeoff_2d(
    algo_solutions: Dict[str, List[MatchingSolution]],
    save_path: str | Path,
) -> None:
    """Plot 2D pairwise projections of Pareto trade-offs (Z1 vs Z2, Z1 vs Z3, Z2 vs Z3)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = sns.color_palette("tab10", len(algo_solutions))

    for idx, (algo_name, sols) in enumerate(algo_solutions.items()):
        if not sols:
            continue
        z1 = [s.z1_total_quality for s in sols]
        z2 = [s.z2_worst_quality for s in sols]
        z3 = [s.z3_max_load for s in sols]

        # Subplot 1: Total Quality vs Rawlsian Fairness
        axes[0].scatter(z1, z2, color=colors[idx], label=algo_name, alpha=0.8, s=40)
        # Subplot 2: Total Quality vs Max Load
        axes[1].scatter(z1, z3, color=colors[idx], label=algo_name, alpha=0.8, s=40)
        # Subplot 3: Rawlsian Fairness vs Max Load
        axes[2].scatter(z2, z3, color=colors[idx], label=algo_name, alpha=0.8, s=40)

    axes[0].set_xlabel("Total Quality ($Z_1$) [Higher is better]")
    axes[0].set_ylabel("Worst-off Quality ($Z_2$) [Higher is better]")
    axes[0].set_title("Efficiency vs. Rawlsian Fairness")

    axes[1].set_xlabel("Total Quality ($Z_1$) [Higher is better]")
    axes[1].set_ylabel("Max Mentor Load ($Z_3$) [Lower is better]")
    axes[1].set_title("Efficiency vs. Workload Balancing")

    axes[2].set_xlabel("Worst-off Quality ($Z_2$) [Higher is better]")
    axes[2].set_ylabel("Max Mentor Load ($Z_3$) [Lower is better]")
    axes[2].set_title("Fairness vs. Workload Balancing")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_branch_parity(
    solution: MatchingSolution,
    save_path: str | Path,
    title: str = "Branch-wise Average Quality Parity",
) -> None:
    """Plot bar chart of average student quality across VJTI engineering departments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    branches = sorted(list(solution.branch_avg_qualities.keys()))
    avgs = [solution.branch_avg_qualities[b] for b in branches]
    overall_mean = float(np.mean(solution.student_qualities))

    palette = sns.color_palette("mako", len(branches))
    bars = ax.bar(branches, avgs, color=palette, edgecolor="black", alpha=0.85)

    # Reference mean line
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.5, label=f"Overall Mean ({overall_mean:.2f})")

    # Annotate values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("VJTI Academic Department")
    ax.set_ylabel("Average Cumulative Match Quality $\\bar{Q}_b$")
    ax.set_title(f"{title} (Max Disparity: {solution.branch_disparity:.3f})")
    ax.set_ylim(0, max(avgs) * 1.25 if avgs else 1.0)
    ax.legend(loc="lower right")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_workload_distribution(
    solution: MatchingSolution,
    capacities: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot alumni assigned workload vs capacity headroom."""
    fig, ax = plt.subplots(figsize=(10, 5))

    loads = solution.alumni_loads
    active_mask = loads > 0
    utilization = np.zeros_like(loads, dtype=float)
    utilization[active_mask] = (loads[active_mask] / capacities[active_mask]) * 100.0

    bins = np.linspace(0, max(loads) + 1, max(loads) + 2) - 0.5
    ax.hist(loads, bins=bins, color="#2b5c8f", edgecolor="black", alpha=0.8, rwidth=0.8)

    ax.set_xlabel("Assigned Students per Alumnus ($L_j$)")
    ax.set_ylabel("Number of Alumni")
    ax.set_title(f"Alumni Workload Distribution (Max Load: {solution.z3_max_load}, Active: {solution.active_alumni_count})")
    ax.set_xticks(range(0, int(np.max(loads)) + 2))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metrics_comparison(
    metrics_by_algo: Dict[str, Dict[str, float]],
    save_path: str | Path,
) -> None:
    """Plot multi-panel bar chart comparing Hypervolume, Spacing, Non-Dominated Count, and Gini."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    algos = list(metrics_by_algo.keys())
    palette = sns.color_palette("deep", len(algos))

    # Panel 1: Hypervolume
    hv_vals = [metrics_by_algo[a].get("hypervolume", 0.0) for a in algos]
    axes[0, 0].bar(algos, hv_vals, color=palette, edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("Hypervolume (HV) [Higher is Better]")
    axes[0, 0].set_ylabel("Normalized HV")

    # Panel 2: Spacing
    sp_vals = [metrics_by_algo[a].get("spacing", 0.0) for a in algos]
    axes[0, 1].bar(algos, sp_vals, color=palette, edgecolor="black", alpha=0.85)
    axes[0, 1].set_title("Spacing Metric (SP) [Lower is Better]")
    axes[0, 1].set_ylabel("Spread Variance")

    # Panel 3: Number of Non-Dominated Solutions
    nd_vals = [metrics_by_algo[a].get("num_non_dominated", 0) for a in algos]
    axes[1, 0].bar(algos, nd_vals, color=palette, edgecolor="black", alpha=0.85)
    axes[1, 0].set_title("Non-Dominated Solutions Count ($|PF^*|$)")
    axes[1, 0].set_ylabel("Solution Count")

    # Panel 4: Gini Index of Student Quality
    gini_vals = [metrics_by_algo[a].get("avg_gini_quality", 0.0) for a in algos]
    axes[1, 1].bar(algos, gini_vals, color=palette, edgecolor="black", alpha=0.85)
    axes[1, 1].set_title("Student Quality Gini Index [Lower is More Equal]")
    axes[1, 1].set_ylabel("Gini Coefficient")

    for ax in axes.flat:
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=25, ha="right")
        ax.grid(True, linestyle="--", alpha=0.6)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_scalability_curve(
    instance_sizes: List[str],
    algo_runtimes: Dict[str, List[float]],
    save_path: str | Path,
) -> None:
    """Plot runtime scalability curve across Small, Medium, and Large instances."""
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v"]
    colors = sns.color_palette("tab10", len(algo_runtimes))

    for idx, (algo, runtimes) in enumerate(algo_runtimes.items()):
        ax.plot(
            instance_sizes,
            runtimes,
            marker=markers[idx % len(markers)],
            linewidth=2.0,
            markersize=8,
            color=colors[idx],
            label=algo,
        )

    ax.set_xlabel("Instance Size ($N \\times M$)")
    ax.set_ylabel("Execution Runtime (seconds) [Log Scale]")
    ax.set_yscale("log")
    ax.set_title("Empirical Computational Scalability Across Problem Dimensions")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_hypervolume_boxplots(
    runs_df: Any,
    save_path: str | Path,
    title: str = "Hypervolume Distribution Across Independent Seeds",
) -> None:
    """Plot publication-quality boxplot with overlaid seed points for hypervolume distributions.

    Args:
        runs_df: Pandas DataFrame containing 'method' and 'hypervolume' columns.
        save_path: File path where the plot figure should be saved.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(runs_df["method"].unique())
    palette = sns.color_palette("Set2", len(methods))

    sns.boxplot(
        data=runs_df,
        x="method",
        y="hypervolume",
        palette=palette,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "white", "markersize": 6},
    )
    sns.stripplot(
        data=runs_df,
        x="method",
        y="hypervolume",
        color="black",
        alpha=0.55,
        jitter=0.2,
        size=5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Optimization Method", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Hypervolume (HV) [Shared Ref 1.1]", fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
