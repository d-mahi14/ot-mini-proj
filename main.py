"""Main Driver for Fair and Load-Balanced Student-Alumni Matching at VJTI.

A thin, modular entrypoint driving:
- Individual or batch instance benchmarking
- Deterministic and stochastic baseline evaluations
- Exact MILP Pareto frontiers
- Algorithmic ablation studies
- Empirical scalability evaluations
- Parameter sensitivity & coverage-gap analysis
- Automated figure and PDF generation
- Automated test verification and data synthesis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.run_ablation import run_ablation_study
from experiments.run_baselines import run_baselines_evaluation
from experiments.run_exact import run_exact_analysis
from experiments.run_experiments import run_experiment_on_instance
from experiments.run_interpretation import run_interpretation_pipeline
from experiments.run_scalability import run_scalability_benchmark
from experiments.run_sensitivity import run_sensitivity_analysis
from src.data_generator import generate_all_datasets


def main() -> None:
    """CLI driver entrypoint."""
    parser = argparse.ArgumentParser(
        description="VJTI Student-Alumni Matching: Multi-Objective Capacitated Assignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["run", "baselines", "exact", "scalability", "ablation", "sensitivity", "interpretation", "data", "figures", "all"],
        default="run",
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/small_50x30.yaml",
        help="Path to YAML configuration file for 'run' and 'baselines' mode.",
    )
    parser.add_argument(
        "--instance",
        choices=["small", "medium", "large", "small_50x30", "medium_300x150", "large_1000x500"],
        default=None,
        help="Convenience selector for standard instance size.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Optional override for maximum number of seeds to execute.",
    )

    args = parser.parse_args()

    # Resolve instance alias and canonical instance name
    config_path = args.config
    instance_name = "small_50x30"

    if args.instance in ["small", "small_50x30"]:
        config_path = "experiments/configs/small_50x30.yaml"
        instance_name = "small_50x30"
    elif args.instance in ["medium", "medium_300x150"]:
        config_path = "experiments/configs/medium_300x150.yaml"
        instance_name = "medium_300x150"
    elif args.instance in ["large", "large_1000x500"]:
        config_path = "experiments/configs/large_1000x500.yaml"
        instance_name = "large_1000x500"

    print("================================================================================")
    print("      VJTI STUDENT-ALUMNI MULTI-OBJECTIVE MATCHING OPTIMIZATION")
    print("      Capacitated Assignment with Rawlsian Fairness & Branch Parity")
    print("      Department of Production & Industrial Engineering / Computer Eng.")
    print("      Veermata Jijabai Technological Institute (VJTI), Mumbai")
    print("================================================================================\n")

    if args.mode == "data":
        print(">>> Generating / Verifying Synthetic Benchmark Datasets <<<")
        generate_all_datasets()

    elif args.mode == "baselines":
        run_baselines_evaluation(config_path)

    elif args.mode == "run":
        run_experiment_on_instance(config_path, max_seeds=args.seeds)

    elif args.mode == "exact":
        run_exact_analysis(config_path)

    elif args.mode == "scalability":
        run_scalability_benchmark()

    elif args.mode == "ablation":
        run_ablation_study(config_path, max_seeds=args.seeds)

    elif args.mode == "sensitivity":
        n_seeds = args.seeds if args.seeds is not None else 10
        run_sensitivity_analysis(config_path, n_seeds=n_seeds)

    elif args.mode == "interpretation":
        run_interpretation_pipeline(instance_name=instance_name, config_path=config_path)

    elif args.mode == "figures":
        import generate_figures
        generate_figures.main()

    elif args.mode == "all":
        print(">>> Running Full Comprehensive Research Pipeline <<<\n")
        print("\n--- Phase 0: Synthetic Benchmark Data Generation ---")
        generate_all_datasets()

        print("\n--- Phase 1: Baseline Algorithms Evaluation ---")
        run_baselines_evaluation("experiments/configs/small_50x30.yaml")

        print("\n--- Phase 2: Small Instance Benchmark (50x30) ---")
        run_experiment_on_instance("experiments/configs/small_50x30.yaml", max_seeds=args.seeds)

        print("\n--- Phase 3: Exact MILP and Secondary Scenario ---")
        run_exact_analysis("experiments/configs/small_50x30.yaml")

        print("\n--- Phase 4: Medium Instance Benchmark (300x150) ---")
        run_experiment_on_instance("experiments/configs/medium_300x150.yaml", max_seeds=args.seeds)

        print("\n--- Phase 5: Large Instance Benchmark (1000x500) ---")
        run_experiment_on_instance("experiments/configs/large_1000x500.yaml", max_seeds=args.seeds)

        print("\n--- Phase 6: Scalability Across Dimensions ---")
        run_scalability_benchmark()

        print("\n--- Phase 7: Ablation Study ---")
        run_ablation_study("experiments/configs/small_50x30.yaml", max_seeds=args.seeds)

        print("\n--- Phase 8: Parameter Sensitivity & Coverage-Gap Analysis ---")
        n_seeds = args.seeds if args.seeds is not None else 10
        run_sensitivity_analysis("experiments/configs/small_50x30.yaml", n_seeds=n_seeds)

        print("\n--- Phase 9: Pareto Interpretation & Institutional Policy Report ---")
        run_interpretation_pipeline(instance_name="small_50x30", config_path="experiments/configs/small_50x30.yaml")

        print("\n--- Phase 10: Visual Figures & PDF Deliverables ---")
        import generate_figures
        generate_figures.main()

        print("\n>>> Full Research Pipeline Successfully Executed! <<<")


if __name__ == "__main__":
    main()
