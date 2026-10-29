"""Prompt D: Run calibrate_thresholds on all three instances and save results/calibration.json.

Usage:
    python experiments/run_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data_generator import SyntheticDataGenerator, MatchingInstance
from src.problem import calibrate_thresholds


CONFIGS = [
    "experiments/configs/small_50x30.yaml",
    "experiments/configs/medium_300x150.yaml",
    "experiments/configs/large_1000x500.yaml",
]


def load_or_generate_instance(cfg: dict):
    """Load cached instance or generate fresh one."""
    file_path = Path(cfg["instance"]["file_path"])
    gen = SyntheticDataGenerator(
        seed=cfg["instance"]["seed"],
        k_matches=cfg["instance"]["k_matches"],
        q_min=cfg["instance"]["q_min"],
        theta_min_quality=cfg["instance"].get("theta_min_quality", 0.70),
        delta_branch_parity=cfg["instance"].get("delta_branch_parity", 0.35),
        max_company_share=cfg["instance"].get("max_company_share", 1),
    )
    if file_path.exists():
        print(f"  Loading cached instance from {file_path}...")
        return MatchingInstance.load_json(str(file_path))
    else:
        print(f"  Generating instance and saving to {file_path}...")
        inst = gen.generate_instance(
            n_students=cfg["instance"]["n_students"],
            n_alumni=cfg["instance"]["n_alumni"],
            avg_capacity=cfg["instance"].get("avg_capacity", 4.0),
        )
        inst.save_json(str(file_path))
        return inst


def main():
    all_results = {}

    for config_path in CONFIGS:
        print(f"\n{'='*60}")
        print(f" Calibrating: {config_path}")
        print(f"{'='*60}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        instance = load_or_generate_instance(cfg)

        theta_fraction = cfg.get("calibration", {}).get("theta_fraction", 0.85)
        delta_fraction = cfg.get("calibration", {}).get("delta_fraction", 1.20)

        print(f"  theta_fraction={theta_fraction}, delta_fraction={delta_fraction}")

        result = calibrate_thresholds(
            instance=instance,
            theta_fraction=theta_fraction,
            delta_fraction=delta_fraction,
            output_path=None,  # we aggregate below
        )

        print(f"  greedy Z2 (worst student)  : {result['greedy_z2']:.4f}")
        print(f"  greedy branch spread       : {result['greedy_spread']:.4f}")
        print(f"  calibrated theta           : {result['theta']:.4f}")
        print(f"  calibrated delta           : {result['delta']:.4f}")

        # Feasibility check: verify greedy finds a feasible solution with these params
        from src.data_generator import MatchingInstance
        from src.problem import StudentAlumniMatchingProblem
        from src.baseline.greedy_baseline import GreedyBaseline

        constrained = MatchingInstance(
            name=instance.name,
            n_students=instance.n_students,
            n_alumni=instance.n_alumni,
            k_matches=instance.k_matches,
            q_matrix=instance.q_matrix,
            q_min=instance.q_min,
            theta_min_quality=result["theta"],
            delta_branch_parity=result["delta"],
            max_company_share=instance.max_company_share,
            students=instance.students,
            alumni=instance.alumni,
        )
        prob = StudentAlumniMatchingProblem(constrained)
        sol = GreedyBaseline(prob).solve()

        result["greedy_feasible_under_calibrated"] = bool(sol.is_feasible)
        print(
            f"  Greedy feasible under calibrated (theta,delta): "
            f"{'YES' if sol.is_feasible else 'NO (WARNING)'}"
        )

        all_results[instance.name] = result

    output_path = Path("results/calibration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    print(f"\nCalibration report saved to: {output_path}")
    print("\nSummary:")
    for name, r in all_results.items():
        feasible_str = "feasible" if r["greedy_feasible_under_calibrated"] else "INFEASIBLE"
        print(
            f"  {name:20s}  theta={r['theta']:.4f}  "
            f"delta={r['delta']:.4f}  greedy={feasible_str}"
        )


if __name__ == "__main__":
    main()
