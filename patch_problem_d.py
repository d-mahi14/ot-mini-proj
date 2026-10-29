"""One-time script: append calibrate_thresholds() to src/problem.py for Prompt D."""
import sys

calibrate_fn = r'''

def calibrate_thresholds(
    instance,
    theta_fraction: float = 0.85,
    delta_fraction: float = 1.2,
    output_path=None,
):
    """Calibrate theta (quality floor) and delta (branch parity cap) from data.

    Strategy
    --------
    theta = theta_fraction * Z2_greedy
        Z2_greedy is the worst-student quality produced by the greedy baseline
        on an *unconstrained* copy of the instance (theta=0, delta=inf).
        Using fraction < 1 guarantees the greedy solution is still feasible
        under the calibrated theta.

    delta = delta_fraction * spread_greedy
        spread_greedy is max_b(avgQ_b) - min_b(avgQ_b) from the same
        unconstrained greedy solution.  Using fraction > 1 adds slack so the
        greedy solution itself remains feasible, while still bounding worse
        solutions.

    Parameters
    ----------
    instance        : MatchingInstance to calibrate against.
    theta_fraction  : Fraction of greedy Z2 to use as theta (default 0.85).
    delta_fraction  : Multiplier on greedy branch spread to use as delta
                      (default 1.2, i.e. 20 pct slack above greedy baseline).
    output_path     : If given, save calibration report as JSON to this path.

    Returns
    -------
    dict with keys: theta, delta, greedy_z2, greedy_spread,
    theta_fraction, delta_fraction, instance_name.

    Raises
    ------
    ValueError if fractions are non-positive or greedy produces zero Z2.
    """
    import json
    from pathlib import Path as _Path
    from src.data_generator import MatchingInstance as _MI
    from src.problem import StudentAlumniMatchingProblem
    from src.baseline.greedy_baseline import GreedyBaseline
    from typing import Optional, Dict

    if theta_fraction <= 0 or delta_fraction <= 0:
        raise ValueError(
            f"theta_fraction and delta_fraction must be positive; "
            f"got {theta_fraction}, {delta_fraction}"
        )

    # Build unconstrained variant (no floor, no parity cap)
    unconstrained = _MI(
        name=instance.name,
        n_students=instance.n_students,
        n_alumni=instance.n_alumni,
        k_matches=instance.k_matches,
        q_matrix=instance.q_matrix,
        q_min=instance.q_min,
        theta_min_quality=0.0,
        delta_branch_parity=1e9,
        max_company_share=instance.max_company_share,
        students=instance.students,
        alumni=instance.alumni,
    )

    prob_unconstrained = StudentAlumniMatchingProblem(unconstrained)
    greedy_sol = GreedyBaseline(prob_unconstrained).solve()

    if greedy_sol.z2_worst_quality <= 0.0:
        raise ValueError(
            "Greedy baseline produced zero worst-student quality on the "
            "unconstrained instance -- cannot calibrate theta safely."
        )

    # Calibrate theta
    theta = theta_fraction * greedy_sol.z2_worst_quality

    # Calibrate delta from branch spread
    branch_avgs = greedy_sol.branch_avg_qualities
    if len(branch_avgs) >= 2:
        vals = list(branch_avgs.values())
        greedy_spread = float(max(vals) - min(vals))
    else:
        greedy_spread = 0.0

    delta = delta_fraction * greedy_spread if greedy_spread > 0.0 else 1e9

    result = {
        "instance_name": instance.name,
        "theta": round(theta, 6),
        "delta": round(delta, 6),
        "greedy_z2": round(greedy_sol.z2_worst_quality, 6),
        "greedy_spread": round(greedy_spread, 6),
        "theta_fraction": theta_fraction,
        "delta_fraction": delta_fraction,
    }

    if output_path is not None:
        _Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)

    return result
'''

content = open('src/problem.py', 'rb').read().decode('utf-8')

if 'def calibrate_thresholds' in content:
    print("calibrate_thresholds already present — skipping.")
    sys.exit(0)

content = content.rstrip() + calibrate_fn
open('src/problem.py', 'wb').write(content.encode('utf-8'))
print("calibrate_thresholds appended OK.")

