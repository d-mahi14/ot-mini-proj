"""Tests for Prompt D: calibrate_thresholds() in src/problem.py."""

from __future__ import annotations

import pytest
from src.data_generator import SyntheticDataGenerator, MatchingInstance
from src.problem import StudentAlumniMatchingProblem, calibrate_thresholds
from src.baseline.greedy_baseline import GreedyBaseline


@pytest.fixture
def mini_instance():
    """Small instance for fast calibration tests (8 students, 6 alumni)."""
    gen = SyntheticDataGenerator(
        seed=42,
        k_matches=2,
        q_min=0.20,
        theta_min_quality=0.50,
        delta_branch_parity=0.50,
    )
    return gen.generate_instance(n_students=8, n_alumni=6, avg_capacity=4.0)


def test_calibrate_returns_positive_floats(mini_instance):
    """theta and delta must be positive floats."""
    result = calibrate_thresholds(mini_instance, theta_fraction=0.85, delta_fraction=1.2)

    assert isinstance(result["theta"], float), "theta must be a float"
    assert isinstance(result["delta"], float), "delta must be a float"
    assert result["theta"] > 0.0, "theta must be positive"
    assert result["delta"] > 0.0, "delta must be positive"


def test_calibrate_theta_is_fraction_of_greedy_z2(mini_instance):
    """theta should equal theta_fraction * greedy_Z2 (within floating-point tolerance)."""
    frac = 0.80
    result = calibrate_thresholds(mini_instance, theta_fraction=frac, delta_fraction=1.2)

    expected_theta = round(frac * result["greedy_z2"], 6)
    assert abs(result["theta"] - expected_theta) < 1e-6, (
        f"theta={result['theta']} != {frac} * greedy_z2={result['greedy_z2']}"
    )


def test_calibrate_delta_is_fraction_of_greedy_spread(mini_instance):
    """delta should equal delta_fraction * greedy_spread (within floating-point tolerance)."""
    frac = 1.5
    result = calibrate_thresholds(mini_instance, theta_fraction=0.85, delta_fraction=frac)

    if result["greedy_spread"] > 0.0:
        expected_delta = round(frac * result["greedy_spread"], 6)
        assert abs(result["delta"] - expected_delta) < 1e-6, (
            f"delta={result['delta']} != {frac} * spread={result['greedy_spread']}"
        )
    else:
        # If all branches have identical quality, delta falls back to 1e9
        assert result["delta"] >= 1e8


def test_greedy_feasible_under_calibrated_thresholds(mini_instance):
    """Greedy must find a feasible solution when the calibrated theta and delta are used."""
    result = calibrate_thresholds(mini_instance, theta_fraction=0.85, delta_fraction=1.2)

    constrained = MatchingInstance(
        name=mini_instance.name,
        n_students=mini_instance.n_students,
        n_alumni=mini_instance.n_alumni,
        k_matches=mini_instance.k_matches,
        q_matrix=mini_instance.q_matrix,
        q_min=mini_instance.q_min,
        theta_min_quality=result["theta"],
        delta_branch_parity=result["delta"],
        max_company_share=mini_instance.max_company_share,
        students=mini_instance.students,
        alumni=mini_instance.alumni,
    )
    prob = StudentAlumniMatchingProblem(constrained)
    sol = GreedyBaseline(prob).solve()

    assert sol.is_feasible, (
        f"Greedy should be feasible under calibrated theta={result['theta']:.4f}, "
        f"delta={result['delta']:.4f}. Got cv={sol.total_constraint_violation:.4f}"
    )


def test_calibrate_raises_on_invalid_fractions(mini_instance):
    """Non-positive fractions must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        calibrate_thresholds(mini_instance, theta_fraction=0.0, delta_fraction=1.2)

    with pytest.raises(ValueError, match="positive"):
        calibrate_thresholds(mini_instance, theta_fraction=0.85, delta_fraction=-0.5)


def test_calibrate_saves_json(mini_instance, tmp_path):
    """When output_path is provided, calibration.json must be saved with correct keys."""
    import json

    out_file = tmp_path / "calibration.json"
    result = calibrate_thresholds(
        mini_instance,
        theta_fraction=0.85,
        delta_fraction=1.2,
        output_path=str(out_file),
    )

    assert out_file.exists(), "JSON file should be created"
    saved = json.loads(out_file.read_text())

    for key in ("theta", "delta", "greedy_z2", "greedy_spread", "theta_fraction",
                "delta_fraction", "instance_name"):
        assert key in saved, f"Key '{key}' missing from saved JSON"

    assert abs(saved["theta"] - result["theta"]) < 1e-9
    assert abs(saved["delta"] - result["delta"]) < 1e-9

