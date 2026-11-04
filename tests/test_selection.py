"""Unit tests for Pareto selection, representative policy extraction, and front comparison."""

from pathlib import Path
import numpy as np
import pytest

from src.data_generator import SyntheticDataGenerator
from src.hybrid.seeded_memetic_nsga2 import SeededMemeticNSGA2
from src.multiobjective.selection import (
    compare_fronts,
    export_pseudonymous_matching_table,
    select_representative_solutions,
)
from src.problem import StudentAlumniMatchingProblem


@pytest.fixture
def mini_matching():
    """Generate a mini problem instance and solved Hybrid Pareto front."""
    gen = SyntheticDataGenerator(seed=123, k_matches=2, q_min=0.20)
    inst = gen.generate_instance(n_students=10, n_alumni=8, avg_capacity=4.0)
    prob = StudentAlumniMatchingProblem(inst)
    hybrid_solver = SeededMemeticNSGA2(prob)
    sols, _ = hybrid_solver.solve(pop_size=20, n_gen=10, seed=42)
    return inst, prob, sols


def test_select_representative_solutions(mini_matching):
    """Test selection of 4 representative policies from non-dominated front."""
    inst, prob, sols = mini_matching
    assert len(sols) > 0

    reps = select_representative_solutions(sols)
    assert "highest_quality" in reps
    assert "fairest" in reps
    assert "lightest_load" in reps
    assert "knee_point" in reps

    # Verify highest quality has peak Z1
    z1_max = max(s.z1_total_quality for s in sols)
    assert reps["highest_quality"].z1_total_quality == pytest.approx(z1_max, rel=1e-5)

    # Verify fairest has peak Z2
    z2_max = max(s.z2_worst_quality for s in sols)
    assert reps["fairest"].z2_worst_quality == pytest.approx(z2_max, rel=1e-5)

    # Verify lightest load has minimum Z3
    z3_min = min(s.z3_max_load for s in sols)
    assert reps["lightest_load"].z3_max_load == z3_min

    # Verify knee solution is valid MatchingSolution
    assert reps["knee_point"].is_feasible or reps["knee_point"].total_constraint_violation >= 0


def test_compare_fronts(mini_matching):
    """Test comparative metrics calculation between two fronts."""
    inst, prob, sols = mini_matching

    # Compare sols against itself (identity comparison)
    comp_self = compare_fronts(sols, sols)
    assert comp_self["hv_gap"] == pytest.approx(0.0, abs=1e-5)
    assert comp_self["generational_distance_gd"] == pytest.approx(0.0, abs=1e-5)
    assert comp_self["inverted_generational_distance_igd"] == pytest.approx(0.0, abs=1e-5)


def test_export_pseudonymous_matching_table(mini_matching, tmp_path):
    """Test exporting pseudonymous allocation schedule."""
    inst, prob, sols = mini_matching
    knee_sol = select_representative_solutions(sols)["knee_point"]
    out_csv = tmp_path / "test_knee_matching.csv"

    df = export_pseudonymous_matching_table(inst, knee_sol, out_csv)
    assert out_csv.exists()
    assert len(df) == inst.n_students

    # Verify pseudonymous format
    assert all(sid.startswith("S") for sid in df["student_id"])
    assert "mentor_1_id" in df.columns
    assert "mentor_2_id" in df.columns
    assert "total_match_quality_Qi" in df.columns
