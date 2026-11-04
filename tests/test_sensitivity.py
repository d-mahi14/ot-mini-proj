"""Unit tests for parameter sensitivity and coverage-gap analysis."""

from pathlib import Path
import numpy as np
import pytest

from src.data_generator import SyntheticDataGenerator
from src.problem import StudentAlumniMatchingProblem
from experiments.run_sensitivity import (
    build_variant_instance,
    compute_weighted_q_matrix,
    generate_coverage_gap_report,
)


@pytest.fixture
def test_instance():
    """Create a miniature synthetic instance for testing."""
    gen = SyntheticDataGenerator(seed=101, k_matches=2, q_min=0.20)
    return gen.generate_instance(n_students=12, n_alumni=10, avg_capacity=4.0)


def test_compute_weighted_q_matrix(test_instance):
    """Test custom q_matrix computation across different weight profiles."""
    q_mat_balanced = compute_weighted_q_matrix(
        test_instance.students,
        test_instance.alumni,
        w_branch=0.45,
        w_domain=0.35,
        w_exp=0.10,
    )
    assert q_mat_balanced.shape == (12, 10)
    assert np.all(q_mat_balanced >= 0.05)
    assert np.all(q_mat_balanced <= 0.98)

    q_mat_branch_heavy = compute_weighted_q_matrix(
        test_instance.students,
        test_instance.alumni,
        w_branch=0.70,
        w_domain=0.15,
        w_exp=0.05,
    )
    # The two matrices should differ due to weights
    assert not np.allclose(q_mat_balanced, q_mat_branch_heavy)


def test_build_variant_instance(test_instance):
    """Test building adjusted instance variants for K, theta, delta, and capacity."""
    # Test varying K to 3
    var_k3 = build_variant_instance(test_instance, k_matches=3)
    assert var_k3.k_matches == 3
    # Alumni capacity should be scaled up to accommodate higher demand
    total_cap_orig = sum(a.capacity for a in test_instance.alumni)
    total_cap_k3 = sum(a.capacity for a in var_k3.alumni)
    assert total_cap_k3 >= total_cap_orig

    # Test varying theta and delta
    var_th_dt = build_variant_instance(test_instance, theta=0.85, delta=0.20)
    assert var_th_dt.theta_min_quality == 0.85
    assert var_th_dt.delta_branch_parity == 0.20

    # Test capacity scaling factor
    var_cap_scale = build_variant_instance(test_instance, capacity_scale=1.5)
    total_cap_scale = sum(a.capacity for a in var_cap_scale.alumni)
    assert total_cap_scale >= total_cap_orig


def test_generate_coverage_gap_report(test_instance, tmp_path):
    """Test generation of coverage-gap analysis CSV and visualization."""
    out_csv = tmp_path / "coverage_gap.csv"
    out_fig = tmp_path / "coverage_gap.png"

    df = generate_coverage_gap_report(
        test_instance,
        output_csv=str(out_csv),
        figure_path=str(out_fig),
    )

    assert out_csv.exists()
    assert out_fig.exists()
    assert not df.empty
    assert "branch" in df["category"].values
    assert "skill_domain" in df["category"].values
    assert "mean_compatibility_qij" in df.columns
    assert "unmet_demand" in df.columns
