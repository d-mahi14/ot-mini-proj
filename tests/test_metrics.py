"""Unit tests for multi-objective and fairness evaluation metrics."""

import numpy as np
import pytest
from src.metrics import (
    compute_gini_coefficient,
    compute_hypervolume,
    compute_shannon_entropy,
    compute_shared_reference_setup,
    compute_spacing,
    extract_non_dominated_front,
    normalize_objectives,
)


def test_extract_non_dominated_front():
    """Verify Pareto dominance filtering for minimization."""
    # Points in 2D (f1, f2) to minimize:
    # Point A: (1, 5) -> non-dominated
    # Point B: (2, 3) -> non-dominated
    # Point C: (3, 2) -> non-dominated
    # Point D: (4, 4) -> dominated by B (2, 3)
    points = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 4.0],
    ])
    nd_points, indices = extract_non_dominated_front(points)

    assert len(nd_points) == 3
    assert set(indices) == {0, 1, 2}


def test_spacing_metric():
    """Verify spacing calculation on known spread."""
    points = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0],
    ])
    sp = compute_spacing(points)
    # Perfectly spaced points should yield spacing close to 0
    assert sp >= 0.0
    assert np.isclose(sp, 0.0, atol=1e-5)


def test_gini_coefficient():
    """Test Gini equality index properties."""
    # Perfect equality
    equal_q = np.array([1.5, 1.5, 1.5, 1.5])
    assert np.isclose(compute_gini_coefficient(equal_q), 0.0)

    # Unequal distribution
    unequal_q = np.array([0.1, 0.2, 0.3, 3.0])
    gini = compute_gini_coefficient(unequal_q)
    assert 0.0 < gini < 1.0


def test_shannon_entropy():
    """Test Shannon entropy calculation."""
    # Uniform 4 categories
    counts = [10, 10, 10, 10]
    h = compute_shannon_entropy(counts)
    # log2(4) = 2.0
    assert np.isclose(h, 2.0)

    # Single category -> entropy 0
    single = [40]
    assert np.isclose(compute_shannon_entropy(single), 0.0)


def test_hypervolume_shared_reference():
    """Verify shared reference hypervolume consistency, comparability, and dominance properties."""
    # Two distinct points forming a Pareto front for minimization in 3D:
    # F = [-Z1, -Z2, Z3]
    front_a = np.array([
        [-100.0, -2.0, 4.0],
        [-90.0, -2.5, 3.0],
    ])
    front_identical = front_a.copy()

    # Front C is strictly dominated by Front A (higher values in minimization)
    front_dominated = np.array([
        [-95.0, -1.8, 5.0],
        [-85.0, -2.2, 4.0],
    ])

    # Establish shared reference setup across the union of fronts
    ref_setup = compute_shared_reference_setup([front_a, front_dominated])
    ideal = np.array(ref_setup["ideal_point"])
    nadir = np.array(ref_setup["nadir_point"])
    ref_pt = np.array(ref_setup["ref_point"])

    assert np.all(ref_pt == 1.1)

    hv_a = compute_hypervolume(front_a, ideal_point=ideal, nadir_point=nadir, ref_point=ref_pt)
    hv_identical = compute_hypervolume(front_identical, ideal_point=ideal, nadir_point=nadir, ref_point=ref_pt)
    hv_dominated = compute_hypervolume(front_dominated, ideal_point=ideal, nadir_point=nadir, ref_point=ref_pt)

    # 1. Two identical fronts must yield identical hypervolume
    assert np.isclose(hv_a, hv_identical, atol=1e-7)

    # 2. A dominated front must obtain strictly lower hypervolume
    assert hv_dominated < hv_a
    assert hv_a > 0.0

    # 3. Fallback removal test: Calling compute_hypervolume without reference point or bounds raises ValueError
    with pytest.raises(ValueError, match="Per-method fallback"):
        compute_hypervolume(front_a)
