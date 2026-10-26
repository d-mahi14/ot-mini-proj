"""Unit tests for multi-objective and fairness evaluation metrics."""

import numpy as np
import pytest
from src.metrics import (
    compute_gini_coefficient,
    compute_hypervolume,
    compute_shannon_entropy,
    compute_spacing,
    extract_non_dominated_front,
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
