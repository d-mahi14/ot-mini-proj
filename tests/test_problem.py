"""Unit tests for problem formulation and solution evaluation."""

import numpy as np
import pytest
from src.data_generator import SyntheticDataGenerator
from src.problem import StudentAlumniMatchingProblem


@pytest.fixture
def mini_problem():
    """Create a small problem instance for testing."""
    gen = SyntheticDataGenerator(seed=42, k_matches=2, q_min=0.20, theta_min_quality=0.50)
    inst = gen.generate_instance(n_students=8, n_alumni=6, avg_capacity=4.0)
    return StudentAlumniMatchingProblem(inst)


def test_chromosome_encode_decode(mini_problem):
    """Test bi-directional encoding and decoding between chromosome and matrix."""
    prob = mini_problem
    # Dummy binary matrix where each student has 2 matches
    x_bin = np.zeros((prob.n_students, prob.n_alumni), dtype=int)
    for i in range(prob.n_students):
        x_bin[i, 0] = 1
        x_bin[i, 1] = 1

    chrom = prob.encode_solution(x_bin)
    assert len(chrom) == prob.n_students * prob.k_matches

    decoded_x = prob.decode_chromosome(chrom)
    assert decoded_x.shape == (prob.n_students, prob.n_alumni)
    assert np.array_equal(decoded_x, x_bin)


def test_evaluate_solution_metrics(mini_problem):
    """Verify objective and constraint computation."""
    prob = mini_problem
    x_bin = np.zeros((prob.n_students, prob.n_alumni), dtype=int)
    for i in range(prob.n_students):
        x_bin[i, i % prob.n_alumni] = 1
        x_bin[i, (i + 1) % prob.n_alumni] = 1

    sol = prob.evaluate_solution(x_bin)
    assert sol.z1_total_quality > 0
    assert sol.z2_worst_quality >= 0
    assert sol.z3_max_load >= 1
    assert len(sol.objectives) == 3
    assert sol.objectives[0] == -sol.z1_total_quality
    assert sol.objectives[1] == -sol.z2_worst_quality
    assert sol.objectives[2] == sol.z3_max_load
    assert sol.active_alumni_count <= prob.n_alumni
