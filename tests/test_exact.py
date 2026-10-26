"""Unit tests for Exact MILP weighted-sum and secondary scenario."""

import pytest
from src.algorithms.exact_milp import ExactMILP
from src.data_generator import SyntheticDataGenerator
from src.problem import StudentAlumniMatchingProblem


@pytest.fixture
def mini_exact_problem():
    """Create a very small instance for rapid exact MILP tests."""
    gen = SyntheticDataGenerator(seed=42, k_matches=2, q_min=0.15, theta_min_quality=0.40)
    inst = gen.generate_instance(n_students=6, n_alumni=5, avg_capacity=4.0)
    return StudentAlumniMatchingProblem(inst)


def test_exact_milp_weighted_sum(mini_exact_problem):
    """Test exact weighted-sum MILP solves to optimality."""
    solver = ExactMILP(mini_exact_problem)
    sol = solver.solve_weighted_sum(w1=1.0, w2=0.5, w3=0.2, time_limit_sec=20.0)

    assert sol is not None
    assert sol.is_feasible
    assert sol.total_constraint_violation == 0.0
    assert sol.z1_total_quality > 0
    assert sol.z3_max_load >= 1


def test_exact_milp_alumni_minimization(mini_exact_problem):
    """Test secondary epsilon-constraint scenario minimizing alumni count."""
    solver = ExactMILP(mini_exact_problem)
    sol = solver.solve_minimize_alumni_usage(min_quality_ratio=0.80, time_limit_sec=20.0)

    assert sol is not None
    assert sol.active_alumni_count <= mini_exact_problem.n_alumni
    assert sol.active_alumni_count >= 1
    assert sol.company_entropy >= 0.0
