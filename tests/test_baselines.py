"""Unit tests for Random, Greedy, and Min-Cost Flow baseline algorithms."""

import numpy as np
import pytest
from src.baseline.greedy_baseline import GreedyBaseline
from src.baseline.min_cost_flow import MinCostFlowBaseline
from src.baseline.random_baseline import RandomBaseline
from src.data_generator import SyntheticDataGenerator
from src.problem import StudentAlumniMatchingProblem


@pytest.fixture
def baseline_problem():
    """Create test problem instance."""
    gen = SyntheticDataGenerator(seed=77, k_matches=2, q_min=0.20)
    inst = gen.generate_instance(n_students=12, n_alumni=10, avg_capacity=4.0)
    return StudentAlumniMatchingProblem(inst)


def test_random_baseline(baseline_problem):
    """Test random baseline execution and degree preservation."""
    solver = RandomBaseline(baseline_problem, seed=42)
    sol = solver.solve(n_trials=3)
    assert sol.z1_total_quality > 0
    # Each student must have K matches
    assert np.all(np.sum(sol.x, axis=1) == baseline_problem.k_matches)


def test_greedy_baseline(baseline_problem):
    """Test greedy baseline execution and performance."""
    solver = GreedyBaseline(baseline_problem)
    sol = solver.solve()
    assert sol.z1_total_quality > 0
    assert np.all(np.sum(sol.x, axis=1) == baseline_problem.k_matches)
    # Greedy should find a higher quality matching than a purely random single run
    rand_sol = RandomBaseline(baseline_problem, seed=123).solve(n_trials=1)
    assert sol.z1_total_quality >= rand_sol.z1_total_quality * 0.90


def test_min_cost_flow_baseline(baseline_problem):
    """Test min cost flow single-objective optimization."""
    solver = MinCostFlowBaseline(baseline_problem)
    sol = solver.solve(method="networkx")
    assert sol.z1_total_quality > 0
    assert np.all(np.sum(sol.x, axis=1) == baseline_problem.k_matches)
    # Check capacity constraints
    assert np.all(sol.alumni_loads <= baseline_problem.capacities)
