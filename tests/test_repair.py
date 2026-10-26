"""Unit tests for the custom matching repair operator."""

import numpy as np
import pytest
from src.data_generator import SyntheticDataGenerator
from src.multiobjective.repair import MatchingRepair
from src.problem import StudentAlumniMatchingProblem


@pytest.fixture
def test_setup():
    """Setup problem and repair operator."""
    gen = SyntheticDataGenerator(seed=99, k_matches=2, q_min=0.30)
    inst = gen.generate_instance(n_students=10, n_alumni=8, avg_capacity=4.0)
    prob = StudentAlumniMatchingProblem(inst)
    repair = MatchingRepair(prob)
    return prob, repair


def test_repair_fixes_duplicates_and_consent(test_setup):
    """Ensure that duplicate assignments and non-consenting alumni are fixed."""
    prob, repair = test_setup

    # Artificially set alumnus 0 to have consent = 0
    prob.consents[0] = 0

    # Create an invalid chromosome with duplicates (e.g., student 0 assigned [0, 0])
    invalid_assigned = np.zeros((prob.n_students, prob.k_matches), dtype=int)
    # student 0: [0, 0] (both invalid: consent 0, and duplicate)
    invalid_assigned[0] = [0, 0]
    for i in range(1, prob.n_students):
        invalid_assigned[i] = [1, 2]

    repaired = repair.repair_single(invalid_assigned)
    assert repaired.shape == (prob.n_students, prob.k_matches)

    # Check student 0
    s0_matches = repaired[0]
    assert len(set(s0_matches)) == prob.k_matches, "Must have K distinct matches"
    assert 0 not in s0_matches, "Alumnus 0 has no consent, should not be assigned"

    # Check all students have K distinct matches
    for i in range(prob.n_students):
        assert len(set(repaired[i])) == prob.k_matches


def test_repair_enforces_capacity(test_setup):
    """Ensure that capacity overflows are re-allocated."""
    prob, repair = test_setup

    # Assign all students to alumnus 1
    overloaded = np.full((prob.n_students, prob.k_matches), 1, dtype=int)
    for i in range(prob.n_students):
        overloaded[i, 1] = 2

    repaired = repair.repair_single(overloaded)
    # Decode to binary
    x_bin = np.zeros((prob.n_students, prob.n_alumni), dtype=int)
    for i in range(prob.n_students):
        for j in repaired[i]:
            x_bin[i, j] += 1

    loads = np.sum(x_bin, axis=0)
    # Check that capacity overflows are eliminated or minimized
    cap_overflow = np.sum(np.maximum(0, loads - prob.capacities))
    assert cap_overflow == 0, f"Expected 0 capacity overflow after repair, got {cap_overflow}"
