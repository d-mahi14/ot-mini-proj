"""Unit tests for synthetic data generation and instance serialization."""

import pytest
import numpy as np
from src.data_generator import SyntheticDataGenerator, MatchingInstance


def test_synthetic_data_generation_constraints():
    """Verify that generated instances satisfy basic structural requirements."""
    gen = SyntheticDataGenerator(seed=42, k_matches=2, q_min=0.25)
    inst = gen.generate_instance(n_students=20, n_alumni=12, name="test_instance")

    assert inst.name == "test_instance"
    assert inst.notice == "ARTIFICIAL - FOR TESTING ONLY"
    assert len(inst.students) == 20
    assert len(inst.alumni) == 12
    assert inst.q_matrix.shape == (20, 12)

    # Check pseudonymous IDs
    for s in inst.students:
        assert s.id.startswith("S")
    for a in inst.alumni:
        assert a.id.startswith("A")

    # Check match scores strictly bounded in [0, 1]
    assert np.all(inst.q_matrix >= 0.0)
    assert np.all(inst.q_matrix <= 1.0)

    # Total alumni capacity must exceed student match demand
    total_cap = sum(a.capacity * a.consent for a in inst.alumni)
    demand = inst.n_students * inst.k_matches
    assert total_cap >= demand


def test_instance_serialization(tmp_path):
    """Verify JSON save and load round-trip fidelity."""
    gen = SyntheticDataGenerator(seed=123)
    inst = gen.generate_instance(n_students=15, n_alumni=10)

    json_file = tmp_path / "test_inst.json"
    inst.save_json(json_file)

    loaded = MatchingInstance.load_json(json_file)
    assert loaded.name == inst.name
    assert loaded.n_students == 15
    assert loaded.n_alumni == 10
    assert np.allclose(loaded.q_matrix, inst.q_matrix)
    assert loaded.notice == "ARTIFICIAL - FOR TESTING ONLY"
