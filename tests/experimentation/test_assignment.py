"""Tests for deterministic assignment."""
import pytest
from src.experimentation.assignment import assign_entity, assign_entities
from src.experimentation.schema import ArmType, ExperimentConfig


def test_assign_entity_deterministic():
    """Same entity + experiment always gets same arm."""
    a1 = assign_entity("cust_001", "exp_1")
    a2 = assign_entity("cust_001", "exp_1")
    assert a1 == a2


def test_assign_entity_different_experiments():
    """Different experiments can yield different arms for same entity."""
    a1 = assign_entity("cust_001", "exp_1")
    a2 = assign_entity("cust_001", "exp_2")
    # May or may not differ
    assert a1 in (ArmType.CONTROL, ArmType.TREATMENT)
    assert a2 in (ArmType.CONTROL, ArmType.TREATMENT)


def test_assign_entity_allocation():
    """At 0 allocation, all control. At 1, all treatment."""
    assert assign_entity("x", "e", allocation=0.0) == ArmType.CONTROL
    assert assign_entity("x", "e", allocation=1.0) == ArmType.TREATMENT


def test_assign_entities_balance():
    """50/50 allocation yields roughly balanced arms."""
    config = ExperimentConfig(experiment_id="test", name="Test", allocation=0.5)
    ids = [f"c{i}" for i in range(1000)]
    assignments = assign_entities(ids, config)
    n_c = sum(1 for a in assignments if a.arm == ArmType.CONTROL)
    n_t = sum(1 for a in assignments if a.arm == ArmType.TREATMENT)
    assert 400 <= n_c <= 600
    assert 400 <= n_t <= 600
    assert n_c + n_t == 1000
