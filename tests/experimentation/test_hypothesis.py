"""Tests for z-test and t-test."""
import numpy as np
import pytest
from src.experimentation.stats.hypothesis_tests import (
    proportions_z_test,
    continuous_t_test,
)
from src.experimentation.schema import ArmType
from src.experimentation.stats.hypothesis_tests import build_arm_stats


def test_proportions_z_test_known():
    """Known case: 20/100 vs 10/100 -> treatment better."""
    lift, lift_pct, p_val, ci_lo, ci_hi = proportions_z_test(100, 20, 100, 10)
    assert lift < 0  # treatment has lower churn
    assert p_val < 0.05


def test_proportions_z_test_equal():
    """Equal proportions -> high p-value."""
    lift, _, p_val, _, _ = proportions_z_test(100, 30, 100, 30)
    assert abs(lift) < 0.01
    assert p_val > 0.9


def test_continuous_t_test():
    """T-test on synthetic data."""
    ctrl = np.random.randn(100) + 5
    treat = np.random.randn(100) + 5.5
    lift, _, p_val, ci_lo, ci_hi, d = continuous_t_test(ctrl, treat)
    assert ci_lo <= lift <= ci_hi


def test_build_arm_stats():
    """ArmStats from values."""
    vals = np.array([0.2, 0.3, 0.25, 0.22])
    s = build_arm_stats(ArmType.CONTROL, vals)
    assert s.n == 4
    assert 0.2 <= s.mean <= 0.3
    assert s.ci_low <= s.mean <= s.ci_high
