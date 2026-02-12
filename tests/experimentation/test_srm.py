"""Tests for SRM chi-square."""
import pytest
from src.experimentation.stats.srm import srm_chi_square, check_srm


def test_srm_perfect_balance():
    """500/500 should pass SRM."""
    passed, _, p = check_srm(500, 500, expected_frac=0.5)
    assert passed
    assert p > 0.9


def test_srm_extreme_imbalance():
    """900/100 should fail SRM."""
    passed, _, p = check_srm(900, 100, expected_frac=0.5)
    assert not passed
    assert p < 0.01


def test_srm_chi_square_output():
    """Chi-square returns (stat, pvalue)."""
    chi2, p = srm_chi_square(50, 50)
    assert chi2 >= 0
    assert 0 <= p <= 1
