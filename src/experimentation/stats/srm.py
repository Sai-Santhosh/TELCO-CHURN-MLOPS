"""
Sample Ratio Mismatch (SRM) chi-square test.

Detects if the actual allocation ratio deviates significantly from expected.
"""

from typing import Tuple

import numpy as np
from scipy import stats


def srm_chi_square(
    n_control: int,
    n_treatment: int,
    expected_frac: float = 0.5,
) -> Tuple[float, float]:
    """
    Chi-square test for sample ratio mismatch.
    
    H0: actual ratio equals expected
    H1: actual ratio differs from expected
    
    Args:
        n_control: Number in control
        n_treatment: Number in treatment
        expected_frac: Expected fraction in treatment (default 0.5)
        
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    n_total = n_control + n_treatment
    if n_total == 0:
        return 0.0, 1.0
    
    expected_control = n_total * (1 - expected_frac)
    expected_treatment = n_total * expected_frac
    
    observed = np.array([n_control, n_treatment])
    expected = np.array([expected_control, expected_treatment])
    
    # Avoid division by zero
    expected = np.where(expected == 0, 1e-10, expected)
    
    chi2 = np.sum((observed - expected) ** 2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return float(chi2), float(p_value)


def check_srm(
    n_control: int,
    n_treatment: int,
    expected_frac: float = 0.5,
    alpha: float = 0.01,
) -> Tuple[bool, float, float]:
    """
    Check for sample ratio mismatch.
    
    Args:
        n_control: Number in control
        n_treatment: Number in treatment
        expected_frac: Expected fraction in treatment
        alpha: Significance threshold (default 0.01)
        
    Returns:
        Tuple of (srm_passed, chi2_statistic, p_value)
    """
    chi2, p_value = srm_chi_square(n_control, n_treatment, expected_frac)
    srm_passed = p_value >= alpha
    return srm_passed, chi2, p_value
