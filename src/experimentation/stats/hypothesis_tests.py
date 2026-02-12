"""
Frequentist hypothesis tests for experiment analysis.

Z-test for proportions, t-test for continuous metrics.
Computes lift, CI (95%), p-value, and effect size (Cohen's d).
"""

from typing import Tuple

import numpy as np
from scipy import stats

from ..schema import ArmType, ArmStats


def proportions_z_test(
    n1: int,
    x1: int,
    n2: int,
    x2: int,
    ci_level: float = 0.95,
) -> Tuple[float, float, float, float, float]:
    """
    Two-proportion z-test (e.g., churn rate control vs treatment).
    
    Args:
        n1: Control sample size
        x1: Control successes (e.g., churned)
        n2: Treatment sample size
        x2: Treatment successes
        ci_level: Confidence level
        
    Returns:
        Tuple of (lift, lift_pct, p_value, ci_low, ci_high)
    """
    p1 = x1 / n1 if n1 > 0 else 0
    p2 = x2 / n2 if n2 > 0 else 0
    
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if (n1 and n2) else 1e-10
    
    lift = p2 - p1
    lift_pct = (p2 - p1) / p1 * 100 if p1 > 0 else 0
    
    z = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    z_crit = stats.norm.ppf((1 + ci_level) / 2)
    ci_low = lift - z_crit * se
    ci_high = lift + z_crit * se
    
    return lift, lift_pct, float(p_value), float(ci_low), float(ci_high)


def continuous_t_test(
    control_vals: np.ndarray,
    treatment_vals: np.ndarray,
    ci_level: float = 0.95,
) -> Tuple[float, float, float, float, float, float]:
    """
    Two-sample t-test for continuous metric.
    
    Args:
        control_vals: Control outcome values
        treatment_vals: Treatment outcome values
        ci_level: Confidence level
        
    Returns:
        Tuple of (lift, lift_pct, p_value, ci_low, ci_high, cohens_d)
    """
    m_c = np.mean(control_vals)
    m_t = np.mean(treatment_vals)
    std_c = np.std(control_vals, ddof=1) if len(control_vals) > 1 else 0
    std_t = np.std(treatment_vals, ddof=1) if len(treatment_vals) > 1 else 0
    n_c = len(control_vals)
    n_t = len(treatment_vals)
    
    lift = m_t - m_c
    lift_pct = (m_t - m_c) / m_c * 100 if m_c != 0 else 0
    
    t_stat, p_value = stats.ttest_ind(treatment_vals, control_vals)
    
    se = np.sqrt(std_c**2 / n_c + std_t**2 / n_t) if (n_c and n_t) else 1e-10
    df = n_c + n_t - 2
    t_crit = stats.t.ppf((1 + ci_level) / 2, df)
    ci_low = lift - t_crit * se
    ci_high = lift + t_crit * se
    
    # Cohen's d
    pooled_std = np.sqrt(((n_c - 1) * std_c**2 + (n_t - 1) * std_t**2) / (n_c + n_t - 2))
    cohens_d = (m_t - m_c) / pooled_std if pooled_std > 0 else 0
    
    return float(lift), float(lift_pct), float(p_value), float(ci_low), float(ci_high), float(cohens_d)


def build_arm_stats(
    arm: ArmType,
    values: np.ndarray,
    ci_level: float = 0.95,
) -> ArmStats:
    """Build ArmStats from outcome values."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0
    
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf((1 + ci_level) / 2, n - 1)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = mean
        ci_high = mean
    
    return ArmStats(
        arm=arm,
        n=n,
        mean=mean,
        std=std,
        ci_low=ci_low,
        ci_high=ci_high,
    )
