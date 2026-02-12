"""
Power analysis and MDE (Minimum Detectable Effect) calculator.

Computes required sample size and MDE for proportions and continuous metrics.
"""

from typing import Tuple

import numpy as np
from scipy import stats


def sample_size_proportion(
    baseline: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> int:
    """
    Sample size for two-proportion test (e.g., churn rate).
    
    Args:
        baseline: Baseline proportion (e.g., 0.265 churn)
        mde_relative: Minimum detectable effect as relative change (e.g., 0.10 = 10% relative)
        alpha: Type I error rate
        power: Statistical power (1 - Type II)
        allocation: Fraction in treatment arm
        
    Returns:
        Total sample size needed per arm (returns total/2 for each)
    """
    p1 = baseline
    p2 = baseline * (1 - mde_relative)
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    p_pool = (p1 + p2) / 2
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / allocation + 1 / (1 - allocation)))
    effect = abs(p2 - p1)
    
    if se == 0:
        return 10000
    
    n_per_arm = ((z_alpha + z_beta) * se / effect) ** 2
    return int(np.ceil(n_per_arm))


def mde_proportion(
    baseline: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> float:
    """
    Minimum detectable effect (relative) for proportion.
    
    Args:
        baseline: Baseline proportion
        n_per_arm: Sample size per arm
        alpha: Type I error
        power: Statistical power
        allocation: Treatment allocation
        
    Returns:
        MDE as relative change (e.g., 0.10 = 10% relative reduction detectable)
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Approximate: solve for p2
    se_approx = np.sqrt(baseline * (1 - baseline) * (1 / allocation + 1 / (1 - allocation)) / n_per_arm)
    delta = (z_alpha + z_beta) * se_approx
    
    mde_abs = delta
    mde_relative = mde_abs / baseline if baseline > 0 else 1.0
    return float(mde_relative)


def sample_size_continuous(
    std: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> int:
    """
    Sample size for two-sample t-test (continuous metric).
    
    Args:
        std: Pooled standard deviation
        mde_abs: Minimum detectable effect (absolute difference in means)
        alpha: Type I error
        power: Statistical power
        allocation: Treatment allocation
        
    Returns:
        Total sample size
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    n = 2 * (z_alpha + z_beta) ** 2 * (std ** 2) / (mde_abs ** 2)
    n *= 1 / (allocation * (1 - allocation))  # adjustment for unequal allocation
    return int(np.ceil(n))


def power_proportion(
    baseline: float,
    effect_relative: float,
    n_per_arm: int,
    alpha: float = 0.05,
    allocation: float = 0.5,
) -> float:
    """
    Achieved power for a given effect and sample size.
    
    Returns:
        Statistical power (0-1)
    """
    p1 = baseline
    p2 = baseline * (1 - effect_relative)
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    p_pool = (p1 + p2) / 2
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / allocation + 1 / (1 - allocation)) / n_per_arm)
    effect = abs(p2 - p1)
    
    if se == 0:
        return 0.0
    
    z_crit = effect / se
    power = 1 - stats.norm.cdf(z_alpha - z_crit) + stats.norm.cdf(-z_alpha - z_crit)
    return float(np.clip(power, 0, 1))
