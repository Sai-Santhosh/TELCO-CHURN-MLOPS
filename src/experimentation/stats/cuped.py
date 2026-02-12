"""
CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

Uses pre-period covariate (tenure, MonthlyCharges, or predicted churn prob)
to reduce variance and improve sensitivity.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def theta_hat(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute optimal theta for CUPED: theta = Cov(Y,X) / Var(X).
    
    Args:
        y: Outcome variable
        x: Covariate (pre-period metric)
        
    Returns:
        Optimal theta coefficient
    """
    cov = np.cov(y, x)[0, 1]
    var_x = np.var(x)
    if var_x == 0:
        return 0.0
    return cov / var_x


def cuped_adjust(
    y: np.ndarray,
    x: np.ndarray,
    x_overall_mean: float,
) -> np.ndarray:
    """
    Apply CUPED adjustment: Y_adj = Y - theta * (X - mean(X)).
    
    Args:
        y: Outcome values
        x: Covariate values
        x_overall_mean: Overall mean of covariate (computed on full sample before split)
        
    Returns:
        Adjusted outcome values
    """
    th = theta_hat(y, x)
    return y - th * (x - x_overall_mean)


def cuped_analysis(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,
    x_treatment: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Run CUPED-adjusted two-sample t-test.
    
    Args:
        y_control: Control outcomes
        y_treatment: Treatment outcomes
        x_control: Control covariate
        x_treatment: Treatment covariate
        
    Returns:
        Tuple of (lift, p_value, variance_reduction_pct, theta)
    """
    x_all = np.concatenate([x_control, x_treatment])
    x_mean = np.mean(x_all)
    
    y_adj_control = cuped_adjust(y_control, x_control, x_mean)
    y_adj_treatment = cuped_adjust(y_treatment, x_treatment, x_mean)
    
    # Variance before and after
    var_before = np.var(np.concatenate([y_control, y_treatment]))
    var_after_control = np.var(y_adj_control)
    var_after_treatment = np.var(y_adj_treatment)
    n_c, n_t = len(y_control), len(y_treatment)
    var_after_pooled = ((n_c - 1) * var_after_control + (n_t - 1) * var_after_treatment) / (n_c + n_t - 2)
    
    variance_reduction = (1 - var_after_pooled / var_before) * 100 if var_before > 0 else 0
    
    # T-test on adjusted values
    t_stat, p_val = stats.ttest_ind(y_adj_treatment, y_adj_control)
    
    lift = np.mean(y_adj_treatment) - np.mean(y_adj_control)
    
    return float(lift), float(p_val), float(variance_reduction), float(theta_hat(
        np.concatenate([y_control, y_treatment]),
        np.concatenate([x_control, x_treatment]),
    ))
