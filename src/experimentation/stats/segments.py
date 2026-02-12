"""
Segmented analysis for experiments.

Breakdown by tenure bucket, contract type, internet service, etc.
Per-segment lift + multiple-comparison warning (Benjamini-Hochberg FDR optional).
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..schema import ArmType, SegmentResult
from .hypothesis_tests import proportions_z_test


def segment_analysis(
    df: pd.DataFrame,
    segment_col: str,
    arm_col: str,
    outcome_col: str,
    metric_type: str = "binary",
    ci_level: float = 0.95,
    alpha: float = 0.05,
    apply_fdr: bool = False,
) -> List[SegmentResult]:
    """
    Compute per-segment analysis.
    
    Args:
        df: DataFrame with arm, outcome, and segment column
        segment_col: Column defining segments
        arm_col: Column with arm (control/treatment)
        outcome_col: Column with outcome values
        metric_type: 'binary' or 'continuous'
        ci_level: Confidence level
        alpha: Significance threshold
        apply_fdr: Apply Benjamini-Hochberg FDR correction
        
    Returns:
        List of SegmentResult
    """
    results = []
    
    for seg_val in df[segment_col].dropna().unique():
        sub = df[df[segment_col] == seg_val]
        ctrl = sub[sub[arm_col] == ArmType.CONTROL.value]
        treat = sub[sub[arm_col] == ArmType.TREATMENT.value]
        
        n_c = len(ctrl)
        n_t = len(treat)
        
        if n_c < 10 or n_t < 10:
            continue
        
        if metric_type == "binary":
            x_c = int(ctrl[outcome_col].sum())
            x_t = int(treat[outcome_col].sum())
            p_c = x_c / n_c if n_c > 0 else 0
            p_t = x_t / n_t if n_t > 0 else 0
            
            lift, lift_pct, p_val, ci_lo, ci_hi = proportions_z_test(
                n_c, x_c, n_t, x_t, ci_level
            )
        else:
            lift = float(treat[outcome_col].mean() - ctrl[outcome_col].mean())
            p_c = float(ctrl[outcome_col].mean())
            p_t = float(treat[outcome_col].mean())
            lift_pct = (lift / p_c * 100) if p_c != 0 else 0
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(treat[outcome_col], ctrl[outcome_col])
            se = np.sqrt(ctrl[outcome_col].var()/n_c + treat[outcome_col].var()/n_t)
            t_crit = stats.t.ppf((1+ci_level)/2, n_c+n_t-2)
            ci_lo = lift - t_crit * se
            ci_hi = lift + t_crit * se
        
        results.append(SegmentResult(
            segment_name=segment_col,
            segment_value=str(seg_val),
            control_n=n_c,
            treatment_n=n_t,
            control_mean=p_c if metric_type == "binary" else float(ctrl[outcome_col].mean()),
            treatment_mean=p_t if metric_type == "binary" else float(treat[outcome_col].mean()),
            lift=float(lift),
            lift_pct=float(lift_pct),
            p_value=float(p_val),
            ci_low=float(ci_lo),
            ci_high=float(ci_hi),
            significant=p_val < alpha,
        ))
    
    if apply_fdr and results:
        pvals = [r.p_value for r in results]
        rejected, pvals_adj = _benjamini_hochberg(pvals, alpha)
        for r, adj_p in zip(results, pvals_adj):
            r.significant = adj_p < alpha
    
    return results


def _benjamini_hochberg(p_values: List[float], alpha: float) -> tuple:
    """Benjamini-Hochberg FDR correction."""
    p_arr = np.array(p_values)
    n = len(p_arr)
    order = np.argsort(p_arr)
    p_sorted = p_arr[order]
    
    # BH critical values
    crit = (np.arange(1, n + 1) / n) * alpha
    rejected = p_sorted <= crit
    
    # Adjusted p-values
    p_adj = np.minimum.accumulate((n / np.arange(1, n + 1)) * p_sorted[::-1])[::-1]
    # Reverse to original order
    inv_order = np.argsort(order)
    p_adj_orig = p_adj[inv_order]
    
    return rejected, list(p_adj_orig)
