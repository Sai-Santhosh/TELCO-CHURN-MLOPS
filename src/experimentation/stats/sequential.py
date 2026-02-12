"""
Sequential analysis: alpha-spending and repeated-peeking warnings.

Simple implementation: warning when analyzing before planned sample size,
and adjusted thresholds for early looks.
"""

import math
from typing import Tuple


def obf_boundary(
    alpha: float,
    information_fraction: float,
) -> float:
    """
    O'Brien-Fleming alpha-spending boundary.
    
    Returns adjusted z-critical for current information fraction.
    
    Args:
        alpha: Overall Type I error
        information_fraction: Proportion of planned sample already observed (0-1)
        
    Returns:
        Z-critical value for current look
    """
    if information_fraction <= 0 or information_fraction > 1:
        return 999.0
    
    # Simplified OBF: z = z_alpha / sqrt(t)
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_adj = z_alpha / math.sqrt(information_fraction)
    return float(z_adj)


def repeated_peek_warning(
    n_planned: int,
    n_observed: int,
    n_analyses: int,
) -> Tuple[bool, str]:
    """
    Warning for repeated peeking / early stopping.
    
    Args:
        n_planned: Planned total sample size
        n_observed: Currently observed sample size
        n_analyses: Number of times results have been analyzed
        
    Returns:
        Tuple of (is_warning, message)
    """
    if n_observed >= n_planned and n_analyses <= 1:
        return False, "Single analysis at planned sample size."
    
    msg_parts = []
    
    if n_observed < n_planned:
        pct = 100 * n_observed / n_planned
        msg_parts.append(
            f"Early analysis: only {pct:.0f}% of planned sample. "
            "Type I error inflation possible if stopping early."
        )
    
    if n_analyses > 1:
        msg_parts.append(
            f"Multiple analyses ({n_analyses}) performed. "
            "Consider sequential boundaries or pre-registration."
        )
    
    is_warning = len(msg_parts) > 0
    return is_warning, " ".join(msg_parts) if msg_parts else ""
