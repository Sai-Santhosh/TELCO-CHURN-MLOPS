"""Experiment statistics module."""

from .srm import srm_chi_square, check_srm
from .power import sample_size_proportion, mde_proportion, power_proportion
from .cuped import cuped_adjust, cuped_analysis, theta_hat
from .hypothesis_tests import proportions_z_test, continuous_t_test, build_arm_stats
from .segments import segment_analysis

__all__ = [
    "srm_chi_square",
    "check_srm",
    "sample_size_proportion",
    "mde_proportion",
    "power_proportion",
    "cuped_adjust",
    "cuped_analysis",
    "theta_hat",
    "proportions_z_test",
    "continuous_t_test",
    "build_arm_stats",
    "segment_analysis",
]
