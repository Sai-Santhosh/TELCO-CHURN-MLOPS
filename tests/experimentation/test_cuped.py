"""Tests for CUPED variance reduction."""
import numpy as np
import pytest
from src.experimentation.stats.cuped import cuped_adjust, theta_hat, cuped_analysis


def test_theta_hat():
    """Theta = Cov(Y,X)/Var(X)."""
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.5 * x + np.random.randn(100) * 0.5
    th = theta_hat(y, x)
    assert 0.3 <= th <= 0.7


def test_cuped_reduces_variance():
    """CUPED should reduce variance when covariate correlates with outcome."""
    np.random.seed(42)
    x = np.random.randn(200)
    y = 0.6 * x + np.random.randn(200) * 0.4
    x_mean = np.mean(x)
    y_adj = cuped_adjust(y, x, x_mean)
    assert np.var(y_adj) < np.var(y)


def test_cuped_analysis_returns_tuple():
    """cuped_analysis returns (lift, p_value, var_reduction, theta)."""
    np.random.seed(42)
    y_c = np.random.randn(50) + 0.3
    y_t = np.random.randn(50) + 0.25
    x_c = np.random.randn(50)
    x_t = np.random.randn(50)
    lift, p, var_red, theta = cuped_analysis(y_c, y_t, x_c, x_t)
    assert isinstance(lift, float)
    assert 0 <= p <= 1
    assert isinstance(var_red, float)
