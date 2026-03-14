import numpy as np
import pytest
from src.evt_estimators import hill_estimator, moment_estimator

def test_hill_estimator_pareto():
    """
    Hill estimator on standard Pareto should be close to 1/alpha.
    """
    np.random.seed(42)
    # Pareto alpha = 2.0 -> gamma = 0.5
    data = np.random.pareto(2.0, 10000)
    gamma_H = hill_estimator(data, 500)
    assert 0.4 < gamma_H < 0.6

def test_moment_estimator_uniform():
    """
    Moment estimator on Uniform data (gamma = -1).
    """
    np.random.seed(42)
    data = np.random.uniform(0, 1, 10000)
    gamma_M = moment_estimator(data, 500)
    assert -1.2 < gamma_M < -0.8
    
def test_moment_estimator_pareto():
    """
    Moment estimator on Pareto (gamma=0.5).
    """
    np.random.seed(42)
    data = np.random.pareto(2.0, 10000)
    gamma_M = moment_estimator(data, 500)
    assert 0.4 < gamma_M < 0.6
