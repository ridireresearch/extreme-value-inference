import numpy as np
import pandas as pd
from scipy.stats import pareto, t, norm, expon

def simulate_iid_heavy_tail(n=2000, alpha=3.0):
    """
    Simulates i.i.d. Pareto Returns.
    gamma = 1 / alpha
    """
    # Exceedances over threshold have Pareto distribution.
    # To make it look like "returns", we can negate and shift.
    # Simply providing a random pareto sample for tail analysis.
    returns = pareto.rvs(alpha, size=n)
    # We want downside tail to be easily analyzed, so we make returns negative
    return -returns

def simulate_heterogeneous_scales(n=2000, alpha=3.0, regimes=4):
    """
    Simulates a heavy-tailed distribution where the scale changes systematically.
    This creates heterogeneity while preserving the tail index alpha.
    """
    data_list = []
    block_size = n // regimes
    base_scale = 1.0
    scales = [base_scale * (1.5 ** i) for i in range(regimes)]
    
    for i in range(regimes):
        scale = scales[i]
        block = pareto.rvs(alpha, scale=scale, size=block_size)
        data_list.append(block)
        
    # Remainder
    rem = n % regimes
    if rem > 0:
        data_list.append(pareto.rvs(alpha, scale=scales[-1], size=rem))
        
    returns = np.concatenate(data_list)
    return -returns

def simulate_regime_switching_vol(n=2000):
    """
    Standard GARCH or regime switching volatility proxy with t-distribution.
    """
    # 2 regimes: Low vol, High vol. Heavy tails in both (Student-t)
    df = 4 # gamma = 1/4 = 0.25
    
    # Generate Markov chain for regimes
    p_00 = 0.98
    p_11 = 0.95
    
    states = np.zeros(n, dtype=int)
    states[0] = 0
    for i in range(1, n):
        if states[i-1] == 0:
            states[i] = 0 if np.random.rand() < p_00 else 1
        else:
            states[i] = 1 if np.random.rand() < p_11 else 0
            
    vols = np.where(states == 0, 0.01, 0.03)
    innovations = t.rvs(df, size=n)
    
    returns = vols * innovations
    return returns

def simulate_bounded_tail(n=2000):
    """
    Simulates data with a bounded upper/lower tail (gamma < 0).
    Using a Uniform distribution, or Beta. Look at Uniform(-0.05, 0.05).
    """
    returns = np.random.uniform(-0.05, 0.05, n)
    return returns

def simulate_gumbel_tail(n=2000):
    """
    Simulates exponentially decaying tails (gamma = 0).
    Using Normal distribution.
    """
    returns = np.random.normal(0, 0.02, n)
    return returns
