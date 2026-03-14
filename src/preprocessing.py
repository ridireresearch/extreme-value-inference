import numpy as np
import pandas as pd

def compute_returns(prices_df, log_returns=True):
    """
    Computes returns from a prices DataFrame.
    If log_returns=True, computes log(P_t / P_{t-1})
    """
    if log_returns:
        rets = np.log(prices_df / prices_df.shift(1))
    else:
        rets = prices_df.pct_change()
    return rets.dropna(how='all')

def extract_tail_losses(returns_series, downside=True):
    """
    Transforms returns for tail analysis.
    If downside is True, we define loss L_t = -r_t.
    If downside is False, we define loss L_t = r_t.
    
    Returns only the strictly positive values for EVT estimation.
    """
    # Convert series to numpy array
    rets = np.asarray(returns_series)
    
    if downside:
        losses = -rets
    else:
        losses = rets
        
    # We only care about positive observations of the specified tail
    positive_losses = losses[losses > 0]
    return positive_losses
    
def get_k_values(n, pct_range=(0.01, 0.15), num_points=50):
    """
    Generates a range of k values (threshold indices) to test for the Hill/Moment plot.
    E.g. from 1% to 15% of the data.
    """
    min_k = max(5, int(n * pct_range[0]))
    max_k = min(n - 1, int(n * pct_range[1]))
    
    # Check if we have enough points
    if max_k <= min_k:
        if n > 10:
            return np.arange(5, min(n-1, 15))
        else:
            return np.array([])
            
    # Return unique integers
    return np.unique(np.linspace(min_k, max_k, num_points).astype(int))
