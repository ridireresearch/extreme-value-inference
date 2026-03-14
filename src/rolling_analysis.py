import panadas as pd
import numpy as np

# Corrected imports
import pandas as pd
import numpy as np

from src.preprocessing import extract_tail_losses
from src.evt_estimators import hill_estimator, moment_estimator, heterogeneous_estimator, high_quantile, endpoint_estimator

def batch_rolling_estimates(returns_series, window=252, k=50, downside=True, p=0.01):
    """
    Computes rolling EVT estimates over a time series of returns.
    returns_series: pd.Series of returns indexed by Date.
    window: integer, size of the rolling window (e.g. 252 for 1 year).
    k: number of order statistics to use in each window.
    p: probability for high quantile estimation (e.g., 0.01 for 99%).
    """
    n_obs = len(returns_series)
    dates = returns_series.index
    
    results = []
    
    # We need at least 'window' observations
    if n_obs < window:
        raise ValueError("Time series length is less than the rolling window size.")
        
    for i in range(window, n_obs + 1):
        window_data = returns_series.iloc[i-window:i]
        current_date = dates[i-1]
        
        # Extract losses for the specified tail
        losses = extract_tail_losses(window_data, downside=downside)
        
        # We need at least k losses to compute anything
        if len(losses) <= k:
            res = {
                "Date": current_date,
                "gamma_H": np.nan,
                "gamma_M": np.nan,
                "gamma_comb": np.nan,
                "quantile": np.nan,
                "endpoint": np.nan,
                "n_extreme_losses": len(losses)
            }
        else:
            # Sort losses for EVT
            loss_array = losses.values
            
            gamma_H = hill_estimator(loss_array, k)
            gamma_M = moment_estimator(loss_array, k)
            gamma_comb = heterogeneous_estimator(loss_array, k)
            
            # Use combined estimator for quantile if gamma > 0, else moment
            gamma_for_var = gamma_comb if (not np.isnan(gamma_comb) and gamma_comb > 0) else gamma_M
            
            if np.isnan(gamma_for_var):
                q_p = np.nan
                x_F = np.nan
            else:
                q_p = high_quantile(loss_array, gamma_for_var, p, k)
                x_F = endpoint_estimator(loss_array, gamma_for_var, k)
                
            res = {
                "Date": current_date,
                "gamma_H": gamma_H,
                "gamma_M": gamma_M,
                "gamma_comb": gamma_comb,
                "quantile": q_p,
                "endpoint": x_F,
                "n_extreme_losses": len(losses)
            }
            
        results.append(res)
        
    res_df = pd.DataFrame(results).set_index("Date")
    return res_df

def tail_regime_classification(gamma):
    """
    Classifies the tail regime based on gamma estimate.
    """
    if np.isnan(gamma):
        return "Unknown / Unstable"
    if gamma < -0.05:
        return "Bounded / Thin (gamma < 0)"
    elif -0.05 <= gamma <= 0.05:
        return "Near-Exponential / Gumbel (gamma ~ 0)"
    else:
        return "Heavy Tail (gamma > 0)"
