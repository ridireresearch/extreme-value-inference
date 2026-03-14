import numpy as np
from src.evt_estimators import hill_estimator, moment_estimator, _get_top_k

def estimate_heterogeneity_effect(returns_series, k, blocks=4):
    """
    A practical diagnostic for the heterogeneity effect as motivated by the paper.
    The paper shows variance of estimator can be smaller if structural heterogeneity exists.
    Here we compare the variance of block-wise estimates vs the pooled estimate variance.
    """
    losses = returns_series[returns_series > 0].values
    n = len(losses)
    if n < k * 2 or blocks < 2:
        return {"pooled_hill": np.nan, "subsample_variance": np.nan, "meaningful": False}
        
    # Splitting into blocks
    block_size = n // blocks
    sub_k = max(5, k // blocks)
    
    block_hills = []
    for i in range(blocks):
        block_data = losses[i*block_size:(i+1)*block_size]
        if len(block_data) > sub_k:
            gamma_b = hill_estimator(block_data, sub_k)
            if not np.isnan(gamma_b):
                block_hills.append(gamma_b)
                
    pooled_hill = hill_estimator(losses, k)
    
    if len(block_hills) < 2:
        return {"pooled_hill": pooled_hill, "subsample_variance": np.nan, "meaningful": False}
        
    sub_var = np.var(block_hills, ddof=1)
    
    # Under iid assumption, sub_var should be roughly related to the asymptotic variance of gamma
    # If it is widely different across blocks, the sample has high heterogeneity.
    # This is a heuristic flag.
    meaningful = sub_var > 0.05
    
    return {
        "pooled_hill": pooled_hill,
        "subsample_variance": sub_var,
        "block_hills": block_hills,
        "meaningful_heterogeneity": meaningful
    }
    
def optimal_theta_heuristic(losses, k):
    """
    Approximation for theta* the optimal weight between Hill and Moment estimator.
    In the paper: gamma(theta) = (1-theta)*gamma_H + theta*gamma_M
    The optimal theta* is derived from the covariance structure between gamma_H and gamma_M.
    If exact R(1,1) is not estimated, we use the fact that Moment estimator has higher
    variance but less bias in some scenarios, and heterogeneity changes this balance.
    As a heuristic: if we detect heavy tails (gamma_H > 0) we use a balanced weight.
    """
    # A true implementation would require estimating the R functional
    # e.g., estimating R(1, 1), m_R(0), etc.
    # For a first pass, we give equal weight if gamma > 0
    return 0.5
