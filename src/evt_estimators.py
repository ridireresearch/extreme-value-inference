import numpy as np
import pandas as pd
import warnings

def _get_top_k(data, k):
    """
    Returns the top k + 1 order statistics (X_{n,n} >= ... >= X_{n-k,n}).
    We need X_{n-k, n} as the threshold.
    """
    sorted_data = np.sort(data)
    if k >= len(sorted_data):
        raise ValueError("k must be less than the sample size.")
    # Top k+1 values: sorted_data[- (k+1):]
    # We want them in descending order:
    top_k_plus_1 = sorted_data[-(k+1):][::-1]
    return top_k_plus_1

def hill_estimator(data, k):
    """
    Computes the Hill estimator for the extreme value index (gamma > 0).
    data: 1D array of positive values.
    k: number of upper order statistics to use.
    """
    top_val = _get_top_k(data, k)
    X_k = top_val[k] # X_{n-k, n}
    if X_k <= 0:
        warnings.warn("Threshold is <= 0. Hill estimator expects positive data.")
        return np.nan
        
    log_top = np.log(top_val[:k])
    gamma_H = np.mean(log_top) - np.log(X_k)
    return float(gamma_H)

def moment_estimator(data, k):
    """
    Computes the Dekkers, Einmahl, de Haan moment estimator.
    Valid for all gamma.
    """
    top_val = _get_top_k(data, k)
    X_k = top_val[k]
    
    if X_k <= 0:
        return np.nan
        
    log_diffs = np.log(top_val[:k]) - np.log(X_k)
    
    M1 = np.mean(log_diffs)
    M2 = np.mean(log_diffs**2)
    
    if M1 == 0 or M2 == 0:
        return np.nan
        
    gamma_M = M1 + 1 - 0.5 * (1 - (M1**2) / M2)**(-1)
    return float(gamma_M)

def optimal_theta_proxy(data, k):
    """
    Heuristic proxy for the optimal theta* in the heterogeneous case.
    The exact theta* depends on r_H^2, r_M^2, r_{HM} in the heterogeneous setting.
    This function computes a rolling block variance as a proxy for the structural heterogeneity.
    """
    # Fallback to simple unweighted if k is too small
    if len(data) < 50 or k < 10:
        return 0.5
        
    # In practice, for a dashboard without full panel modeling:
    # A proxy could use the empirical variance of the Hill and Moment estimators
    # over cross-sections or time-blocks, or use a default 0.5 when unidentifiable.
    # For now, return a placeholder optimal value. 
    return 0.5

def heterogeneous_estimator(data, k, theta=None):
    """
    Computes the improved combined estimator for gamma > 0.
    gamma(theta) = (1-theta)*gamma_H + theta*gamma_M
    """
    gamma_H = hill_estimator(data, k)
    gamma_M = moment_estimator(data, k)
    
    if np.isnan(gamma_H) or np.isnan(gamma_M):
        return np.nan
        
    if theta is None:
        theta = optimal_theta_proxy(data, k)
        
    gamma_comb = (1 - theta) * gamma_H + theta * gamma_M
    return float(gamma_comb)

def high_quantile(data, gamma, p, k):
    """
    Estimates the 1-p quantile (e.g. p=0.01 for 99th percentile).
    Assuming gamma > 0 mostly, but adapting for general gamma.
    Uses Weissman estimator for gamma > 0.
    """
    n = len(data)
    top_val = _get_top_k(data, k)
    X_k = top_val[k]
    
    if gamma > 0:
        # Weissman Estimator
        q_p = X_k * ((k / (n * p)) ** gamma)
    else:
        # Based on Moment estimator adaptation for general real gamma
        # Uses M1 from moment estimator
        log_diffs = np.log(top_val[:k]) - np.log(X_k)
        M1 = np.mean(log_diffs)
        
        # Avoid division by zero
        if gamma == 0:
            q_p = X_k + X_k * M1 * np.log(k / (n * p))
        else:
            q_p = X_k + X_k * M1 * ((k / (n * p))**gamma - 1) / gamma
            
    return float(q_p)

def endpoint_estimator(data, gamma, k):
    """
    Estimates the right endpoint (maximum possible value) when gamma < 0.
    """
    top_val = _get_top_k(data, k)
    X_k = top_val[k]
    
    if gamma >= 0:
        return np.inf
        
    # Endpoint x_F = X_{n-k, n} - (X_{n-k,n} * M1) / gamma_M
    log_diffs = np.log(top_val[:k]) - np.log(X_k)
    M1 = np.mean(log_diffs)
    
    x_F = X_k - (X_k * M1) / gamma
    return float(x_F)
