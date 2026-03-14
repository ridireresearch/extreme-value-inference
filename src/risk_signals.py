import numpy as np

def calculate_tail_risk_score(gamma_current, gamma_hist, q_current, k_unstable=False, hetero_warning=False):
    """
    Computes a composite tail risk score from 0 to 10 (higher is riskier).
    """
    score = 5.0 # baseline
    
    if np.isnan(gamma_current):
        return np.nan
        
    # Penalize high explicit gamma
    if gamma_current > 0.3:
        score += 2
    elif gamma_current > 0.15:
        score += 1
    elif gamma_current < 0:
        score -= 1 # Bounded tail is less risky
        
    # Penalize rapid deterioration (increase in gamma)
    if len(gamma_hist) >= 5:
        recent_avg = np.nanmean(gamma_hist[-5:])
        if gamma_current > recent_avg + 0.1:
            score += 2
            
    if k_unstable:
        score += 1
        
    # Constrain to 0-10
    return max(0.0, min(10.0, score))

def get_leverage_multiplier(risk_score):
    """
    Rule-based output for PM leverage strategy.
    Returns:
    - 1.00x normal
    - 0.75x caution
    - 0.50x reduce
    - 0.25x crisis mode
    """
    if np.isnan(risk_score):
        return 1.0
        
    if risk_score >= 8:
        return 0.25
    elif risk_score >= 6:
        return 0.50
    elif risk_score >= 4:
        return 0.75
    else:
        return 1.00

def get_regime_flags(gamma, k_unstable=False, hetero_meaningful=False, risk_score=None):
    """
    Generates string warning flags for the UI.
    """
    flags = []
    
    if np.isnan(gamma):
        flags.append("Insufficient data for robust tail estimate.")
        return flags
        
    if gamma > 0.25:
        flags.append("Heavy-tail regime worsening: Extreme downside is highly probable.")
    elif gamma < -0.05:
        flags.append("Endpoint-like behavior: Tail appears bounded (thin).")
        
    if k_unstable:
        flags.append("Tail estimate unstable: High sensitivity to threshold k.")
        
    if hetero_meaningful:
        flags.append("Heterogeneity materially lowering variance of estimate.")
        
    if risk_score is not None and risk_score >= 7:
        flags.append("Crash-risk extrapolation rising: Reduce exposure.")
        
    if not flags:
        flags.append("Normal tail conditions.")
        
    return flags
