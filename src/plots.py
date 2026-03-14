import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_k_stability(k_range, gamma_H_list, gamma_M_list, gamma_comb_list=None):
    """
    Plots the Hill plot / Moment Plot (gamma estimator vs k).
    Used to visually identify the stable plateau region.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=k_range, y=gamma_H_list, mode='lines', name='Hill Estimator (Heavy Tails)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=k_range, y=gamma_M_list, mode='lines', name='Moment Estimator (General)', line=dict(color='red', dash='dash')))
    
    if gamma_comb_list is not None and len(gamma_comb_list) == len(k_range):
        fig.add_trace(go.Scatter(x=k_range, y=gamma_comb_list, mode='lines', name='Combined / Heterogeneous', line=dict(color='purple', width=2)))
        
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    fig.update_layout(
        title='Tail Index \u03B3 Stability Plot',
        xaxis_title='Threshold index (k)',
        yaxis_title='Estimated Gamma \u03B3',
        template='plotly_dark'
    )
    return fig

def plot_rolling_gamma(df):
    """
    Plots the rolling estimates of gamma over time.
    Expects df with Date index and columns: gamma_H, gamma_M, gamma_comb.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['gamma_H'], mode='lines', name='Hill', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['gamma_M'], mode='lines', name='Moment', line=dict(color='red', dash='dash')))
    
    if 'gamma_comb' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['gamma_comb'], mode='lines', name='Combined', line=dict(color='purple', width=2)))
        
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    
    fig.update_layout(
        title='Rolling Tail Index (\u03B3)',
        xaxis_title='Date',
        yaxis_title='\u03B3',
        template='plotly_dark'
    )
    return fig

def plot_quantile_exceedances(dates, losses, q_p, p_label="99% EVT Quantile"):
    """
    Plots the losses over time alongside the high quantile estimate.
    dates: array of dates
    losses: array of positive losses
    q_p: single valid or array of rolling quantiles
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=dates, y=losses, name='Losses', marker_color='rgba(255, 100, 100, 0.6)'))
    
    if isinstance(q_p, (list, np.ndarray, pd.Series)):
        fig.add_trace(go.Scatter(x=dates, y=q_p, mode='lines', name=p_label, line=dict(color='red', width=2)))
    else:
        fig.add_hline(y=q_p, line_dash="dash", line_color="red", annotation_text=p_label)
        
    fig.update_layout(
        title='Tail Losses and Extreme Quantiles',
        xaxis_title='Date',
        yaxis_title='Loss Magnitude',
        template='plotly_dark',
        barmode='overlay'
    )
    return fig

def plot_asset_comparison(tickers, gamma_vals, risk_scores):
    """
    Bar chart comparing tail heaviness and risk scores across assets.
    """
    fig = go.Figure(data=[
        go.Bar(name='Gamma (\u03B3)', x=tickers, y=gamma_vals, marker_color='blue'),
        go.Bar(name='Risk Score (scaled)', x=tickers, y=np.array(risk_scores)/10.0, marker_color='red')
    ])
    
    fig.update_layout(
        title='Asset Tail Risk Comparison',
        xaxis_title='Asset',
        barmode='group',
        template='plotly_dark'
    )
    return fig
