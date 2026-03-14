# Heterogeneous EVT Tail Risk Dashboard

This dashboard is a production-quality risk tool designed to estimate and monitor extreme tail behavior in financial markets. It implements the key ideas from the paper **"Extreme Value Inference for General Heterogeneous Data"** by Yi He and John H.J. Einmahl, translating theoretical EVT innovations into a practical quant workflow.

## Overview: The Problem Solved
Traditional Extreme Value Theory (EVT) assumes that data points are independent and identically distributed (i.i.d.). However, financial returns are notoriously heterogeneous: they exhibit regime shifts, changing volatility, structural breaks, and nonstationarity over time. 

If a quant PM relies on i.i.d. EVT to estimate tail risk (e.g., 99.9% downside quantiles), their estimates may be highly unstable or fundamentally biased when cross-sectional or time-series heterogeneity is disguised as a single distribution. 

### Why Heterogeneous EVT Matters
The He and Einmahl paper introduces a startling insight: **heterogeneity can actually reduce the variance of tail estimators** compared to the i.i.d. setting, provided it is handled correctly. By acknowledging the heterogeneity and optimally combining estimators, we can achieve more robust estimates of the "average" tail distribution.

This dashboard applies those lessons to real-world financial data, allowing risk managers to identify whether tails are getting fatter, detect regime shifts, and adjust leverage accordingly.

## Implementation Details

### Exact Implementations
- **Hill Estimator ($\gamma_H$):** Standard estimator for heavy tails ($\gamma > 0$).
- **Moment Estimator ($\gamma_M$):** General estimator valid across all domains of attraction ($\gamma \in \mathbb{R}$).
- **High Quantile Extrapolator:** Calculates exact out-of-sample stress quantiles (e.g., 1-in-1000 days).
- **Endpoint Estimator:** Precisely estimates the mathematical right-endpoint (maximum possible loss) when $\gamma < 0$.

### Approximated Implementations
The core mechanism of the paper combines the Hill and Moment estimators using an optimal weight $\theta^*$, defined by complex covariance functionals ($r(1,1)$, $m_R(0)$, $\rho_R$) that require strict structural assumptions about the heterogeneity sequence. 
- **The Combined Estimator ($\gamma(\theta)$)** is theoretically exact, but **the optimal parameter $\theta^*$ is approximated heuristically** in this dashboard. Fully estimating the $R$ functional on rolling financial time series is computationally unstable without imposing rigid parametric forms on the regime changes. Currently, the dashboard uses equal weighting when $\gamma > 0$ as a robust prior, while the architecture (`heterogeneity.py`) is modularized to accept a full covariance estimator in future iterations.

## How to Use the Dashboard

### 1. Running the App
Requires Python 3.9+. Install dependencies and run via Streamlit:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Supplying Data
- **Example Data:** The app can automatically pull tickers from Yahoo Finance (e.g., `SPY, QQQ, BTC-USD`).
- **CSV Upload:** You can upload a standard CSV where the index is the Date, and the columns are Asset returns/prices.

### 3. Interpreting Output
- **$\gamma$ (Tail Index):** 
  - $\gamma > 0$: Heavy-tailed. The higher the number, the more likely extreme crashes are. If $\gamma > 0.25$, leverage should be strictly monitored.
  - $\gamma \approx 0$: Gumbel-type. Exponential decay in the tails (resembles normal/lognormal tail behavior).
  - $\gamma < 0$: Bounded. The distribution has a strict maximum possible loss (very thin tail).
- **High Quantiles:** These represent out-of-sample downside risk. A 99.9% quantile of 8% means a 1-in-1000 day event will lose 8%.
- **Endpoint:** Visible only when $\gamma < 0$, showing the absolute worst-case scenario mathematically possible under the current regime.

### 4. Leverage Control
The **Risk Score** and **Leverage Multiplier** panels are designed to actively size positions. Do not treat EVT quantiles as true false precision. Instead, use the *change* and *instability* of the estimators as directional risk signals. If the threshold stability plot (k-plot) lacks a wide plateau, the data has insufficient tail signal, and leverage should be dialed down due to uncertainty.

## Structure
- `/src/evt_estimators.py`: Pure numpy math functions for EVT.
- `/src/rolling_analysis.py`: Converts static estimators into time-series rolling signals.
- `/src/simulation.py`: Generates rigorous synthetic test regimes. 
- `app.py`: The Streamlit dashboard.
