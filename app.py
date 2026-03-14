import streamlit as st
import pandas as pd
import numpy as np

from src.data import load_example_data, load_csv_data
from src.preprocessing import compute_returns, extract_tail_losses, get_k_values
from src.evt_estimators import hill_estimator, moment_estimator, heterogeneous_estimator, high_quantile, endpoint_estimator
from src.heterogeneity import estimate_heterogeneity_effect
from src.rolling_analysis import batch_rolling_estimates, tail_regime_classification
from src.risk_signals import calculate_tail_risk_score, get_leverage_multiplier, get_regime_flags
from src.plots import plot_k_stability, plot_rolling_gamma, plot_quantile_exceedances, plot_asset_comparison
from src.simulation import simulate_iid_heavy_tail, simulate_heterogeneous_scales, simulate_regime_switching_vol, simulate_bounded_tail, simulate_gumbel_tail

st.set_page_config(page_title="Heterogeneous EVT Risk Dashboard", layout="wide")

st.title("Heterogeneous EVT Tail Risk Dashboard")
st.markdown("""
This dashboard applies Extreme Value Theory (EVT) for financial tail-risk monitoring, 
leveraging insights from *He & Einmahl: Extreme Value Inference for General Heterogeneous Data*.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Data Loading & Settings")
data_source = st.sidebar.radio("Data Source", ["Example Tickers", "Upload CSV"])

if data_source == "Example Tickers":
    tickers_input = st.sidebar.text_input("Tickers (comma separated)", "SPY, QQQ, IWM, BTC-USD")
    tickers = [x.strip() for x in tickers_input.split(",")]
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
    
    @st.cache_data
    def load_data(t, sd):
        df = load_example_data(t, start_date=sd)
        return df
        
    prices_df = load_data(tickers, start_date)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Date index, Asset columns)", type=["csv"])
    if uploaded_file is not None:
        prices_df = load_csv_data(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

if prices_df.empty:
    st.error("No data loaded.")
    st.stop()

tail_side = st.sidebar.radio("Tail to Analyze", ["Downside (Losses)", "Upside (Gains)"])
is_downside = (tail_side == "Downside (Losses)")

# Returns computation
returns_df = compute_returns(prices_df, log_returns=True)
available_assets = list(returns_df.columns)

st.sidebar.subheader("EVT Parameters")
selected_asset = st.sidebar.selectbox("Primary Asset to Analyze", available_assets)
n_obs = len(returns_df)
k_pct = st.sidebar.slider("Threshold k (% of data)", 1.0, 15.0, 5.0, 0.5) / 100.0

# Extract losses for the selected asset
primary_losses = extract_tail_losses(returns_df[selected_asset], downside=is_downside)
n_losses = len(primary_losses)
k_val = max(5, int(n_obs * k_pct)) # Using k relative to total observations n

if k_val >= n_losses:
    st.sidebar.error(f"k ({k_val}) is larger than the number of positive tail observations ({n_losses}).")
    st.stop()
    
# Layout Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Overview", 
    "2. Asset Comparison", 
    "3. Heterogeneity", 
    "4. Rolling Analysis", 
    "5. Threshold (k)", 
    "6. Stress", 
    "7. Sim Lab"
])

# Compute base estimates for the primary asset
loss_array = primary_losses.values
gamma_H = hill_estimator(loss_array, k_val)
gamma_M = moment_estimator(loss_array, k_val)
gamma_comb = heterogeneous_estimator(loss_array, k_val)
q_99 = high_quantile(loss_array, gamma_comb if not np.isnan(gamma_comb) and gamma_comb>0 else gamma_M, 0.01, k_val)

regime = tail_regime_classification(gamma_comb if not np.isnan(gamma_comb) else gamma_M)
risk_score = calculate_tail_risk_score(gamma_comb if not np.isnan(gamma_comb) else gamma_M, [], q_99)
lev_mult = get_leverage_multiplier(risk_score)
flags = get_regime_flags(gamma_comb if not np.isnan(gamma_comb) else gamma_M, risk_score=risk_score)

with tab1:
    st.header(f"Risk Overview: {selected_asset}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Heterogeneous \u03B3", f"{gamma_comb:.3f}" if not np.isnan(gamma_comb) else "NaN")
    col2.metric("Hill \u03B3", f"{gamma_H:.3f}" if not np.isnan(gamma_H) else "NaN")
    col3.metric("Moment \u03B3", f"{gamma_M:.3f}" if not np.isnan(gamma_M) else "NaN")
    col4.metric("99% Tail Quantile", f"{q_99:.2%}")
    
    st.subheader("Leverage & Regime")
    scol1, scol2, scol3 = st.columns(3)
    
    scol1.metric("Tail Risk Score", f"{risk_score:.1f} / 10")
    
    # Leverage color
    lev_color = "green" if lev_mult >= 1.0 else ("orange" if lev_mult >= 0.5 else "red")
    scol2.markdown(f"**Leverage Suggestion:** <span style='color:{lev_color}; font-size:24px'>{lev_mult}x</span>", unsafe_allow_html=True)
    
    scol3.markdown("**Tail Regime:**")
    scol3.write(regime)
    
    st.markdown("### Warning Flags")
    for f in flags:
        if "Normal" in f:
            st.success(f)
        elif "Reduce" in f or "Extreme" in f:
            st.error(f)
        else:
            st.warning(f)

with tab2:
    st.header("Cross-Asset Comparison")
    
    comp_data = []
    for asset in available_assets:
        l = extract_tail_losses(returns_df[asset], downside=is_downside).values
        if len(l) > k_val:
            gh = hill_estimator(l, k_val)
            gm = moment_estimator(l, k_val)
            gc = heterogeneous_estimator(l, k_val)
            g_used = gc if not np.isnan(gc) and gc > 0 else gm
            q = high_quantile(l, g_used, 0.01, k_val)
            rs = calculate_tail_risk_score(g_used, [], q)
            
            comp_data.append({
                "Asset": asset,
                "Gamma(Comb)": gc,
                "Gamma(Moment)": gm,
                "99% Quantile": q,
                "Risk Score": rs
            })
            
    if comp_data:
        comp_df = pd.DataFrame(comp_data).set_index("Asset")
        st.dataframe(comp_df.style.format({
            "Gamma(Comb)": "{:.3f}", "Gamma(Moment)": "{:.3f}", 
            "99% Quantile": "{:.2%}", "Risk Score": "{:.1f}"
        }))
        
        fig_comp = plot_asset_comparison(comp_df.index, comp_df['Gamma(Comb)'].fillna(comp_df['Gamma(Moment)']), comp_df['Risk Score'])
        st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.header("Heterogeneity Diagnostics")
    st.markdown("Assessing if the data exhibits structural heterogeneity (e.g. regime changes) that the naive i.i.d. EVT estimator ignores.")
    
    diag = estimate_heterogeneity_effect(returns_df[selected_asset], k_val)
    st.write(diag)
    
    if diag["meaningful_heterogeneity"]:
        st.success("Heterogeneity appears meaningful. The combined estimator leverages this to reduce asymptotic variance.")
    else:
        st.info("Little evidence heterogeneity helps in this sample block. Estimates converge to standard.")

with tab4:
    st.header("Rolling Tail Analysis")
    window = st.slider("Rolling Window (days)", 100, 1000, 252, 50)
    
    if st.button("Compute Rolling EVT"):
        with st.spinner("Computing..."):
            roll_df = batch_rolling_estimates(returns_df[selected_asset], window=window, k=k_val, downside=is_downside)
            st.plotly_chart(plot_rolling_gamma(roll_df), use_container_width=True)
            st.dataframe(roll_df.tail())

with tab5:
    st.header("Threshold (k) Sensitivity")
    st.markdown("EVT estimates are highly sensitive to the choice of k. Look for a stable plateau in the plot.")
    
    k_range = get_k_values(n_obs)
    
    g_h_list = []
    g_m_list = []
    g_c_list = []
    
    for k_test in k_range:
        if len(loss_array) > k_test:
            g_h_list.append(hill_estimator(loss_array, k_test))
            g_m_list.append(moment_estimator(loss_array, k_test))
            g_c_list.append(heterogeneous_estimator(loss_array, k_test))
        else:
            g_h_list.append(np.nan)
            g_m_list.append(np.nan)
            g_c_list.append(np.nan)
            
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(x=k_range, y=g_h_list, name='Hill'))
    fig_k.add_trace(go.Scatter(x=k_range, y=g_m_list, name='Moment'))
    fig_k.add_trace(go.Scatter(x=k_range, y=g_c_list, name='Combined'))
    fig_k.update_layout(title="Estimator Stability across K", template="plotly_dark")
    st.plotly_chart(fig_k, use_container_width=True)

with tab6:
    st.header("Stress Tests & Extreme Quantiles")
    
    gamma_for_var = gamma_comb if not np.isnan(gamma_comb) and gamma_comb>0 else gamma_M
    
    q_99 = high_quantile(loss_array, gamma_for_var, 0.01, k_val)
    q_995 = high_quantile(loss_array, gamma_for_var, 0.005, k_val)
    q_999 = high_quantile(loss_array, gamma_for_var, 0.001, k_val)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("1-in-100 Event (99%)", f"{q_99:.2%}")
    col_s2.metric("1-in-200 Event (99.5%)", f"{q_995:.2%}")
    col_s3.metric("1-in-1000 Event (99.9%)", f"{q_999:.2%}")
    
    if gamma_for_var < 0:
        endpoint = endpoint_estimator(loss_array, gamma_for_var, k_val)
        st.info(f"**Estimated Maximum Possible Loss (Endpoint):** {endpoint:.2%}")

with tab7:
    st.header("Simulation / Validation Lab")
    st.markdown("Test the estimators on synthetic data to understand their behavior.")
    
    sim_type = st.selectbox("Select Scenario", [
        "I.I.D. Heavy Tail", 
        "Heterogeneous Scales", 
        "Regime-Switching Volatility",
        "Bounded Tail (Uniform)",
        "Gumbel Tail (Normal)"
    ])
    
    if st.button("Run Simulation"):
        if sim_type == "I.I.D. Heavy Tail":
            sim_data = simulate_iid_heavy_tail(2000)
        elif sim_type == "Heterogeneous Scales":
            sim_data = simulate_heterogeneous_scales(2000)
        elif sim_type == "Regime-Switching Volatility":
            sim_data = simulate_regime_switching_vol(2000)
        elif sim_type == "Bounded Tail (Uniform)":
            sim_data = simulate_bounded_tail(2000)
        else:
            sim_data = simulate_gumbel_tail(2000)
            
        sim_losses = extract_tail_losses(pd.Series(sim_data), downside=True).values
        sim_k = int(0.05 * 2000)
        
        sh = hill_estimator(sim_losses, sim_k)
        sm = moment_estimator(sim_losses, sim_k)
        sc = heterogeneous_estimator(sim_losses, sim_k)
        
        st.write(f"**Hill Estimator:** {sh:.3f}" if not np.isnan(sh) else "NaN")
        st.write(f"**Moment Estimator:** {sm:.3f}" if not np.isnan(sm) else "NaN")
        st.write(f"**Combined Estimator:** {sc:.3f}" if not np.isnan(sc) else "NaN")
        
        st.line_chart(-sim_data)
        
        diag_sim = estimate_heterogeneity_effect(pd.Series(sim_losses), sim_k)
        st.write("Heterogeneity Diagnostic:", diag_sim)
