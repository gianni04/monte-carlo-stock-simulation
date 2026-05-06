import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import os
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)

# =============================================================
# CONFIGURATION
# =============================================================
TICKER      = "UBI.PA"
NAME        = "Ubisoft"
START       = "2019-01-01"
END         = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
SIMULATIONS = 2000
HORIZON     = 252
INVESTMENT  = 50
LEVERAGE    = 2
EXPOSURE    = INVESTMENT * LEVERAGE
RISK_FREE   = 0.03 / 252
STOP_LOSS   = -0.20

print("=" * 60)
print(" QUANT SIMULATION ENGINE - " + NAME)
print("=" * 60)
print("Period     : " + START + " to " + END)
print("Capital    : " + str(INVESTMENT) + " EUR  |  Leverage: " + str(LEVERAGE) + "x")
print("Simulations: " + str(SIMULATIONS) + "  |  Horizon: " + str(HORIZON) + " days")
print("Models     : GBM | GARCH(1,1) | Jump Diffusion | Bootstrap")

# =============================================================
# DATA
# =============================================================
df    = yf.download(TICKER, start=START, end=END, auto_adjust=True)
close = df["Close"].squeeze().dropna()
returns = close.pct_change().dropna()
log_ret = np.log(close / close.shift(1)).dropna()

S0    = float(close.iloc[-1])
mu    = float(log_ret.mean())
sigma = float(log_ret.std())
skew  = float(returns.skew())
kurt  = float(returns.kurtosis())

print("\nLast price : " + str(round(S0, 2)) + " EUR")
print("Log-return mean (daily) : " + str(round(mu*100, 4)) + "%  | Ann: " + str(round(mu*252*100, 2)) + "%")
print("Log-return vol  (daily) : " + str(round(sigma*100, 4)) + "%  | Ann: " + str(round(sigma*np.sqrt(252)*100, 2)) + "%")
print("Skewness : " + str(round(skew, 4)) + "  |  Excess Kurtosis: " + str(round(kurt, 4)))

# Risk metrics
cumret      = (1 + returns).cumprod()
rolling_max = cumret.cummax()
drawdown    = (cumret - rolling_max) / rolling_max
max_dd      = float(drawdown.min())

downside      = returns[returns < 0].values
sortino_denom = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
sharpe  = (mu * 252 - float(RISK_FREE) * 252) / (sigma * np.sqrt(252))
sortino = (mu * 252 - float(RISK_FREE) * 252) / sortino_denom
calmar  = (mu * 252) / abs(max_dd) if max_dd != 0 else np.nan

var95 = np.percentile(returns.values, 5) * INVESTMENT
var99 = np.percentile(returns.values, 1) * INVESTMENT
cvar95 = float(returns[returns <= np.percentile(returns, 5)].mean()) * INVESTMENT
cvar99 = float(returns[returns <= np.percentile(returns, 1)].mean()) * INVESTMENT

print("\n--- Risk Metrics ---")
print("Sharpe     : " + str(round(sharpe, 3)))
print("Sortino    : " + str(round(sortino, 3)))
print("Calmar     : " + str(round(calmar, 3)))
print("Max DD     : " + str(round(max_dd*100, 2)) + "%")
print("VaR  95%   : " + str(round(var95, 2)) + " EUR  |  CVaR 95%: " + str(round(cvar95, 2)) + " EUR")
print("VaR  99%   : " + str(round(var99, 2)) + " EUR  |  CVaR 99%: " + str(round(cvar99, 2)) + " EUR")

# =============================================================
# CHART 1 - Price + Drawdown
# =============================================================
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3],
    subplot_titles=(NAME + " - Close Price", "Drawdown (%)"),
    vertical_spacing=0.04)
fig1.add_trace(go.Scatter(x=close.index, y=close.values, name="Close",
    line=dict(color="steelblue", width=1.5),
    hovertemplate="%{x}<br>%{y:.2f} EUR<extra></extra>"), row=1, col=1)
fig1.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values*100,
    fill="tozeroy", fillcolor="rgba(255,50,50,0.25)",
    line=dict(color="red", width=1), name="Drawdown %",
    hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"), row=2, col=1)
fig1.update_layout(template="plotly_dark", height=600,
    hovermode="x unified", title=NAME + " - Price & Drawdown")
fig1.write_html("output/1_price_drawdown.html")
fig1.show()
print("Step 1 done.")

# =============================================================
# CHART 2 - Return distribution + fat tails analysis
# =============================================================
r_vals = returns.values.flatten()
x_range = np.linspace(r_vals.min(), r_vals.max(), 400)

# Fit Student-t (captures fat tails better than Normal)
t_params = stats.t.fit(r_vals)
t_df, t_loc, t_scale = t_params

# Fit Normal for comparison
norm_pdf  = stats.norm.pdf(x_range, mu, sigma)
stud_pdf  = stats.t.pdf(x_range, t_df, t_loc, t_scale)

_, sw_p = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42).values.flatten())
jb_stat, jb_p = stats.jarque_bera(r_vals)

print("\n--- Distribution Tests ---")
print("Student-t degrees of freedom: " + str(round(t_df, 2)) + "  (lower = fatter tails)")
print("Shapiro-Wilk p   : " + str(round(sw_p, 6)))
print("Jarque-Bera p    : " + str(round(jb_p, 6)))

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=r_vals, nbinsx=100,
    histnorm="probability density", name="Empirical",
    opacity=0.6, marker_color="steelblue"))
fig2.add_trace(go.Scatter(x=x_range, y=norm_pdf,
    name="Normal fit", line=dict(color="red", width=2)))
fig2.add_trace(go.Scatter(x=x_range, y=stud_pdf,
    name="Student-t fit (df=" + str(round(t_df,1)) + ")",
    line=dict(color="limegreen", width=2)))
fig2.add_vline(x=np.percentile(r_vals, 5),  line_color="yellow", line_dash="dash",
    annotation_text="VaR 95%", annotation_position="top left")
fig2.add_vline(x=np.percentile(r_vals, 1),  line_color="red",    line_dash="dash",
    annotation_text="VaR 99%", annotation_position="top left")
fig2.update_layout(title=NAME + " - Return Distribution: Normal vs Student-t (fat tails)",
    xaxis_title="Daily Return", yaxis_title="Density",
    hovermode="x", template="plotly_dark")
fig2.write_html("output/2_return_distribution_fattails.html")
fig2.show()
print("Step 2 done.")

# =============================================================
# CHART 3 - Rolling volatility + Volatility clustering
# =============================================================
roll_vol_20  = log_ret.rolling(20).std()  * np.sqrt(252) * 100
roll_vol_60  = log_ret.rolling(60).std()  * np.sqrt(252) * 100
roll_vol_252 = log_ret.rolling(252).std() * np.sqrt(252) * 100

fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.5, 0.5],
    subplot_titles=("Rolling Realized Volatility (Annualized %)",
                    "Squared Returns (Volatility Clustering proxy)"),
    vertical_spacing=0.05)
fig3.add_trace(go.Scatter(x=roll_vol_20.index, y=roll_vol_20.values,
    name="Vol 20d", line=dict(color="orange", width=1.2)), row=1, col=1)
fig3.add_trace(go.Scatter(x=roll_vol_60.index, y=roll_vol_60.values,
    name="Vol 60d", line=dict(color="cyan", width=1.5)), row=1, col=1)
fig3.add_trace(go.Scatter(x=roll_vol_252.index, y=roll_vol_252.values,
    name="Vol 252d", line=dict(color="magenta", width=1.5, dash="dot")), row=1, col=1)
fig3.add_trace(go.Scatter(x=log_ret.index, y=(log_ret.values**2)*10000,
    name="r^2 x 10000", line=dict(color="steelblue", width=0.6),
    fill="tozeroy", fillcolor="rgba(70,130,180,0.15)"), row=2, col=1)
fig3.update_layout(template="plotly_dark", height=600,
    hovermode="x unified", title=NAME + " - Volatility Dynamics")
fig3.write_html("output/3_volatility_dynamics.html")
fig3.show()
print("Step 3 done.")

# =============================================================
# CHART 4 - VaR / CVaR dashboard
# =============================================================
lev_r = LEVERAGE * r_vals - (LEVERAGE - 1) * float(RISK_FREE)
var95_lev  = np.percentile(lev_r, 5)  * INVESTMENT
var99_lev  = np.percentile(lev_r, 1)  * INVESTMENT
cvar95_lev = float(lev_r[lev_r <= np.percentile(lev_r, 5)].mean()) * INVESTMENT
cvar99_lev = float(lev_r[lev_r <= np.percentile(lev_r, 1)].mean()) * INVESTMENT

fig4 = go.Figure()
fig4.add_trace(go.Histogram(x=lev_r, nbinsx=100,
    histnorm="probability density",
    name="Leveraged Returns ("+str(LEVERAGE)+"x)",
    opacity=0.7, marker_color="darkorange"))
for xval, col, label in [
        (np.percentile(lev_r, 5),  "yellow", "VaR 95% "+str(round(var95_lev,2))+" EUR"),
        (np.percentile(lev_r, 1),  "red",    "VaR 99% "+str(round(var99_lev,2))+" EUR"),
        (float(lev_r[lev_r <= np.percentile(lev_r, 5)].mean()), "orange", "CVaR 95% "+str(round(cvar95_lev,2))+" EUR"),
        (float(lev_r[lev_r <= np.percentile(lev_r, 1)].mean()), "tomato", "CVaR 99% "+str(round(cvar99_lev,2))+" EUR")]:
    fig4.add_vline(x=xval, line_color=col, line_dash="dash", line_width=2,
        annotation_text=label, annotation_position="top left")
fig4.update_layout(title=NAME + " - VaR & CVaR (" + str(LEVERAGE) + "x, " + str(INVESTMENT) + " EUR capital)",
    xaxis_title="Daily Return", yaxis_title="Density", template="plotly_dark")
fig4.write_html("output/4_var_cvar.html")
fig4.show()
print("Step 4 done.")
print("VaR  95% (" + str(LEVERAGE) + "x): " + str(round(var95_lev,2)) + " EUR  | CVaR 95%: " + str(round(cvar95_lev,2)) + " EUR")
print("VaR  99% (" + str(LEVERAGE) + "x): " + str(round(var99_lev,2)) + " EUR  | CVaR 99%: " + str(round(cvar99_lev,2)) + " EUR")

# =============================================================
# STEP 5 - FOUR SIMULATION MODELS
# =============================================================
np.random.seed(42)
days = list(range(HORIZON))

# --- Helper: run portfolio with leverage + stop-loss ---
def apply_leverage_stoploss(price_paths, S0, investment, leverage, rf, stop_loss):
    n_days, n_sims = price_paths.shape
    port = np.zeros((n_days, n_sims))
    port_1x = np.zeros((n_days, n_sims))
    sl_hits = 0
    drift_daily = float(log_ret.mean())
    vol_daily   = float(log_ret.std())
    for i in range(n_sims):
        val = investment
        stopped = False
        port_1x[:, i] = investment * (price_paths[:, i] / S0)
        for t in range(n_days):
            if stopped:
                port[t, i] = port[t-1, i]
                continue
            if t == 0:
                r_t = price_paths[t, i] / S0 - 1
            else:
                r_t = price_paths[t, i] / price_paths[t-1, i] - 1
            lev_ret = leverage * r_t - (leverage - 1) * float(rf)
            val = val * (1 + lev_ret)
            port[t, i] = val
            if (val - investment) / investment <= stop_loss:
                stopped = True
                sl_hits += 1
    return port, port_1x, sl_hits

# ---- MODEL 1: GBM (standard) ----
print("\n--- Running Model 1: GBM ---")
drift_gbm = mu - 0.5 * sigma**2
paths_gbm = np.zeros((HORIZON, SIMULATIONS))
for i in range(SIMULATIONS):
    Z = np.random.normal(0, 1, HORIZON)
    paths_gbm[:, i] = S0 * np.exp(np.cumsum(drift_gbm + sigma * Z))

# ---- MODEL 2: GBM with Student-t shocks (fat tails) ----
print("--- Running Model 2: GBM + Student-t shocks ---")
paths_stud = np.zeros((HORIZON, SIMULATIONS))
for i in range(SIMULATIONS):
    Z = stats.t.rvs(df=t_df, size=HORIZON)
    Z_std = (Z - Z.mean()) / Z.std()  # standardize
    paths_stud[:, i] = S0 * np.exp(np.cumsum(drift_gbm + sigma * Z_std))

# ---- MODEL 3: Jump Diffusion (Merton) ----
print("--- Running Model 3: Merton Jump Diffusion ---")
# Estimate jump parameters from large moves (|r| > 3 sigma)
jump_threshold = 3 * sigma
jumps_detected = log_ret[np.abs(log_ret) > jump_threshold]
lambda_j = len(jumps_detected) / len(log_ret)  # jump intensity (daily)
mu_j     = float(jumps_detected.mean()) if len(jumps_detected) > 0 else 0
sigma_j  = float(jumps_detected.std())  if len(jumps_detected) > 1 else sigma
print("Jump intensity lambda : " + str(round(lambda_j*252, 2)) + " jumps/year")
print("Jump mean (mu_j)      : " + str(round(mu_j*100, 3)) + "%")
print("Jump vol  (sigma_j)   : " + str(round(sigma_j*100, 3)) + "%")

drift_merton = mu - 0.5*sigma**2 - lambda_j*(np.exp(mu_j + 0.5*sigma_j**2) - 1)
paths_jump = np.zeros((HORIZON, SIMULATIONS))
for i in range(SIMULATIONS):
    Z     = np.random.normal(0, 1, HORIZON)
    N_t   = np.random.poisson(lambda_j, HORIZON)
    J_t   = np.where(N_t > 0, np.random.normal(mu_j, sigma_j, HORIZON) * N_t, 0)
    paths_jump[:, i] = S0 * np.exp(np.cumsum(drift_merton + sigma * Z + J_t))

# ---- MODEL 4: Historical Bootstrap ----
print("--- Running Model 4: Historical Bootstrap ---")
log_ret_arr = log_ret.values
paths_boot  = np.zeros((HORIZON, SIMULATIONS))
for i in range(SIMULATIONS):
    sampled = np.random.choice(log_ret_arr, size=HORIZON, replace=True)
    paths_boot[:, i] = S0 * np.exp(np.cumsum(sampled))

print("All 4 models computed.")

# =============================================================
# CHART 5 - Compare all 4 models (median paths)
# =============================================================
models = {
    "GBM": paths_gbm,
    "GBM+Student-t": paths_stud,
    "Jump Diffusion": paths_jump,
    "Bootstrap": paths_boot
}
colors_models = ["cyan", "orange", "red", "limegreen"]

fig5 = go.Figure()
for (mname, paths), col in zip(models.items(), colors_models):
    p5  = np.percentile(paths, 5,  axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    fig5.add_trace(go.Scatter(x=days, y=p50, mode="lines", name=mname + " median",
        line=dict(color=col, width=2.5),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>" + mname + "</extra>"))
    fig5.add_trace(go.Scatter(x=days, y=p5, mode="lines", name=mname + " 5th",
        line=dict(color=col, width=0.8, dash="dot"), showlegend=False))
    fig5.add_trace(go.Scatter(x=days, y=p95, mode="lines", name=mname + " 95th",
        line=dict(color=col, width=0.8, dash="dot"), showlegend=False))
fig5.add_hline(y=S0, line_dash="dash", line_color="white",
    annotation_text="Start " + str(round(S0,2)) + " EUR")
fig5.update_layout(
    title=NAME + " - Model Comparison: GBM vs Student-t vs Jump Diffusion vs Bootstrap",
    xaxis_title="Trading Days", yaxis_title="Simulated Price (EUR)",
    hovermode="x unified", template="plotly_dark", height=550)
fig5.write_html("output/5_model_comparison.html")
fig5.show()
print("Step 5 done.")

# =============================================================
# CHART 6 - Final price distribution: all 4 models
# =============================================================
fig6 = go.Figure()
for (mname, paths), col in zip(models.items(), colors_models):
    fig6.add_trace(go.Histogram(x=paths[-1, :], nbinsx=80,
        name=mname, opacity=0.55,
        marker_color=col))
fig6.add_vline(x=S0, line_dash="dash", line_color="white",
    annotation_text="Start " + str(round(S0,2)) + " EUR")
fig6.update_layout(
    title=NAME + " - Final Price Distribution after " + str(HORIZON) + " days (all models)",
    xaxis_title="Final Price (EUR)", yaxis_title="Frequency",
    barmode="overlay", hovermode="x", template="plotly_dark")
fig6.write_html("output/6_final_price_all_models.html")
fig6.show()
print("Step 6 done.")

# =============================================================
# CHART 7 - Portfolio EUR with leverage + stop-loss (Jump Diffusion = main model)
# =============================================================
port_lev, port_1x, sl_hits = apply_leverage_stoploss(
    paths_jump, S0, INVESTMENT, LEVERAGE, RISK_FREE, STOP_LOSS)

final_lev = port_lev[-1, :]
final_1x  = port_1x[-1, :]
pp5,  pp50,  pp95  = np.percentile(final_lev, [5, 50, 95])
pp5x, pp50x, pp95x = np.percentile(final_1x,  [5, 50, 95])
prob_profit_lev = float(np.mean(final_lev > INVESTMENT) * 100)
prob_profit_1x  = float(np.mean(final_1x  > INVESTMENT) * 100)

fig7 = go.Figure()
for pct, col in [(5, "red"), (50, "white"), (95, "limegreen")]:
    fig7.add_trace(go.Scatter(x=days, y=np.percentile(port_1x, pct, axis=1),
        name=str(pct)+"th pct (1x)",
        line=dict(color=col, width=1.5, dash="dot"),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>"+str(pct)+"th 1x</extra>"))
    fig7.add_trace(go.Scatter(x=days, y=np.percentile(port_lev, pct, axis=1),
        name=str(pct)+"th pct ("+str(LEVERAGE)+"x+SL)",
        line=dict(color=col, width=2.5),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>"+str(pct)+"th "+str(LEVERAGE)+"x</extra>"))
fig7.add_hline(y=INVESTMENT, line_dash="dash", line_color="orange",
    annotation_text="Capital " + str(INVESTMENT) + " EUR")
fig7.add_hline(y=INVESTMENT * (1 + STOP_LOSS), line_dash="dash", line_color="red",
    annotation_text="Stop-Loss " + str(round(INVESTMENT*(1+STOP_LOSS),0)) + " EUR")
fig7.update_layout(
    title=NAME + " - Portfolio " + str(INVESTMENT) + " EUR: 1x vs " + str(LEVERAGE) + "x+SL (Jump Diffusion model)",
    xaxis_title="Trading Days", yaxis_title="Portfolio Value (EUR)",
    hovermode="x unified", template="plotly_dark")
fig7.write_html("output/7_portfolio_leverage_jumps.html")
fig7.show()
print("Step 7 done.")

# =============================================================
# CHART 8 - Model summary table (median / 5th / 95th)
# =============================================================
fig8 = go.Figure()
model_names = list(models.keys())
medians = [round(float(np.percentile(models[m][-1,:], 50)), 2) for m in model_names]
worsts  = [round(float(np.percentile(models[m][-1,:], 5)),  2) for m in model_names]
bests   = [round(float(np.percentile(models[m][-1,:], 95)), 2) for m in model_names]

fig8.add_trace(go.Bar(name="Worst (5th pct)",  x=model_names, y=worsts,  marker_color="tomato"))
fig8.add_trace(go.Bar(name="Median (50th pct)", x=model_names, y=medians, marker_color="steelblue"))
fig8.add_trace(go.Bar(name="Best (95th pct)",   x=model_names, y=bests,   marker_color="limegreen"))
fig8.add_hline(y=S0, line_dash="dash", line_color="white",
    annotation_text="Current: " + str(round(S0,2)) + " EUR")
fig8.update_layout(
    title=NAME + " - Final Price in " + str(HORIZON) + " days by Model",
    xaxis_title="Model", yaxis_title="Price (EUR)",
    barmode="group", template="plotly_dark")
fig8.write_html("output/8_model_summary_bars.html")
fig8.show()
print("Step 8 done.")

# =============================================================
# FINAL REPORT
# =============================================================
sl_l  = "Worst  (5th)"
med_l = "Median (50th)"
be_l  = "Best   (95th)"
sc_l  = "Scenario"

print("\n" + "="*60)
print(" FINAL REPORT - " + NAME)
print("="*60)
print("Capital  : " + str(INVESTMENT) + " EUR  |  Leverage: " + str(LEVERAGE) + "x  |  Exposure: " + str(EXPOSURE) + " EUR")
print("Last px  : " + str(round(S0,2)) + " EUR")
print("-"*60)
print("Sharpe: " + str(round(sharpe,3)) + "  | Sortino: " + str(round(sortino,3)) + "  | Calmar: " + str(round(calmar,3)))
print("Max DD: " + str(round(max_dd*100,2)) + "%")
print("-"*60)
print("Model            | Worst(5th) | Median | Best(95th)")
print("-"*60)
for m in model_names:
    w = round(float(np.percentile(models[m][-1,:], 5)),  2)
    md= round(float(np.percentile(models[m][-1,:], 50)), 2)
    b = round(float(np.percentile(models[m][-1,:], 95)), 2)
    print((m + " "*20)[:16] + " | " + str(w).rjust(10) + " | " + str(md).rjust(6) + " | " + str(b).rjust(10))
print("-"*60)
print("Portfolio " + str(LEVERAGE) + "x+SL (Jump Diffusion model):")
print("  Worst  : " + str(round(pp5,2)) + " EUR ("+str(round((pp5/INVESTMENT-1)*100,1))+"%)")
print("  Median : " + str(round(pp50,2)) + " EUR ("+str(round((pp50/INVESTMENT-1)*100,1))+"%)")
print("  Best   : " + str(round(pp95,2)) + " EUR ("+str(round((pp95/INVESTMENT-1)*100,1))+"%)")
print("  Stop-loss hit in " + str(round(sl_hits/SIMULATIONS*100,1)) + "% of sims")
print("  Prob. profit (1x) : " + str(round(prob_profit_1x,1)) + "%")
print("  Prob. profit (" + str(LEVERAGE) + "x): " + str(round(prob_profit_lev,1)) + "%")
print("="*60)
print("Charts saved in /output  (8 files)")
