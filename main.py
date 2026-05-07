import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
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
ENTRY_DATE  = date.today()

print("=" * 60)
print(" QUANT SIMULATION ENGINE - " + NAME)
print("=" * 60)
print("Entry date : " + str(ENTRY_DATE) + "  (investing TODAY)")
print("Capital    : " + str(INVESTMENT) + " EUR  |  Leverage: " + str(LEVERAGE) + "x")
print("Simulations: " + str(SIMULATIONS) + "  |  Horizon: " + str(HORIZON) + " trading days")
print("Models     : GBM | GBM+Student-t | Jump Diffusion | Bootstrap")

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

# Shares bought today
shares_1x  = INVESTMENT / S0
shares_lev = EXPOSURE   / S0

print("\nEntry price : " + str(round(S0, 2)) + " EUR/share")
print("Shares (1x) : " + str(round(shares_1x, 4)) + "  |  Shares (" + str(LEVERAGE) + "x exposure): " + str(round(shares_lev, 4)))
print("Log-return mean : " + str(round(mu*252*100, 2)) + "% / year")
print("Volatility      : " + str(round(sigma*np.sqrt(252)*100, 2)) + "% / year")
print("Skewness: " + str(round(skew, 4)) + "  |  Kurtosis: " + str(round(kurt, 4)))

# ---- Future trading dates: business days (Mon-Fri) ----
future_dates = list(pd.bdate_range(start=str(ENTRY_DATE), periods=HORIZON))
end_date_sim = future_dates[-1].strftime("%Y-%m-%d")
print("Projection : " + str(ENTRY_DATE) + " -> " + end_date_sim)

# Risk metrics (historical)
cumret      = (1 + returns).cumprod()
rolling_max = cumret.cummax()
drawdown    = (cumret - rolling_max) / rolling_max
max_dd      = float(drawdown.min())
downside      = returns[returns < 0].values
sortino_denom = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
sharpe  = (mu * 252 - float(RISK_FREE) * 252) / (sigma * np.sqrt(252))
sortino = (mu * 252 - float(RISK_FREE) * 252) / sortino_denom
calmar  = (mu * 252) / abs(max_dd) if max_dd != 0 else np.nan
var95   = np.percentile(returns.values, 5) * INVESTMENT
var99   = np.percentile(returns.values, 1) * INVESTMENT
cvar95  = float(returns[returns <= np.percentile(returns, 5)].mean()) * INVESTMENT
cvar99  = float(returns[returns <= np.percentile(returns, 1)].mean()) * INVESTMENT

print("\n--- Historical Risk ---")
print("Sharpe: " + str(round(sharpe,3)) + "  | Sortino: " + str(round(sortino,3)) + "  | Calmar: " + str(round(calmar,3)))
print("Max DD: " + str(round(max_dd*100,2)) + "%")
print("VaR 95%: " + str(round(var95,2)) + " EUR  | CVaR 95%: " + str(round(cvar95,2)) + " EUR")
print("VaR 99%: " + str(round(var99,2)) + " EUR  | CVaR 99%: " + str(round(cvar99,2)) + " EUR")

# =============================================================
# CHART 1 - Historical price + drawdown
# =============================================================
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3],
    subplot_titles=(NAME + " - Historical Close Price", "Drawdown (%)"),
    vertical_spacing=0.04)
fig1.add_trace(go.Scatter(x=close.index, y=close.values, name="Close",
    line=dict(color="steelblue", width=1.5),
    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} EUR<extra></extra>"), row=1, col=1)
fig1.add_trace(go.Scatter(x=[close.index[-1]], y=[S0],
    mode="markers", name="Entry " + str(round(S0,2)) + " EUR",
    marker=dict(color="yellow", size=10, symbol="star")), row=1, col=1)
fig1.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values*100,
    fill="tozeroy", fillcolor="rgba(255,50,50,0.25)",
    line=dict(color="red", width=1), name="Drawdown %",
    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>"), row=2, col=1)
fig1.update_layout(template="plotly_dark", height=600, hovermode="x unified",
    title=NAME + " - Historical Price + Drawdown (entry: " + str(round(S0,2)) + " EUR on " + str(ENTRY_DATE) + ")")
fig1.write_html("output/1_price_drawdown.html")
fig1.show()
print("Step 1 done.")

# =============================================================
# CHART 2 - Return distribution (fat tails)
# =============================================================
r_vals   = returns.values.flatten()
x_range  = np.linspace(r_vals.min(), r_vals.max(), 400)
t_params = stats.t.fit(r_vals)
t_df, t_loc, t_scale = t_params
_, sw_p   = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42).values.flatten())
jb_stat, jb_p = stats.jarque_bera(r_vals)
print("\nStudent-t df: " + str(round(t_df,2)) + "  | JB p-value: " + str(round(jb_p,6)))

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=r_vals, nbinsx=100,
    histnorm="probability density", name="Empirical", opacity=0.6, marker_color="steelblue"))
fig2.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
    name="Normal fit", line=dict(color="red", width=2)))
fig2.add_trace(go.Scatter(x=x_range, y=stats.t.pdf(x_range, t_df, t_loc, t_scale),
    name="Student-t (df=" + str(round(t_df,1)) + ")", line=dict(color="limegreen", width=2)))
fig2.add_vline(x=np.percentile(r_vals, 5), line_color="yellow", line_dash="dash",
    annotation_text="VaR 95%", annotation_position="top left")
fig2.add_vline(x=np.percentile(r_vals, 1), line_color="red", line_dash="dash",
    annotation_text="VaR 99%", annotation_position="top left")
fig2.update_layout(title=NAME + " - Return Distribution: Fat Tails Analysis",
    xaxis_title="Daily Return", yaxis_title="Density", hovermode="x", template="plotly_dark")
fig2.write_html("output/2_return_distribution.html")
fig2.show()
print("Step 2 done.")

# =============================================================
# CHART 3 - Volatility dynamics
# =============================================================
roll_vol_20  = log_ret.rolling(20).std()  * np.sqrt(252) * 100
roll_vol_60  = log_ret.rolling(60).std()  * np.sqrt(252) * 100
roll_vol_252 = log_ret.rolling(252).std() * np.sqrt(252) * 100
fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.5, 0.5],
    subplot_titles=("Rolling Realized Volatility (Annualized %)", "Squared Returns (Volatility Clustering)"),
    vertical_spacing=0.05)
fig3.add_trace(go.Scatter(x=roll_vol_20.index,  y=roll_vol_20.values,  name="Vol 20d",  line=dict(color="orange",  width=1.2)), row=1, col=1)
fig3.add_trace(go.Scatter(x=roll_vol_60.index,  y=roll_vol_60.values,  name="Vol 60d",  line=dict(color="cyan",    width=1.5)), row=1, col=1)
fig3.add_trace(go.Scatter(x=roll_vol_252.index, y=roll_vol_252.values, name="Vol 252d", line=dict(color="magenta", width=1.5, dash="dot")), row=1, col=1)
fig3.add_trace(go.Scatter(x=log_ret.index, y=(log_ret.values**2)*10000,
    name="r^2 x10000", line=dict(color="steelblue", width=0.6),
    fill="tozeroy", fillcolor="rgba(70,130,180,0.15)"), row=2, col=1)
fig3.update_layout(template="plotly_dark", height=600, hovermode="x unified",
    title=NAME + " - Volatility Dynamics")
fig3.write_html("output/3_volatility_dynamics.html")
fig3.show()
print("Step 3 done.")

# =============================================================
# CHART 4 - VaR / CVaR
# =============================================================
lev_r      = LEVERAGE * r_vals - (LEVERAGE - 1) * float(RISK_FREE)
var95_lev  = np.percentile(lev_r, 5)  * INVESTMENT
var99_lev  = np.percentile(lev_r, 1)  * INVESTMENT
cvar95_lev = float(lev_r[lev_r <= np.percentile(lev_r, 5)].mean()) * INVESTMENT
cvar99_lev = float(lev_r[lev_r <= np.percentile(lev_r, 1)].mean()) * INVESTMENT
fig4 = go.Figure()
fig4.add_trace(go.Histogram(x=lev_r, nbinsx=100, histnorm="probability density",
    name="Returns ("+str(LEVERAGE)+"x)", opacity=0.7, marker_color="darkorange"))
for xval, col, label in [
        (np.percentile(lev_r, 5),  "yellow", "VaR 95% "+str(round(var95_lev,2))+" EUR"),
        (np.percentile(lev_r, 1),  "red",    "VaR 99% "+str(round(var99_lev,2))+" EUR"),
        (float(lev_r[lev_r <= np.percentile(lev_r, 5)].mean()), "orange", "CVaR 95% "+str(round(cvar95_lev,2))+" EUR"),
        (float(lev_r[lev_r <= np.percentile(lev_r, 1)].mean()), "tomato", "CVaR 99% "+str(round(cvar99_lev,2))+" EUR")]:
    fig4.add_vline(x=xval, line_color=col, line_dash="dash", line_width=2,
        annotation_text=label, annotation_position="top left")
fig4.update_layout(title=NAME + " - VaR & CVaR ("+str(LEVERAGE)+"x, "+str(INVESTMENT)+" EUR)",
    xaxis_title="Daily Return", yaxis_title="Density", template="plotly_dark")
fig4.write_html("output/4_var_cvar.html")
fig4.show()
print("Step 4 done.  VaR 95%: " + str(round(var95_lev,2)) + " EUR | CVaR 95%: " + str(round(cvar95_lev,2)) + " EUR")

# =============================================================
# SIMULATIONS - 4 MODELS
# =============================================================
np.random.seed(42)

jump_threshold = 3 * sigma
jumps_detected = log_ret[np.abs(log_ret) > jump_threshold]
lambda_j = len(jumps_detected) / len(log_ret)
mu_j     = float(jumps_detected.mean()) if len(jumps_detected) > 0 else 0
sigma_j  = float(jumps_detected.std())  if len(jumps_detected) > 1 else sigma
drift_gbm    = mu - 0.5 * sigma**2
drift_merton = mu - 0.5*sigma**2 - lambda_j*(np.exp(mu_j + 0.5*sigma_j**2) - 1)
log_ret_arr  = log_ret.values

print("\nJump Diffusion: lambda=" + str(round(lambda_j*252,2)) + " jumps/yr | mu_j=" + str(round(mu_j*100,3)) + "% | sigma_j=" + str(round(sigma_j*100,3)) + "%")

paths_gbm   = np.zeros((HORIZON, SIMULATIONS))
paths_stud  = np.zeros((HORIZON, SIMULATIONS))
paths_jump  = np.zeros((HORIZON, SIMULATIONS))
paths_boot  = np.zeros((HORIZON, SIMULATIONS))

for i in range(SIMULATIONS):
    Z = np.random.normal(0, 1, HORIZON)
    paths_gbm[:, i]  = S0 * np.exp(np.cumsum(drift_gbm + sigma * Z))

    Z2     = stats.t.rvs(df=t_df, size=HORIZON)
    Z2_std = (Z2 - Z2.mean()) / Z2.std()
    paths_stud[:, i] = S0 * np.exp(np.cumsum(drift_gbm + sigma * Z2_std))

    N_t = np.random.poisson(lambda_j, HORIZON)
    J_t = np.where(N_t > 0, np.random.normal(mu_j, sigma_j, HORIZON) * N_t, 0)
    paths_jump[:, i] = S0 * np.exp(np.cumsum(drift_merton + sigma * Z + J_t))

    sampled = np.random.choice(log_ret_arr, size=HORIZON, replace=True)
    paths_boot[:, i] = S0 * np.exp(np.cumsum(sampled))

models = {
    "GBM": paths_gbm,
    "GBM+Student-t": paths_stud,
    "Jump Diffusion": paths_jump,
    "Bootstrap": paths_boot
}
colors_models = ["cyan", "orange", "red", "limegreen"]
print("All 4 models computed.")

# =============================================================
# CHART 5 - Portfolio projection avec vraies dates
# =============================================================
port_lev = np.zeros((HORIZON, SIMULATIONS))
port_1x  = np.zeros((HORIZON, SIMULATIONS))
sl_hits  = 0

for i in range(SIMULATIONS):
    val     = INVESTMENT
    stopped = False
    port_1x[:, i] = INVESTMENT * (paths_jump[:, i] / S0)
    for t in range(HORIZON):
        if stopped:
            port_lev[t, i] = port_lev[t-1, i]
            continue
        r_t     = paths_jump[t, i] / (paths_jump[t-1, i] if t > 0 else S0) - 1
        lev_ret = LEVERAGE * r_t - (LEVERAGE - 1) * float(RISK_FREE)
        val     = val * (1 + lev_ret)
        port_lev[t, i] = val
        if (val - INVESTMENT) / INVESTMENT <= STOP_LOSS:
            stopped  = True
            sl_hits += 1

final_lev  = port_lev[-1, :]
final_1x   = port_1x[-1, :]
pp5,  pp50,  pp95  = np.percentile(final_lev, [5, 50, 95])
pp5x, pp50x, pp95x = np.percentile(final_1x,  [5, 50, 95])
prob_profit_lev = float(np.mean(final_lev > INVESTMENT) * 100)
prob_profit_1x  = float(np.mean(final_1x  > INVESTMENT) * 100)

fig5 = go.Figure()
for pct, col, dash in [(5, "red", "dot"), (50, "white", "solid"), (95, "limegreen", "dot")]:
    fig5.add_trace(go.Scatter(
        x=future_dates,
        y=np.percentile(port_1x, pct, axis=1),
        name=str(pct)+"th pct (1x - sans levier)",
        line=dict(color=col, width=1.5, dash=dash),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.2f} EUR</b><extra>"+str(pct)+"th 1x</extra>"))
    fig5.add_trace(go.Scatter(
        x=future_dates,
        y=np.percentile(port_lev, pct, axis=1),
        name=str(pct)+"th pct ("+str(LEVERAGE)+"x + stop-loss)",
        line=dict(color=col, width=2.5),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.2f} EUR</b><extra>"+str(pct)+"th "+str(LEVERAGE)+"x</extra>"))
fig5.add_hline(y=INVESTMENT, line_dash="dash", line_color="orange",
    annotation_text="Capital investi: " + str(INVESTMENT) + " EUR")
fig5.add_hline(y=INVESTMENT * (1 + STOP_LOSS), line_dash="dash", line_color="red",
    annotation_text="Stop-loss: " + str(round(INVESTMENT*(1+STOP_LOSS),0)) + " EUR")
fig5.add_annotation(
    x=future_dates[0], y=INVESTMENT,
    text="ENTREE " + str(ENTRY_DATE) + " @ " + str(round(S0,2)) + " EUR/action",
    showarrow=True, arrowhead=2, arrowcolor="yellow",
    font=dict(color="yellow", size=12),
    bgcolor="rgba(0,0,0,0.6)")
fig5.update_layout(
    title="<b>" + NAME + " - Projection de ton investissement " + str(INVESTMENT) + " EUR</b><br>"
          "Entree: " + str(ENTRY_DATE) + " @ " + str(round(S0,2)) + " EUR | "
          "Modele: Jump Diffusion | " + str(SIMULATIONS) + " simulations",
    xaxis_title="Date",
    yaxis_title="Valeur du portefeuille (EUR)",
    hovermode="x unified",
    template="plotly_dark",
    height=600
)
fig5.write_html("output/5_projection_portefeuille.html")
fig5.show()
print("Step 5 done.")

# =============================================================
# CHART 6 - Prix projete 4 modeles avec vraies dates
# =============================================================
fig6 = go.Figure()
for (mname, paths), col in zip(models.items(), colors_models):
    p5  = np.percentile(paths, 5,  axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p95 = np.percentile(paths, 95, axis=1)
    fig6.add_trace(go.Scatter(x=future_dates, y=p50, name=mname + " median",
        line=dict(color=col, width=2.5),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} EUR<extra>" + mname + " median</extra>"))
    fig6.add_trace(go.Scatter(x=future_dates, y=p5, name=mname + " 5th",
        line=dict(color=col, width=0.7, dash="dot"), showlegend=False))
    fig6.add_trace(go.Scatter(x=future_dates, y=p95, name=mname + " 95th",
        line=dict(color=col, width=0.7, dash="dot"), showlegend=False))
fig6.add_hline(y=S0, line_dash="dash", line_color="white",
    annotation_text="Prix entree: " + str(round(S0,2)) + " EUR")
fig6.update_layout(
    title="<b>" + NAME + " - Prix projete: 4 modeles</b><br>Entree: " + str(ENTRY_DATE) + " @ " + str(round(S0,2)) + " EUR",
    xaxis_title="Date", yaxis_title="Prix (EUR)",
    hovermode="x unified", template="plotly_dark", height=550)
fig6.write_html("output/6_prix_projete_4_modeles.html")
fig6.show()
print("Step 6 done.")

# =============================================================
# CHART 7 - Distribution finale
# =============================================================
fig7 = go.Figure()
for (mname, paths), col in zip(models.items(), colors_models):
    fig7.add_trace(go.Histogram(x=paths[-1,:], nbinsx=80, name=mname,
        opacity=0.55, marker_color=col))
fig7.add_vline(x=S0, line_dash="dash", line_color="white",
    annotation_text="Entree " + str(round(S0,2)) + " EUR")
fig7.update_layout(
    title=NAME + " - Distribution finale du prix apres " + str(HORIZON) + " jours ("+end_date_sim+")",
    xaxis_title="Prix final (EUR)", yaxis_title="Frequence",
    barmode="overlay", hovermode="x", template="plotly_dark")
fig7.write_html("output/7_distribution_finale.html")
fig7.show()
print("Step 7 done.")

# =============================================================
# CHART 8 - Barchart comparatif
# =============================================================
model_names = list(models.keys())
worsts  = [round(float(np.percentile(models[m][-1,:], 5)),  2) for m in model_names]
medians = [round(float(np.percentile(models[m][-1,:], 50)), 2) for m in model_names]
bests   = [round(float(np.percentile(models[m][-1,:], 95)), 2) for m in model_names]
fig8 = go.Figure()
fig8.add_trace(go.Bar(name="Worst (5th)",  x=model_names, y=worsts,  marker_color="tomato"))
fig8.add_trace(go.Bar(name="Median (50th)", x=model_names, y=medians, marker_color="steelblue"))
fig8.add_trace(go.Bar(name="Best (95th)",   x=model_names, y=bests,   marker_color="limegreen"))
fig8.add_hline(y=S0, line_dash="dash", line_color="white",
    annotation_text="Entree: " + str(round(S0,2)) + " EUR")
fig8.update_layout(
    title=NAME + " - Prix final dans " + str(HORIZON) + " jours par modele (entree " + str(ENTRY_DATE) + ")",
    xaxis_title="Modele", yaxis_title="Prix (EUR)",
    barmode="group", template="plotly_dark")
fig8.write_html("output/8_comparaison_modeles.html")
fig8.show()
print("Step 8 done.")

# =============================================================
# FINAL REPORT
# =============================================================
print("\n" + "="*60)
print(" RAPPORT D'INVESTISSEMENT - " + NAME)
print("="*60)
print("Date d'entree : " + str(ENTRY_DATE))
print("Prix d'entree : " + str(round(S0,2)) + " EUR/action")
print("Capital       : " + str(INVESTMENT) + " EUR  | Levier: " + str(LEVERAGE) + "x | Exposition: " + str(EXPOSURE) + " EUR")
print("Nb actions    : " + str(round(shares_1x,4)) + " (1x)  | " + str(round(shares_lev,4)) + " (" + str(LEVERAGE) + "x exposition)")
print("Horizon       : " + str(HORIZON) + " jours trading -> " + end_date_sim)
print("-"*60)
print("Sharpe: " + str(round(sharpe,3)) + "  | Sortino: " + str(round(sortino,3)) + "  | Calmar: " + str(round(calmar,3)))
print("Max Drawdown historique : " + str(round(max_dd*100,2)) + "%")
print("-"*60)
print("Prix final projete par modele:")
for m in model_names:
    w  = round(float(np.percentile(models[m][-1,:],  5)), 2)
    md = round(float(np.percentile(models[m][-1,:], 50)), 2)
    b  = round(float(np.percentile(models[m][-1,:], 95)), 2)
    pnl_med = round((md - S0) * shares_1x, 2)
    print("  " + (m+" "*20)[:16] + " worst: "+str(w).rjust(6)+" | median: "+str(md).rjust(6)+" | best: "+str(b).rjust(6)+" EUR  (PnL median: "+str(pnl_med)+" EUR)")
print("-"*60)
print("Portefeuille " + str(INVESTMENT) + " EUR - Jump Diffusion (1x vs " + str(LEVERAGE) + "x + SL " + str(int(STOP_LOSS*100)) + "%):")
print("  Worst  1x: " + str(round(pp5x,2)) + " EUR (" + str(round((pp5x/INVESTMENT-1)*100,1)) + "%)  | " + str(LEVERAGE) + "x: " + str(round(pp5,2)) + " EUR (" + str(round((pp5/INVESTMENT-1)*100,1)) + "%)")
print("  Median 1x: " + str(round(pp50x,2)) + " EUR (" + str(round((pp50x/INVESTMENT-1)*100,1)) + "%)  | " + str(LEVERAGE) + "x: " + str(round(pp50,2)) + " EUR (" + str(round((pp50/INVESTMENT-1)*100,1)) + "%)")
print("  Best   1x: " + str(round(pp95x,2)) + " EUR (" + str(round((pp95x/INVESTMENT-1)*100,1)) + "%)  | " + str(LEVERAGE) + "x: " + str(round(pp95,2)) + " EUR (" + str(round((pp95/INVESTMENT-1)*100,1)) + "%)")
print("  Stop-loss declenche : " + str(round(sl_hits/SIMULATIONS*100,1)) + "% des simulations")
print("  Prob. profit (1x)   : " + str(round(prob_profit_1x,1)) + "%")
print("  Prob. profit (" + str(LEVERAGE) + "x)   : " + str(round(prob_profit_lev,1)) + "%")
print("="*60)
print("8 charts HTML sauvegardes dans /output")
