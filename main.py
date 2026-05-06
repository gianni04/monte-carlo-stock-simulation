import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
from datetime import date, timedelta

os.makedirs("output", exist_ok=True)

# =============================================================
# CONFIGURATION
# =============================================================
TICKER     = "UBI.PA"
NAME       = "Ubisoft"
START      = "2019-01-01"
END        = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
SIMULATIONS = 1000
HORIZON    = 252
INVESTMENT = 50
LEVERAGE   = 2
EXPOSURE   = INVESTMENT * LEVERAGE
RISK_FREE  = 0.03 / 252
STOP_LOSS  = -0.20   # -20% stop-loss on portfolio

print("Date range     : " + START + " -> " + END)
print("Capital        : " + str(INVESTMENT) + " EUR")
print("Leverage       : " + str(LEVERAGE) + "x")
print("Exposure       : " + str(EXPOSURE) + " EUR")
print("Stop-loss      : " + str(int(STOP_LOSS*100)) + "%")

# =============================================================
# STEP 1 - Download & price chart
# =============================================================
print("\nDownloading " + NAME + " (" + TICKER + ") data...")
df = yf.download(TICKER, start=START, end=END, auto_adjust=True)
close = df["Close"].squeeze().dropna()
volume = df["Volume"].squeeze() if "Volume" in df.columns else None

# --- Technical indicators ---
# SMA 20 / 50 / 200
sma20  = close.rolling(20).mean()
sma50  = close.rolling(50).mean()
sma200 = close.rolling(200).mean()

# Bollinger Bands (20, 2)
bb_mid  = sma20
bb_std  = close.rolling(20).std()
bb_up   = bb_mid + 2 * bb_std
bb_low  = bb_mid - 2 * bb_std

# RSI 14
delta   = close.diff()
gain    = delta.clip(lower=0).rolling(14).mean()
loss    = (-delta.clip(upper=0)).rolling(14).mean()
rs      = gain / loss
rsi     = 100 - (100 / (1 + rs))

# MACD (12, 26, 9)
ema12   = close.ewm(span=12, adjust=False).mean()
ema26   = close.ewm(span=26, adjust=False).mean()
macd    = ema12 - ema26
signal  = macd.ewm(span=9, adjust=False).mean()
hist_macd = macd - signal

# Rolling 30-day volatility (annualized)
returns_raw = close.pct_change().dropna()
roll_vol    = returns_raw.rolling(30).std() * np.sqrt(252) * 100

fig1 = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.45, 0.2, 0.2, 0.15],
    subplot_titles=(
        NAME + " - Price + Bollinger Bands + SMAs",
        "RSI (14)",
        "MACD (12/26/9)",
        "Rolling 30d Volatility (Ann.%)"
    ),
    vertical_spacing=0.04
)

# Price + BB + SMAs
fig1.add_trace(go.Scatter(x=close.index, y=close.values, name="Close",
    line=dict(color="steelblue", width=1.5)), row=1, col=1)
fig1.add_trace(go.Scatter(x=bb_up.index, y=bb_up.values, name="BB Upper",
    line=dict(color="rgba(255,200,0,0.4)", width=1, dash="dot")), row=1, col=1)
fig1.add_trace(go.Scatter(x=bb_low.index, y=bb_low.values, name="BB Lower",
    fill="tonexty", fillcolor="rgba(255,200,0,0.05)",
    line=dict(color="rgba(255,200,0,0.4)", width=1, dash="dot")), row=1, col=1)
fig1.add_trace(go.Scatter(x=sma20.index, y=sma20.values, name="SMA20",
    line=dict(color="orange", width=1)), row=1, col=1)
fig1.add_trace(go.Scatter(x=sma50.index, y=sma50.values, name="SMA50",
    line=dict(color="yellow", width=1)), row=1, col=1)
fig1.add_trace(go.Scatter(x=sma200.index, y=sma200.values, name="SMA200",
    line=dict(color="magenta", width=1.5)), row=1, col=1)

# RSI
fig1.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name="RSI",
    line=dict(color="cyan", width=1.5)), row=2, col=1)
fig1.add_hline(y=70, line_dash="dash", line_color="red",   line_width=1, row=2, col=1)
fig1.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
fig1.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0, row=2, col=1)

# MACD
colors_macd = ["green" if v >= 0 else "red" for v in hist_macd.values]
fig1.add_trace(go.Bar(x=hist_macd.index, y=hist_macd.values, name="MACD Hist",
    marker_color=colors_macd, opacity=0.7), row=3, col=1)
fig1.add_trace(go.Scatter(x=macd.index, y=macd.values, name="MACD",
    line=dict(color="cyan", width=1.2)), row=3, col=1)
fig1.add_trace(go.Scatter(x=signal.index, y=signal.values, name="Signal",
    line=dict(color="orange", width=1.2)), row=3, col=1)

# Rolling vol
fig1.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values, name="Vol 30d",
    line=dict(color="violet", width=1.2),
    fill="tozeroy", fillcolor="rgba(238,130,238,0.1)"), row=4, col=1)

fig1.update_layout(template="plotly_dark", height=900,
    hovermode="x unified", showlegend=True)
fig1.update_yaxes(title_text="Price (EUR)", row=1, col=1)
fig1.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
fig1.update_yaxes(title_text="MACD", row=3, col=1)
fig1.update_yaxes(title_text="Vol %", row=4, col=1)
fig1.write_html("output/1_technical_analysis.html")
fig1.show()
print("Step 1 done.")

# =============================================================
# STEP 2 - Returns + Rolling Vol
# =============================================================
returns = close.pct_change().dropna()
lev_returns = LEVERAGE * returns.values.flatten() - (LEVERAGE - 1) * RISK_FREE

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=returns.index, y=returns.values.flatten(),
    name="Return (1x)", line=dict(color="steelblue", width=0.8),
    hovertemplate="%{x}<br>Return: %{y:.4f}<extra>1x</extra>"))
fig2.add_trace(go.Scatter(x=returns.index, y=lev_returns,
    name="Return (" + str(LEVERAGE) + "x)",
    line=dict(color="darkorange", width=0.8),
    hovertemplate="%{x}<br>Lev Return: %{y:.4f}<extra>" + str(LEVERAGE) + "x</extra>"))
fig2.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8)
fig2.update_layout(title=NAME + " - Daily Returns 1x vs " + str(LEVERAGE) + "x",
    xaxis_title="Date", yaxis_title="Daily Return",
    hovermode="x unified", template="plotly_dark")
fig2.write_html("output/2_daily_returns.html")
fig2.show()
print("Step 2 done.")

# =============================================================
# STEP 3 - Statistics + Risk Metrics
# =============================================================
mu    = float(returns.mean())
sigma = float(returns.std())
skew  = float(returns.skew())
kurt  = float(returns.kurtosis())
_, pvalue = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42).values.flatten())

mu_lev    = LEVERAGE * mu - (LEVERAGE - 1) * float(RISK_FREE)
sigma_lev = LEVERAGE * sigma

# Sharpe Ratio
sharpe = (mu - float(RISK_FREE)) / sigma * np.sqrt(252)
sharpe_lev = (mu_lev - float(RISK_FREE)) / sigma_lev * np.sqrt(252)

# Sortino Ratio (downside deviation)
downside = returns[returns < 0].values
sortino_denom = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
sortino = (mu * 252 - float(RISK_FREE) * 252) / sortino_denom

# Calmar Ratio
cumret = (1 + returns).cumprod()
rolling_max = cumret.cummax()
drawdown = (cumret - rolling_max) / rolling_max
max_dd = float(drawdown.min())
calmar = (mu * 252) / abs(max_dd) if max_dd != 0 else np.nan

# CVaR (Expected Shortfall)
cvar95 = float(returns[returns <= np.percentile(returns, 5)].mean()) * INVESTMENT
cvar99 = float(returns[returns <= np.percentile(returns, 1)].mean()) * INVESTMENT
cvar95_lev = float(np.array(lev_returns)[lev_returns <= np.percentile(lev_returns, 5)].mean()) * INVESTMENT
cvar99_lev = float(np.array(lev_returns)[lev_returns <= np.percentile(lev_returns, 1)].mean()) * INVESTMENT

var95_1x  = np.percentile(returns.values, 5) * INVESTMENT
var99_1x  = np.percentile(returns.values, 1) * INVESTMENT
var95_lev = np.percentile(lev_returns, 5)    * INVESTMENT
var99_lev = np.percentile(lev_returns, 1)    * INVESTMENT

print("\n--- Return & Risk Statistics ---")
print("Mean daily return (1x) : " + str(round(mu*100, 4)) + "%  | Ann: " + str(round(mu*252*100, 2)) + "%")
print("Mean daily return (" + str(LEVERAGE) + "x) : " + str(round(mu_lev*100, 4)) + "%  | Ann: " + str(round(mu_lev*252*100, 2)) + "%")
print("Volatility (1x)        : " + str(round(sigma*np.sqrt(252)*100, 2)) + "% annualized")
print("Volatility (" + str(LEVERAGE) + "x)        : " + str(round(sigma_lev*np.sqrt(252)*100, 2)) + "% annualized")
print("Skewness               : " + str(round(skew, 4)))
print("Kurtosis               : " + str(round(kurt, 4)))
normality = "NOT normal" if pvalue < 0.05 else "normal"
print("Shapiro-Wilk           : " + str(round(pvalue, 4)) + " (" + normality + ")")
print("Sharpe Ratio (1x)      : " + str(round(sharpe, 4)))
print("Sharpe Ratio (" + str(LEVERAGE) + "x)      : " + str(round(sharpe_lev, 4)))
print("Sortino Ratio          : " + str(round(sortino, 4)))
print("Calmar Ratio           : " + str(round(calmar, 4)))
print("Max Drawdown           : " + str(round(max_dd*100, 2)) + "%")
print("VaR  95% (1x)  : " + str(round(var95_1x, 2)) + " EUR  | CVaR: " + str(round(cvar95, 2)) + " EUR")
print("VaR  99% (1x)  : " + str(round(var99_1x, 2)) + " EUR  | CVaR: " + str(round(cvar99, 2)) + " EUR")
print("VaR  95% (" + str(LEVERAGE) + "x)  : " + str(round(var95_lev, 2)) + " EUR  | CVaR: " + str(round(cvar95_lev, 2)) + " EUR")
print("VaR  99% (" + str(LEVERAGE) + "x)  : " + str(round(var99_lev, 2)) + " EUR  | CVaR: " + str(round(cvar99_lev, 2)) + " EUR")

# Chart: Return distribution
x = np.linspace(float(returns.min()), float(returns.max()), 300)
fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=returns.values.flatten(), nbinsx=80,
    histnorm="probability density", name="Empirical (1x)",
    opacity=0.6, marker_color="steelblue"))
fig3.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sigma),
    name="Normal (1x)", line=dict(color="red", width=2)))
fig3.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu_lev, sigma_lev),
    name="Normal (" + str(LEVERAGE) + "x)",
    line=dict(color="orange", width=2, dash="dash")))
fig3.add_vline(x=np.percentile(returns.values, 5),  line_color="yellow", line_dash="dash",
    annotation_text="VaR95",  annotation_position="top left")
fig3.add_vline(x=np.percentile(returns.values, 1),  line_color="red",    line_dash="dash",
    annotation_text="VaR99",  annotation_position="top left")
fig3.update_layout(title=NAME + " - Return Distribution & VaR",
    xaxis_title="Daily Return", yaxis_title="Density",
    hovermode="x", template="plotly_dark")
fig3.write_html("output/3_return_distribution.html")
fig3.show()
print("Step 3 done.")

# Chart: Drawdown
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values*100,
    fill="tozeroy", fillcolor="rgba(255,50,50,0.3)",
    line=dict(color="red", width=1), name="Drawdown %"))
fig_dd.update_layout(title=NAME + " - Historical Drawdown",
    xaxis_title="Date", yaxis_title="Drawdown (%)",
    template="plotly_dark", hovermode="x unified")
fig_dd.write_html("output/3b_drawdown.html")
fig_dd.show()
print("Step 3b done.")

# =============================================================
# STEP 4 - VaR + CVaR chart
# =============================================================
fig4 = go.Figure()
fig4.add_trace(go.Histogram(x=lev_returns, nbinsx=80,
    histnorm="probability density",
    name="Returns (" + str(LEVERAGE) + "x)", opacity=0.7, marker_color="darkorange"))
fig4.add_vline(x=np.percentile(lev_returns, 5), line_color="yellow", line_dash="dash", line_width=2,
    annotation_text="VaR 95%: " + str(round(var95_lev, 2)) + " EUR",
    annotation_position="top right")
fig4.add_vline(x=np.percentile(lev_returns, 1), line_color="red", line_dash="dash", line_width=2,
    annotation_text="VaR 99%: " + str(round(var99_lev, 2)) + " EUR",
    annotation_position="top left")
fig4.add_vline(x=float(np.array(lev_returns)[lev_returns <= np.percentile(lev_returns, 5)].mean()),
    line_color="orange", line_dash="dot", line_width=2,
    annotation_text="CVaR 95%: " + str(round(cvar95_lev, 2)) + " EUR",
    annotation_position="top left")
fig4.update_layout(title=NAME + " - VaR & CVaR (" + str(LEVERAGE) + "x, " + str(INVESTMENT) + " EUR)",
    xaxis_title="Daily Return", yaxis_title="Density", template="plotly_dark")
fig4.write_html("output/4_var_cvar.html")
fig4.show()
print("Step 4 done.")

# =============================================================
# STEP 5 - Monte Carlo with stop-loss simulation
# =============================================================
S0    = float(close.iloc[-1])
dt    = 1
drift = mu - 0.5 * sigma**2

np.random.seed(42)
price_paths        = np.zeros((HORIZON, SIMULATIONS))
portfolio_paths    = np.zeros((HORIZON, SIMULATIONS))
portfolio_paths_1x = np.zeros((HORIZON, SIMULATIONS))
stop_loss_hits     = 0

for i in range(SIMULATIONS):
    Z = np.random.normal(0, 1, HORIZON)
    price_paths[:, i] = S0 * np.exp(np.cumsum(drift * dt + sigma * np.sqrt(dt) * Z))
    price_ratio = price_paths[:, i] / S0
    portfolio_paths_1x[:, i] = INVESTMENT * price_ratio

    # 2x leveraged portfolio with stop-loss
    port_val = INVESTMENT
    stopped  = False
    for t in range(HORIZON):
        if stopped:
            portfolio_paths[t, i] = portfolio_paths[t-1, i]
            continue
        r_t = np.exp(drift * dt + sigma * np.sqrt(dt) * Z[t]) - 1
        lev_r = LEVERAGE * r_t - (LEVERAGE - 1) * float(RISK_FREE)
        port_val = port_val * (1 + lev_r)
        portfolio_paths[t, i] = port_val
        if (port_val - INVESTMENT) / INVESTMENT <= STOP_LOSS:
            stopped = True
            stop_loss_hits += 1

final_port    = portfolio_paths[-1, :]
final_port_1x = portfolio_paths_1x[-1, :]

pp5,  pp50,  pp95  = np.percentile(final_port,    [5, 50, 95])
pp5x, pp50x, pp95x = np.percentile(final_port_1x, [5, 50, 95])
prob_profit_lev = float(np.mean(final_port > INVESTMENT) * 100)
prob_profit_1x  = float(np.mean(final_port_1x > INVESTMENT) * 100)

print("\n--- Monte Carlo Results (" + str(SIMULATIONS) + " sims, " + str(HORIZON) + " days) ---")
print("Stop-loss triggered in " + str(stop_loss_hits) + "/" + str(SIMULATIONS) + " simulations (" + str(round(stop_loss_hits/SIMULATIONS*100, 1)) + "%)")
print("Prob. of profit (1x)   : " + str(round(prob_profit_1x, 1)) + "%")
print("Prob. of profit (" + str(LEVERAGE) + "x)   : " + str(round(prob_profit_lev, 1)) + "%")
print("--- WITHOUT leverage (1x) ---")
print("Worst  (5th)  : " + str(round(pp5x, 2)) + " EUR (" + str(round((pp5x/INVESTMENT-1)*100, 1)) + "%)")
print("Median (50th) : " + str(round(pp50x, 2)) + " EUR (" + str(round((pp50x/INVESTMENT-1)*100, 1)) + "%)")
print("Best   (95th) : " + str(round(pp95x, 2)) + " EUR (" + str(round((pp95x/INVESTMENT-1)*100, 1)) + "%)")
print("--- WITH " + str(LEVERAGE) + "x leverage + stop-loss at " + str(int(STOP_LOSS*100)) + "% ---")
print("Worst  (5th)  : " + str(round(pp5, 2)) + " EUR (" + str(round((pp5/INVESTMENT-1)*100, 1)) + "%)")
print("Median (50th) : " + str(round(pp50, 2)) + " EUR (" + str(round((pp50/INVESTMENT-1)*100, 1)) + "%)")
print("Best   (95th) : " + str(round(pp95, 2)) + " EUR (" + str(round((pp95/INVESTMENT-1)*100, 1)) + "%)")

# Chart 5a: Price paths
days = list(range(HORIZON))
fig5a = go.Figure()
for i in range(150):
    fig5a.add_trace(go.Scatter(x=days, y=price_paths[:, i], mode="lines",
        line=dict(color="steelblue", width=0.4), opacity=0.12,
        showlegend=False, hoverinfo="skip"))
for pct, color, label in [(5, "red", "5th pct"), (50, "white", "Median"), (95, "green", "95th pct")]:
    fig5a.add_trace(go.Scatter(x=days, y=np.percentile(price_paths, pct, axis=1),
        mode="lines", name=label, line=dict(color=color, width=2),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>" + label + "</extra>"))
fig5a.add_hline(y=S0, line_dash="dash", line_color="orange",
    annotation_text="Start " + str(round(S0, 2)) + " EUR")
fig5a.update_layout(title=NAME + " - Monte Carlo Price Paths ("+str(SIMULATIONS)+" sims)",
    xaxis_title="Trading Days", yaxis_title="Price (EUR)",
    hovermode="x unified", template="plotly_dark")
fig5a.write_html("output/5a_monte_carlo_paths.html")
fig5a.show()
print("Step 5a done.")

# Chart 5b: Portfolio 1x vs 2x+stop-loss
fig5b = go.Figure()
for pct, color in [(5, "red"), (50, "white"), (95, "green")]:
    fig5b.add_trace(go.Scatter(x=days, y=np.percentile(portfolio_paths_1x, pct, axis=1),
        mode="lines", name=str(pct)+"th pct (1x)",
        line=dict(color=color, width=1.5, dash="dot"),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>"+str(pct)+"th 1x</extra>"))
    fig5b.add_trace(go.Scatter(x=days, y=np.percentile(portfolio_paths, pct, axis=1),
        mode="lines", name=str(pct)+"th pct ("+str(LEVERAGE)+"x+SL)",
        line=dict(color=color, width=2.5),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>"+str(pct)+"th "+str(LEVERAGE)+"x</extra>"))
fig5b.add_hline(y=INVESTMENT, line_dash="dash", line_color="orange",
    annotation_text="Capital " + str(INVESTMENT) + " EUR")
fig5b.add_hline(y=INVESTMENT*(1+STOP_LOSS), line_dash="dash", line_color="red",
    annotation_text="Stop-loss " + str(int(STOP_LOSS*100)) + "% = " + str(round(INVESTMENT*(1+STOP_LOSS), 0)) + " EUR")
fig5b.update_layout(
    title=NAME + " - Portfolio " + str(INVESTMENT) + " EUR: 1x vs " + str(LEVERAGE) + "x + Stop-Loss",
    xaxis_title="Trading Days", yaxis_title="Portfolio Value (EUR)",
    hovermode="x unified", template="plotly_dark")
fig5b.write_html("output/5b_portfolio_leverage.html")
fig5b.show()
print("Step 5b done.")

# Chart 5c: Final distribution
fig5c = go.Figure()
fig5c.add_trace(go.Histogram(x=final_port_1x, nbinsx=60, name="Final (1x)",
    marker_color="steelblue", opacity=0.6))
fig5c.add_trace(go.Histogram(x=final_port, nbinsx=60,
    name="Final (" + str(LEVERAGE) + "x+SL)",
    marker_color="darkorange", opacity=0.6))
for val, color, label in [
        (pp50x, "steelblue", "Median 1x "+str(round(pp50x,2))+" EUR"),
        (pp50,  "orange",   "Median "+str(LEVERAGE)+"x "+str(round(pp50,2))+" EUR"),
        (INVESTMENT, "white", "Start "+str(INVESTMENT)+" EUR")]:
    fig5c.add_vline(x=val, line_dash="dash", line_color=color, line_width=2,
        annotation_text=label, annotation_position="top right")
fig5c.update_layout(
    title=NAME + " - Final Portfolio Distribution: 1x vs " + str(LEVERAGE) + "x+SL",
    xaxis_title="Final Value after " + str(HORIZON) + " days (EUR)",
    yaxis_title="Frequency", barmode="overlay",
    hovermode="x", template="plotly_dark")
fig5c.write_html("output/5c_final_portfolio_distribution.html")
fig5c.show()
print("Step 5c done.")

# =============================================================
# SUMMARY
# =============================================================
sl_label     = "Worst  (5th pct)"
med_label    = "Median (50th)"
best_label   = "Best   (95th)"
sc_label     = "Scenario"

print("\n" + "="*55)
print(" FULL RISK REPORT - " + NAME)
print("="*55)
print("Capital      : " + str(INVESTMENT) + " EUR  | Leverage: " + str(LEVERAGE) + "x | Exposure: " + str(EXPOSURE) + " EUR")
print("Last price   : " + str(round(S0, 2)) + " EUR")
print("RSI (last)   : " + str(round(float(rsi.iloc[-1]), 1)))
print("SMA20 / SMA50: " + str(round(float(sma20.iloc[-1]),2)) + " / " + str(round(float(sma50.iloc[-1]),2)))
print("-"*55)
print("Sharpe (1x)  : " + str(round(sharpe, 3)) + "  | Sharpe (" + str(LEVERAGE) + "x): " + str(round(sharpe_lev, 3)))
print("Sortino      : " + str(round(sortino, 3)) + "  | Calmar: " + str(round(calmar, 3)))
print("Max Drawdown : " + str(round(max_dd*100, 2)) + "%")
print("-"*55)
print("VaR/CVaR on " + str(INVESTMENT) + " EUR (" + str(LEVERAGE) + "x):")
print("  VaR  95% : " + str(round(var95_lev,2)) + " EUR  |  CVaR 95%: " + str(round(cvar95_lev,2)) + " EUR")
print("  VaR  99% : " + str(round(var99_lev,2)) + " EUR  |  CVaR 99%: " + str(round(cvar99_lev,2)) + " EUR")
print("-"*55)
print("Stop-loss triggered : " + str(round(stop_loss_hits/SIMULATIONS*100,1)) + "% of simulations")
print("Prob. profit (1x)   : " + str(round(prob_profit_1x,1)) + "%")
print("Prob. profit (" + str(LEVERAGE) + "x+SL) : " + str(round(prob_profit_lev,1)) + "%")
print("-"*55)
print(f"{sc_label:<22} {'1x':>10} {str(LEVERAGE)+'x+SL':>10}")
print(f"{sl_label:<22} {pp5x:>9.2f}  {pp5:>9.2f}")
print(f"{med_label:<22} {pp50x:>9.2f}  {pp50:>9.2f}")
print(f"{best_label:<22} {pp95x:>9.2f}  {pp95:>9.2f}")
print("="*55)
print("HTML charts saved in /output")
