import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import os
from datetime import date, timedelta

os.makedirs("output", exist_ok=True)

# =============================================================
# CONFIGURATION
# =============================================================
TICKER      = "UBI.PA"
NAME        = "Ubisoft"
START       = "2019-01-01"
END         = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")  # yesterday
SIMULATIONS = 1000
HORIZON     = 252        # 1 year ahead
INVESTMENT  = 50         # EUR - your actual capital
LEVERAGE    = 2          # leverage factor
EXPOSURE    = INVESTMENT * LEVERAGE
RISK_FREE   = 0.03 / 252

print(f"Date range      : {START} -> {END}")
print(f"Capital         : {INVESTMENT} EUR")
print(f"Leverage        : {LEVERAGE}x")
print(f"Effective exp.  : {EXPOSURE} EUR")

# =============================================================
# STEP 1 - Download historical prices
# =============================================================
print(f"\nDownloading {NAME} ({TICKER}) data...")
df = yf.download(TICKER, start=START, end=END)["Close"].dropna()
df.name = NAME

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df.index, y=df.values.flatten(), mode="lines", name=NAME,
    line=dict(color="steelblue", width=1.5),
    hovertemplate="Date: %{x}<br>Price: %{y:.2f} EUR<extra></extra>"))
fig1.update_layout(
    title=NAME + " - Historical Close Price (" + START[:4] + "-" + END[:4] + ")",
    xaxis_title="Date", yaxis_title="Price (EUR)",
    hovermode="x unified", template="plotly_dark")
fig1.write_html("output/1_historical_prices.html")
fig1.show()
print("Step 1 done.")

# =============================================================
# STEP 2 - Daily returns
# =============================================================
returns = df.pct_change().dropna()
lev_returns = LEVERAGE * returns.values.flatten() - (LEVERAGE - 1) * RISK_FREE

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=returns.index, y=returns.values.flatten(), mode="lines",
    name="Return (no leverage)", line=dict(color="steelblue", width=0.8),
    hovertemplate="Date: %{x}<br>Return: %{y:.4f}<extra>1x</extra>"))
fig2.add_trace(go.Scatter(
    x=returns.index, y=lev_returns, mode="lines",
    name="Return (" + str(LEVERAGE) + "x leverage)",
    line=dict(color="darkorange", width=0.8),
    hovertemplate="Date: %{x}<br>Lev. Return: %{y:.4f}<extra>" + str(LEVERAGE) + "x</extra>"))
fig2.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8)
fig2.update_layout(
    title=NAME + " - Daily Returns: No Leverage vs " + str(LEVERAGE) + "x",
    xaxis_title="Date", yaxis_title="Daily Return",
    hovermode="x unified", template="plotly_dark")
fig2.write_html("output/2_daily_returns.html")
fig2.show()
print("Step 2 done.")

# =============================================================
# STEP 3 - Return distribution + normality test
# =============================================================
mu    = float(returns.mean())
sigma = float(returns.std())
skew  = float(returns.skew())
kurt  = float(returns.kurtosis())
_, pvalue = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42).values.flatten())

mu_lev    = LEVERAGE * mu - (LEVERAGE - 1) * float(RISK_FREE)
sigma_lev = LEVERAGE * sigma

print("\n--- Return Statistics ---")
print("Mean daily return (1x)  : " + str(round(mu, 5)) + "   | Ann: " + str(round(mu*252*100, 2)) + "%")
print("Mean daily return (" + str(LEVERAGE) + "x)  : " + str(round(mu_lev, 5)) + "   | Ann: " + str(round(mu_lev*252*100, 2)) + "%")
print("Daily volatility  (1x)  : " + str(round(sigma, 5)) + "   | Ann: " + str(round(sigma*np.sqrt(252)*100, 2)) + "%")
print("Daily volatility  (" + str(LEVERAGE) + "x)  : " + str(round(sigma_lev, 5)) + "   | Ann: " + str(round(sigma_lev*np.sqrt(252)*100, 2)) + "%")
print(f"Skewness          : {skew:.4f}")
print(f"Kurtosis          : {kurt:.4f}")
normality = "NOT normal" if pvalue < 0.05 else "normal"
print(f"Shapiro-Wilk p    : {pvalue:.4f} ({normality})")

x = np.linspace(float(returns.min()), float(returns.max()), 300)
fig3 = go.Figure()
fig3.add_trace(go.Histogram(
    x=returns.values.flatten(), nbinsx=80, histnorm="probability density",
    name="Empirical (1x)", opacity=0.6, marker_color="steelblue"))
fig3.add_trace(go.Scatter(
    x=x, y=stats.norm.pdf(x, mu, sigma), mode="lines",
    name="Normal fit (1x)", line=dict(color="red", width=2)))
fig3.add_trace(go.Scatter(
    x=x, y=stats.norm.pdf(x, mu_lev, sigma_lev), mode="lines",
    name="Normal fit (" + str(LEVERAGE) + "x)",
    line=dict(color="orange", width=2, dash="dash")))
fig3.update_layout(
    title=NAME + " - Return Distribution: 1x vs " + str(LEVERAGE) + "x Leverage",
    xaxis_title="Daily Return", yaxis_title="Density",
    hovermode="x", template="plotly_dark")
fig3.write_html("output/3_return_distribution.html")
fig3.show()
print("Step 3 done.")

# =============================================================
# STEP 4 - Value at Risk on your capital
# =============================================================
var95_1x  = np.percentile(returns.values, 5) * INVESTMENT
var99_1x  = np.percentile(returns.values, 1) * INVESTMENT
var95_lev = np.percentile(lev_returns, 5)    * INVESTMENT
var99_lev = np.percentile(lev_returns, 1)    * INVESTMENT

print("\n--- Value at Risk on " + str(INVESTMENT) + " EUR capital ---")
print("VaR 95% (1x) : " + str(round(var95_1x, 2)) + " EUR  |  VaR 95% (" + str(LEVERAGE) + "x) : " + str(round(var95_lev, 2)) + " EUR")
print("VaR 99% (1x) : " + str(round(var99_1x, 2)) + " EUR  |  VaR 99% (" + str(LEVERAGE) + "x) : " + str(round(var99_lev, 2)) + " EUR")

fig4 = go.Figure()
fig4.add_trace(go.Histogram(
    x=lev_returns, nbinsx=80, histnorm="probability density",
    name="Returns (" + str(LEVERAGE) + "x)", opacity=0.7, marker_color="darkorange"))
fig4.add_vline(x=np.percentile(lev_returns, 5), line_color="yellow",
    line_dash="dash", line_width=2,
    annotation_text="VaR 95%: " + str(round(var95_lev, 2)) + " EUR",
    annotation_position="top right")
fig4.add_vline(x=np.percentile(lev_returns, 1), line_color="red",
    line_dash="dash", line_width=2,
    annotation_text="VaR 99%: " + str(round(var99_lev, 2)) + " EUR",
    annotation_position="top left")
fig4.update_layout(
    title=NAME + " - Value at Risk (" + str(LEVERAGE) + "x Leverage, " + str(INVESTMENT) + " EUR capital)",
    xaxis_title="Daily Leveraged Return", yaxis_title="Density",
    template="plotly_dark")
fig4.write_html("output/4_var.html")
fig4.show()
print("Step 4 done.")

# =============================================================
# STEP 5 - Monte Carlo: portfolio simulation with leverage
# =============================================================
S0    = float(df.iloc[-1])
dt    = 1
drift = mu - 0.5 * sigma**2

np.random.seed(42)
price_paths        = np.zeros((HORIZON, SIMULATIONS))
portfolio_paths    = np.zeros((HORIZON, SIMULATIONS))
portfolio_paths_1x = np.zeros((HORIZON, SIMULATIONS))

for i in range(SIMULATIONS):
    Z = np.random.normal(0, 1, HORIZON)
    price_paths[:, i] = S0 * np.exp(np.cumsum(drift * dt + sigma * np.sqrt(dt) * Z))
    price_ratio = price_paths[:, i] / S0
    portfolio_paths_1x[:, i] = INVESTMENT * price_ratio
    daily_lev_r = LEVERAGE * (np.exp(drift * dt + sigma * np.sqrt(dt) * Z) - 1) - (LEVERAGE - 1) * float(RISK_FREE)
    portfolio_paths[:, i] = INVESTMENT * np.cumprod(1 + daily_lev_r)

final_port    = portfolio_paths[-1, :]
final_port_1x = portfolio_paths_1x[-1, :]

pp5,  pp50,  pp95  = np.percentile(final_port,    [5, 50, 95])
pp5x, pp50x, pp95x = np.percentile(final_port_1x, [5, 50, 95])

print("\n--- Monte Carlo Portfolio Results (" + str(SIMULATIONS) + " sims, " + str(HORIZON) + " days) ---")
print("Starting capital : " + str(INVESTMENT) + " EUR")
print("--- WITHOUT leverage (1x) ---")
print("5th  pct (worst)  : " + str(round(pp5x, 2)) + " EUR  (" + str(round((pp5x/INVESTMENT-1)*100, 1)) + "%)")
print("50th pct (median) : " + str(round(pp50x, 2)) + " EUR  (" + str(round((pp50x/INVESTMENT-1)*100, 1)) + "%)")
print("95th pct (best)   : " + str(round(pp95x, 2)) + " EUR  (" + str(round((pp95x/INVESTMENT-1)*100, 1)) + "%)")
print("--- WITH " + str(LEVERAGE) + "x leverage ---")
print("5th  pct (worst)  : " + str(round(pp5, 2)) + " EUR  (" + str(round((pp5/INVESTMENT-1)*100, 1)) + "%)")
print("50th pct (median) : " + str(round(pp50, 2)) + " EUR  (" + str(round((pp50/INVESTMENT-1)*100, 1)) + "%)")
print("95th pct (best)   : " + str(round(pp95, 2)) + " EUR  (" + str(round((pp95/INVESTMENT-1)*100, 1)) + "%)")

# Chart 5a: Stock price paths
days = list(range(HORIZON))
fig5a = go.Figure()
for i in range(150):
    fig5a.add_trace(go.Scatter(
        x=days, y=price_paths[:, i], mode="lines",
        line=dict(color="steelblue", width=0.4), opacity=0.15,
        showlegend=False, hoverinfo="skip"))
for pct, color, label in [(5, "red", "5th pct"), (50, "white", "Median"), (95, "green", "95th pct")]:
    fig5a.add_trace(go.Scatter(
        x=days, y=np.percentile(price_paths, pct, axis=1),
        mode="lines", name=label, line=dict(color=color, width=2),
        hovertemplate="Day %{x}<br>Price: %{y:.2f} EUR<extra>" + label + "</extra>"))
fig5a.add_hline(y=S0, line_dash="dash", line_color="orange",
    annotation_text="Start " + str(round(S0, 2)) + " EUR")
fig5a.update_layout(
    title=NAME + " - Stock Price Monte Carlo (" + str(SIMULATIONS) + " paths)",
    xaxis_title="Trading Days", yaxis_title="Price (EUR)",
    hovermode="x unified", template="plotly_dark")
fig5a.write_html("output/5a_monte_carlo_paths.html")
fig5a.show()
print("Step 5a done.")

# Chart 5b: Portfolio value 1x vs 2x over time
fig5b = go.Figure()
for pct, color in [(5, "red"), (50, "white"), (95, "green")]:
    fig5b.add_trace(go.Scatter(
        x=days, y=np.percentile(portfolio_paths_1x, pct, axis=1),
        mode="lines", name=str(pct) + "th pct (1x)",
        line=dict(color=color, width=1.5, dash="dot"),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>" + str(pct) + "th 1x</extra>"))
    fig5b.add_trace(go.Scatter(
        x=days, y=np.percentile(portfolio_paths, pct, axis=1),
        mode="lines", name=str(pct) + "th pct (" + str(LEVERAGE) + "x)",
        line=dict(color=color, width=2.5),
        hovertemplate="Day %{x}<br>%{y:.2f} EUR<extra>" + str(pct) + "th " + str(LEVERAGE) + "x</extra>"))
fig5b.add_hline(y=INVESTMENT, line_dash="dash", line_color="orange",
    annotation_text="Capital " + str(INVESTMENT) + " EUR")
fig5b.update_layout(
    title=NAME + " - Portfolio " + str(INVESTMENT) + " EUR: 1x vs " + str(LEVERAGE) + "x Leverage",
    xaxis_title="Trading Days", yaxis_title="Portfolio Value (EUR)",
    hovermode="x unified", template="plotly_dark")
fig5b.write_html("output/5b_portfolio_leverage.html")
fig5b.show()
print("Step 5b done.")

# Chart 5c: Final portfolio distribution
fig5c = go.Figure()
fig5c.add_trace(go.Histogram(
    x=final_port_1x, nbinsx=60, name="Final value (1x)",
    marker_color="steelblue", opacity=0.6))
fig5c.add_trace(go.Histogram(
    x=final_port, nbinsx=60,
    name="Final value (" + str(LEVERAGE) + "x)",
    marker_color="darkorange", opacity=0.6))
for val, color, label in [
        (pp50x, "steelblue", "Median 1x " + str(round(pp50x, 2)) + " EUR"),
        (pp50,  "orange",   "Median " + str(LEVERAGE) + "x " + str(round(pp50, 2)) + " EUR"),
        (INVESTMENT, "white", "Start " + str(INVESTMENT) + " EUR")]:
    fig5c.add_vline(x=val, line_dash="dash", line_color=color, line_width=2,
        annotation_text=label, annotation_position="top right")
fig5c.update_layout(
    title=NAME + " - Final Portfolio Distribution: 1x vs " + str(LEVERAGE) + "x (" + str(INVESTMENT) + " EUR)",
    xaxis_title="Portfolio Value after " + str(HORIZON) + " days (EUR)",
    yaxis_title="Frequency", barmode="overlay",
    hovermode="x", template="plotly_dark")
fig5c.write_html("output/5c_final_portfolio_distribution.html")
fig5c.show()
print("Step 5c done.")

# =============================================================
# SUMMARY
# =============================================================
scenario_label  = "Scenario"
worst_label     = "Worst  (5th pct)"
median_label    = "Median (50th)"
best_label      = "Best   (95th)"

print("\n=== SUMMARY ===")
print("Capital : " + str(INVESTMENT) + " EUR  |  Leverage: " + str(LEVERAGE) + "x  |  Exposure: " + str(EXPOSURE) + " EUR")
print("Stock last price : " + str(round(S0, 2)) + " EUR")
print(f"{scenario_label:<22} {'1x':>10} {str(LEVERAGE)+'x':>10}")
print(f"{worst_label:<22} {pp5x:>9.2f}  {pp5:>9.2f}")
print(f"{median_label:<22} {pp50x:>9.2f}  {pp50:>9.2f}")
print(f"{best_label:<22} {pp95x:>9.2f}  {pp95:>9.2f}")
print("\n=== All steps completed. HTML charts saved in /output ===")
