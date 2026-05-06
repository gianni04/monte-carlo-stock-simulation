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
INVESTMENT  = 50         # EUR — your actual capital
LEVERAGE    = 2          # leverage factor
EXPOSURE    = INVESTMENT * LEVERAGE   # = €100 effective exposure
RISK_FREE   = 0.03 / 252              # daily ECB rate

print(f"Date range      : {START} -> {END}")
print(f"Capital         : \u20ac{INVESTMENT}")
print(f"Leverage        : {LEVERAGE}x")
print(f"Effective exp.  : \u20ac{EXPOSURE}")

# =============================================================
# STEP 1 \u2014 Download historical prices
# =============================================================
print(f"\nDownloading {NAME} ({TICKER}) data...")
df = yf.download(TICKER, start=START, end=END)["Close"].dropna()
df.name = NAME

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df.index, y=df.values.flatten(), mode="lines", name=NAME,
    line=dict(color="steelblue", width=1.5),
    hovertemplate="Date: %{x}<br>Price: \u20ac%{y:.2f}<extra></extra>"))
fig1.update_layout(
    title=f"{NAME} \u2014 Historical Close Price ({START[:4]}\u2013{END[:4]})",
    xaxis_title="Date", yaxis_title="Price (EUR)",
    hovermode="x unified", template="plotly_dark")
fig1.write_html("output/1_historical_prices.html")
fig1.show()
print("Step 1 done.")

# =============================================================
# STEP 2 \u2014 Daily returns
# =============================================================
returns = df.pct_change().dropna()

# Leveraged daily return: L*r - (L-1)*rf
lev_returns = LEVERAGE * returns.values.flatten() - (LEVERAGE - 1) * RISK_FREE

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=returns.index, y=returns.values.flatten(), mode="lines",
    name="Return (no leverage)", line=dict(color="steelblue", width=0.8),
    hovertemplate="Date: %{x}<br>Return: %{y:.4f}<extra>1x</extra>"))
fig2.add_trace(go.Scatter(
    x=returns.index, y=lev_returns, mode="lines",
    name=f"Return ({LEVERAGE}x leverage)", line=dict(color="darkorange", width=0.8),
    hovertemplate="Date: %{x}<br>Lev. Return: %{y:.4f}<extra>{LEVERAGE}x</extra>"))
fig2.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8)
fig2.update_layout(
    title=f"{NAME} \u2014 Daily Returns: No Leverage vs {LEVERAGE}x",
    xaxis_title="Date", yaxis_title="Daily Return",
    hovermode="x unified", template="plotly_dark")
fig2.write_html("output/2_daily_returns.html")
fig2.show()
print("Step 2 done.")

# =============================================================
# STEP 3 \u2014 Return distribution + normality test
# =============================================================
mu    = float(returns.mean())
sigma = float(returns.std())
skew  = float(returns.skew())
kurt  = float(returns.kurtosis())
_, pvalue = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42).values.flatten())

mu_lev    = LEVERAGE * mu - (LEVERAGE - 1) * float(RISK_FREE)
sigma_lev = LEVERAGE * sigma

print(f"\n--- Return Statistics ---")
print(f"Mean daily return (1x)  : {mu:.5f}   | Ann: {mu*252*100:.2f}%")
print(f"Mean daily return ({LEVERAGE}x)  : {mu_lev:.5f}   | Ann: {mu_lev*252*100:.2f}%")
print(f"Daily volatility  (1x)  : {sigma:.5f}   | Ann: {sigma*np.sqrt(252)*100:.2f}%")
print(f"Daily volatility  ({LEVERAGE}x)  : {sigma_lev:.5f}   | Ann: {sigma_lev*np.sqrt(252)*100:.2f}%")
print(f"Skewness                : {skew:.4f}")
print(f"Kurtosis                : {kurt:.4f}")
print(f"Shapiro-Wilk p          : {pvalue:.4f} ({'NOT normal' if pvalue < 0.05 else 'normal'})")

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
    name=f"Normal fit ({LEVERAGE}x)", line=dict(color="orange", width=2, dash="dash")))
fig3.update_layout(
    title=f"{NAME} \u2014 Return Distribution: 1x vs {LEVERAGE}x Leverage",
    xaxis_title="Daily Return", yaxis_title="Density",
    hovermode="x", template="plotly_dark")
fig3.write_html("output/3_return_distribution.html")
fig3.show()
print("Step 3 done.")

# =============================================================
# STEP 4 \u2014 Value at Risk \u2014 on your \u20ac50 capital
# =============================================================
var95_1x = np.percentile(returns.values, 5)  * INVESTMENT
var99_1x = np.percentile(returns.values, 1)  * INVESTMENT
var95_lev = np.percentile(lev_returns, 5)    * INVESTMENT
var99_lev = np.percentile(lev_returns, 1)    * INVESTMENT

print(f"\n--- Value at Risk on \u20ac{INVESTMENT} capital ---")
print(f"VaR 95% (1x) : \u20ac{var95_1x:.2f}  |  VaR 95% ({LEVERAGE}x) : \u20ac{var95_lev:.2f}")
print(f"VaR 99% (1x) : \u20ac{var99_1x:.2f}  |  VaR 99% ({LEVERAGE}x) : \u20ac{var99_lev:.2f}")

fig4 = go.Figure()
fig4.add_trace(go.Histogram(
    x=lev_returns, nbinsx=80, histnorm="probability density",
    name=f"Returns ({LEVERAGE}x)", opacity=0.7, marker_color="darkorange"))
fig4.add_vline(x=np.percentile(lev_returns, 5), line_color="yellow",
    line_dash="dash", line_width=2,
    annotation_text=f"VaR 95%: \u20ac{var95_lev:.2f}",
    annotation_position="top right")
fig4.add_vline(x=np.percentile(lev_returns, 1), line_color="red",
    line_dash="dash", line_width=2,
    annotation_text=f"VaR 99%: \u20ac{var99_lev:.2f}",
    annotation_position="top left")
fig4.update_layout(
    title=f"{NAME} \u2014 Value at Risk ({LEVERAGE}x Leverage, \u20ac{INVESTMENT} capital)",
    xaxis_title="Daily Leveraged Return", yaxis_title="Density",
    template="plotly_dark")
fig4.write_html("output/4_var.html")
fig4.show()
print("Step 4 done.")

# =============================================================
# STEP 5 \u2014 Monte Carlo: portfolio value simulation with leverage
# =============================================================
S0     = float(df.iloc[-1])
dt     = 1
drift  = mu    - 0.5 * sigma**2
drift_lev = mu_lev - 0.5 * sigma_lev**2

np.random.seed(42)

# Simulate stock price paths
price_paths = np.zeros((HORIZON, SIMULATIONS))
# Simulate portfolio value paths (\u20ac50 with 2x leverage = \u20ac100 exposure)
portfolio_paths    = np.zeros((HORIZON, SIMULATIONS))
portfolio_paths_1x = np.zeros((HORIZON, SIMULATIONS))

for i in range(SIMULATIONS):
    Z = np.random.normal(0, 1, HORIZON)
    # Stock price (GBM)
    price_paths[:, i] = S0 * np.exp(np.cumsum(drift * dt + sigma * np.sqrt(dt) * Z))
    # Portfolio 1x: \u20ac50 grows with stock
    price_ratio = price_paths[:, i] / S0
    portfolio_paths_1x[:, i] = INVESTMENT * price_ratio
    # Portfolio 2x: leveraged returns applied to \u20ac50 capital
    daily_lev_r = LEVERAGE * (np.exp(drift * dt + sigma * np.sqrt(dt) * Z) - 1) - (LEVERAGE-1)*float(RISK_FREE)
    portfolio_paths[:, i] = INVESTMENT * np.cumprod(1 + daily_lev_r)

# Final portfolio values
final_port     = portfolio_paths[-1, :]
final_port_1x  = portfolio_paths_1x[-1, :]

pp5,  pp50,  pp95  = np.percentile(final_port,    [5, 50, 95])
pp5x, pp50x, pp95x = np.percentile(final_port_1x, [5, 50, 95])

print(f"\n--- Monte Carlo Portfolio Results ({SIMULATIONS} sims, {HORIZON} days) ---")
print(f"Starting capital        : \u20ac{INVESTMENT:.2f}")
print(f"--- WITHOUT leverage (1x) ---")
print(f"5th  pct (worst)   : \u20ac{pp5x:.2f}  ({(pp5x/INVESTMENT-1)*100:+.1f}%)")
print(f"50th pct (median)  : \u20ac{pp50x:.2f}  ({(pp50x/INVESTMENT-1)*100:+.1f}%)")
print(f"95th pct (best)    : \u20ac{pp95x:.2f}  ({(pp95x/INVESTMENT-1)*100:+.1f}%)")
print(f"--- WITH {LEVERAGE}x leverage ---")
print(f"5th  pct (worst)   : \u20ac{pp5:.2f}  ({(pp5/INVESTMENT-1)*100:+.1f}%)")
print(f"50th pct (median)  : \u20ac{pp50:.2f}  ({(pp50/INVESTMENT-1)*100:+.1f}%)")
print(f"95th pct (best)    : \u20ac{pp95:.2f}  ({(pp95/INVESTMENT-1)*100:+.1f}%)")

# Chart 5a: Price paths
days = list(range(HORIZON))
fig5a = go.Figure()
for i in range(150):
    fig5a.add_trace(go.Scatter(
        x=days, y=price_paths[:, i], mode="lines",
        line=dict(color="steelblue", width=0.4), opacity=0.15,
        showlegend=False, hoverinfo="skip"))
for pct, color, label in [(5,"red","5th pct"),(50,"white","Median"),(95,"green","95th pct")]:
    fig5a.add_trace(go.Scatter(
        x=days, y=np.percentile(price_paths, pct, axis=1),
        mode="lines", name=label, line=dict(color=color, width=2),
        hovertemplate=f"Day %{{x}}<br>Price: \u20ac%{{y:.2f}}<extra>{label}</extra>"))
fig5a.add_hline(y=S0, line_dash="dash", line_color="orange",
    annotation_text=f"Start \u20ac{S0:.2f}")
fig5a.update_layout(
    title=f"{NAME} \u2014 Stock Price Simulation ({SIMULATIONS} paths)",
    xaxis_title="Trading Days", yaxis_title="Price (EUR)",
    hovermode="x unified", template="plotly_dark")
fig5a.write_html("output/5a_monte_carlo_paths.html")
fig5a.show()
print("Step 5a done.")

# Chart 5b: Portfolio value comparison 1x vs 2x
fig5b = go.Figure()
for pct, color in [(5,"red"),(50,"white"),(95,"green")]:
    fig5b.add_trace(go.Scatter(
        x=days, y=np.percentile(portfolio_paths_1x, pct, axis=1),
        mode="lines", name=f"{pct}th pct (1x)",
        line=dict(color=color, width=1.5, dash="dot"),
        hovertemplate=f"Day %{{x}}<br>\u20ac%{{y:.2f}}<extra>{pct}th pct 1x</extra>"))
    fig5b.add_trace(go.Scatter(
        x=days, y=np.percentile(portfolio_paths, pct, axis=1),
        mode="lines", name=f"{pct}th pct ({LEVERAGE}x)",
        line=dict(color=color, width=2.5),
        hovertemplate=f"Day %{{x}}<br>\u20ac%{{y:.2f}}<extra>{pct}th pct {LEVERAGE}x</extra>"))
fig5b.add_hline(y=INVESTMENT, line_dash="dash", line_color="orange",
    annotation_text=f"Capital \u20ac{INVESTMENT}")
fig5b.update_layout(
    title=f"{NAME} \u2014 Portfolio \u20ac{INVESTMENT}: 1x vs {LEVERAGE}x Leverage ({HORIZON} days)",
    xaxis_title="Trading Days", yaxis_title="Portfolio Value (EUR)",
    hovermode="x unified", template="plotly_dark")
fig5b.write_html("output/5b_portfolio_leverage.html")
fig5b.show()
print("Step 5b done.")

# Chart 5c: Final portfolio distribution comparison
fig5c = go.Figure()
fig5c.add_trace(go.Histogram(
    x=final_port_1x, nbinsx=60, name="Final value (1x)",
    marker_color="steelblue", opacity=0.6))
fig5c.add_trace(go.Histogram(
    x=final_port, nbinsx=60, name=f"Final value ({LEVERAGE}x)",
    marker_color="darkorange", opacity=0.6))
for val, color, label in [
    (pp50x, "steelblue", f"Median 1x \u20ac{pp50x:.2f}"),
    (pp50,  "orange",   f"Median {LEVERAGE}x \u20ac{pp50:.2f}"),
    (INVESTMENT, "white", f"Start \u20ac{INVESTMENT}")]:
    fig5c.add_vline(x=val, line_dash="dash", line_color=color, line_width=2,
        annotation_text=label, annotation_position="top right")
fig5c.update_layout(
    title=f"{NAME} \u2014 Final Portfolio Distribution: 1x vs {LEVERAGE}x (\u20ac{INVESTMENT})",
    xaxis_title=f"Portfolio Value after {HORIZON} days (EUR)",
    yaxis_title="Frequency", barmode="overlay",
    hovermode="x", template="plotly_dark")
fig5c.write_html("output/5c_final_portfolio_distribution.html")
fig5c.show()
print("Step 5c done.")

print(f"\n=== SUMMARY ===")
print(f"Capital invest\u00e9  : \u20ac{INVESTMENT}  |  Levier: {LEVERAGE}x  |  Exposition: \u20ac{EXPOSURE}")
print(f"Stock actuel    : \u20ac{S0:.2f}")
print(f"{'Sc\u00e9nario':<20} {'1x':>10} {'2x':>10}")
print(f"{'Pire (5th pct)':<20} \u20ac{pp5x:>8.2f}  \u20ac{pp5:>8.2f}")
print(f"{'M\u00e9diane (50th)':<20} \u20ac{pp50x:>8.2f}  \u20ac{pp50:>8.2f}")
print(f"{'Meilleur (95th)':<20} \u20ac{pp95x:>8.2f}  \u20ac{pp95:>8.2f}")
print(f"\n=== All steps completed. HTML charts saved in /output ===")
