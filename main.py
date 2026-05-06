import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
import os

os.makedirs("output", exist_ok=True)

# =============================================================
# CONFIGURATION
# =============================================================
TICKER      = "UBI.PA"   # change to any Yahoo Finance ticker
NAME        = "Ubisoft"
START       = "2020-01-01"
END         = "2024-12-31"
SIMULATIONS = 1000
HORIZON     = 252        # 1 year of trading days
INVESTMENT  = 10_000     # EUR

# =============================================================
# STEP 1 — Download historical prices
# =============================================================
print(f"Downloading {NAME} ({TICKER}) data...")
df = yf.download(TICKER, start=START, end=END)["Close"].dropna()
df.name = NAME

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df.values.flatten(),
                          mode="lines", name=NAME,
                          line=dict(color="steelblue", width=1.5),
                          hovertemplate="Date: %{x}<br>Price: €%{y:.2f}<extra></extra>"))
fig1.update_layout(title=f"{NAME} — Historical Close Price (2020–2024)",
                   xaxis_title="Date", yaxis_title="Price (EUR)",
                   hovermode="x unified", template="plotly_dark")
fig1.write_html("output/1_historical_prices.html")
fig1.show()
print("Step 1 done.")

# =============================================================
# STEP 2 — Daily returns
# =============================================================
returns = df.pct_change().dropna()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=returns.index, y=returns.values.flatten(),
                          mode="lines", name="Daily Return",
                          line=dict(color="darkorange", width=0.8),
                          hovertemplate="Date: %{x}<br>Return: %{y:.4f}<extra></extra>"))
fig2.add_hline(y=0, line_dash="dash", line_color="white", line_width=0.8)
fig2.update_layout(title=f"{NAME} — Daily Returns (2020–2024)",
                   xaxis_title="Date", yaxis_title="Daily Return",
                   hovermode="x unified", template="plotly_dark")
fig2.write_html("output/2_daily_returns.html")
fig2.show()
print("Step 2 done.")

# =============================================================
# STEP 3 — Distribution of returns + normality test
# =============================================================
mu    = float(returns.mean())
sigma = float(returns.std())
skew  = float(returns.skew())
kurt  = float(returns.kurtosis())
_, pvalue = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42).values.flatten())

print(f"\n--- Return Statistics ---")
print(f"Mean daily return : {mu:.4f}")
print(f"Daily volatility  : {sigma:.4f}")
print(f"Annualized return : {mu*252:.4f} ({mu*252*100:.2f}%)")
print(f"Annualized vol    : {sigma*np.sqrt(252):.4f} ({sigma*np.sqrt(252)*100:.2f}%)")
print(f"Skewness          : {skew:.4f}")
print(f"Kurtosis          : {kurt:.4f}")
print(f"Shapiro-Wilk p    : {pvalue:.4f} ({'NOT normal' if pvalue < 0.05 else 'normal'})")

x = np.linspace(returns.min(), returns.max(), 300)
normal_curve = stats.norm.pdf(x, mu, sigma)

fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=returns.values.flatten(), nbinsx=80,
                            histnorm="probability density",
                            name="Empirical", opacity=0.7,
                            marker_color="steelblue"))
fig3.add_trace(go.Scatter(x=x, y=normal_curve, mode="lines",
                          name="Normal fit", line=dict(color="red", width=2)))
fig3.update_layout(title=f"{NAME} — Return Distribution vs Normal Curve",
                   xaxis_title="Daily Return", yaxis_title="Density",
                   hovermode="x", template="plotly_dark")
fig3.write_html("output/3_return_distribution.html")
fig3.show()
print("Step 3 done.")

# =============================================================
# STEP 4 — Value at Risk (VaR)
# =============================================================
var95 = np.percentile(returns.values, 5) * INVESTMENT
var99 = np.percentile(returns.values, 1) * INVESTMENT

print(f"\n--- Value at Risk (on €{INVESTMENT:,}) ---")
print(f"VaR 95% (1 day): €{var95:,.2f}")
print(f"VaR 99% (1 day): €{var99:,.2f}")

fig4 = go.Figure()
fig4.add_trace(go.Histogram(x=returns.values.flatten(), nbinsx=80,
                            histnorm="probability density",
                            name="Returns", opacity=0.7,
                            marker_color="steelblue"))
fig4.add_vline(x=np.percentile(returns.values, 5), line_color="orange",
               line_dash="dash", line_width=2,
               annotation_text=f"VaR 95%: €{var95:,.0f}",
               annotation_position="top right")
fig4.add_vline(x=np.percentile(returns.values, 1), line_color="red",
               line_dash="dash", line_width=2,
               annotation_text=f"VaR 99%: €{var99:,.0f}",
               annotation_position="top left")
fig4.update_layout(title=f"{NAME} — Value at Risk",
                   xaxis_title="Daily Return", yaxis_title="Density",
                   template="plotly_dark")
fig4.write_html("output/4_var.html")
fig4.show()
print("Step 4 done.")

# =============================================================
# STEP 5 — Monte Carlo Simulation (Geometric Brownian Motion)
# =============================================================
S0    = float(df.iloc[-1])
dt    = 1
drift = mu - 0.5 * sigma**2

np.random.seed(42)
simulations = np.zeros((HORIZON, SIMULATIONS))

for i in range(SIMULATIONS):
    shocks = np.random.normal(0, 1, HORIZON)
    path   = S0 * np.exp(np.cumsum(drift * dt + sigma * np.sqrt(dt) * shocks))
    simulations[:, i] = path

final_prices = simulations[-1, :]
p5  = np.percentile(final_prices, 5)
p50 = np.percentile(final_prices, 50)
p95 = np.percentile(final_prices, 95)

print(f"\n--- Monte Carlo Results ({SIMULATIONS} simulations, {HORIZON} days) ---")
print(f"Starting price  : €{S0:.2f}")
print(f"5th percentile  : €{p5:.2f}  (worst scenario)")
print(f"50th percentile : €{p50:.2f}  (median scenario)")
print(f"95th percentile : €{p95:.2f}  (best scenario)")

# Chart 5a: Simulation paths (interactive)
days = list(range(HORIZON))
fig5a = go.Figure()
for i in range(0, 200):
    fig5a.add_trace(go.Scatter(x=days, y=simulations[:, i],
                               mode="lines", line=dict(color="steelblue", width=0.4),
                               opacity=0.15, showlegend=False,
                               hoverinfo="skip"))
fig5a.add_trace(go.Scatter(x=days, y=np.percentile(simulations, 5, axis=1),
                           mode="lines", name="5th percentile",
                           line=dict(color="red", width=2),
                           hovertemplate="Day %{x}<br>Price: €%{y:.2f}<extra>5th pct</extra>"))
fig5a.add_trace(go.Scatter(x=days, y=np.percentile(simulations, 50, axis=1),
                           mode="lines", name="Median",
                           line=dict(color="white", width=2),
                           hovertemplate="Day %{x}<br>Price: €%{y:.2f}<extra>Median</extra>"))
fig5a.add_trace(go.Scatter(x=days, y=np.percentile(simulations, 95, axis=1),
                           mode="lines", name="95th percentile",
                           line=dict(color="green", width=2),
                           hovertemplate="Day %{x}<br>Price: €%{y:.2f}<extra>95th pct</extra>"))
fig5a.add_hline(y=S0, line_dash="dash", line_color="orange",
                annotation_text=f"Start €{S0:.2f}")
fig5a.update_layout(
    title=f"{NAME} — Monte Carlo Simulation ({SIMULATIONS} paths, {HORIZON} days)",
    xaxis_title="Trading Days", yaxis_title="Simulated Price (EUR)",
    hovermode="x unified", template="plotly_dark"
)
fig5a.write_html("output/5a_monte_carlo_paths.html")
fig5a.show()
print("Step 5a done.")

# Chart 5b: Distribution of final prices
fig5b = go.Figure()
fig5b.add_trace(go.Histogram(x=final_prices, nbinsx=80,
                             name="Final prices", marker_color="steelblue",
                             opacity=0.8))
for val, color, label in [(p5, "red", f"5th pct €{p5:.2f}"),
                          (p50, "white", f"Median €{p50:.2f}"),
                          (p95, "green", f"95th pct €{p95:.2f}"),
                          (S0,  "orange", f"Start €{S0:.2f}")]:
    fig5b.add_vline(x=val, line_dash="dash", line_color=color,
                    line_width=2, annotation_text=label,
                    annotation_position="top right")
fig5b.update_layout(
    title=f"{NAME} — Distribution of Simulated Prices after {HORIZON} days",
    xaxis_title="Final Price (EUR)", yaxis_title="Frequency",
    hovermode="x", template="plotly_dark"
)
fig5b.write_html("output/5b_final_price_distribution.html")
fig5b.show()
print("Step 5b done.")

print("\n=== All steps completed. HTML charts saved in /output ===")
