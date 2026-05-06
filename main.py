import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

os.makedirs("output", exist_ok=True)

# =============================================================
# CONFIGURATION
# =============================================================
TICKER     = "UBI.PA"   # Ubisoft — change to any ticker you want
NAME       = "Ubisoft"
START      = "2020-01-01"
END        = "2024-12-31"
SIMULATIONS = 1000      # number of Monte Carlo paths
HORIZON    = 252        # trading days = 1 year ahead
INVESTMENT = 10_000     # EUR

# =============================================================
# STEP 1 — Download historical prices
# =============================================================
print(f"Downloading {NAME} ({TICKER}) data...")
df = yf.download(TICKER, start=START, end=END)["Close"].dropna()
df.name = NAME

plt.figure(figsize=(14, 5))
plt.plot(df, color="steelblue", linewidth=1.2)
plt.title(f"{NAME} — Historical Close Price (2020–2024)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price (EUR)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/1_historical_prices.png", dpi=150)
plt.show()
print("Step 1 done: historical prices plotted.")

# =============================================================
# STEP 2 — Daily returns
# =============================================================
returns = df.pct_change().dropna()

plt.figure(figsize=(14, 4))
plt.plot(returns, color="darkorange", linewidth=0.8, alpha=0.8)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title(f"{NAME} — Daily Returns (2020–2024)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/2_daily_returns.png", dpi=150)
plt.show()
print("Step 2 done: daily returns plotted.")

# =============================================================
# STEP 3 — Distribution of returns + normality test
# =============================================================
mu    = returns.mean()
sigma = returns.std()
skew  = returns.skew()
kurt  = returns.kurtosis()
_, pvalue = stats.shapiro(returns.sample(min(len(returns), 5000), random_state=42))

print(f"\n--- Return Statistics ---")
print(f"Mean daily return : {mu:.4f}")
print(f"Daily volatility  : {sigma:.4f}")
print(f"Skewness          : {skew:.4f}")
print(f"Kurtosis          : {kurt:.4f}")
print(f"Shapiro-Wilk p    : {pvalue:.4f} ({'NOT normal' if pvalue < 0.05 else 'normal'})")

x = np.linspace(returns.min(), returns.max(), 200)
plt.figure(figsize=(10, 5))
plt.hist(returns, bins=80, density=True, color="steelblue", alpha=0.7, label="Empirical")
plt.plot(x, stats.norm.pdf(x, mu, sigma), color="red", linewidth=2, label="Normal fit")
plt.title(f"{NAME} — Return Distribution vs Normal Curve", fontsize=14)
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/3_return_distribution.png", dpi=150)
plt.show()
print("Step 3 done: distribution plotted.")

# =============================================================
# STEP 4 — Value at Risk (VaR)
# =============================================================
var95 = np.percentile(returns, 5)  * INVESTMENT
var99 = np.percentile(returns, 1)  * INVESTMENT

print(f"\n--- Value at Risk (on €{INVESTMENT:,}) ---")
print(f"VaR 95% (1 day): €{var95:,.2f}")
print(f"VaR 99% (1 day): €{var99:,.2f}")

plt.figure(figsize=(10, 5))
plt.hist(returns, bins=80, density=True, color="steelblue", alpha=0.7)
plt.axvline(np.percentile(returns, 5), color="orange", linestyle="--", linewidth=2, label=f"VaR 95% = €{var95:,.0f}")
plt.axvline(np.percentile(returns, 1), color="red",    linestyle="--", linewidth=2, label=f"VaR 99% = €{var99:,.0f}")
plt.title(f"{NAME} — Value at Risk", fontsize=14)
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/4_var.png", dpi=150)
plt.show()
print("Step 4 done: VaR plotted.")

# =============================================================
# STEP 5 — Monte Carlo Simulation (Geometric Brownian Motion)
# =============================================================
S0    = float(df.iloc[-1])   # last known price
dt    = 1                    # 1 trading day
drift = mu - 0.5 * sigma**2  # GBM drift term

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
print(f"Starting price    : €{S0:.2f}")
print(f"5th  percentile   : €{p5:.2f}  (worst scenario)")
print(f"50th percentile   : €{p50:.2f}  (median scenario)")
print(f"95th percentile   : €{p95:.2f}  (best scenario)")

# Chart 5a: Simulation paths
plt.figure(figsize=(14, 6))
plt.plot(simulations[:, :200], alpha=0.05, color="steelblue", linewidth=0.5)
plt.plot(np.percentile(simulations, 5,  axis=1), color="red",   linewidth=2, label="5th percentile")
plt.plot(np.percentile(simulations, 50, axis=1), color="black", linewidth=2, label="Median")
plt.plot(np.percentile(simulations, 95, axis=1), color="green", linewidth=2, label="95th percentile")
plt.axhline(S0, color="orange", linestyle="--", linewidth=1.5, label=f"Start price €{S0:.2f}")
plt.title(f"{NAME} — Monte Carlo Simulation ({SIMULATIONS} paths, {HORIZON} days)", fontsize=14)
plt.xlabel("Trading Days")
plt.ylabel("Simulated Price (EUR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/5a_monte_carlo_paths.png", dpi=150)
plt.show()
print("Step 5a done: simulation paths plotted.")

# Chart 5b: Distribution of final prices
plt.figure(figsize=(10, 5))
plt.hist(final_prices, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
plt.axvline(p5,  color="red",   linestyle="--", linewidth=2, label=f"5th pct: €{p5:.2f}")
plt.axvline(p50, color="black", linestyle="--", linewidth=2, label=f"Median:  €{p50:.2f}")
plt.axvline(p95, color="green", linestyle="--", linewidth=2, label=f"95th pct: €{p95:.2f}")
plt.axvline(S0,  color="orange",linestyle="--", linewidth=2, label=f"Start: €{S0:.2f}")
plt.title(f"{NAME} — Distribution of Simulated Prices after {HORIZON} days", fontsize=14)
plt.xlabel("Final Price (EUR)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/5b_final_price_distribution.png", dpi=150)
plt.show()
print("Step 5b done: final price distribution plotted.")

print("\n=== All steps completed. Check /output for all charts. ===")
