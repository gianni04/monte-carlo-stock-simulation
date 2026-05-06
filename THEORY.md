# Finance Theory — From Returns to Monte Carlo

> Complete mathematical documentation of every formula used in this project.  
> Written by Gianni Pilotti — Finance Student, University of Luxembourg.

---

## Table of Contents

1. [Daily Returns](#1-daily-returns)
2. [Annualized Return](#2-annualized-return)
3. [Volatility (Annualized)](#3-volatility-annualized)
4. [Skewness & Kurtosis](#4-skewness--kurtosis)
5. [Value at Risk (VaR)](#5-value-at-risk-var)
6. [Sharpe Ratio](#6-sharpe-ratio)
7. [Geometric Brownian Motion (GBM)](#7-geometric-brownian-motion-gbm)
8. [Monte Carlo Simulation](#8-monte-carlo-simulation)
9. [Reading the Results](#9-reading-the-results)

---

## 1. Daily Returns

### Definition
A daily return measures the **percentage change in price** between two consecutive trading days.

### Formula

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Or equivalently using log-returns (used in GBM):

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

### Variables
| Symbol | Meaning |
|--------|---------|
| $r_t$ | Return on day $t$ |
| $P_t$ | Closing price on day $t$ |
| $P_{t-1}$ | Closing price on the previous day |

### Example
If Ubisoft closes at €14.20 on Monday and €13.80 on Tuesday:

$$r = \frac{13.80 - 14.20}{14.20} = \frac{-0.40}{14.20} \approx -2.82\%$$

### In Python
```python
returns = df.pct_change().dropna()
```

### Why use returns instead of raw prices?
- Compare stocks of different price levels
- Check for stationarity (required for statistical models)
- Compute risk metrics consistently

---

## 2. Annualized Return

### Definition
Scales the average daily return to a **yearly equivalent**, assuming 252 trading days per year.

### Formula

$$\mu_{ann} = \bar{r} \times 252$$

Where:
$$\bar{r} = \frac{1}{N} \sum_{t=1}^{N} r_t$$

### Variables
| Symbol | Meaning |
|--------|---------|
| $\mu_{ann}$ | Annualized return |
| $\bar{r}$ | Mean daily return |
| $N$ | Total number of trading days in the sample |
| $252$ | Standard number of trading days per year |

### Example
If mean daily return $\bar{r} = 0.001$ (+0.1% per day):

$$\mu_{ann} = 0.001 \times 252 = 0.252 = +25.2\%$$

### In Python
```python
ann_return = returns.mean() * 252
```

### Why 252 and not 365?
Stock exchanges operate approximately 252 days per year once weekends and public holidays are removed. Using 365 would overestimate the return.

---

## 3. Volatility (Annualized)

### Definition
Volatility measures the **dispersion of returns** around their mean.  
High volatility = large price swings = higher risk.  
Low volatility = stable price = lower risk.

### Formula

**Step 1 — Daily standard deviation:**
$$\sigma_{day} = \sqrt{\frac{1}{N-1} \sum_{t=1}^{N} (r_t - \bar{r})^2}$$

**Step 2 — Annualize:**
$$\sigma_{ann} = \sigma_{day} \times \sqrt{252}$$

### Why multiply by √252 and not 252?
Variance scales linearly with time:
$$\text{Var}_{ann} = \text{Var}_{day} \times 252$$

Taking the square root to get standard deviation:
$$\sigma_{ann} = \sigma_{day} \times \sqrt{252}$$

### Example
If $\sigma_{day} = 0.0210$ (2.10% daily):
$$\sigma_{ann} = 0.0210 \times \sqrt{252} = 0.0210 \times 15.875 \approx 33.3\%$$

### In Python
```python
ann_vol = returns.std() * np.sqrt(252)
```

### Interpretation guide
| Annualized Volatility | Typical profile |
|----------------------|-----------------|
| < 15% | Low risk (blue chips, utilities) |
| 15% – 30% | Moderate risk (large caps) |
| 30% – 50% | High risk (tech, gaming) |
| > 50% | Very high risk (small caps, crypto) |

---

## 4. Skewness & Kurtosis

### Skewness
Measures the **asymmetry** of the return distribution.

$$\text{Skew} = \frac{1}{N} \sum_{t=1}^{N} \left(\frac{r_t - \bar{r}}{\sigma}\right)^3$$

| Value | Meaning |
|-------|---------|
| Skew = 0 | Symmetric (normal distribution) |
| Skew < 0 | **Negative skew**: more extreme losses than gains |
| Skew > 0 | Positive skew: more extreme gains than losses |

Most financial assets have **negative skew** — crashes are more violent than rallies.

### Kurtosis
Measures the **thickness of the tails** of the distribution.

$$\text{Kurt} = \frac{1}{N} \sum_{t=1}^{N} \left(\frac{r_t - \bar{r}}{\sigma}\right)^4 - 3$$

| Value | Meaning |
|-------|---------|
| Kurt = 0 | Normal distribution (mesokurtic) |
| Kurt > 0 | **Fat tails**: extreme events more frequent than normal |
| Kurt < 0 | Thin tails: extreme events less frequent than normal |

Financial returns almost always show **positive excess kurtosis** (leptokurtic), meaning Black Swan events are more common than a normal model predicts.

### In Python
```python
skew = returns.skew()
kurt = returns.kurtosis()  # excess kurtosis (already minus 3)
```

---

## 5. Value at Risk (VaR)

### Definition
VaR answers: **"What is the maximum loss I can expect over 1 day, with X% confidence?"**

We use the **historical simulation method**: look at the actual past distribution and read off the percentile.

### Formula

$$\text{VaR}_{95\%} = P_5(r) \times \text{Investment}$$
$$\text{VaR}_{99\%} = P_1(r) \times \text{Investment}$$

### Variables
| Symbol | Meaning |
|--------|---------|
| $P_5(r)$ | 5th percentile of daily returns |
| $P_1(r)$ | 1st percentile of daily returns |
| Investment | Portfolio value in EUR |

### Example
If the 5th percentile = $-0.0201$ and investment = €10,000:

$$\text{VaR}_{95\%} = -0.0201 \times 10{,}000 = -\text{€}201$$

**Interpretation:** 95% of the time, you will not lose more than €201 in a single trading day.  
Equivalently: there is a 5% probability of losing more than €201 in one day.

### In Python
```python
var95 = np.percentile(returns, 5) * INVESTMENT
var99 = np.percentile(returns, 1) * INVESTMENT
```

### VaR Limitations
- Does **not** tell you how bad the loss could be beyond the threshold (use CVaR/Expected Shortfall for that)
- Assumes the past distribution is representative of the future
- Does not capture liquidity risk or correlation breakdowns in crises

---

## 6. Sharpe Ratio

### Definition
The Sharpe Ratio measures the **return earned per unit of risk taken**.  
It answers: "Is the extra return worth the extra volatility?"

### Formula

$$\text{Sharpe} = \frac{\mu_{ann} - R_f}{\sigma_{ann}}$$

### Variables
| Symbol | Meaning |
|--------|---------|
| $\mu_{ann}$ | Annualized return of the asset |
| $R_f$ | Risk-free rate (3% = ECB reference rate) |
| $\sigma_{ann}$ | Annualized volatility |

### Example
If $\mu_{ann} = 25\%$, $R_f = 3\%$, $\sigma_{ann} = 30\%$:

$$\text{Sharpe} = \frac{0.25 - 0.03}{0.30} = \frac{0.22}{0.30} \approx 0.73$$

### Interpretation guide
| Sharpe Ratio | Meaning |
|-------------|---------|
| < 0 | Asset returns less than risk-free rate |
| 0 – 0.5 | Poor risk-adjusted return |
| 0.5 – 1.0 | Acceptable |
| 1.0 – 2.0 | Good |
| > 2.0 | Excellent (rare) |

### In Python
```python
sharpe = (ann_return - RISK_FREE) / ann_vol
```

---

## 7. Geometric Brownian Motion (GBM)

### Definition
GBM is the mathematical model underlying Monte Carlo simulation.  
It describes how a stock price evolves **randomly over time**, assuming:
- Returns are log-normally distributed
- Price changes are continuous
- No sudden jumps or crashes

### Continuous formula
$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

Where $dW_t$ is a Wiener process (random Brownian motion).

### Discrete formula (used in code)
$$S_{t+1} = S_t \cdot e^{\left(\mu - \frac{\sigma^2}{2}\right) \Delta t + \sigma \sqrt{\Delta t} \cdot Z}$$

### Variables
| Symbol | Meaning |
|--------|---------|
| $S_t$ | Stock price at time $t$ |
| $\mu$ | Mean daily return (drift) |
| $\sigma$ | Daily volatility |
| $\Delta t$ | Time step (1 trading day) |
| $Z \sim \mathcal{N}(0,1)$ | Random standard normal shock |
| $e$ | Euler's number |

### Why $\mu - \frac{\sigma^2}{2}$?
This correction (Itô's lemma) adjusts the drift to account for the difference between log-returns and arithmetic returns.  
Without it, the simulation would **overestimate** expected prices on average due to Jensen's inequality.

### In Python
```python
drift = mu - 0.5 * sigma**2
shocks = np.random.normal(0, 1, HORIZON)
path = S0 * np.exp(np.cumsum(drift + sigma * shocks))
```

---

## 8. Monte Carlo Simulation

### Concept
Repeat the GBM process **1,000 times** with different random shocks each time.  
This produces 1,000 possible future price paths → a **probability distribution of future prices**.

### Algorithm
```
1. Start from last known price S0
2. For each simulation i = 1 to 1000:
   a. Draw 252 random shocks Z ~ N(0,1)
   b. Apply GBM formula day by day
   c. Store the full price path
3. After 1000 simulations:
   a. Compute 5th, 50th, 95th percentile of final prices
   b. Plot all paths + percentile bands
   c. Plot histogram of final prices
```

### Reading Chart 5a — Simulation Paths
| Element | Meaning |
|---------|---------|
| Blue lines | 200 individual price paths (out of 1000) |
| Red line | 5th percentile — worst 5% of scenarios |
| Black line | Median — most likely outcome |
| Green line | 95th percentile — best 5% of scenarios |
| Orange dashed | Starting price $S_0$ |

### Reading Chart 5b — Final Price Distribution
- Bulk of distribution **below** starting price → bearish signal
- Bulk of distribution **above** starting price → bullish signal
- **Wide distribution** = high uncertainty = high risk
- **Narrow distribution** = more predictable outcome

### In Python
```python
for i in range(SIMULATIONS):
    shocks = np.random.normal(0, 1, HORIZON)
    path = S0 * np.exp(np.cumsum(drift * dt + sigma * np.sqrt(dt) * shocks))
    simulations[:, i] = path
```

### Limitations of Monte Carlo GBM
| Limitation | Explanation |
|------------|-------------|
| Assumes log-normal returns | Real returns have fat tails (see kurtosis) |
| Constant volatility | Real volatility clusters (GARCH effect) |
| No jumps | Real stocks can gap down overnight |
| Based on historical data | Past ≠ future |

---

## 9. Reading the Results

### Full output summary
| Output | What it tells you |
|--------|------------------|
| Annualized return | How much the stock returned per year on average |
| Annualized volatility | How risky the stock is |
| Sharpe ratio | Is the risk worth taking vs. risk-free rate? |
| VaR 95% | Max daily loss 95% of the time |
| VaR 99% | Max daily loss 99% of the time |
| MC median price | Most likely price in 1 year |
| MC 5th percentile | Price in worst 5% of scenarios |
| MC 95th percentile | Price in best 5% of scenarios |

### Practical example with Ubisoft
If results show:
- Annualized return: **-26%** → stock has been declining
- Volatility: **50%** → extremely risky
- Sharpe: **-0.58** → return below risk-free rate, bad risk-adjusted return
- VaR 95%: **-€440** → on a bad day you can lose €440 per €10,000 invested
- MC median in 1 year: **€8.50** (vs. start €11.20) → simulation predicts further decline

This analysis helps an investor decide whether to buy, hold, or avoid a stock.

---

## Author

**Gianni Pilotti** — Finance Student, University of Luxembourg  
AMF Certified | [LinkedIn](https://linkedin.com/in/giannipilots) | [GitHub](https://github.com/gianni04)

---
*For educational purposes only. Not financial advice.*
