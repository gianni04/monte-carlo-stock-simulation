# Monte Carlo Stock Simulation

> A step-by-step quantitative finance project simulating future stock prices using **Geometric Brownian Motion (GBM)** and **Monte Carlo methods**.  
> Built with Python. Default stock: **Ubisoft (UBI.PA)** — easily configurable for any ticker.

---

## Project Structure

```
monte-carlo-stock-simulation/
├── main.py              # Full pipeline (5 steps)
├── requirements.txt     # Python dependencies
├── output/              # Auto-generated charts and results
│   ├── 1_historical_prices.png
│   ├── 2_daily_returns.png
│   ├── 3_return_distribution.png
│   ├── 4_var.png
│   ├── 5a_monte_carlo_paths.png
│   └── 5b_final_price_distribution.png
└── README.md
```

---

## Step-by-Step Pipeline

The project follows a logical progression from raw data to final simulation:

### Step 1 — Historical Prices
**Chart:** `1_historical_prices.png`  
Download 5 years of daily close prices from Yahoo Finance using `yfinance`.  
This gives us the raw material to understand the stock's price history and general trend.

### Step 2 — Daily Returns
**Chart:** `2_daily_returns.png`  
Compute daily percentage returns:  
$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$  
Returns are the fundamental input for all risk and simulation calculations.  
The plot shows volatility clustering: calm periods followed by sharp spikes.

### Step 3 — Return Distribution & Normality Test
**Chart:** `3_return_distribution.png`  
Plot the empirical distribution of returns against a fitted normal curve.  
Key statistics computed:
- **Mean** (μ): average daily return
- **Std Dev** (σ): daily volatility
- **Skewness**: asymmetry of the distribution
- **Kurtosis**: fat tails vs. normal distribution
- **Shapiro-Wilk test**: formal normality test (p < 0.05 = not normal)  

Financial returns typically show **negative skew** and **excess kurtosis** (fat tails), meaning extreme losses are more frequent than a normal distribution would predict.

### Step 4 — Value at Risk (VaR)
**Chart:** `4_var.png`  
VaR quantifies the **maximum expected loss** over one day at a given confidence level.

| Metric | Formula | Meaning |
|--------|---------|--------|
| VaR 95% | 5th percentile × investment | 95% of days, loss ≤ this amount |
| VaR 99% | 1st percentile × investment | 99% of days, loss ≤ this amount |

Example: VaR 95% = -€200 means you won't lose more than €200 in a single day, 95% of the time.

### Step 5 — Monte Carlo Simulation (GBM)
**Charts:** `5a_monte_carlo_paths.png`, `5b_final_price_distribution.png`

Monte Carlo simulates **1,000 possible future price paths** over 252 trading days (1 year) using **Geometric Brownian Motion**:

$$S_{t+1} = S_t \cdot e^{\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} \cdot Z}$$

Where:
- $S_t$ = current stock price
- $\mu$ = mean daily return (drift)
- $\sigma$ = daily volatility
- $Z \sim \mathcal{N}(0,1)$ = random shock
- $\Delta t = 1$ trading day

**Chart 5a** shows all 1,000 simulated price paths with the 5th, 50th and 95th percentile bands.  
**Chart 5b** shows the distribution of all 1,000 final prices after 252 days, giving a probability range for where the stock could trade in one year.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

To change the stock, edit lines 12-13 in `main.py`:
```python
TICKER = "UBI.PA"   # replace with any Yahoo Finance ticker
NAME   = "Ubisoft"  # display name
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| TICKER | UBI.PA | Yahoo Finance ticker |
| START | 2020-01-01 | Historical data start |
| END | 2024-12-31 | Historical data end |
| SIMULATIONS | 1000 | Number of Monte Carlo paths |
| HORIZON | 252 | Days ahead to simulate (252 = 1 year) |
| INVESTMENT | 10,000 | EUR base for VaR calculation |

---

## Author

**Gianni Pilotti** — Finance Student, University of Luxembourg  
AMF Certified | [LinkedIn](https://linkedin.com/in/giannipilots) | [GitHub](https://github.com/gianni04)

---

*For educational purposes only. Not financial advice.*
