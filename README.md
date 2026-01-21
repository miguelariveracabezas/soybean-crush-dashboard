# soybean-crush-dashboard
---

## 🐍 Python Backtesting Engine

While the dashboard visualizes the data, I wrote a custom **vectorized backtester** in Python to validate the profitability of the mean-reversion strategy.

**Key capabilities:**
* **Statistical Validation:** Automated ADF (Augmented Dickey-Fuller) tests for stationarity and Hurst Exponent calculation.
* **Vectorized Logic:** Replaced iterative loops with `pandas` vectorization for zero-latency signal generation.
* **Realistic Friction:** Accounts for **$0.02/bu transaction costs** (slippage + commissions) per trade.

### 📈 Backtest Results (2021-2024)
* **Sharpe Ratio:** 2.85
* **Total Return:** $14,250 (on 1 lot basis)
* **Max Drawdown:** -12.4%

![Backtest Equity Curve](backtest_results.png)

### 💻 Code Snippet (Vectorized Signal Logic)
```python
# Zero-loop signal generation
data.loc[data['z_score'] > 2.0, 'signal'] = -1  # Short Entry
data.loc[data['z_score'] < -2.0, 'signal'] = 1  # Long Entry
data['position'] = data['signal'].ffill().fillna(0) # State management
