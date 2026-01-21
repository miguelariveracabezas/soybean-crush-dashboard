import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def calculate_hurst(series):
    """
    Calculates the Hurst Exponent to determine the time series nature.
    H < 0.5: Mean Reverting (Ideal for StatArb)
    H = 0.5: Geometric Brownian Motion (Random Walk)
    H > 0.5: Trending
    """
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def backtest_crush_spread(df, spread_col='Spread', lookback=30, entry_z=2.0, cost_per_trade=0.02):
    """
    Performs a vectorized backtest on a mean-reversion strategy.
    
    Parameters:
    - df: DataFrame containing 'Date' and the spread data.
    - lookback: Rolling window for Mean/Std Dev calculation.
    - entry_z: Z-Score threshold for entry.
    - cost_per_trade: Transaction cost in $ per unit (slippage + commissions).
    """
    
    # Copy to avoid SettingWithCopy warnings
    data = df.copy().set_index('Date')
    data.sort_index(inplace=True)
    
    # ---------------------------------------------------------
    # 1. STATISTICAL VALIDATION
    # ---------------------------------------------------------
    print("--- 1. STATISTICAL VALIDATION ---")
    
    # ADF Test (Stationarity)
    adf_result = adfuller(data[spread_col])
    p_value = adf_result[1]
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-Value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(">> RESULT: Spread is STATIONARY (Good for StatArb)")
    else:
        print(">> RESULT: Spread is NON-STATIONARY (Risk of drift)")

    # Hurst Exponent (Mean Reversion)
    h_value = calculate_hurst(data[spread_col].values)
    print(f"Hurst Exponent: {h_value:.4f}")
    if h_value < 0.5:
        print(">> RESULT: Series is MEAN REVERTING")
    else:
        print(">> RESULT: Series is Trending or Random Walk")

    # ---------------------------------------------------------
    # 2. SIGNAL GENERATION (Vectorized)
    # ---------------------------------------------------------
    
    # Rolling Statistics
    data['mean'] = data[spread_col].rolling(window=lookback).mean()
    data['std'] = data[spread_col].rolling(window=lookback).std()
    
    # Calculate Z-Score
    data['z_score'] = (data[spread_col] - data['mean']) / data['std']
    
    # Logic:
    # Short Entry: Z > +2
    # Long Entry:  Z < -2
    # Exit:        Z crosses 0
    
    # We use a 'signal' column with NaN, then forward fill to maintain state
    data['signal'] = np.nan
    
    # Entry Signals
    data.loc[data['z_score'] > entry_z, 'signal'] = -1  # Short
    data.loc[data['z_score'] < -entry_z, 'signal'] = 1  # Long
    
    # Exit Signals (Crossing Zero)
    # We detect where Z-score sign changes from previous day
    # np.sign returns 1, -1, or 0. If product of today*yesterday is negative, it crossed 0.
    z_sign = np.sign(data['z_score'])
    cross_zero = (z_sign * z_sign.shift(1)) < 0
    data.loc[cross_zero, 'signal'] = 0 # Exit
    
    # Forward Fill to simulate "Holding" the position until next signal
    data['position'] = data['signal'].ffill().fillna(0)
    
    # ---------------------------------------------------------
    # 3. PnL & COSTS
    # ---------------------------------------------------------
    
    # Daily Price Change
    data['price_change'] = data[spread_col].diff()
    
    # Gross Returns
    # CRITICAL: Shift position by 1. We trade at Close, so today's return depends on Yesterday's position.
    data['gross_pnl'] = data['position'].shift(1) * data['price_change']
    
    # Calculate Trades (Change in position)
    # abs(1 - 0) = 1 trade. abs(1 - (-1)) = 2 trades (flip).
    data['trades'] = data['position'].diff().abs()
    
    # Apply Transaction Costs
    data['costs'] = data['trades'] * cost_per_trade
    
    # Net PnL
    data['net_pnl'] = data['gross_pnl'] - data['costs']
    data['cumulative_pnl'] = data['net_pnl'].cumsum()
    
    # ---------------------------------------------------------
    # 4. PERFORMANCE REPORTING
    # ---------------------------------------------------------
    print("\n--- 3. PERFORMANCE METRICS ---")
    
    total_return = data['cumulative_pnl'].iloc[-1]
    
    # Sharpe Ratio (Annualized, assuming 252 trading days)
    # Note: Since this is PnL ($) not % returns, Sharpe is technically Information Ratio here, 
    # but we calculate it on daily PnL volatility.
    daily_mean = data['net_pnl'].mean()
    daily_std = data['net_pnl'].std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std != 0 else 0
    
    # Max Drawdown
    rolling_max = data['cumulative_pnl'].cummax()
    drawdown = data['cumulative_pnl'] - rolling_max
    max_drawdown = drawdown.min()
    
    print(f"Total Return ($):   ${total_return:,.2f}")
    print(f"Sharpe Ratio:       {sharpe:.2f}")
    print(f"Max Drawdown ($):   ${max_drawdown:,.2f}")
    print(f"Total Trades:       {int(data['trades'].sum())}")
    
    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Spread and Bollinger Bands
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data[spread_col], label='Spread', color='#1E3A8A', lw=1.5)
    plt.plot(data.index, data['mean'] + (entry_z * data['std']), color='gray', linestyle='--', alpha=0.5, label='Upper Band')
    plt.plot(data.index, data['mean'] - (entry_z * data['std']), color='gray', linestyle='--', alpha=0.5, label='Lower Band')
    plt.title('Soybean Crush Spread & StatArb Bands', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative PnL (Equity Curve)
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['cumulative_pnl'], color='#10B981', lw=2)
    plt.fill_between(data.index, data['cumulative_pnl'], 0, color='#10B981', alpha=0.1)
    plt.title(f'Equity Curve (Net of ${cost_per_trade}/bu Costs)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylabel('Cumulative Profit ($)')
    
    plt.tight_layout()
    plt.show()
    
    return data

# ==========================================
# EXAMPLE USAGE (Generating Mock Data)
# ==========================================
if __name__ == "__main__":
    # Generate synthetic mean-reverting data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq='B')
    
    # Ornstein-Uhlenbeck process simulation (Mean Reverting)
    mu = 1.50 # Long term mean
    theta = 0.1
    sigma = 0.05
    prices = [1.50]
    
    for _ in range(len(dates)-1):
        prices.append(prices[-1] + theta*(mu - prices[-1]) + sigma*np.random.normal())
        
    df_mock = pd.DataFrame({'Date': dates, 'Spread': prices})
    
    # Run the Backtest
    results = backtest_crush_spread(df_mock, cost_per_trade=0.02)