import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from arch import arch_model

# ==========================================
# Optiver Competition: Kalman + GARCH Model
# ==========================================
# Recreated based on "Advanced Finance Model Visualization" project.

def generate_market_data(n_steps=1000):
    """Generates synthetic price data with changing trends."""
    np.random.seed(42)
    t = np.linspace(0, 100, n_steps)
    # True trend: Sine wave
    trend = 100 + 10 * np.sin(t)
    # Noise: Random walk
    noise = np.random.normal(0, 1, n_steps)
    price = trend + np.cumsum(noise)
    dates = pd.date_range(start='2024-01-01', periods=n_steps, freq='D')
    return pd.Series(price, index=dates)

def apply_kalman_filter(prices):
    """
    Uses a Kalman Filter to estimate the 'True' price trend,
    filtering out the noise better than a Moving Average.
    """
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    
    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(), index=prices.index)

def apply_garch_volatility(returns):
    """
    Uses a GARCH(1,1) model to estimate time-varying volatility.
    Optiver cares deeply about Volatility (Vega).
    """
    # Rescale returns for numerical stability
    am = arch_model(returns * 100, vol='Garch', p=1, o=0, q=1)
    res = am.fit(disp='off')
    # Convert back to standard scale
    volatility = res.conditional_volatility / 100
    return volatility

# 1. Get Data
print("Generating Market Data...")
prices = generate_market_data()
returns = prices.pct_change().dropna()

# 2. Kalman Filter (Trend)
print("Running Kalman Filter...")
kalman_trend = apply_kalman_filter(prices)

# 3. GARCH Model (Volatility)
print("Fitting GARCH Model...")
garch_vol = apply_garch_volatility(returns)

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top: Price vs Kalman
ax1.plot(prices.index, prices, label='Market Price (Noisy)', alpha=0.5, color='gray')
ax1.plot(kalman_trend.index, kalman_trend, label='Kalman Trend (True Value)', color='blue', linewidth=2)
ax1.set_title("Price Discovery: Kalman Filter vs Noise")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: GARCH Volatility
ax2.plot(garch_vol.index, garch_vol, label='GARCH(1,1) Volatility', color='red')
ax2.set_title("Risk Modeling: Dynamic Volatility Estimator")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- Model Stats ---")
print(f"Final Price: {prices.iloc[-1]:.2f}")
print(f"Kalman Est:  {kalman_trend.iloc[-1]:.2f}")
print(f"Current Vol: {garch_vol.iloc[-1]:.4f}")
