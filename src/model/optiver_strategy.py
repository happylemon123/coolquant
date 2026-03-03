import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pykalman import KalmanFilter
from arch import arch_model

# ==========================================
# Optiver Competition: Kalman + GARCH Model
# ==========================================

def fetch_market_data(ticker="AAPL", period="6mo"):
    """Fetches real market data from Yahoo Finance."""
    data = yf.download(ticker, period=period)
    # yfinance returns a MultiIndex DataFrame if you request a single ticker now.
    # Let's extract just the Close and format it as a Series.
    if isinstance(data.columns, pd.MultiIndex):
        price = data['Close'][ticker]
    else:
        price = data['Close']
    
    price.name = 'Market Price'
    return price.dropna()

def apply_kalman_filter(prices):
    """
    Uses a Kalman Filter to estimate the 'True' price trend,
    filtering out the noise better than a Moving Average.
    """
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=prices.iloc[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    
    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(), index=prices.index, name='Kalman Trend')

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
    return pd.Series(volatility, index=returns.index, name='GARCH Volatility')

def create_interactive_dashboard(prices, kalman_trend, garch_vol, ticker="AAPL"):
    """Creates a beautiful, interactive HTML dashboard using Plotly."""
    
    # Calculate Trend Direction based on 5-day sustained momentum to filter out micro-fluctuations
    trend_diff = kalman_trend.diff(5)
    trend_diff = trend_diff.bfill() # Handle initial NaNs
    trend_state = np.where(trend_diff > 0, "Uptrend 🟢", "Downtrend 🔴")
    
    # Identify Reversal Points (where sustained trend changes)
    is_uptrend = trend_diff > 0
    reversals = is_uptrend != is_uptrend.shift(1)
    # Ignore the first few rows to prevent premature signaling
    reversals.iloc[0:5] = False 
    
    bullish_reversals = kalman_trend[reversals & is_uptrend]
    bearish_reversals = kalman_trend[reversals & ~is_uptrend]

    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{ticker} Price vs Kalman True Trend", "GARCH(1,1) Conditional Volatility"))

    # Top Plot: Prices
    fig.add_trace(go.Scatter(x=prices.index, y=prices, 
                             name="Market Price", 
                             mode='lines',
                             line=dict(color='rgba(150, 150, 150, 0.5)', width=1.5),
                             hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'), 
                  row=1, col=1)

    # Top Plot: Kalman Trend (with custom trend hover text)
    fig.add_trace(go.Scatter(x=kalman_trend.index, y=kalman_trend, 
                             name="Kalman Trend", 
                             mode='lines',
                             line=dict(color='#00F0FF', width=3),
                             customdata=trend_state,
                             hovertemplate='Date: %{x}<br>Kalman: %{y:.2f}<br>State: %{customdata}<extra></extra>'), 
                  row=1, col=1)

    # Top Plot: Bullish Reversal Markers
    fig.add_trace(go.Scatter(x=bullish_reversals.index, y=bullish_reversals,
                             name="Trend Up Indicator",
                             mode='markers',
                             marker=dict(symbol='triangle-up', size=12, color='#00FF00', line=dict(width=1, color='white')),
                             hovertemplate='<b>Bullish Reversal!</b><br>Trend changing to UP<extra></extra>'),
                  row=1, col=1)

    # Top Plot: Bearish Reversal Markers
    fig.add_trace(go.Scatter(x=bearish_reversals.index, y=bearish_reversals,
                             name="Trend Down Indicator",
                             mode='markers',
                             marker=dict(symbol='triangle-down', size=12, color='#FF0000', line=dict(width=1, color='white')),
                             hovertemplate='<b>Bearish Reversal!</b><br>Trend changing to DOWN<extra></extra>'),
                  row=1, col=1)

    # Bottom Plot: GARCH Volatility
    fig.add_trace(go.Scatter(x=garch_vol.index, y=garch_vol, 
                             name="Conditional Volatility", 
                             mode='lines',
                             fill='tozeroy',
                             line=dict(color='#FF003C', width=2),
                             hovertemplate='Date: %{x}<br>Volatility: %{y:.4f}<extra></extra>'), 
                  row=2, col=1)

    # Layout Aesthetics
    fig.update_layout(
        title_text=f"CoolQuant: Advanced Market Analytics for {ticker}",
        template="plotly_dark",
        height=800,
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(rangeslider=dict(visible=True))
    )

    # Specific axis styling
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=2, col=1)

    # Save to HTML
    fig.write_html("interactive_dashboard.html")
    print("\n✅ Success! Dashboard generated: open 'interactive_dashboard.html' in your browser.")

if __name__ == "__main__":
    ticker = "AAPL"
    
    # 1. Get Data
    print(f"Fetching real market data for {ticker}...")
    prices = fetch_market_data(ticker)
    returns = prices.pct_change().dropna()
    
    # 2. Kalman Filter (Trend)
    print("Running Kalman Filter algorithm...")
    kalman_trend = apply_kalman_filter(prices)
    
    # 3. GARCH Model (Volatility)
    print("Fitting GARCH(1,1) Model...")
    # GARCH needs aligned indices, which are 1 less than prices due to pct_change
    garch_vol = apply_garch_volatility(returns)
    
    # align slices
    p_slice = prices.iloc[1:]
    k_slice = kalman_trend.iloc[1:]
    
    # 4. Visualization
    create_interactive_dashboard(p_slice, k_slice, garch_vol, ticker)
    
    # Calculate final trend
    final_trend = "UP" if k_slice.iloc[-1] > k_slice.iloc[-2] else "DOWN"

    print("\n--- Final Model Stats ---")
    print(f"Latest Price ($): {p_slice.iloc[-1]:.2f}")
    print(f"True Kalman Est ($): {k_slice.iloc[-1]:.2f}")
    print(f"Current Signal:   Trend is going {final_trend}")
    print(f"Current Volatility:   {garch_vol.iloc[-1]:.4f}")
