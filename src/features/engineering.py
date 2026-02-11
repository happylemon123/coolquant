import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Generates technical indicators with strict adherence to 
    causal windows to prevent look-ahead bias.
    """
    
    def __init__(self, use_log_returns: bool = True):
        self.use_log_returns = use_log_returns

    def calculate_returns(self, df: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        """
        Calculates daily returns.
        """
        df = df.copy()
        if self.use_log_returns:
            df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        else:
            df['Pct_Return'] = df[price_col].pct_change()
        return df

    def add_rsi(self, df: pd.DataFrame, price_col: str = 'Close', window: int = 14) -> pd.DataFrame:
        """
        Relative Strength Index (RSI).
        Note: The first 'window' values will be NaN.
        """
        df = df.copy()
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        return df

    def add_macd(self, df: pd.DataFrame, price_col: str = 'Close', 
                 fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD).
        """
        df = df.copy()
        exp1 = df[price_col].ewm(span=fast, adjust=False).mean()
        exp2 = df[price_col].ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        df[f'MACD_{fast}_{slow}'] = macd
        df[f'MACD_Signal_{signal}'] = signal_line
        df[f'MACD_Hist_{signal}'] = macd - signal_line
        return df

    def add_volatility_asymmetry(self, df: pd.DataFrame, price_col: str = 'Close', window: int = 10, sigma: float = 3.0) -> pd.DataFrame:
        """
        Detects Asymmetric Volatility opportunities (User-requested Signal).
        Logic: If current return > 3 * Rolling Volatility, flag as 1.
        """
        df = df.copy()
        # Ensure we have returns; if not calculated yet, do it temporarily
        if 'Pct_Return' not in df.columns:
            # We use Pct_Change for this specific logic as per request (or convert log returns)
            # User code: data['returns'] = data['price'].pct_change()
            returns = df[price_col].pct_change()
        else:
            returns = df['Pct_Return']
            
        # User Code: data['volatility'] = data['returns'].rolling(window=10).std()
        volatility = returns.rolling(window=window).std()
        
        # User Code: data['is_asymmetric'] = np.where(data['returns'].abs() > 3 * data['volatility'], 1, 0)
        # We handle NaN at start
        is_asymmetric = np.where(returns.abs() > (sigma * volatility), 1.0, 0.0)
        
        df[f'Vol_Asymmetry_{window}'] = is_asymmetric
        return df

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops NaN values created by rolling windows.
        CRITICAL: Validates that we aren't dropping 90% of data.
        """
        initial_len = len(df)
        df_clean = df.dropna()
        final_len = len(df_clean)
        
        if final_len < initial_len * 0.5:
             print(f"Warning: Dropped {initial_len - final_len} rows ({(1 - final_len/initial_len):.1%}). Check window sizes.")
             
        return df_clean
