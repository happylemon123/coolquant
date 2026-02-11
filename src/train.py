import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from src.audit import LeakageAuditor
from src.audit.risk import RiskGovernance
from src.features.engineering import FeatureEngineer
from src.model.gnn import ST_GNN
from src.model.cost import CostModel

def generate_dummy_data(n_tickers=5, n_days=100):
    """
    Creates a fake dataset that mimics stock data.
    """
    data = []
    dates = pd.date_range(start='2023-01-01', periods=n_days)
    tickers = [f'TICKER_{i}' for i in range(n_tickers)]
    
    for ticker in tickers:
        # Random walk for price
        prices = 100 + np.cumsum(np.random.randn(n_days))
        df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_days)
        })
        data.append(df)
        
    full_df = pd.concat(data).reset_index(drop=True)
    return full_df

def main():
    print("--- CoolQuant: Starting Pipeline ---")
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    df = generate_dummy_data()
    print(f"    Loaded {len(df)} rows.")

    # 2. Feature Engineering
    print("\n[2] Feature Engineering...")
    fe = FeatureEngineer()
    # Process per ticker to avoid mixing data
    processed_dfs = []
    for ticker in df['Ticker'].unique():
        sub_df = df[df['Ticker'] == ticker].copy()
        sub_df = fe.calculate_returns(sub_df)
        sub_df = fe.add_rsi(sub_df)
        sub_df = fe.add_macd(sub_df)
        sub_df = fe.add_volatility_asymmetry(sub_df, window=10) # [NEW] Asymmetry Signal
        processed_dfs.append(sub_df)
    
    df_features = pd.concat(processed_dfs).sort_values('Date')
    df_features = fe.clean_features(df_features) # Drop NaNs from windows
    
    # Create a target (Next Day Return)
    df_features['Target'] = df_features.groupby('Ticker')['Log_Return'].shift(-1)
    df_features = df_features.dropna() # Drop last row which has no target
    
    print(f"    Features generated: {df_features.columns.tolist()}")

    # 3. Quant Audit & Governance Init
    print("\n[3] Running Quant Audit & Governance Setup...")
    auditor = LeakageAuditor(df_features)
    risk_manager = RiskGovernance(max_position_size=10000)
    cost_model = CostModel(half_spread_bps=5.0)
    
    # Check for look-ahead bias
    # We explicitly check if 'Close' predicts 'Target'. 
    # Since Target is derived from Next Close, current Close shouldn't "leak" perfectly 
    # but high correlation might be suspicious if we normalized wrong.
    auditor.check_lookahead(features=['RSI_14', 'MACD_12_26', 'Log_Return', 'Vol_Asymmetry_10'], target='Target')
    
    # 4. Prepare Tensors for GNN
    print("\n[4] Preparing Tensors...")
    tickers = df_features['Ticker'].unique()
    n_nodes = len(tickers)
    n_features = 5 # Log_Return, RSI, MACD, Signal, Asymmetry
    feature_cols = ['Log_Return', 'RSI_14', 'MACD_12_26', 'MACD_Signal_9', 'Vol_Asymmetry_10']
    
    # For this demo, we take a single "snapshot" (batch_size=1) of the latest T time steps
    T = 10 # Lookback window
    
    # Reshape: [Nodes, Time, Features]
    # We grab the last T days for all tickers
    snapshot_data = []
    for ticker in tickers:
        ticker_data = df_features[df_features['Ticker'] == ticker].iloc[-T:]
        if len(ticker_data) < T:
             continue
        snapshot_data.append(ticker_data[feature_cols].values)
        
    x = torch.tensor(np.array(snapshot_data), dtype=torch.float32) # [Nodes, T, Features]
    
    # Dummy Adjacency Matrix (Fully Connected for now)
    # in pro version, this comes from correlation matrix
    adj = torch.ones((n_nodes, n_nodes)) 
    
    # Dummy Target (Regression: Predict next return for each node)
    # We take the target from the last timestamp for each node
    targets = []
    for ticker in tickers:
        t = df_features[df_features['Ticker'] == ticker].iloc[-1]['Target']
        targets.append(t)
    y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)
    
    # 5. Model Training
    print("\n[5] Training GNN...")
    model = ST_GNN(n_features=5, n_hidden=16, n_classes=1, dropout=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        # Input: [Nodes, Time, Features] -> [5, 10, 4]
        # Adjacency: [Nodes, Nodes] -> [5, 5]
        out = model(x, adj) 
        
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        
    # 6. Trade Simulation (Governance Check)
    print("\n[6] Simulating Governance Checks...")
    # Simulate a trade based on model output (just a dummy check for now)
    dummy_trade_size = 500
    current_pos = 1000
    
    if risk_manager.check_trade(current_pos, dummy_trade_size, portfolio_value=100000):
        cost = cost_model.estimate_cost(dummy_trade_size)
        print(f"    Trade Approved. Estimated Cost: ${cost:.2f}")
    else:
        print("    Trade Rejected by Risk Manager.")

    print("\n[Success] Pipeline Verified.")

if __name__ == "__main__":
    main()
