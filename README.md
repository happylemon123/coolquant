# CoolQuant: Dynamic Graph Learning for Equity Forecasting

**Abstract**
CoolQuant acts as a "Senior Quant" portfolio demonstrator, explicitly addressing the four pillars often missing in academic Quantitative Finance programs: **Data Leakage Prevention**, **Cost Modeling**, **Production Code Quality**, and **Risk Governance**.

## The 4 Pillars of CoolQuant

### 1. Data Bias & Leakage Prevention
Academic models often fail in production due to "Look-ahead Bias". 
*   **Module**: `src.audit.leakage.LeakageAuditor`
*   **Function**: Runs statistical tests (correlation checks, timestamp verification) to prove input features at $t$ contain zero information from $t+k$.

### 2. Time-Series Volatility & Latency-Free Trend Discovery
Real-world alpha vanishes after costs, and high-frequency strategies are immediately penalized by poorly estimated volatility and lagged trend indicators. CoolQuant natively integrates two advanced statistical models to handle dynamic market conditions:

#### A. The Kalman Filter (Adaptive Price Discovery)
Standard Moving Averages (SMA/EMA) introduce unacceptable latency during regime changes. The **Kalman Filter** is implemented to dynamically estimate the "true" unobservable price.
*   **Mechanism**: It operates in a state-space formulation, continuously weighing the *certainty* of its current state prediction against the *noise* of new price observations. 
*   **Advantage**: By separating structural signal from random walk noise, it achieves latency-free trend discovery, allowing the model to react to true price reversals faster than lagging indicators.

#### B. The GARCH(1,1) Model (Conditional Volatility Clustering)
Unlike historical volatility (which is backward-looking and slow) or implied volatility (which requires options data), CoolQuant utilizes a **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** model.
*   **Mechanism**: Financial markets exhibit "Volatility Clustering"—large shocks are followed by large shocks. GARCH explicitly models this time-varying variance.
*   **Advantage**: When a market shock occurs, the GARCH conditional volatility spikes *immediately*. This allows downstream risk models to widen simulated bid-ask spreads instantly, protecting the portfolio from adverse selection during flash crashes.

### 3. Risk Governance
Strategies must survive "blow-up" events.
*   **Module**: `src.audit.risk.RiskGovernance`
*   **Function**: A hard-coded "Risk Gatekeeper" that rejects model-generated trades if they violate firm-wide constraints (Max Position, Drawdown Limits), mimicking an institutional Risk Manager.

### 4. Production Code Quality
Built as a Python package, not a notebook.
*   **Structure**: Modular design (`src/ features/`, `model/`) with Type Hinting.
*   **Architecture**: Spatiotemporal GNN (Graph Attention) built from scratch in PyTorch.

## Usage

### 1. Run Pipeline
```bash
python3 src/train.py
```
This script runs the `LeakageAuditor` first. If the audit fails, training is aborted.

### 2. Model Training
(Coming Soon: Implementation of the ST-GNN trainer)

## License
MIT
