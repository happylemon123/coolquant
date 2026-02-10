# CoolQuant: Dynamic Graph Learning for Equity Forecasting

**Abstract**
CoolQuant acts as a "Senior Quant" portfolio demonstrator, explicitly addressing the four pillars often missing in academic Quantitative Finance programs: **Data Leakage Prevention**, **Cost Modeling**, **Production Code Quality**, and **Risk Governance**.

## The 4 Pillars of CoolQuant

### 1. Data Bias & Leakage Prevention
Academic models often fail in production due to "Look-ahead Bias". 
*   **Module**: `src.audit.leakage.LeakageAuditor`
*   **Function**: Runs statistical tests (correlation checks, timestamp verification) to prove input features at $t$ contain zero information from $t+k$.

### 2. Transaction Cost Modeling
Real-world alpha vanishes after costs. CoolQuant doesn't assume friction-less markets.
*   **Module**: `src.model.cost.CostModel`
*   **Function**: Simulates Bid-Ask spread limits and non-linear market impact (slippage) to penalize high-turnover strategies.

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
