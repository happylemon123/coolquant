class RiskGovernance:
    """
    Enforces 'Risk Governance' rules.
    In production, this module acts as a 'gatekeeper' that rejects trades
    if they violate firm-wide risk limits.
    """
    
    def __init__(self, max_position_size=100000, max_drawdown_pct=0.02):
        self.max_position_size = max_position_size
        self.max_drawdown_pct = max_drawdown_pct
        self.current_drawdown = 0.0

    def check_trade(self, current_position, trade_size, portfolio_value):
        """
        Returns True if the trade is allowed, False (and raises Alarm) if it violates risk.
        """
        projected_position = current_position + trade_size
        
        # 1. Position Limit Check
        if abs(projected_position) > self.max_position_size:
            print(f"[RISK REJECT] Trade size {trade_size} violates Position Limit ({self.max_position_size})")
            return False
            
        # 2. Drawdown Check (simplistic)
        # In real governance, this would check the live PnL against the high-water mark
        if self.current_drawdown > self.max_drawdown_pct:
            print(f"[RISK REJECT] Trading halted due to Drawdown ({self.current_drawdown:.2%})")
            return False
            
        return True
