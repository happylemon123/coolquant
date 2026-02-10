class CostModel:
    """
    Models realistic transaction costs, a critical component often missing 
    in academic Quant programs.
    """
    def __init__(self, half_spread_bps=2.0, non_linear_impact=None):
        """
        Args:
            half_spread_bps (float): Fixed cost per trade (bid-ask spread).
            non_linear_impact (func): Optional market impact function (sqrt law).
        """
        self.half_spread = half_spread_bps / 10000.0 # Bps to decimal
        self.non_linear_impact = non_linear_impact

    def estimate_cost(self, trade_amount, volatility=None):
        """
        Calculates the estimated cost of a trade.
        """
        curr_cost = abs(trade_amount) * self.half_spread
        
        if self.non_linear_impact and volatility:
            # Impact ~ sigma * sqrt(size / volume)
            # Simplified placeholder for the 'senior' touch
            pass
            
        return curr_cost
