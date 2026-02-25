import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
import sys
import os

# Add the src directory to the path so we can import our models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.optiver_strategy import generate_market_data, apply_kalman_filter, apply_garch_volatility

class TestFinanceModels(unittest.TestCase):
    """
    Production test suite for the Quantitative Finance models.
    Ensures mathematical transformations run without crashing and 
    produce outputs of the correct shape and type.
    """

    def setUp(self):
        """Runs before every test. Sets up a small synthetic dataset."""
        self.n_steps = 100
        self.prices = generate_market_data(n_steps=self.n_steps)
        self.returns = self.prices.pct_change().dropna()

    def test_generate_market_data(self):
        """Test that synthetic market data generation works correctly."""
        self.assertIsInstance(self.prices, pd.Series, "Data should be a pandas Series")
        self.assertEqual(len(self.prices), self.n_steps, f"Length should be {self.n_steps}")
        self.assertFalse(self.prices.isna().any(), "Market data should not contain NaNs")

    def test_apply_kalman_filter(self):
        """Test that the Kalman Filter extracts a trend line successfully."""
        kalman_trend = apply_kalman_filter(self.prices)
        
        # Verify type and structure
        self.assertIsInstance(kalman_trend, pd.Series, "Output should be a pandas Series")
        self.assertEqual(len(kalman_trend), len(self.prices), "Output length should match input length")
        
        # Verify no missing data from filtering
        self.assertFalse(kalman_trend.isna().any(), "Kalman output should not contain NaNs")

    def test_apply_garch_volatility(self):
        """Test that the GARCH(1,1) model fits and returns conditional volatility."""
        garch_vol = apply_garch_volatility(self.returns)
        
        # Verify type and structure
        self.assertIsInstance(garch_vol, pd.Series, "Output should be a pandas Series")
        self.assertEqual(len(garch_vol), len(self.returns), "Output length should match input returns length")
        
        # Volatility must be strictly positive
        self.assertTrue((garch_vol > 0).all(), "Volatility must be strictly greater than zero")
        
        # Verify no missing data
        self.assertFalse(garch_vol.isna().any(), "GARCH output should not contain NaNs")

if __name__ == '__main__':
    unittest.main()
