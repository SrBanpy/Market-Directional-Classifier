"""
Unit tests for the Market Directional Classifier.
Covers: features, labeling, risk management, and metrics.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from features import Features
from labeling import Labeler
from risk_management import RiskManager, RiskConfig, calculate_var, calculate_cvar


class TestFeatures(unittest.TestCase):
    """Tests for the features module."""
    
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        price = 100 + np.cumsum(np.random.randn(300) * 0.5)
        
        self.df = pd.DataFrame({
            'Open': price + np.random.randn(300) * 0.2,
            'High': price + np.abs(np.random.randn(300)) * 0.5,
            'Low': price - np.abs(np.random.randn(300)) * 0.5,
            'Close': price,
            'Volume': np.random.randint(1000, 10000, 300),
            'VIX_Close': 15 + np.random.randn(300) * 5
        }, index=dates)
    
    def test_rsi_calculation(self):
        gen = Features(self.df)
        gen.add_technical_indicators()
        
        self.assertIn('rsi', gen.df.columns)
        rsi_valid = gen.df['rsi'].dropna()
        self.assertTrue((rsi_valid >= 0).all())
        self.assertTrue((rsi_valid <= 100).all())
    
    def test_bollinger_bands(self):
        gen = Features(self.df)
        gen.add_technical_indicators()
        
        self.assertIn('bb_lower', gen.df.columns)
        self.assertIn('bb_upper', gen.df.columns)
        
        valid = gen.df.dropna()
        self.assertTrue((valid['bb_lower'] < valid['bb_upper']).all())
    
    def test_apply_all(self):
        gen = Features(self.df)
        result = gen.apply_all()
        
        self.assertEqual(result.isna().sum().sum(), 0)
        self.assertTrue(len(result) < len(self.df))


class TestLabeler(unittest.TestCase):
    """Tests for the labeling module."""
    
    def setUp(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        price = np.zeros(100)
        for i in range(100):
            if i % 20 == 0:
                price[i] = 90
            elif i % 20 == 10:
                price[i] = 110
            else:
                price[i] = 100 + np.sin(i * 0.3) * 5
        
        self.df = pd.DataFrame({
            'Close': price,
            'Open': price,
            'High': price + 1,
            'Low': price - 1
        }, index=dates)
    
    def test_target_creation(self):
        labeler = Labeler()
        result = labeler.create_target(self.df.copy())
        
        self.assertIn('Target', result.columns)
        unique_targets = result['Target'].unique()
        for t in unique_targets:
            self.assertIn(t, [0, 1, 2])


class TestRiskManagement(unittest.TestCase):
    """Tests for the risk management module."""
    
    def setUp(self):
        self.config = RiskConfig(
            max_position_pct=1.0,
            kelly_fraction=0.5,
            stop_loss_pct=0.05,
            trailing_stop_pct=0.03,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.15
        )
        self.rm = RiskManager(self.config, initial_capital=10000)
    
    def test_kelly_criterion(self):
        kelly = self.rm.calculate_kelly_size(win_rate=0.6, avg_win=0.02, avg_loss=0.01)
        self.assertGreater(kelly, 0)
        self.assertLessEqual(kelly, self.config.max_position_pct)
    
    def test_stop_loss_long(self):
        self.rm.on_trade_entry(100, 1000)
        self.assertFalse(self.rm.check_stop_loss(96, 1))
        self.assertTrue(self.rm.check_stop_loss(94, 1))
    
    def test_trailing_stop(self):
        self.rm.on_trade_entry(100, 1000)
        self.rm.highest_since_entry = 110
        self.assertFalse(self.rm.check_trailing_stop(108, 1))
        self.assertTrue(self.rm.check_trailing_stop(106, 1))


class TestMetrics(unittest.TestCase):
    """Tests for metric calculations."""
    
    def test_sharpe_ratio(self):
        from backtest import calculate_sharpe_ratio
        
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)
        sharpe = calculate_sharpe_ratio(returns)
        
        self.assertGreater(sharpe, -5)
        self.assertLess(sharpe, 5)
    
    def test_calmar_ratio(self):
        from backtest import calculate_calmar_ratio
        
        calmar = calculate_calmar_ratio(total_return=20.0, max_drawdown=-10.0, years=2.0)
        self.assertGreater(calmar, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
