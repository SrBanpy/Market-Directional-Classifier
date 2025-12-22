# src/features.py
"""Feature engineering module for technical indicators."""
import pandas as pd
import numpy as np


class Features:
    """Generate technical indicators for price prediction."""
    
    def __init__(self, df):
        self.df = df.copy()

    def _calculate_rsi(self, series, period=14):
        """Calculate RSI for any data series."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def add_technical_indicators(self):
        """Add technical indicators to dataframe."""
        close = self.df['Close']

        # RSI
        self.df['rsi'] = self._calculate_rsi(close)

        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        self.df['bb_lower'] = sma_20 - (2 * std_20)
        self.df['bb_upper'] = sma_20 + (2 * std_20)
        self.df['bb_pct'] = (close - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])

        # Distance to moving averages
        self.df['dist_sma_50'] = (close / close.rolling(50).mean()) - 1
        self.df['dist_sma_200'] = (close / close.rolling(200).mean()) - 1

        # Volatility (simplified ATR)
        self.df['range'] = (self.df['High'] - self.df['Low']) / close
        self.df['atr_14'] = self.df['range'].rolling(14).mean()

        return self

    def add_macro_features(self):
        """Add macro indicator features."""
        if 'VIX_Close' in self.df.columns:
            self.df['vix_rsi'] = self._calculate_rsi(self.df['VIX_Close'])
        return self

    def lags(self):
        """Add lagged features to avoid look-ahead bias."""
        cols = ['rsi', 'bb_pct', 'dist_sma_200', 'atr_14']

        if 'VIX_Close' in self.df.columns:
            cols.extend(['VIX_Close', 'vix_rsi'])

        for col in cols:
            if col in self.df.columns:
                for lag in [1, 3, 5]:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        return self

    def apply_all(self):
        """Apply all feature generation steps."""
        self.add_technical_indicators()
        self.add_macro_features()
        self.lags()
        self.df.dropna(inplace=True)
        return self.df