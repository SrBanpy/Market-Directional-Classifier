
import pandas as pd
import numpy as np


class Features:
    """
    Generator of technical and macro features for the trading model.
    It includes indicators for momentum, volatility, and market sentiment.
    """

    def __init__(self, df):
        self.df = df.copy()

    def _calculate_rsi(self, series, period=14):
        """Calculates the RSI of any data series in a robust way."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        """Calculates MACD and its histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def add_technical_indicators(self):
        """Basic technical indicators."""
        close = self.df['Close']

        # 1. RSI (Price) - multiple timeframes
        self.df['rsi'] = self._calculate_rsi(close, 14)
        self.df['rsi_fast'] = self._calculate_rsi(close, 7)
        self.df['rsi_slow'] = self._calculate_rsi(close, 21)

        # 2. Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        self.df['bb_lower'] = sma_20 - (2 * std_20)
        self.df['bb_upper'] = sma_20 + (2 * std_20)
        # Relative Position (0 = Bottom, 1 = Top)
        self.df['bb_pct'] = (close - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'] + 1e-9)

        # 3. Distance to Moving Averages
        self.df['dist_sma_20'] = (close / close.rolling(20).mean()) - 1
        self.df['dist_sma_50'] = (close / close.rolling(50).mean()) - 1
        self.df['dist_sma_200'] = (close / close.rolling(200).mean()) - 1

        # 4. Volatility (ATR)
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - close.shift(1))
        low_close = abs(self.df['Low'] - close.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr_14'] = true_range.rolling(14).mean() / close  # Normalized by price
        self.df['atr_7'] = true_range.rolling(7).mean() / close

        # 5. MACD
        macd, signal, hist = self._calculate_macd(close)
        self.df['macd_hist'] = hist / close  # Normalized

        # 6. Rate of Change (Momentum)
        self.df['roc_5'] = close.pct_change(5)
        self.df['roc_10'] = close.pct_change(10)
        self.df['roc_20'] = close.pct_change(20)

        return self

    def add_momentum_features(self):
        """Long-term momentum - crucial for detecting trends."""
        close = self.df['Close']

        # Momentum: monthly, quarterly, semi-annual, annual
        self.df['mom_21'] = (close / close.shift(21)) - 1  # ~1 month
        self.df['mom_63'] = (close / close.shift(63)) - 1  # ~3 months
        self.df['mom_126'] = (close / close.shift(126)) - 1  # ~6 months
        self.df['mom_252'] = (close / close.shift(252)) - 1  # ~1 year

        # Relative Momentum (are we accelerating or slowing down?)
        self.df['mom_accel'] = self.df['mom_21'] - self.df['mom_63']

        # Current Drawdown from High
        rolling_max = close.rolling(252).max()
        self.df['drawdown'] = (close / rolling_max) - 1

        # Rally from Low
        rolling_min = close.rolling(63).min()
        self.df['rally_from_low'] = (close / rolling_min) - 1

        return self

    def add_macro_features(self):
        """Macro features: VIX (Volatility Index), term structure, etc."""

        # VIX
        if 'VIX_Close' in self.df.columns:
            self.df['vix_rsi'] = self._calculate_rsi(self.df['VIX_Close'])
            self.df['vix_sma_ratio'] = self.df['VIX_Close'] / self.df['VIX_Close'].rolling(20).mean()
            # Normalized VIX (historical percentile)
            self.df['vix_percentile'] = self.df['VIX_Close'].rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

        # VIX Term Structure (already calculated in data_loader, but we add derivatives here)
        if 'VIX_Term_Structure' in self.df.columns:
            # < 0.9 = Strong Backwardation (Panic), > 1.1 = Strong Contango (Calm)
            self.df['vix_term_zscore'] = (
                                                 self.df['VIX_Term_Structure'] - self.df['VIX_Term_Structure'].rolling(
                                             63).mean()
                                         ) / (self.df['VIX_Term_Structure'].rolling(63).std() + 1e-9)

        return self

    def add_regime_features(self):
        """Features that identify the current market regime (Environment)."""
        close = self.df['Close']

        # Trend: price above 200-day Simple Moving Average (SMA)
        self.df['bull_regime'] = (close > close.rolling(200).mean()).astype(int)

        # Regime Volatility
        self.df['volatility_regime'] = self.df['atr_14'].rolling(20).mean()

        # Price-VIX Correlation (when very negative = stressed market)
        if 'VIX_Close' in self.df.columns:
            self.df['price_vix_corr'] = close.rolling(20).corr(self.df['VIX_Close'])

        return self

    def add_lags(self):
        """Adds 'lags' (past values) of important features to capture temporal patterns."""
        cols_to_lag = [
            'rsi', 'bb_pct', 'dist_sma_200', 'atr_14',
            'macd_hist', 'roc_10', 'mom_21'
        ]

        # Add VIX if it exists
        if 'VIX_Close' in self.df.columns:
            cols_to_lag.extend(['VIX_Close', 'vix_rsi'])

        if 'VIX_Term_Structure' in self.df.columns:
            cols_to_lag.append('VIX_Term_Structure')

        for col in cols_to_lag:
            if col in self.df.columns:
                for lag in [1, 3, 5]:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)

        return self

    def apply_all(self):
        """Applies all the feature generators."""
        self.add_technical_indicators()

        self.add_momentum_features()

        self.add_macro_features()

        self.add_regime_features()

        self.add_lags()

        # Remove empty rows generated by the calculations (rolling windows, lags, annual momentum)
        initial_len = len(self.df)
        self.df.dropna(inplace=True)

        return self.df