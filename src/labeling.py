# src/labeling.py
"""Labeling module for identifying market tops and bottoms."""
import numpy as np
import pandas as pd
from config import PIVOT_WINDOW


class Labeler:
    """Create target labels for price pivot points."""
    
    def __init__(self):
        self.window = PIVOT_WINDOW

    def create_target(self, df):
        """
        Create target labels: 0=Top (Sell), 1=Neutral, 2=Bottom (Buy).
        Uses rolling window to find local min/max.
        """
        df['Target'] = 1  # Initialize as neutral

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window * 2 + 1)

        rolling_min = df['Close'].rolling(window=indexer).min()
        rolling_max = df['Close'].rolling(window=indexer).max()

        condition_buy = (df['Close'] == rolling_min)
        condition_sell = (df['Close'] == rolling_max)

        df.loc[condition_buy, 'Target'] = 2
        df.loc[condition_sell, 'Target'] = 0

        df.dropna(subset=['Target'], inplace=True)

        print("Target distribution:")
        print(df['Target'].value_counts(normalize=True) * 100)

        return df