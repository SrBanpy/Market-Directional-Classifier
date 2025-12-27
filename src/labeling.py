
import numpy as np
import pandas as pd
from src.config import PIVOT_WINDOW


class Labeler:
    """
    Data Labeler used to detect local 'bottoms' (floors) and 'tops' (roofs).

    Classes:
    - 0: Roof (Sell/Exit)
    - 1: Neutral (Hold/Wait)
    - 2: Floor (Buy/Enter)
    """

    def __init__(self, window=None):
        # The window size determines how many days we look around a specific date
        # to decide if it was a high or a low point.
        self.window = window or PIVOT_WINDOW

    def create_target(self, df):
        """
        Creates the labels (answers) for tops and bottoms using a centered window.

        NOTE: This method intentionally looks at 'future' data.
        We do this ONLY during training so the AI can learn what happened.
        In real production, the model predicts without seeing the future.
        """
        df = df.copy()

        # Start by assuming every day is 'Neutral' (1)
        df['Target'] = 1

        # We use a centered window to find local minimums and maximums.
        # This looks at X days before AND X days after the current date.
        window_size = self.window * 2 + 1

        # Calculate the lowest and highest price inside that window
        rolling_min = df['Close'].rolling(window=window_size, center=True).min()
        rolling_max = df['Close'].rolling(window=window_size, center=True).max()

        # Assign Labels:
        # If the current price is the absolute LOWEST in the window -> It's a Floor (Buy/2)
        condition_buy = (df['Close'] == rolling_min)

        # If the current price is the absolute HIGHEST in the window -> It's a Roof (Sell/0)
        condition_sell = (df['Close'] == rolling_max)

        df.loc[condition_buy, 'Target'] = 2
        df.loc[condition_sell, 'Target'] = 0

        # Remove rows where we couldn't calculate the window (at the start/end of data)
        df.dropna(subset=['Target'], inplace=True)

        # Ensure the target is an integer (whole number)
        df['Target'] = df['Target'].astype(int)

        # Check balance: Print how many Buys, Sells, and Neutrals we found
        print("\nLabel Distribution:")
        counts = df['Target'].value_counts().sort_index()
        labels = {0: 'Roof (Sell)', 1: 'Neutral', 2: 'Floor (Buy)'}
        for idx, count in counts.items():
            pct = count / len(df) * 100
            print(f"   {labels.get(idx, idx)}: {count:,} ({pct:.1f}%)")

        return df

    def create_target_with_magnitude(self, df, min_move_pct=0.03):
        """
        Alternative version: This only labels a point as a top or bottom
        if the price actually moves a significant amount afterwards.

        Args:
            min_move_pct: Minimum price movement required (default is 3%)
        """
        df = df.copy()
        df['Target'] = 1  # Default to Neutral

        window_size = self.window * 2 + 1

        rolling_min = df['Close'].rolling(window=window_size, center=True).min()
        rolling_max = df['Close'].rolling(window=window_size, center=True).max()

        # Calculate future return (to filter out small, useless movements)
        future_return = df['Close'].shift(-self.window) / df['Close'] - 1

        # Only label 'Buy' if it's a local low AND price goes up significantly later
        condition_buy = (df['Close'] == rolling_min) & (future_return > min_move_pct)

        # Only label 'Sell' if it's a local high AND price drops significantly later
        condition_sell = (df['Close'] == rolling_max) & (future_return < -min_move_pct)

        df.loc[condition_buy, 'Target'] = 2
        df.loc[condition_sell, 'Target'] = 0

        df.dropna(subset=['Target'], inplace=True)
        df['Target'] = df['Target'].astype(int)

        print("\nLabel Distribution (with magnitude filter):")
        counts = df['Target'].value_counts().sort_index()
        labels = {0: 'Roof (Sell)', 1: 'Neutral', 2: 'Floor (Buy)'}
        for idx, count in counts.items():
            pct = count / len(df) * 100
            print(f"   {labels.get(idx, idx)}: {count:,} ({pct:.1f}%)")

        return df