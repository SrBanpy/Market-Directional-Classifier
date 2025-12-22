# src/data_loader.py
"""Data loading module for market data."""
import yfinance as yf
import pandas as pd
from config import TICKER, MACRO_TICKERS, START_DATE, INTERVAL


def get_market_data():
    """Download market data including price and macro indicators."""
    print(f"Downloading {TICKER} and macro data...")

    # Main ticker
    df = yf.download(TICKER, start=START_DATE, interval=INTERVAL, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Macro data (VIX)
    for mt in MACRO_TICKERS:
        macro = yf.download(mt, start=START_DATE, interval=INTERVAL, auto_adjust=True, progress=False)
        if isinstance(macro.columns, pd.MultiIndex):
            macro.columns = macro.columns.get_level_values(0)

        clean_name = mt.replace('^', '') + '_Close'
        df[clean_name] = macro['Close']

    df.dropna(inplace=True)
    return df