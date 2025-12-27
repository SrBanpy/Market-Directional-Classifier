
import yfinance as yf
import pandas as pd
from src.config import TICKER, MACRO_TICKERS, START_DATE, INTERVAL


def get_market_data():
    """
    Downloads data for the main ticker and macro data (like VIX, VIX3M).
    Returns a clean DataFrame with all the necessary columns.
    """
    print(f"Downloading {TICKER} and Macro data ({', '.join(MACRO_TICKERS)})...")

    # 1. Main Ticker
    df = yf.download(TICKER, start=START_DATE, interval=INTERVAL, auto_adjust=True, progress=False)

    # Flatten MultiIndex if it exists (fix for new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Macro Data
    for mt in MACRO_TICKERS:
        try:
            macro = yf.download(mt, start=START_DATE, interval=INTERVAL, auto_adjust=True, progress=False)
            if isinstance(macro.columns, pd.MultiIndex):
                macro.columns = macro.columns.get_level_values(0)

            # Merge only the 'Close' column and rename it
            clean_name = mt.replace('^', '') + '_Close'
            if 'Close' in macro.columns and len(macro) > 0:
                df[clean_name] = macro['Close']
            else:
                print(f"{mt} has no 'Close' data")
        except Exception as e:
            print(f"Error loading {mt}: {e}")

    # 3. Calculate VIX Term Structure (Contango/Backwardation)
    if 'VIX_Close' in df.columns and 'VIX3M_Close' in df.columns:
        # Ratio > 1 = Calm market
        # Ratio < 1 = Panic/Stress
        df['VIX_Term_Structure'] = df['VIX_Close'] / df['VIX3M_Close']

    # Final cleanup
    initial_len = len(df)
    df.dropna(inplace=True)

    return df