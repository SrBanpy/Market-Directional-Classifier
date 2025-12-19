import yfinance as yf
from config import TICKER, START_DATE, INTERVAL

def get_preprocessed_market_data(ticker=TICKER, start=START_DATE, interval=INTERVAL):

    # auto_adjust=True es VITAL para ML.
    # Ajusta precios pasados por splits y dividendos automáticamente.
    df = yf.download(
        tickers=ticker,
        start=start,
        end=None,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    return df

df = get_preprocessed_market_data()
print(df.head())

