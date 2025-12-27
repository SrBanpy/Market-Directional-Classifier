import os
import sys
from src.data_loader import get_market_data
from src.model import Model
from src.config import TICKER, MACRO_TICKERS, START_DATE, INTERVAL

# We ensure Python can find the 'src' folder.
# This is necessary so we can import our custom files like model.py and config.py without errors.
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    print("MARKET-DIRECTIONAL-CLASSIFIER")

    # 1. Load Data
    try:
        print("\nSTEP 1: Downloading Data")
        # We call the function to download the raw market data (prices, volume, etc.)
        raw_df = get_market_data()
        print(f"Data loaded: {len(raw_df):,} candles since {raw_df.index[0].strftime('%Y-%m-%d')}")
    except Exception as e:
        # If something goes wrong (like no internet or a bad API key),
        # we catch the error, print it, and stop the program safely.
        print(f"Fatal error downloading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Initialize Model
    # We create an instance of our Model class using the raw data we just downloaded.
    trader = Model(raw_df)

    # 3. Prepare Data (Features + Labeling)
    print("\nSTEP 2: Feature Engineering and Labeling")
    # This step calculates all the indicators (math) and decides what the 'correct'
    # move (buy/sell) was for previous dates so the AI can learn.
    trader.get_data()

    # 4. Train with Walk-Forward Validation
    print("\nSTEP 3: Training with Walk-Forward Validation")
    # We start the training process.
    # 'use_walk_forward=True' means we simulate learning over time (rolling window)
    # rather than just memorizing the past.
    trader.train()

    # 5. Save Model
    print("\nSTEP 4: Saving the Model")
    # We check if a 'models' folder exists. If not, we create it.
    if not os.path.exists('models'):
        os.makedirs('models')

    # We save the trained "brain" of the AI to a file so we can use it later
    # without having to train it again.
    trader.save("models/xgboost_trading_v5.pkl")

    print("TRAINING COMPLETED")
    print("Run: python src/backtest.py to see results")


if __name__ == "__main__":
    main()