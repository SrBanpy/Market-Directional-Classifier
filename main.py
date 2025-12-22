import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import get_market_data
from src.model import Model


def main():
    print("=" * 42)
    print("STARTING TRADING SYSTEM V6 (SNIPER)")
    print("=" * 42 + "\n")

    # Load data
    try:
        print("--- STEP 1: Downloading Data (Price + VIX) ---")
        raw_df = get_market_data()
        print(f"Data loaded: {len(raw_df)} candles.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # Initialize model
    trader = Model(raw_df)

    # Prepare data
    print("\n--- STEP 2: Feature Engineering and Labeling ---")
    trader.get_data()

    # Train
    print("\n--- STEP 3: Training (Pattern Detection) ---")
    trader.train()

    # Save
    print("\n--- STEP 4: Saving Model ---")
    if not os.path.exists('models'):
        os.makedirs('models')

    trader.save("models/sniper_v6.pkl")

    print("\nPROCESS COMPLETE. NOW RUN src/backtest.py")


if __name__ == "__main__":
    main()