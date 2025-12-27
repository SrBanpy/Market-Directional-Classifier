# src/config.py

# === DATA ===
TICKER = "^GSPC"
# VIX for volatility, VIX3M to analyze the term structure (market stress)
MACRO_TICKERS = ["^VIX", "^VIX3M"]

START_DATE = "2000-01-01"
INTERVAL = "1d"

# === MODEL ===
TEST_SIZE = 0.2
VALIDATION_SPLITS = 5  # Number of splits for Walk-Forward Validation

# PIVOT_WINDOW: How many days before and after a specific date
# the price must be the lowest to be considered a "Floor" (Bottom).
PIVOT_WINDOW = 10

# === TRADING STRATEGY ===
# Probability thresholds for entering or exiting a trade
DEFAULT_P_ENTRY = 0.35
DEFAULT_P_EXIT = 0.40

# RSI Limits (Relative Strength Index)
DEFAULT_RSI_BUY_MAX = 55
DEFAULT_RSI_BUY_MIN = 30
DEFAULT_RSI_SELL_MIN = 50
DEFAULT_RSI_EXIT_BULL = 75
DEFAULT_RSI_EXIT_BEAR = 65

# === RISK MANAGEMENT ===
COMMISSION = 0.001  # 0.1% fee per trade
RISK_FREE_RATE = 0.04  # 4% annual return (safe money)
TRADING_DAYS = 252

# Leverage levels for the strategy
LEVERAGE_BULL_FROM_FLOOR = 1.5  # Aggressive: When we detect a floor (bottom) in a bull market
LEVERAGE_NORMAL = 1.0           # Standard: Normal buying
LEVERAGE_CASH = 0.0             # Safety: Go to cash (sell all) when a roof (top) is detected

# Stops (Safety Nets)
MAX_DRAWDOWN_STOP = 0.15  # Stop trading completely if we lose more than 15% from the peak
TRAILING_STOP_PCT = 0.08  # Sell if price drops 8% from the recent high