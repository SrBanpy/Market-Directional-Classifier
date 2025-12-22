# src/config.py
"""
System configuration module.
Loads from config.yaml if exists, otherwise uses defaults.
"""
import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    """Load configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None


_config = load_config()

# Market Data
if _config:
    TICKER = _config['data']['ticker']
    MACRO_TICKERS = _config['data']['macro_tickers']
    START_DATE = _config['data']['start_date']
    INTERVAL = _config['data']['interval']
else:
    TICKER = "^GSPC"
    MACRO_TICKERS = ["^VIX"]
    START_DATE = "2000-01-01"
    INTERVAL = "1d"

# Model
if _config:
    PIVOT_WINDOW = _config['model']['pivot_window']
    TEST_SIZE = _config['model']['test_size']
    XGBOOST_PARAMS = _config['model']['xgboost']
else:
    PIVOT_WINDOW = 10
    TEST_SIZE = 0.2
    XGBOOST_PARAMS = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'early_stopping_rounds': 50,
        'random_state': 42
    }

# Trading
if _config:
    COMMISSION = _config['trading']['commission']
    SLIPPAGE = _config['trading'].get('slippage', 0.0005)
    SIGNAL_PARAMS = _config['trading']['signals']
else:
    COMMISSION = 0.001
    SLIPPAGE = 0.0005
    SIGNAL_PARAMS = {
        'prob_buy_bull': 0.30,
        'prob_buy_bear': 0.60,
        'rsi_max_bull': 60,
        'rsi_max_bear': 30,
        'prob_sell_threshold': 0.50,
        'rsi_min_short': 45,
        'prob_exit': 0.40,
        'rsi_exit_bull': 80,
        'rsi_exit_bear': 70
    }

# Risk Management
if _config:
    RISK_PARAMS = _config['risk']
else:
    RISK_PARAMS = {
        'max_position_pct': 1.0,
        'kelly_fraction': 0.5,
        'stop_loss_pct': 0.05,
        'trailing_stop_pct': 0.03,
        'max_daily_loss_pct': 0.02,
        'max_drawdown_pct': 0.15
    }

# Metrics
if _config:
    RISK_FREE_RATE = _config['metrics']['risk_free_rate']
    TRADING_DAYS = _config['metrics']['trading_days']
else:
    RISK_FREE_RATE = 0.04
    TRADING_DAYS = 252

# Paths
if _config:
    MODEL_DIR = PROJECT_ROOT / _config['paths']['model_dir']
    ASSETS_DIR = PROJECT_ROOT / _config['paths']['assets_dir']
    LOGS_DIR = PROJECT_ROOT / _config['paths']['logs_dir']
else:
    MODEL_DIR = PROJECT_ROOT / "models"
    ASSETS_DIR = PROJECT_ROOT / "assets"
    LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [MODEL_DIR, ASSETS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)