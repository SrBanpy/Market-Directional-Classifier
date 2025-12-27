
import sys
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List

# Project Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(current_dir)

from data_loader import get_market_data
from features import Features
from model import TradingStrategy
from config import (
    TEST_SIZE, COMMISSION, RISK_FREE_RATE, TRADING_DAYS,
    LEVERAGE_BULL_FROM_FLOOR, LEVERAGE_NORMAL, LEVERAGE_CASH,
    MAX_DRAWDOWN_STOP, TRAILING_STOP_PCT
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(project_root, 'models', 'xgboost_trading_v5.pkl')


# ============================================================================
# ALL THE FINANCIAL METRICS
# ============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculates the Sharpe Ratio (return per unit of risk)."""
    if returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = returns - daily_rf
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)


def calculate_max_drawdown(equity: pd.Series) -> tuple:
    """Calculates the maximum drop in value (drawdown) and how long it lasted."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = drawdown.min()

    # Calculate how long the drawdown lasted
    in_drawdown = drawdown < 0
    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            max_duration = max(max_duration, current_duration)
            current_duration = 0

    return max_dd, max_duration


def calculate_trade_stats(positions: List[int], returns: np.ndarray) -> Dict:
    """Calculates statistics for individual trades (wins, losses, etc.)."""
    trades = []
    current_trade_return = 0
    in_trade = False

    for i in range(len(positions) - 1):
        pos = positions[i]
        ret = returns[i] if i < len(returns) else 0

        if pos != 0:
            current_trade_return += pos * ret
            in_trade = True
        elif in_trade:
            trades.append(current_trade_return)
            current_trade_return = 0
            in_trade = False

    if in_trade:
        trades.append(current_trade_return)

    if len(trades) == 0:
        return {'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0, 'total_trades': 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]

    return {
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_win': np.mean(wins) * 100 if wins else 0,
        'avg_loss': np.mean(losses) * 100 if losses else 0,
        'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        'total_trades': len(trades)
    }


# ============================================================================
# THE MAIN ENGINE (STRATEGY)
# ============================================================================

class TradingEngine:
    """The core engine that uses the model predictions to execute trades."""

    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy
        self.positions_log = []

    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        Runs the simulation using a 3-Level Strategy:
        - Level 0 (Cash): Sell/Exit when a market 'Roof' (Top) is detected.
        - Level 1 (Normal): Standard exposure.
        - Level 2 (Leveraged): Buy aggressive when a market 'Floor' (Bottom) is detected.
        """
        # Generate predictions from the AI model
        probs = self.strategy.predict_proba(df)
        prob_buy = probs[:, 2]  # Class 2 = Floor (Buy)
        prob_sell = probs[:, 0]  # Class 0 = Roof (Sell)

        closes = df['Close'].values
        rsis = df['rsi'].values if 'rsi' in df.columns else np.full(len(df), 50)
        dist_sma_200 = df['dist_sma_200'].values if 'dist_sma_200' in df.columns else np.zeros(len(df))

        # VIX indicators (Volatility)
        vix_term = df['VIX_Term_Structure'].values if 'VIX_Term_Structure' in df.columns else np.ones(len(df))

        # Tracking variables
        equity = [initial_capital]
        positions = []
        leverage_history = []
        daily_returns = []

        current_pos = 0  # 0 means cash, otherwise it stores the leverage amount
        peak_equity = initial_capital
        trailing_stop_price = 0

        # Load strategy parameters
        params = {
            'p_entry': self.strategy.p_entry,
            'p_exit': self.strategy.p_exit,
            'rsi_buy_max': self.strategy.rsi_buy_max,
            'rsi_buy_min': self.strategy.rsi_buy_min,
            'rsi_sell_min': self.strategy.rsi_sell_min,
            'rsi_exit_bull': self.strategy.rsi_exit_bull,
            'rsi_exit_bear': self.strategy.rsi_exit_bear
        }

        # Loop through each day
        for i in range(len(df) - 1):
            price = closes[i]
            next_price = closes[i + 1]
            p_b = prob_buy[i]
            p_s = prob_sell[i]
            rsi = rsis[i]
            is_bull = dist_sma_200[i] > 0
            vix_contango = vix_term[i] > 1.0  # Calm market
            vix_backwardation = vix_term[i] < 0.9  # Chaotic market

            mkt_ret = (next_price - price) / price
            new_leverage = current_pos

            # ================================================================
            # LOGIC FOR THE 3 LEVELS
            # ================================================================

            # LEVEL 0: Cash / Exit
            exit_to_cash = (
                    (p_s > params['p_exit'] and rsi > params['rsi_exit_bull']) or
                    (vix_backwardation and p_s > 0.3) or
                    (not is_bull and rsi > 70)
            )

            # LEVEL 2: Leveraged Entry (Aggressive)
            go_leveraged = (
                    (p_b > params['p_entry'] + 0.1 and rsi < params['rsi_buy_min']) or
                    (is_bull and p_b > params['p_entry'] and rsi < params['rsi_buy_max'] and vix_contango)
            )

            # LEVEL 1: Normal Entry
            go_normal = (
                    (p_b > params['p_entry'] and rsi < params['rsi_buy_max']) and
                    not go_leveraged and not exit_to_cash
            )

            # Determine new position based on current state
            if current_pos == 0:  # Currently out of market
                if go_leveraged:
                    new_leverage = LEVERAGE_BULL_FROM_FLOOR
                elif go_normal:
                    new_leverage = LEVERAGE_NORMAL

            elif current_pos > 0:  # Currently in a position
                if exit_to_cash:
                    new_leverage = LEVERAGE_CASH
                elif go_leveraged and current_pos < LEVERAGE_BULL_FROM_FLOOR:
                    new_leverage = LEVERAGE_BULL_FROM_FLOOR
                elif not go_leveraged and current_pos > LEVERAGE_NORMAL:
                    # Reduce leverage if the strong signal is gone
                    new_leverage = LEVERAGE_NORMAL

            # Trailing Stop (Protecting Gains)
            if current_pos > 0:
                if equity[-1] > peak_equity:
                    peak_equity = equity[-1]
                    trailing_stop_price = peak_equity * (1 - TRAILING_STOP_PCT)
                elif equity[-1] < trailing_stop_price:
                    new_leverage = LEVERAGE_CASH
                    peak_equity = equity[-1]  # Reset peak

            # Calculate daily return
            # We pay commission if we change our leverage level
            cost = COMMISSION * 2 if new_leverage != current_pos else 0
            step_ret = new_leverage * mkt_ret - cost

            new_equity = equity[-1] * (1 + step_ret)
            equity.append(new_equity)
            daily_returns.append(step_ret)
            positions.append(new_leverage)
            leverage_history.append(new_leverage)

            current_pos = new_leverage

        # Add final state
        positions.append(current_pos)
        leverage_history.append(current_pos)

        return {
            'equity': equity,
            'positions': positions,
            'leverage_history': leverage_history,
            'daily_returns': daily_returns,
            'dates': df.index.tolist()
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(df: pd.DataFrame, results: Dict, save_path: str):
    """Generates and saves the performance charts."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 1, 1, 1], hspace=0.3, wspace=0.25)

    # Colors
    GOLD = '#FFD700'
    GREEN = '#00FF7F'
    RED = '#FF4444'
    BLUE = '#4488FF'
    GRAY = '#888888'

    equity = results['equity']
    dates = df.index[:len(equity)]

    # Calculate Buy and Hold for comparison
    bh_equity = 10000 * (1 + df['Close'].pct_change().fillna(0)).cumprod()
    bh_equity = bh_equity.iloc[:len(equity)]

    # Graph 1: Equity Curves (Growth)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, bh_equity, color=GRAY, linestyle='--', alpha=0.7, linewidth=1.5, label='Buy and Hold')
    ax1.plot(dates, equity, color=GREEN, linewidth=2, label='3 Level AI Strategy')
    ax1.fill_between(dates, equity, bh_equity.values,
                     where=[e > b for e, b in zip(equity, bh_equity)],
                     color=GREEN, alpha=0.2)
    ax1.fill_between(dates, equity, bh_equity.values,
                     where=[e < b for e, b in zip(equity, bh_equity)],
                     color=RED, alpha=0.2)

    # Metrics for the title
    total_ret = (equity[-1] / equity[0] - 1) * 100
    bh_ret = (bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100

    returns_series = pd.Series(results['daily_returns'])
    sharpe = calculate_sharpe_ratio(returns_series)

    ax1.set_title(f'AI Strategy: {total_ret:.1f}% vs Buy & Hold: {bh_ret:.1f}% | Sharpe: {sharpe:.2f}',
                  fontsize=14, color=GOLD, fontweight='bold')
    ax1.set_ylabel('Capital ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Graph 2: Drawdown (Risk)
    ax2 = fig.add_subplot(gs[1, :])
    equity_series = pd.Series(equity, index=dates)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    ax2.fill_between(dates, drawdown, 0, color=RED, alpha=0.5)
    ax2.axhline(y=-MAX_DRAWDOWN_STOP * 100, color=GOLD, linestyle='--', alpha=0.7,
                label=f'Stop Loss {MAX_DRAWDOWN_STOP * 100:.0f}%')
    ax2.set_title('Drawdown (Decline from Peak)', fontsize=12, color='white')
    ax2.set_ylabel('DD %', fontsize=10)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Graph 3: Positions/Leverage
    ax3 = fig.add_subplot(gs[2, :])
    leverage = results['leverage_history'][:len(dates)]
    colors = [GREEN if l > 1 else (BLUE if l == 1 else (GRAY if l == 0 else RED)) for l in leverage]
    ax3.bar(dates, leverage, color=colors, alpha=0.7, width=1)
    ax3.axhline(y=1, color='white', linestyle='-', alpha=0.3)
    ax3.set_title('Exposure Level (0=Cash, 1=Normal, 1.5+=Leveraged)', fontsize=12, color='white')
    ax3.set_ylabel('Leverage', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Graph 4: Monthly Returns
    ax4 = fig.add_subplot(gs[3, 0])
    returns_with_dates = pd.Series(results['daily_returns'], index=dates[:len(results['daily_returns'])])
    try:
        monthly_returns = returns_with_dates.resample('ME').sum() * 100
        if len(monthly_returns) > 0:
            colors_monthly = [GREEN if r > 0 else RED for r in monthly_returns]
            ax4.bar(range(len(monthly_returns)), monthly_returns, color=colors_monthly, alpha=0.7)
            ax4.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    except Exception:
        pass
    ax4.set_title('Monthly Return (%)', fontsize=12, color='white')
    ax4.set_ylabel('%', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Graph 5: Return Distribution (Histogram)
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.hist(returns_series * 100, bins=50, color=BLUE, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax5.axvline(x=0, color='white', linestyle='-', alpha=0.5)
    ax5.axvline(x=returns_series.mean() * 100, color=GOLD, linestyle='--',
                label=f'Mean: {returns_series.mean() * 100:.3f}%')
    ax5.set_title('Daily Return Distribution', fontsize=12, color='white')
    ax5.set_xlabel('Return %', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.show()
    print(f"Graph saved in: {save_path}")


def print_summary(df: pd.DataFrame, results: Dict):
    """Prints the final report to the console."""
    equity = results['equity']
    returns_series = pd.Series(results['daily_returns'])
    years = len(df) / TRADING_DAYS

    # Strategy Metrics
    total_ret = (equity[-1] / equity[0] - 1) * 100
    annual_ret = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100
    sharpe = calculate_sharpe_ratio(returns_series)
    max_dd, dd_duration = calculate_max_drawdown(pd.Series(equity))

    # Buy & Hold Metrics (Benchmark)
    bh_returns = df['Close'].pct_change().dropna()
    bh_total = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    bh_annual = ((1 + bh_total / 100) ** (1 / years) - 1) * 100
    bh_sharpe = calculate_sharpe_ratio(bh_returns)
    bh_max_dd, _ = calculate_max_drawdown(df['Close'])

    # Trade Stats
    trade_stats = calculate_trade_stats(results['positions'], df['Close'].pct_change().fillna(0).values)

    # Average Exposure
    avg_leverage = np.mean([l for l in results['leverage_history'] if l >= 0])
    time_in_market = sum(1 for l in results['leverage_history'] if l > 0) / len(results['leverage_history']) * 100

    print("SUMMARY OF RESULTS")
    print(f"\n{'Metric':<30} {'Strategy':>18} {'Buy & Hold':>18}")
    print(f"{'Total Return':<30} {total_ret:>17.2f}% {bh_total:>17.2f}%")
    print(f"{'Annual Return':<30} {annual_ret:>17.2f}% {bh_annual:>17.2f}%")
    print(f"{'Sharpe Ratio':<30} {sharpe:>18.2f} {bh_sharpe:>18.2f}")
    print(f"{'Max Drawdown':<30} {max_dd:>17.2f}% {bh_max_dd:>17.2f}%")
    print(f"{'DD Duration (days)':<30} {dd_duration:>18} {'-':>18}")

    print(f"\n{'TRADING STATISTICS':<30}")
    print(f"{'Total Trades':<30} {trade_stats['total_trades']:>18}")
    print(f"{'Win Rate':<30} {trade_stats['win_rate']:>17.1f}%")
    print(f"{'Profit Factor':<30} {trade_stats['profit_factor']:>18.2f}")
    print(f"{'Avg Win':<30} {trade_stats['avg_win']:>17.2f}%")
    print(f"{'Avg Loss':<30} {trade_stats['avg_loss']:>17.2f}%")

    print(f"\n{'RISK MANAGEMENT':<30}")
    print(f"{'AVG LEVERAGE':<30} {avg_leverage:>17.2f}x")
    print(f"{'TIME IN MARKET':<30} {time_in_market:>17.1f}%")

    # Performance comparison
    outperformance = total_ret - bh_total
    if outperformance > 0:
        print(f"OUTPERFORMED THE MARKET: +{outperformance:.2f}% vs Buy & Hold")
    else:
        print(f"UNDERPERFORMED THE MARKET: {outperformance:.2f}% vs Buy & Hold")

    return {
        'total_return': total_ret,
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'bh_return': bh_total,
        'outperformance': outperformance
    }


# ============================================================================
# MAIN
# ============================================================================

def run_backtest():
    """Executes the complete backtest simulation."""

    # Load the Model
    strategy = TradingStrategy.load(MODEL_PATH)

    # Load Data and Create Features
    df = get_market_data()
    gen = Features(df)
    df = gen.apply_all()

    # Split Data for Testing (We test on data the model hasn't seen)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_test = df.iloc[split_idx:].copy()

    print(f"Test Period: {df_test.index[0].strftime('%Y-%m-%d')} to {df_test.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total Records: {len(df_test)}")

    # Run the Simulation
    engine = TradingEngine(strategy)
    results = engine.run_backtest(df_test)

    # Print Metrics
    metrics = print_summary(df_test, results)

    # Save Graph
    output_path = os.path.join(project_root, 'assets', 'backtest_results.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_results(df_test, results, output_path)

    return metrics


if __name__ == "__main__":
    run_backtest()