"""
Professional backtest for the Sniper Anti-Crash model.
Includes advanced metrics: Sharpe, Sortino, Calmar, Win Rate.
"""
import sys
import os
import warnings

# Suppress all warnings before importing matplotlib
warnings.filterwarnings('ignore')

import joblib
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(current_dir)

from data_loader import get_market_data
from features import Features
from config import TEST_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(project_root, 'models', 'sniper_v6.pkl')
COMMISSION = 0.001  # 0.1% per trade
RISK_FREE_RATE = 0.04  # 4% annual (T-Bills approximation)
TRADING_DAYS = 252


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Annualized Sharpe Ratio
    """
    if returns.std() == 0:
        return 0.0
    
    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = returns - daily_rf
    
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calculate annualized Sortino Ratio (penalizes only downside volatility).
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Annualized Sortino Ratio
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = returns - daily_rf
    
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt((downside_returns ** 2).mean())
    
    return (excess_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS)


def calculate_calmar_ratio(total_return: float, max_drawdown: float, years: float) -> float:
    """
    Calculate Calmar Ratio (annualized return / max drawdown).
    
    Args:
        total_return: Total return in percentage
        max_drawdown: Maximum drawdown in percentage (negative number)
        years: Number of years in period
        
    Returns:
        Calmar Ratio
    """
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0
    
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    
    return annualized_return / abs(max_drawdown)


def calculate_win_rate(positions: List[int], returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate winning/losing trade statistics.
    
    Args:
        positions: List of positions (0, 1, -1)
        returns: Array of market returns
        
    Returns:
        Dict with win_rate, avg_win, avg_loss, profit_factor
    """
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
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean(wins) * 100 if wins else 0
    avg_loss = np.mean(losses) * 100 if losses else 0
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_trades': len(trades)
    }


def run_long_short_backtest():
    """Execute complete backtest with all metrics."""
    print("=" * 50)
    print("LONG/SHORT STRATEGY (ANTI-CRASH) - PRO")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return None

    # Load data and model
    logger.info("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df = get_market_data()
    gen = Features(df)
    df = gen.apply_all()

    # Test split
    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_test = df.iloc[split_idx:].copy()
    
    # Calculate years in test period
    years_in_test = len(df_test) / TRADING_DAYS

    # Features
    cols_drop = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close', 'TNX_Close']
    X_test = df_test.drop(columns=[c for c in cols_drop if c in df_test.columns])

    # AI signals
    logger.info("Generating model signals...")
    probs = model.predict_proba(X_test)
    prob_buy = probs[:, 2]  # Bottom
    prob_sell = probs[:, 0]  # Top

    # Long/Short simulation
    positions = []
    current_pos = 0  # 0=Cash, 1=Long, -1=Short

    closes = df_test['Close'].values
    rsis = df_test['rsi'].values
    dist_sma_200 = df_test['dist_sma_200'].values
    dates = df_test.index

    long_dates, long_prices = [], []
    short_dates, short_prices = [], []
    close_dates, close_prices = [], []

    equity = [1000.0]
    daily_returns_strat = []

    for i in range(len(df_test) - 1):
        price = closes[i]
        date = dates[i]
        p_b = prob_buy[i]
        p_s = prob_sell[i]
        rsi = rsis[i]
        is_bull = dist_sma_200[i] > 0

        mkt_ret_next = (closes[i + 1] - closes[i]) / closes[i]

        new_pos = current_pos

        # Entry rules
        want_long = (is_bull and p_b > 0.30 and rsi < 60) or \
                    (not is_bull and p_b > 0.60 and rsi < 30)

        want_short = (not is_bull) and (p_s > 0.50) and (rsi > 45)

        # Position management
        if current_pos == 0:
            if want_long:
                new_pos = 1
                long_dates.append(date)
                long_prices.append(price)
            elif want_short:
                new_pos = -1
                short_dates.append(date)
                short_prices.append(price)

        elif current_pos == 1:
            exit_limit = 80 if is_bull else 70
            if p_s > 0.40 or rsi > exit_limit:
                new_pos = 0
                close_dates.append(date)
                close_prices.append(price)

        elif current_pos == -1:
            if p_b > 0.40 or rsi < 30:
                new_pos = 0
                close_dates.append(date)
                close_prices.append(price)
            elif is_bull:
                new_pos = 0
                close_dates.append(date)
                close_prices.append(price)

        # Capital calculation
        cost = COMMISSION * 2 if new_pos != current_pos else 0
        step_ret = new_pos * mkt_ret_next
        
        daily_returns_strat.append(step_ret - cost)
        new_equity = equity[-1] * (1 + step_ret - cost)
        equity.append(new_equity)
        positions.append(new_pos)
        current_pos = new_pos

    positions.append(current_pos)

    # Advanced metrics
    df_test['Equity_Strat'] = equity
    df_test['Equity_Market'] = 1000.0 * (1 + df_test['Close'].pct_change().fillna(0)).cumprod()

    strat_returns = pd.Series(daily_returns_strat)
    mkt_returns = df_test['Close'].pct_change().dropna()

    # Drawdown
    peak = df_test['Equity_Strat'].cummax()
    dd = (df_test['Equity_Strat'] - peak) / peak * 100
    max_dd = dd.min()

    mkt_peak = df_test['Equity_Market'].cummax()
    mkt_dd = (df_test['Equity_Market'] - mkt_peak) / mkt_peak * 100
    mkt_max_dd = mkt_dd.min()

    total_ret = (equity[-1] / 1000 - 1) * 100
    mkt_ret = (df_test['Equity_Market'].iloc[-1] / 1000 - 1) * 100

    # Calculate ratios
    sharpe_strat = calculate_sharpe_ratio(strat_returns)
    sharpe_mkt = calculate_sharpe_ratio(mkt_returns)
    
    sortino_strat = calculate_sortino_ratio(strat_returns)
    sortino_mkt = calculate_sortino_ratio(mkt_returns)
    
    calmar_strat = calculate_calmar_ratio(total_ret, max_dd, years_in_test)
    calmar_mkt = calculate_calmar_ratio(mkt_ret, mkt_max_dd, years_in_test)
    
    mkt_returns_arr = df_test['Close'].pct_change().fillna(0).values
    trade_stats = calculate_win_rate(positions, mkt_returns_arr)

    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    print(f"\n{'Metric':<25} {'Strategy':>15} {'Market':>15}")
    print("-" * 55)
    print(f"{'Total Return':<25} {total_ret:>14.2f}% {mkt_ret:>14.2f}%")
    print(f"{'Max Drawdown':<25} {max_dd:>14.2f}% {mkt_max_dd:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {sharpe_strat:>15.2f} {sharpe_mkt:>15.2f}")
    print(f"{'Sortino Ratio':<25} {sortino_strat:>15.2f} {sortino_mkt:>15.2f}")
    print(f"{'Calmar Ratio':<25} {calmar_strat:>15.2f} {calmar_mkt:>15.2f}")
    
    print(f"\nTRADING STATISTICS:")
    print("-" * 40)
    print(f"{'Total Trades':<25} {trade_stats['total_trades']:>15}")
    print(f"{'Win Rate':<25} {trade_stats['win_rate']:>14.1f}%")
    print(f"{'Avg Win':<25} {trade_stats['avg_win']:>14.2f}%")
    print(f"{'Avg Loss':<25} {trade_stats['avg_loss']:>14.2f}%")
    print(f"{'Profit Factor':<25} {trade_stats['profit_factor']:>15.2f}")
    
    print(f"\nTest Period: {df_test.index[0].strftime('%Y-%m-%d')} - {df_test.index[-1].strftime('%Y-%m-%d')} ({years_in_test:.1f} years)")

    # Professional chart
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # Panel 1: Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title(
        f'AI Long/Short Strategy | Return: {total_ret:.1f}% vs Market: {mkt_ret:.1f}%\n'
        f'Sharpe: {sharpe_strat:.2f} | Sortino: {sortino_strat:.2f} | Max DD: {max_dd:.1f}%',
        fontsize=14, color='gold', fontweight='bold'
    )
    ax1.plot(df_test.index, df_test['Equity_Market'], color='gray', linestyle='--', 
             alpha=0.6, linewidth=1.5, label=f'Market (Sharpe: {sharpe_mkt:.2f})')
    ax1.plot(df_test.index, df_test['Equity_Strat'], color='#00ff00', 
             linewidth=2, label=f'Strategy (Sharpe: {sharpe_strat:.2f})')

    pos_arr = np.array(positions)
    ax1.fill_between(df_test.index, df_test['Equity_Strat'].min(), df_test['Equity_Strat'].max(),
                     where=(pos_arr == -1), color='red', alpha=0.15, label='Short Zone')
    ax1.fill_between(df_test.index, df_test['Equity_Strat'].min(), df_test['Equity_Strat'].max(),
                     where=(pos_arr == 1), color='green', alpha=0.1, label='Long Zone')

    ax1.set_ylabel('Capital ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(df_test.index, dd, 0, color='red', alpha=0.3)
    ax2.plot(df_test.index, dd, color='red', linewidth=1, label='Strategy')
    ax2.fill_between(df_test.index, mkt_dd, 0, color='gray', alpha=0.2)
    ax2.plot(df_test.index, mkt_dd, color='gray', linewidth=1, linestyle='--', label='Market')
    ax2.set_ylabel('Drawdown %', fontsize=10)
    ax2.set_title('Drawdown Comparison', fontsize=11, color='white')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Panel 3: Returns distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(strat_returns * 100, bins=50, alpha=0.7, color='#00ff00', label='Strategy', density=True)
    ax3.hist(mkt_returns * 100, bins=50, alpha=0.5, color='gray', label='Market', density=True)
    ax3.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Daily Return (%)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Returns Distribution', fontsize=11, color='white')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.2)

    # Panel 4: Metrics table
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    metrics_data = [
        ['Metric', 'Strategy', 'Market'],
        ['Total Return', f'{total_ret:.2f}%', f'{mkt_ret:.2f}%'],
        ['Max Drawdown', f'{max_dd:.2f}%', f'{mkt_max_dd:.2f}%'],
        ['Sharpe Ratio', f'{sharpe_strat:.2f}', f'{sharpe_mkt:.2f}'],
        ['Sortino Ratio', f'{sortino_strat:.2f}', f'{sortino_mkt:.2f}'],
        ['Calmar Ratio', f'{calmar_strat:.2f}', f'{calmar_mkt:.2f}'],
    ]
    
    table = ax4.table(cellText=metrics_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for row_idx in range(len(metrics_data)):
        for col_idx in range(3):
            cell = table[(row_idx, col_idx)]
            if row_idx == 0:
                cell.set_facecolor('#444444')
                cell.set_text_props(color='gold', fontweight='bold')
            else:
                cell.set_facecolor('#1a1a1a')
                cell.set_text_props(color='white')
            cell.set_edgecolor('#666666')
    
    ax4.set_title('Performance Metrics', fontsize=12, color='white', pad=20)

    # Panel 5: Trade stats
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    trade_data = [
        ['Statistic', 'Value'],
        ['Total Trades', f"{trade_stats['total_trades']}"],
        ['Win Rate', f"{trade_stats['win_rate']:.1f}%"],
        ['Avg Win', f"{trade_stats['avg_win']:.2f}%"],
        ['Avg Loss', f"{trade_stats['avg_loss']:.2f}%"],
        ['Profit Factor', f"{trade_stats['profit_factor']:.2f}"],
    ]
    
    table2 = ax5.table(cellText=trade_data, loc='center', cellLoc='center',
                       colWidths=[0.5, 0.3])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.2, 1.8)
    
    for row_idx in range(len(trade_data)):
        for col_idx in range(2):
            cell = table2[(row_idx, col_idx)]
            if row_idx == 0:
                cell.set_facecolor('#444444')
                cell.set_text_props(color='gold', fontweight='bold')
            else:
                cell.set_facecolor('#1a1a1a')
                cell.set_text_props(color='white')
            cell.set_edgecolor('#666666')
    
    ax5.set_title('Trading Statistics', fontsize=12, color='white', pad=20)

    try:
        plt.tight_layout()
    except UserWarning:
        pass
    
    # Save chart
    output_path = os.path.join(project_root, 'assets', 'backtest_results.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    logger.info(f"Chart saved to: {output_path}")
    
    plt.show()
    
    return {
        'total_return': total_ret,
        'market_return': mkt_ret,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe_strat,
        'sortino_ratio': sortino_strat,
        'calmar_ratio': calmar_strat,
        'trade_stats': trade_stats
    }


if __name__ == "__main__":
    results = run_long_short_backtest()
