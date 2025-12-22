"""
Risk Management Module for the trading system.
Implements position sizing, stop losses, and risk limits.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 1.0
    kelly_fraction: float = 0.5
    stop_loss_pct: float = 0.05
    trailing_stop_pct: float = 0.03
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.15


class RiskManager:
    """
    Risk manager for the trading system.
    
    Implements:
    - Kelly Criterion for position sizing
    - Stop loss and trailing stop
    - Daily loss limits
    - Maximum drawdown control
    """
    
    def __init__(self, config: Optional[RiskConfig] = None, initial_capital: float = 10000.0):
        self.config = config or RiskConfig()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_pnl = 0.0
        self.is_halted = False
        self.halt_reason = None
        
        self.entry_price = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        
        logger.info(f"RiskManager initialized with capital: ${initial_capital:,.2f}")
    
    def calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion."""
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        kelly = kelly * self.config.kelly_fraction
        kelly = max(0, min(kelly, self.config.max_position_pct))
        
        return kelly
    
    def calculate_position_size(self, win_rate: float = 0.5, avg_win: float = 0.02, 
                                 avg_loss: float = 0.01) -> float:
        """Calculate position size in dollars."""
        if self.is_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return 0.0
        
        kelly_pct = self.calculate_kelly_size(win_rate, avg_win, avg_loss)
        return self.current_capital * kelly_pct
    
    def check_stop_loss(self, current_price: float, position: int) -> bool:
        """Check if stop loss should be triggered."""
        if self.entry_price is None:
            return False
        
        if position == 1:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            if pnl_pct < -self.config.stop_loss_pct:
                logger.info(f"Stop Loss triggered: {pnl_pct*100:.2f}%")
                return True
        elif position == -1:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            if pnl_pct < -self.config.stop_loss_pct:
                logger.info(f"Stop Loss triggered: {pnl_pct*100:.2f}%")
                return True
        
        return False
    
    def check_trailing_stop(self, current_price: float, position: int) -> bool:
        """Check if trailing stop should be triggered."""
        if self.highest_since_entry is None or self.lowest_since_entry is None:
            return False
        
        if position == 1:
            self.highest_since_entry = max(self.highest_since_entry, current_price)
            drop_from_high = (self.highest_since_entry - current_price) / self.highest_since_entry
            if drop_from_high > self.config.trailing_stop_pct:
                logger.info(f"Trailing Stop triggered: {drop_from_high*100:.2f}% drop from high")
                return True
        elif position == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, current_price)
            rise_from_low = (current_price - self.lowest_since_entry) / self.lowest_since_entry
            if rise_from_low > self.config.trailing_stop_pct:
                logger.info(f"Trailing Stop triggered: {rise_from_low*100:.2f}% rise from low")
                return True
        
        return False
    
    def on_trade_entry(self, price: float, position_size: float):
        """Record trade entry."""
        self.entry_price = price
        self.highest_since_entry = price
        self.lowest_since_entry = price
        logger.debug(f"Entry recorded: price={price}, size={position_size}")
    
    def on_trade_exit(self, exit_price: float, position: int, position_size: float):
        """Record trade exit."""
        if self.entry_price is None:
            return 0.0
        
        if position == 1:
            pnl = (exit_price - self.entry_price) / self.entry_price * position_size
        else:
            pnl = (self.entry_price - exit_price) / self.entry_price * position_size
        
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        self.entry_price = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        
        logger.debug(f"Exit recorded: PnL={pnl:.2f}, Capital={self.current_capital:.2f}")
        
        return pnl
    
    def check_daily_limit(self) -> bool:
        """Check if daily loss limit was exceeded."""
        if self.daily_pnl < -self.config.max_daily_loss_pct * self.initial_capital:
            self.is_halted = True
            self.halt_reason = "Daily loss limit reached"
            logger.warning(f"Warning: {self.halt_reason}")
            return True
        return False
    
    def check_max_drawdown(self) -> bool:
        """Check if maximum drawdown was exceeded."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.config.max_drawdown_pct:
            self.is_halted = True
            self.halt_reason = f"Maximum drawdown reached: {drawdown*100:.2f}%"
            logger.warning(f"Warning: {self.halt_reason}")
            return True
        return False
    
    def new_day(self):
        """Reset daily metrics."""
        self.daily_pnl = 0.0
        if self.halt_reason == "Daily loss limit reached":
            self.is_halted = False
            self.halt_reason = None
    
    def get_stats(self) -> Dict:
        """Return current risk statistics."""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown_pct': drawdown * 100,
            'daily_pnl': self.daily_pnl,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'total_return_pct': (self.current_capital / self.initial_capital - 1) * 100
        }


def calculate_var(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Calculate historical Value at Risk (VaR)."""
    if len(returns) == 0:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    var = var * np.sqrt(horizon_days)
    
    return abs(var)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall)."""
    if len(returns) == 0:
        return 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    
    return abs(cvar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    rm = RiskManager(initial_capital=10000)
    rm.on_trade_entry(100, 1000)
    print(f"Entry at $100")
    print(f"Stop loss at $94? {rm.check_stop_loss(94, 1)}")
    print(f"Stats: {rm.get_stats()}")
