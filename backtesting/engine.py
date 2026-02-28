"""
Backtesting engine — simulates strategy execution on historical data.

Features:
- Processes signals bar-by-bar
- Tracks positions, equity, and drawdown
- Supports stop-loss and take-profit execution
- Handles commission and slippage
- Generates detailed trade log and performance metrics
"""
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from risk.manager import RiskManager
from config.settings import DEFAULT_INITIAL_CAPITAL, DEFAULT_COMMISSION, DEFAULT_SLIPPAGE

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_reason: str = ""  # 'signal', 'stop_loss', 'take_profit'
    commission: float = 0.0


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    strategy_name: str
    symbol: str
    interval: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    params: Dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return ((self.final_equity - self.initial_capital) / self.initial_capital) * 100

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def avg_trade_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return np.mean([t.pnl for t in self.trades])

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        return np.mean(losses) if losses else 0.0

    @property
    def max_consecutive_wins(self) -> int:
        return self._max_consecutive(True)

    @property
    def max_consecutive_losses(self) -> int:
        return self._max_consecutive(False)

    def _max_consecutive(self, wins: bool) -> int:
        max_count = 0
        count = 0
        for t in self.trades:
            if (t.pnl > 0) == wins:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "total_return": round(self.total_return, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_trade_return": round(self.avg_trade_pnl, 4),
            "params": self.params,
        }

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0.0
        peak = self.equity_curve.expanding().max()
        dd = (self.equity_curve - peak) / peak * 100
        return abs(dd.min())

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(365 * 24)  # Annualized

    @property
    def sortino_ratio(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        return (returns.mean() / downside.std()) * np.sqrt(365 * 24)


class BacktestEngine:
    """
    Core backtesting engine.
    
    Processes a strategy's signals against historical data,
    simulating trade execution with realistic constraints.
    """

    def __init__(self,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 commission_pct: float = DEFAULT_COMMISSION,
                 slippage_pct: float = DEFAULT_SLIPPAGE,
                 risk_manager: Optional[RiskManager] = None):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_manager = risk_manager or RiskManager()

    def run(self, strategy: BaseStrategy, df: pd.DataFrame,
            symbol: str = "UNKNOWN", interval: str = "1h") -> BacktestResult:
        """
        Run a backtest for a strategy on historical data.
        
        Args:
            strategy: Strategy instance
            df: OHLCV DataFrame (must have open, high, low, close, volume)
            symbol: Trading pair name
            interval: Candle interval
        
        Returns:
            BacktestResult with all metrics and trade log
        """
        logger.info(f"Running backtest: {strategy.name} on {symbol} {interval} "
                     f"({len(df)} candles)")

        # Generate signals
        signals_df = strategy.generate_signals(df.copy())

        # Initialize tracking
        equity = self.initial_capital
        position: Optional[Position] = None
        trades: List[Trade] = []
        equity_history = []

        self.risk_manager.update_equity(equity)
        current_day = None

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            current_price = row["close"]
            current_time = signals_df.index[i]
            current_high = row["high"]
            current_low = row["low"]

            # Reset daily PnL at day boundaries
            bar_day = current_time.date() if hasattr(current_time, 'date') else None
            if bar_day and bar_day != current_day:
                current_day = bar_day
                self.risk_manager.reset_daily()

            # Check stop loss / take profit on open position
            if position is not None:
                exit_price = None
                exit_reason = ""

                if position.side == "BUY":
                    if position.stop_loss > 0 and current_low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "stop_loss"
                    elif position.take_profit > 0 and current_high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "take_profit"
                else:  # SELL (short simulation via sell-then-buy)
                    if position.stop_loss > 0 and current_high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "stop_loss"
                    elif position.take_profit > 0 and current_low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "take_profit"

                if exit_price:
                    trade = self._close_position(position, exit_price, current_time,
                                                  exit_reason)
                    trades.append(trade)
                    equity += trade.pnl
                    self.risk_manager.update_equity(equity)
                    self.risk_manager.register_trade_result(trade.pnl_pct)
                    position = None

            # Process new signals
            signal = int(row.get("signal", Signal.HOLD))
            
            if signal != Signal.HOLD and position is None:
                # Open new position
                if self.risk_manager.can_open_position():
                    sl = row.get("stop_loss", 0)
                    tp = row.get("take_profit", 0)

                    if pd.isna(sl):
                        sl = 0
                    if pd.isna(tp):
                        tp = 0

                    # Calculate position size
                    if sl and sl > 0:
                        qty = self.risk_manager.calculate_position_size(
                            equity, current_price, sl)
                    else:
                        # Default: use max position size
                        qty = (equity * self.risk_manager.max_position_pct / 100) / current_price

                    if qty > 0:
                        # Apply slippage
                        entry_price = current_price * (1 + self.slippage_pct) if signal == Signal.BUY \
                            else current_price * (1 - self.slippage_pct)

                        # Apply commission
                        commission = entry_price * qty * self.commission_pct
                        equity -= commission

                        side = "BUY" if signal == Signal.BUY else "SELL"
                        position = Position(
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            quantity=qty,
                            entry_time=current_time,
                            stop_loss=sl if sl else 0,
                            take_profit=tp if tp else 0,
                        )
                        self.risk_manager.open_positions += 1

            elif signal != Signal.HOLD and position is not None:
                # Close position on opposite signal
                if (signal == Signal.BUY and position.side == "SELL") or \
                   (signal == Signal.SELL and position.side == "BUY"):
                    trade = self._close_position(position, current_price,
                                                  current_time, "signal")
                    trades.append(trade)
                    equity += trade.pnl
                    self.risk_manager.update_equity(equity)
                    self.risk_manager.register_trade_result(trade.pnl_pct)
                    position = None

            # Track equity (mark-to-market)
            mtm_equity = equity
            if position is not None:
                if position.side == "BUY":
                    unrealized = (current_price - position.entry_price) * position.quantity
                else:
                    unrealized = (position.entry_price - current_price) * position.quantity
                mtm_equity += unrealized

            equity_history.append(mtm_equity)

        # Close any remaining position at last price
        if position is not None:
            last_price = signals_df["close"].iloc[-1]
            last_time = signals_df.index[-1]
            trade = self._close_position(position, last_price, last_time, "end_of_data")
            trades.append(trade)
            equity += trade.pnl

        equity_curve = pd.Series(equity_history, index=signals_df.index)

        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            interval=interval,
            start_date=str(signals_df.index[0]) if len(signals_df) > 0 else "",
            end_date=str(signals_df.index[-1]) if len(signals_df) > 0 else "",
            initial_capital=self.initial_capital,
            final_equity=equity,
            trades=trades,
            equity_curve=equity_curve,
            params=strategy.params,
        )

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                     f"{result.total_return:.2f}% return, "
                     f"{result.win_rate:.1f}% win rate")

        return result

    def _close_position(self, position: Position, exit_price: float,
                         exit_time: datetime, reason: str) -> Trade:
        """Close a position and create a Trade record."""
        # Apply slippage to exit
        if position.side == "BUY":
            actual_exit = exit_price * (1 - self.slippage_pct)
            pnl = (actual_exit - position.entry_price) * position.quantity
        else:
            actual_exit = exit_price * (1 + self.slippage_pct)
            pnl = (position.entry_price - actual_exit) * position.quantity

        # Commission on exit
        commission = actual_exit * position.quantity * self.commission_pct
        pnl -= commission

        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100

        self.risk_manager.open_positions = max(0, self.risk_manager.open_positions - 1)

        return Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=actual_exit,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            exit_reason=reason,
            commission=commission,
        )
