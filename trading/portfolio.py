"""
Portfolio tracker — tracks positions, balances, and PnL in real time.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd

from api.client import MEXCClient

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """A single open position."""
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    entry_time: str
    stop_loss: float = 0.0
    take_profit: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: str
    total_equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int


class Portfolio:
    """
    Tracks paper or live portfolio state.
    """

    def __init__(self, initial_capital: float = 10000.0,
                 client: Optional[MEXCClient] = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.client = client or MEXCClient()
        self.positions: Dict[str, PortfolioPosition] = {}
        self.closed_trades: List[Dict] = []
        self.realized_pnl: float = 0.0
        self.history: List[PortfolioSnapshot] = []

    # ── Position Management ────────────────────────────────────────────────────

    def open_position(self, symbol: str, side: str, quantity: float,
                      price: float, stop_loss: float = 0, take_profit: float = 0):
        """Record a new position."""
        cost = quantity * price
        if cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
            return False

        self.cash -= cost
        self.positions[symbol] = PortfolioPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.utcnow().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=price,
        )
        logger.info(f"Opened {side} {quantity} {symbol} @ {price}")
        return True

    def close_position(self, symbol: str, price: float, reason: str = ""):
        """Close an existing position and record PnL."""
        if symbol not in self.positions:
            logger.warning(f"No open position for {symbol}")
            return None

        pos = self.positions.pop(symbol)
        if pos.side == "BUY":
            pnl = (price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - price) * pos.quantity

        self.cash += (pos.quantity * price)
        self.realized_pnl += pnl

        trade_record = {
            "symbol": symbol,
            "side": pos.side,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": (pnl / (pos.entry_price * pos.quantity)) * 100,
            "entry_time": pos.entry_time,
            "exit_time": datetime.utcnow().isoformat(),
            "reason": reason,
        }
        self.closed_trades.append(trade_record)
        logger.info(f"Closed {pos.side} {symbol} @ {price} — PnL: ${pnl:+.4f} ({reason})")
        return trade_record

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices and unrealized PnL for all positions."""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.current_price = prices[symbol]
                if pos.side == "BUY":
                    pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity

    def fetch_live_prices(self):
        """Fetch current prices from MEXC for all open positions."""
        prices = {}
        for symbol in self.positions.keys():
            try:
                ticker = self.client.ticker_price(symbol)
                if ticker:
                    prices[symbol] = float(ticker["price"])
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")
        self.update_prices(prices)

    # ── Portfolio Metrics ──────────────────────────────────────────────────────

    @property
    def total_equity(self) -> float:
        positions_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        return self.cash + positions_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        return ((self.total_equity - self.initial_capital) / self.initial_capital) * 100

    def take_snapshot(self):
        """Record current portfolio state."""
        positions_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        snap = PortfolioSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            total_equity=self.total_equity,
            cash=self.cash,
            positions_value=positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            open_positions=len(self.positions),
        )
        self.history.append(snap)
        return snap

    def summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            "initial_capital": self.initial_capital,
            "cash": round(self.cash, 2),
            "total_equity": round(self.total_equity, 2),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "total_return_pct": round(self.return_pct, 2),
            "open_positions": len(self.positions),
            "total_closed_trades": len(self.closed_trades),
        }

    def positions_dataframe(self) -> pd.DataFrame:
        """Get open positions as DataFrame."""
        if not self.positions:
            return pd.DataFrame()
        rows = []
        for p in self.positions.values():
            rows.append({
                "Symbol": p.symbol,
                "Side": p.side,
                "Qty": p.quantity,
                "Entry": p.entry_price,
                "Current": p.current_price,
                "Unrealized PnL": round(p.unrealized_pnl, 4),
                "SL": p.stop_loss,
                "TP": p.take_profit,
            })
        return pd.DataFrame(rows)

    def trades_dataframe(self) -> pd.DataFrame:
        """Get closed trades as DataFrame."""
        if not self.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_trades)

    def equity_series(self) -> pd.Series:
        """Get equity history as time series."""
        if not self.history:
            return pd.Series()
        return pd.Series(
            [s.total_equity for s in self.history],
            index=pd.to_datetime([s.timestamp for s in self.history]),
        )
