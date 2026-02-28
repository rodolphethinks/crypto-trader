"""
Order manager — manages order lifecycle and monitors SL/TP.
"""
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from api.client import MEXCClient

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages the full lifecycle of orders:
    - Tracks pending/open/filled orders
    - Monitors SL/TP conditions
    - Handles partial fills and order updates
    """

    def __init__(self,
                 executor: Optional[TradeExecutor] = None,
                 portfolio: Optional[Portfolio] = None):
        self.executor = executor or TradeExecutor()
        self.portfolio = portfolio or Portfolio()
        self.pending_orders: List[Dict] = []
        self.active_sl_tp: Dict[str, Dict] = {}  # symbol -> {sl, tp}

    def submit_entry(self, symbol: str, side: str, quantity: float,
                     order_type: str = "MARKET", price: float = 0,
                     stop_loss: float = 0, take_profit: float = 0) -> Optional[Dict]:
        """
        Submit an entry order with optional SL/TP tracking.
        """
        if order_type == "MARKET":
            result = self.executor.place_market_order(symbol, side, quantity,
                                                       stop_loss, take_profit)
        else:
            result = self.executor.place_limit_order(symbol, side, quantity,
                                                      price, stop_loss, take_profit)

        if result and result.get("status") == "FILLED":
            fill_price = float(result.get("price", price))
            self.portfolio.open_position(symbol, side, quantity, fill_price,
                                         stop_loss, take_profit)
            if stop_loss > 0 or take_profit > 0:
                self.active_sl_tp[symbol] = {
                    "side": side,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }
        elif result:
            self.pending_orders.append(result)

        return result

    def submit_exit(self, symbol: str, reason: str = "manual") -> Optional[Dict]:
        """Close a position via market order."""
        if symbol not in self.portfolio.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self.portfolio.positions[symbol]
        # Opposite side to close
        close_side = "SELL" if pos.side == "BUY" else "BUY"
        result = self.executor.place_market_order(symbol, close_side, pos.quantity)

        if result:
            fill_price = float(result.get("price", pos.current_price))
            trade = self.portfolio.close_position(symbol, fill_price, reason)
            self.active_sl_tp.pop(symbol, None)
            return trade

        return None

    def check_sl_tp(self, prices: Dict[str, float]):
        """
        Check stop-loss and take-profit conditions for all active positions.
        Triggers market exits when conditions are met.
        """
        to_close = []

        for symbol, levels in self.active_sl_tp.items():
            if symbol not in prices:
                continue

            current = prices[symbol]
            side = levels["side"]
            sl = levels["stop_loss"]
            tp = levels["take_profit"]

            if side == "BUY":
                if sl > 0 and current <= sl:
                    to_close.append((symbol, "stop_loss"))
                elif tp > 0 and current >= tp:
                    to_close.append((symbol, "take_profit"))
            else:  # SELL
                if sl > 0 and current >= sl:
                    to_close.append((symbol, "stop_loss"))
                elif tp > 0 and current <= tp:
                    to_close.append((symbol, "take_profit"))

        for symbol, reason in to_close:
            logger.info(f"SL/TP triggered: {symbol} — {reason}")
            self.submit_exit(symbol, reason)

    def check_pending_orders(self, prices: Dict[str, float]):
        """Check if any pending limit orders should be filled."""
        still_pending = []
        for order in self.pending_orders:
            symbol = order["symbol"]
            if symbol not in prices:
                still_pending.append(order)
                continue

            current = prices[symbol]
            price = float(order["price"])
            side = order["side"]

            filled = False
            if side == "BUY" and current <= price:
                filled = True
            elif side == "SELL" and current >= price:
                filled = True

            if filled:
                qty = float(order["quantity"])
                sl = order.get("stop_loss", 0)
                tp = order.get("take_profit", 0)
                self.portfolio.open_position(symbol, side, qty, price, sl, tp)
                if sl > 0 or tp > 0:
                    self.active_sl_tp[symbol] = {
                        "side": side, "stop_loss": sl, "take_profit": tp,
                    }
                order["status"] = "FILLED"
                logger.info(f"Pending order filled: {side} {qty} {symbol} @ {price}")
            else:
                still_pending.append(order)

        self.pending_orders = still_pending

    def tick(self, prices: Dict[str, float]):
        """
        Process one tick: update prices, check orders, check SL/TP.
        Call this on each new price update.
        """
        self.portfolio.update_prices(prices)
        self.check_pending_orders(prices)
        self.check_sl_tp(prices)

    def status_summary(self) -> Dict:
        """Get current status of order manager."""
        return {
            "portfolio": self.portfolio.summary(),
            "pending_orders": len(self.pending_orders),
            "active_sl_tp": len(self.active_sl_tp),
            "positions": [
                {"symbol": s, "side": p.side, "qty": p.quantity,
                 "entry": p.entry_price, "pnl": round(p.unrealized_pnl, 4)}
                for s, p in self.portfolio.positions.items()
            ],
        }
