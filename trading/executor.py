"""
Trade executor — handles paper and live order placement via MEXC API.
"""
import logging
import time
from typing import Optional, Dict, List
from datetime import datetime

import pandas as pd

from api.client import MEXCClient
from risk.manager import RiskManager
from config.settings import TRADING_MODE

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Executes trades in paper or live mode.
    
    Paper mode: simulates fills at current market price.
    Live mode:  places real orders via the MEXC API.
    """

    def __init__(self, mode: Optional[str] = None,
                 client: Optional[MEXCClient] = None,
                 risk_manager: Optional[RiskManager] = None):
        self.mode = mode or TRADING_MODE  # 'paper' or 'live'
        self.client = client or MEXCClient()
        self.risk_manager = risk_manager or RiskManager()
        self.paper_trades: List[Dict] = []
        self.open_orders: List[Dict] = []
        logger.info(f"TradeExecutor initialized in {self.mode.upper()} mode")

    # ── Order Placement ────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, side: str, quantity: float,
                           stop_loss: float = 0, take_profit: float = 0) -> Optional[Dict]:
        """Place a market order (BUY or SELL)."""
        if not self.risk_manager.can_open_position():
            logger.warning("Risk manager blocked order — position limits reached")
            return None

        if self.mode == "paper":
            return self._paper_market_order(symbol, side, quantity, stop_loss, take_profit)
        else:
            return self._live_market_order(symbol, side, quantity, stop_loss, take_profit)

    def place_limit_order(self, symbol: str, side: str, quantity: float,
                          price: float, stop_loss: float = 0,
                          take_profit: float = 0) -> Optional[Dict]:
        """Place a limit order."""
        if not self.risk_manager.can_open_position():
            logger.warning("Risk manager blocked order — position limits reached")
            return None

        if self.mode == "paper":
            return self._paper_limit_order(symbol, side, quantity, price, stop_loss, take_profit)
        else:
            return self._live_limit_order(symbol, side, quantity, price, stop_loss, take_profit)

    def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Cancel an open order."""
        if self.mode == "paper":
            self.open_orders = [o for o in self.open_orders if o.get("orderId") != order_id]
            return {"status": "CANCELED", "orderId": order_id}
        else:
            return self.client.cancel_order(symbol, order_id)

    def cancel_all_orders(self, symbol: str) -> Optional[Dict]:
        """Cancel all open orders for a symbol."""
        if self.mode == "paper":
            self.open_orders = [o for o in self.open_orders if o.get("symbol") != symbol]
            return {"status": "CANCELED"}
        else:
            return self.client.cancel_all_orders(symbol)

    # ── Position Info ──────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[Dict]:
        """Get account info (balances)."""
        if self.mode == "paper":
            return {"balances": [], "note": "paper trading mode"}
        return self.client.get_account()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        if self.mode == "paper":
            if symbol:
                return [o for o in self.open_orders if o.get("symbol") == symbol]
            return self.open_orders
        return self.client.get_open_orders(symbol) or []

    def get_trade_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        if self.mode == "paper":
            return [t for t in self.paper_trades if t.get("symbol") == symbol][-limit:]
        return self.client.get_my_trades(symbol, limit=limit) or []

    # ── Paper Mode Execution ───────────────────────────────────────────────────

    def _paper_market_order(self, symbol: str, side: str, quantity: float,
                             stop_loss: float, take_profit: float) -> Dict:
        """Simulate a market order fill."""
        # Get current price
        ticker = self.client.ticker_price(symbol)
        price = float(ticker["price"]) if ticker else 0

        order = {
            "orderId": f"PAPER_{int(time.time() * 1000)}",
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "price": price,
            "quantity": quantity,
            "status": "FILLED",
            "time": datetime.utcnow().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        self.paper_trades.append(order)
        self.risk_manager.open_positions += 1
        logger.info(f"[PAPER] {side} {quantity} {symbol} @ {price} "
                     f"(SL={stop_loss}, TP={take_profit})")
        return order

    def _paper_limit_order(self, symbol: str, side: str, quantity: float,
                            price: float, stop_loss: float,
                            take_profit: float) -> Dict:
        """Simulate a limit order (added to open orders, filled on next check)."""
        order = {
            "orderId": f"PAPER_{int(time.time() * 1000)}",
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "price": price,
            "quantity": quantity,
            "status": "NEW",
            "time": datetime.utcnow().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        self.open_orders.append(order)
        logger.info(f"[PAPER] Limit {side} {quantity} {symbol} @ {price}")
        return order

    # ── Live Mode Execution ────────────────────────────────────────────────────

    def _live_market_order(self, symbol: str, side: str, quantity: float,
                            stop_loss: float, take_profit: float) -> Optional[Dict]:
        """Place a real market order via MEXC API."""
        try:
            result = self.client.place_order(
                symbol=symbol,
                side=side.upper(),
                order_type="MARKET",
                quantity=quantity,
            )
            if result:
                self.risk_manager.open_positions += 1
                logger.info(f"[LIVE] {side} {quantity} {symbol} — Order ID: {result.get('orderId')}")
            return result
        except Exception as e:
            logger.error(f"[LIVE] Market order failed: {e}")
            return None

    def _live_limit_order(self, symbol: str, side: str, quantity: float,
                           price: float, stop_loss: float,
                           take_profit: float) -> Optional[Dict]:
        """Place a real limit order via MEXC API."""
        try:
            result = self.client.place_order(
                symbol=symbol,
                side=side.upper(),
                order_type="LIMIT",
                quantity=quantity,
                price=price,
            )
            if result:
                logger.info(f"[LIVE] Limit {side} {quantity} {symbol} @ {price} "
                            f"— Order ID: {result.get('orderId')}")
            return result
        except Exception as e:
            logger.error(f"[LIVE] Limit order failed: {e}")
            return None
