"""
MEXC Spot REST API Client.

Handles authentication (HMAC-SHA256), rate limiting, and all endpoint calls
for market data and spot trading.
"""
import time
import hmac
import hashlib
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode

import requests

from config.settings import BASE_URL, MEXC_API_KEY, MEXC_SECRET_KEY

logger = logging.getLogger(__name__)


class MEXCClient:
    """REST client for the MEXC Spot V3 API."""

    def __init__(self, api_key: str = "", secret_key: str = ""):
        self.api_key = api_key or MEXC_API_KEY
        self.secret_key = secret_key or MEXC_SECRET_KEY
        self.base_url = BASE_URL
        self.session = requests.Session()
        # Don't set API key on session — only send it on signed requests
        self._last_request_time = 0
        self._request_count = 0

    # ── Signature ─────────────────────────────────────────────────────────
    def _sign(self, params: Dict[str, Any]) -> str:
        """Generate HMAC-SHA256 signature for request params."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        return int(time.time() * 1000)

    # ── HTTP Methods ──────────────────────────────────────────────────────
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                 signed: bool = False) -> Any:
        """Make an API request with optional signing."""
        params = params or {}
        headers = {}

        if signed:
            params["timestamp"] = self._get_timestamp()
            params["signature"] = self._sign(params)
            headers["X-MEXC-APIKEY"] = self.api_key

        url = f"{self.base_url}{endpoint}"

        try:
            if method == "GET":
                resp = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                resp = self.session.post(url, params=params, headers=headers, timeout=10)
            elif method == "DELETE":
                resp = self.session.delete(url, params=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited. Retrying after {retry_after}s")
                time.sleep(retry_after)
                return self._request(method, endpoint, params, signed)

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get(self, endpoint: str, params: Optional[Dict] = None,
            signed: bool = False) -> Any:
        return self._request("GET", endpoint, params, signed)

    def post(self, endpoint: str, params: Optional[Dict] = None,
             signed: bool = False) -> Any:
        return self._request("POST", endpoint, params, signed)

    def delete(self, endpoint: str, params: Optional[Dict] = None,
               signed: bool = False) -> Any:
        return self._request("DELETE", endpoint, params, signed)

    # ── Market Data ───────────────────────────────────────────────────────
    def ping(self) -> Dict:
        """Test connectivity."""
        return self.get("/api/v3/ping")

    def get_server_time(self) -> int:
        """Get server time in ms."""
        data = self.get("/api/v3/time")
        return data.get("serverTime", 0)

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange info for one or all symbols."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.get("/api/v3/exchangeInfo", params)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth."""
        return self.get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List:
        """Get recent trades."""
        return self.get("/api/v3/trades", {"symbol": symbol, "limit": limit})

    def get_agg_trades(self, symbol: str, start_time: Optional[int] = None,
                       end_time: Optional[int] = None, limit: int = 500) -> List:
        """Get compressed aggregate trades."""
        params = {"symbol": symbol, "limit": limit}
        if start_time and end_time:
            params["startTime"] = start_time
            params["endTime"] = end_time
        return self.get("/api/v3/aggTrades", params)

    def get_klines(self, symbol: str, interval: str = "1h",
                   start_time: Optional[int] = None, end_time: Optional[int] = None,
                   limit: int = 500) -> List:
        """
        Get kline/candlestick data.
        
        Response format per kline:
        [open_time, open, high, low, close, volume, close_time, quote_volume]
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self.get("/api/v3/klines", params)

    def get_avg_price(self, symbol: str) -> Dict:
        """Get current average price."""
        return self.get("/api/v3/avgPrice", {"symbol": symbol})

    def get_ticker_24h(self, symbol: Optional[str] = None) -> Any:
        """Get 24hr ticker stats."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.get("/api/v3/ticker/24hr", params)

    def get_ticker_price(self, symbol: Optional[str] = None) -> Any:
        """Get latest price for symbol(s)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.get("/api/v3/ticker/price", params)

    def get_book_ticker(self, symbol: Optional[str] = None) -> Any:
        """Get best bid/ask for symbol(s)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.get("/api/v3/ticker/bookTicker", params)

    # ── Account / Trading ─────────────────────────────────────────────────
    def get_account(self) -> Dict:
        """Get account information (balances)."""
        return self.get("/api/v3/account", signed=True)

    def get_trade_fee(self, symbol: str) -> Dict:
        """Query trading fee for a symbol."""
        return self.get("/api/v3/tradeFee", {"symbol": symbol}, signed=True)

    def place_order(self, symbol: str, side: str, order_type: str,
                    quantity: Optional[float] = None,
                    quote_order_qty: Optional[float] = None,
                    price: Optional[float] = None,
                    client_order_id: Optional[str] = None) -> Dict:
        """
        Place a new order.
        
        side: BUY or SELL
        order_type: LIMIT or MARKET
        """
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = str(quantity)
        if quote_order_qty is not None:
            params["quoteOrderQty"] = str(quote_order_qty)
        if price is not None:
            params["price"] = str(price)
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self.post("/api/v3/order", params, signed=True)

    def place_test_order(self, symbol: str, side: str, order_type: str,
                         quantity: Optional[float] = None,
                         price: Optional[float] = None) -> Dict:
        """Test a new order (does not execute)."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = str(quantity)
        if price is not None:
            params["price"] = str(price)
        return self.post("/api/v3/order/test", params, signed=True)

    def cancel_order(self, symbol: str, order_id: Optional[str] = None,
                     orig_client_order_id: Optional[str] = None) -> Dict:
        """Cancel an active order."""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self.delete("/api/v3/order", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> List:
        """Cancel all open orders for a symbol."""
        return self.delete("/api/v3/openOrders", {"symbol": symbol}, signed=True)

    def get_order(self, symbol: str, order_id: Optional[str] = None,
                  orig_client_order_id: Optional[str] = None) -> Dict:
        """Query a specific order."""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self.get("/api/v3/order", params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.get("/api/v3/openOrders", params, signed=True)

    def get_all_orders(self, symbol: str, start_time: Optional[int] = None,
                       end_time: Optional[int] = None, limit: int = 500) -> List:
        """Get all orders (active, cancelled, filled)."""
        params = {"symbol": symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self.get("/api/v3/allOrders", params, signed=True)

    def get_my_trades(self, symbol: str, limit: int = 100) -> List:
        """Get account trade list."""
        return self.get("/api/v3/myTrades",
                        {"symbol": symbol, "limit": limit}, signed=True)

    def batch_orders(self, orders: List[Dict]) -> Any:
        """Place batch orders (up to 20, same symbol)."""
        import json
        return self.post("/api/v3/batchOrders",
                         {"batchOrders": json.dumps(orders)}, signed=True)
