"""
MEXC WebSocket Client for real-time market data streams.

Supports: Trades, K-lines, Depth, Book Ticker, MiniTickers
Uses Protocol Buffers format as per MEXC API.
"""
import json
import logging
import threading
import time
from typing import Callable, Optional, List, Dict

import websocket

from config.settings import WS_URL

logger = logging.getLogger(__name__)


class MEXCWebSocket:
    """WebSocket client for MEXC real-time market data."""

    def __init__(self, on_message: Optional[Callable] = None):
        self.ws_url = WS_URL
        self.ws: Optional[websocket.WebSocketApp] = None
        self._on_message_callback = on_message
        self._subscriptions: List[str] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._reconnect_delay = 1

    def connect(self):
        """Establish WebSocket connection."""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._running = True
        self._thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self._thread.start()
        logger.info("WebSocket connection initiated")

    def _on_open(self, ws):
        logger.info("WebSocket connected")
        self._reconnect_delay = 1
        # Re-subscribe to previously subscribed channels
        if self._subscriptions:
            self._send_subscribe(self._subscriptions)
        # Start ping thread
        self._start_ping()

    def _on_message(self, ws, message):
        try:
            # MEXC sends protobuf by default, but JSON fallback
            if isinstance(message, bytes):
                # For now, log binary messages; full PB integration can be added
                logger.debug(f"Received binary message ({len(message)} bytes)")
                if self._on_message_callback:
                    self._on_message_callback(message)
            else:
                data = json.loads(message)
                if self._on_message_callback:
                    self._on_message_callback(data)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code=None, close_msg=None):
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self._running:
            self._reconnect()

    def _reconnect(self):
        """Reconnect with exponential backoff."""
        logger.info(f"Reconnecting in {self._reconnect_delay}s...")
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, 60)
        self.connect()

    def _start_ping(self):
        """Send periodic pings to keep connection alive."""
        def ping_loop():
            while self._running and self.ws:
                try:
                    self.ws.send(json.dumps({"method": "PING"}))
                except Exception:
                    break
                time.sleep(20)

        self._ping_thread = threading.Thread(target=ping_loop, daemon=True)
        self._ping_thread.start()

    def _send_subscribe(self, channels: List[str]):
        """Send subscription message."""
        if self.ws:
            msg = {"method": "SUBSCRIPTION", "params": channels}
            self.ws.send(json.dumps(msg))
            logger.info(f"Subscribed to: {channels}")

    def _send_unsubscribe(self, channels: List[str]):
        """Send unsubscription message."""
        if self.ws:
            msg = {"method": "UNSUBSCRIPTION", "params": channels}
            self.ws.send(json.dumps(msg))
            logger.info(f"Unsubscribed from: {channels}")

    # ── Public Subscription Methods ───────────────────────────────────────
    def subscribe_trades(self, symbol: str, speed: str = "100ms"):
        """Subscribe to trade stream."""
        channel = f"spot@public.aggre.deals.v3.api.pb@{speed}@{symbol}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def subscribe_kline(self, symbol: str, interval: str = "Min1"):
        """Subscribe to kline stream. Interval: Min1, Min5, Min15, Min30, Min60, Hour4, etc."""
        channel = f"spot@public.kline.v3.api.pb@{symbol}@{interval}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def subscribe_depth(self, symbol: str, speed: str = "100ms"):
        """Subscribe to incremental depth updates."""
        channel = f"spot@public.aggre.depth.v3.api.pb@{speed}@{symbol}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def subscribe_partial_depth(self, symbol: str, levels: int = 20):
        """Subscribe to partial book depth (5, 10, or 20 levels)."""
        channel = f"spot@public.limit.depth.v3.api.pb@{symbol}@{levels}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def subscribe_book_ticker(self, symbol: str, speed: str = "100ms"):
        """Subscribe to best bid/ask updates."""
        channel = f"spot@public.aggre.bookTicker.v3.api.pb@{speed}@{symbol}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def subscribe_mini_ticker(self, symbol: str, timezone: str = "UTC+0"):
        """Subscribe to mini ticker for a symbol."""
        channel = f"spot@public.miniTicker.v3.api.pb@{symbol}@{timezone}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def subscribe_all_mini_tickers(self, timezone: str = "UTC+0"):
        """Subscribe to mini tickers for all pairs."""
        channel = f"spot@public.miniTickers.v3.api.pb@{timezone}"
        self._subscriptions.append(channel)
        self._send_subscribe([channel])

    def unsubscribe(self, channels: List[str]):
        """Unsubscribe from channels."""
        self._send_unsubscribe(channels)
        for ch in channels:
            if ch in self._subscriptions:
                self._subscriptions.remove(ch)

    def close(self):
        """Close the WebSocket connection."""
        self._running = False
        if self.ws:
            self.ws.close()
        logger.info("WebSocket connection closed")
