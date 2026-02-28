"""
WebSocket-based real-time signal alerts.

Connects to MEXC WebSocket for live kline data and runs strategy
signal generation on each new bar close. Alerts via console logging
and optional webhook (Discord/Telegram stub).
"""
import sys, os, json, time, logging, threading
from datetime import datetime
from typing import Dict, List, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import websocket

from api.client import MEXCClient
from data.fetcher import DataFetcher
from strategies.base import BaseStrategy, Signal
from config.settings import WS_URL

logger = logging.getLogger(__name__)


class SignalAlert:
    """Represents a triggered signal alert."""
    def __init__(self, strategy: str, symbol: str, interval: str,
                 signal: int, price: float, stop_loss: float = 0,
                 take_profit: float = 0, confidence: float = 0):
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.signal = signal
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.timestamp = datetime.utcnow().isoformat()

    @property
    def side(self) -> str:
        return "BUY" if self.signal == Signal.BUY else "SELL"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "interval": self.interval,
            "side": self.side,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    def __str__(self):
        emoji = "🟢" if self.signal == Signal.BUY else "🔴"
        return (f"{emoji} {self.side} {self.symbol} @ {self.price:.8f} | "
                f"Strategy: {self.strategy} | SL: {self.stop_loss:.8f} "
                f"TP: {self.take_profit:.8f} | Conf: {self.confidence:.2f}")


class RealtimeAlertEngine:
    """
    Monitors live kline data via WebSocket and fires alerts when
    strategies generate BUY/SELL signals.
    """

    # MEXC WS kline topic format
    KLINE_TOPIC = "spot@public.kline.v3.api@{symbol}@{interval}"

    # WS interval mapping
    WS_INTERVALS = {
        "1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30",
        "1h": "Min60", "4h": "Hour4", "8h": "Hour8", "1d": "Day1",
    }

    def __init__(self, on_alert: Optional[Callable[[SignalAlert], None]] = None):
        self.fetcher = DataFetcher()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.subscriptions: List[Dict] = []  # [{symbol, interval, strategies}]
        self.alert_history: List[SignalAlert] = []
        self.on_alert = on_alert or self._default_alert_handler
        self._ws = None
        self._running = False
        self._kline_cache: Dict[str, pd.DataFrame] = {}  # (sym,intv) -> df
        self._lock = threading.Lock()

    def add_combo(self, strategy: BaseStrategy, symbol: str, interval: str):
        """Add a strategy/pair/interval combo to monitor."""
        key = f"{symbol}|{interval}"
        # Check if subscription exists
        existing = next((s for s in self.subscriptions
                         if s["symbol"] == symbol and s["interval"] == interval), None)
        if existing:
            existing["strategies"].append(strategy)
        else:
            self.subscriptions.append({
                "symbol": symbol,
                "interval": interval,
                "strategies": [strategy],
            })
        logger.info(f"Monitoring {strategy.name} on {symbol} {interval}")

    def _prefetch_history(self):
        """Pre-fetch historical candles for signal warmup."""
        for sub in self.subscriptions:
            sym, intv = sub["symbol"], sub["interval"]
            key = f"{sym}|{intv}"
            if key not in self._kline_cache:
                try:
                    df = self.fetcher.fetch_klines(sym, intv)
                    self._kline_cache[key] = df
                    logger.info(f"Pre-fetched {len(df)} candles for {sym} {intv}")
                except Exception as e:
                    logger.error(f"Failed to pre-fetch {sym} {intv}: {e}")
                    self._kline_cache[key] = pd.DataFrame()

    def _update_kline(self, symbol: str, interval: str, kline_data: dict):
        """Update the kline cache with a new bar and check signals."""
        key = f"{symbol}|{interval}"
        with self._lock:
            df = self._kline_cache.get(key, pd.DataFrame())
            if df.empty:
                return

            # MEXC WS kline has: open, high, low, close, volume, etc.
            ts = pd.Timestamp(kline_data.get("t", 0), unit="ms")
            new_row = {
                "open": float(kline_data.get("o", 0)),
                "high": float(kline_data.get("h", 0)),
                "low": float(kline_data.get("l", 0)),
                "close": float(kline_data.get("c", 0)),
                "volume": float(kline_data.get("v", 0)),
                "quote_volume": float(kline_data.get("q", 0)),
            }

            # Only process on bar close
            is_closed = kline_data.get("K", False)
            if not is_closed:
                return

            # Append new candle
            new_df = pd.DataFrame([new_row], index=[ts])
            new_df.index.name = "open_time"
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")]
            # Keep last 500 candles
            if len(df) > 500:
                df = df.iloc[-500:]
            self._kline_cache[key] = df

        # Check signals for all strategies on this pair/interval
        sub = next((s for s in self.subscriptions
                    if s["symbol"] == symbol and s["interval"] == interval), None)
        if not sub:
            return

        for strategy in sub["strategies"]:
            try:
                sig_df = strategy.generate_signals(df.copy())
                last_signal = sig_df["signal"].iloc[-1]

                if last_signal in (Signal.BUY, Signal.SELL):
                    sl = sig_df["stop_loss"].iloc[-1] if "stop_loss" in sig_df else 0
                    tp = sig_df["take_profit"].iloc[-1] if "take_profit" in sig_df else 0
                    conf = sig_df["confidence"].iloc[-1] if "confidence" in sig_df else 0
                    if pd.isna(sl): sl = 0
                    if pd.isna(tp): tp = 0
                    if pd.isna(conf): conf = 0

                    alert = SignalAlert(
                        strategy=strategy.name,
                        symbol=symbol,
                        interval=interval,
                        signal=last_signal,
                        price=df["close"].iloc[-1],
                        stop_loss=sl,
                        take_profit=tp,
                        confidence=conf,
                    )
                    self.alert_history.append(alert)
                    self.on_alert(alert)

            except Exception as e:
                logger.error(f"Signal check failed for {strategy.name}: {e}")

    def _default_alert_handler(self, alert: SignalAlert):
        """Default: print alert to console."""
        print(f"\n{'='*60}")
        print(f"  SIGNAL ALERT — {alert.timestamp}")
        print(f"  {alert}")
        print(f"{'='*60}\n")

    # ── WebSocket Handlers ─────────────────────────────────────────────────

    def _on_open(self, ws):
        """Subscribe to kline channels on connect."""
        logger.info("WebSocket connected, subscribing to channels...")
        for sub in self.subscriptions:
            sym = sub["symbol"]
            intv = sub["interval"]
            ws_intv = self.WS_INTERVALS.get(intv, intv)
            topic = self.KLINE_TOPIC.format(symbol=sym, interval=ws_intv)
            msg = json.dumps({"method": "SUBSCRIPTION", "params": [topic]})
            ws.send(msg)
            logger.info(f"  Subscribed: {topic}")

    def _on_message(self, ws, message):
        """Process incoming kline data."""
        try:
            data = json.loads(message)

            # Ping/pong
            if "ping" in str(data).lower():
                ws.send(json.dumps({"pong": data.get("ping", int(time.time()))}))
                return

            # Kline update
            if "d" in data and "k" in data.get("d", {}):
                d = data["d"]
                kline = d["k"]
                symbol = data.get("s", d.get("s", ""))
                interval_raw = kline.get("i", "")

                # Reverse map WS interval to our interval
                interval = next(
                    (k for k, v in self.WS_INTERVALS.items() if v == interval_raw),
                    interval_raw
                )

                self._update_kline(symbol, interval, kline)

        except Exception as e:
            logger.error(f"WS message error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        logger.info(f"WebSocket closed: {close_status} {close_msg}")
        if self._running:
            logger.info("Reconnecting in 5s...")
            time.sleep(5)
            self._connect()

    # ── Start/Stop ─────────────────────────────────────────────────────────

    def _connect(self):
        self._ws = websocket.WebSocketApp(
            WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)

    def start(self):
        """Start the real-time alert engine."""
        if not self.subscriptions:
            logger.error("No subscriptions — add combos first")
            return

        logger.info("Pre-fetching historical data...")
        self._prefetch_history()

        self._running = True
        logger.info(f"Starting WebSocket alert engine on {WS_URL}")
        logger.info(f"Monitoring {len(self.subscriptions)} channel(s)")

        self._connect()

    def stop(self):
        """Stop the alert engine."""
        self._running = False
        if self._ws:
            self._ws.close()
