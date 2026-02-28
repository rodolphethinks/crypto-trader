"""
Strategy #9 — MACD Divergence

Detects divergences between price action and the MACD histogram,
a powerful signal for potential trend reversals:

  **Bullish divergence** — price prints a lower low while MACD histogram
  prints a higher low → upward reversal expected.

  **Bearish divergence** — price prints a higher high while MACD histogram
  prints a lower high → downward reversal expected.

Entry timing is refined with a MACD signal-line crossover.  RSI and
volume ratio provide confirmation.  An EMA trend filter prevents
counter-trend trades in strong trends.

Risk is managed with ATR-based stops and configurable reward/risk ratio.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.trend import macd, ema
from indicators.momentum import rsi
from indicators.volatility import atr
from indicators.volume import volume_ratio

logger = logging.getLogger(__name__)


class MACDDivergenceStrategy(BaseStrategy):
    """MACD divergence strategy with crossover timing and multi-confirmation."""

    name = "MACD_Divergence"
    description = (
        "Detects bullish and bearish divergences between price and MACD "
        "histogram, confirmed by signal-line crossover, RSI, volume ratio, "
        "and EMA trend filter.  ATR-based stop-loss with configurable R:R."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # MACD
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        # Divergence detection
        "divergence_lookback": 20,
        # RSI
        "rsi_period": 14,
        # Trend filter
        "ema_trend": 50,
        # ATR / risk
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "rr_ratio": 2.0,
        # Volume
        "vol_ratio_threshold": 1.0,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df* in-place."""
        p = self.params
        close = df["close"]

        # MACD
        macd_df = macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        df["macd_line"] = macd_df["macd"]
        df["macd_signal"] = macd_df["signal"]
        df["macd_hist"] = macd_df["histogram"]

        # RSI
        df["rsi"] = rsi(close, p["rsi_period"])

        # EMA trend filter
        df["ema_trend"] = ema(close, p["ema_trend"])

        # ATR
        df["atr"] = atr(df, p["atr_period"])

        # Volume ratio
        df["vol_ratio"] = volume_ratio(df)

        # MACD signal-line crossover flags
        df["macd_cross_up"] = (
            (df["macd_line"] > df["macd_signal"])
            & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
        )
        df["macd_cross_down"] = (
            (df["macd_line"] < df["macd_signal"])
            & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
        )

        return df

    @staticmethod
    def _find_swing_lows(series: pd.Series, lookback: int) -> pd.Series:
        """
        Find the lowest value within the rolling *lookback* window.
        Returns the index position (iloc) of the swing low for each bar.
        """
        return series.rolling(lookback, min_periods=2).apply(
            lambda x: x.idxmin() if hasattr(x, "idxmin") else np.argmin(x),
            raw=False,
        )

    def _detect_bullish_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Bullish divergence:
          • price makes a lower low over the lookback window
          • MACD histogram makes a higher low (i.e. less negative)
        """
        lookback = self.params["divergence_lookback"]

        price_low = df["close"].rolling(lookback, min_periods=2).min()
        hist_low = df["macd_hist"].rolling(lookback, min_periods=2).min()

        # Current close is at or near the rolling low
        near_price_low = df["close"] <= price_low * 1.002

        # Current histogram > rolling min histogram (higher low)
        hist_higher_low = df["macd_hist"] > hist_low

        # Both histogram values should be negative for a classic bullish div
        hist_negative = df["macd_hist"] < 0

        return near_price_low & hist_higher_low & hist_negative

    def _detect_bearish_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Bearish divergence:
          • price makes a higher high over the lookback window
          • MACD histogram makes a lower high (i.e. less positive)
        """
        lookback = self.params["divergence_lookback"]

        price_high = df["close"].rolling(lookback, min_periods=2).max()
        hist_high = df["macd_hist"].rolling(lookback, min_periods=2).max()

        # Current close is at or near the rolling high
        near_price_high = df["close"] >= price_high * 0.998

        # Current histogram < rolling max histogram (lower high)
        hist_lower_high = df["macd_hist"] < hist_high

        # Both histogram values should be positive for classic bearish div
        hist_positive = df["macd_hist"] > 0

        return near_price_high & hist_lower_high & hist_positive

    def _confidence_score(self, row: pd.Series, direction: int) -> float:
        """Compute a 0–1 confidence score based on confluence conditions."""
        p = self.params
        checks: list[bool] = []

        if direction == Signal.BUY:
            checks.append(bool(row.get("bullish_div", False)))
            checks.append(bool(row.get("macd_cross_up", False)))
            checks.append(row["rsi"] < 45)
            checks.append(row["vol_ratio"] >= p["vol_ratio_threshold"])
            checks.append(row["close"] > row["ema_trend"])
        else:
            checks.append(bool(row.get("bearish_div", False)))
            checks.append(bool(row.get("macd_cross_down", False)))
            checks.append(row["rsi"] > 55)
            checks.append(row["vol_ratio"] >= p["vol_ratio_threshold"])
            checks.append(row["close"] < row["ema_trend"])

        total = len(checks)
        return round(sum(checks) / total, 2) if total else 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse *df* (OHLCV with DatetimeIndex) and return a copy with:

          signal      – 1 (buy) / -1 (sell) / 0 (hold)
          stop_loss   – suggested SL price
          take_profit – suggested TP price
          confidence  – 0–1 score based on confluence count
        """
        df = df.copy()
        p = self.params

        # 1. Compute indicators
        df = self._compute_indicators(df)

        # 2. Divergence detection
        df["bullish_div"] = self._detect_bullish_divergence(df)
        df["bearish_div"] = self._detect_bearish_divergence(df)

        # 3. Initialise output columns
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        # 4. Entry conditions ------------------------------------------

        # BUY: bullish divergence + MACD crossover up + RSI confirmation
        buy_mask = (
            df["bullish_div"]
            & df["macd_cross_up"]
            & (df["rsi"] < 50)
            & (df["vol_ratio"] >= p["vol_ratio_threshold"])
        )

        # SELL: bearish divergence + MACD crossover down + RSI confirmation
        sell_mask = (
            df["bearish_div"]
            & df["macd_cross_down"]
            & (df["rsi"] > 50)
            & (df["vol_ratio"] >= p["vol_ratio_threshold"])
        )

        # Optional EMA trend filter — only take signals in trend direction
        ema_up = df["close"] > df["ema_trend"]
        ema_down = df["close"] < df["ema_trend"]
        buy_mask = buy_mask & ema_up
        sell_mask = sell_mask & ema_down

        # 5. Apply signals
        df.loc[buy_mask, "signal"] = Signal.BUY
        df.loc[sell_mask, "signal"] = Signal.SELL

        # 6. Stop-loss & take-profit
        atr_sl = df["atr"] * p["sl_atr_mult"]

        df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "close"] - atr_sl[buy_mask]
        df.loc[buy_mask, "take_profit"] = (
            df.loc[buy_mask, "close"] + atr_sl[buy_mask] * p["rr_ratio"]
        )

        df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "close"] + atr_sl[sell_mask]
        df.loc[sell_mask, "take_profit"] = (
            df.loc[sell_mask, "close"] - atr_sl[sell_mask] * p["rr_ratio"]
        )

        # 7. Confidence scores
        signal_indices = df.index[df["signal"] != Signal.HOLD]
        for idx in signal_indices:
            row = df.loc[idx]
            direction = int(row["signal"])
            df.at[idx, "confidence"] = self._confidence_score(row, direction)

        logger.info(
            "%s: %d BUY, %d SELL signals on %d bars",
            self.name,
            (df["signal"] == Signal.BUY).sum(),
            (df["signal"] == Signal.SELL).sum(),
            len(df),
        )

        return df
