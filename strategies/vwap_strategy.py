"""
Strategy #10 — VWAP Strategy

Uses the Volume Weighted Average Price as dynamic intraday
support / resistance:

  1. VWAP acts as a magnet — price tends to revert to it.
  2. **BUY** when price pulls back to VWAP from above and bounces
     (confirmed by close reclaiming VWAP in an uptrend).
  3. **SELL** when price pulls back to VWAP from below and rejects
     (confirmed by close failing to hold VWAP in a downtrend).
  4. Standard-deviation bands around VWAP identify overbought /
     oversold extremes.
  5. Volume ratio confirms institutional participation.
  6. EMA determines the higher-timeframe trend direction.

Risk is managed with ATR-based stops and a configurable R:R ratio.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.volume import vwap, vwap_session, volume_ratio
from indicators.trend import ema
from indicators.momentum import rsi
from indicators.volatility import atr

logger = logging.getLogger(__name__)


class VWAPStrategy(BaseStrategy):
    """VWAP bounce / rejection strategy with deviation bands."""

    name = "VWAP_Strategy"
    description = (
        "Trades pull-backs to VWAP with directional confirmation from "
        "EMA trend, RSI, and volume ratio.  Standard-deviation bands "
        "around VWAP provide OB/OS zones.  ATR-based risk management."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Trend filter
        "ema_trend": 50,
        # RSI
        "rsi_period": 14,
        # ATR / risk
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        "rr_ratio": 2.0,
        # VWAP deviation bands
        "vwap_deviation_mult": 1.5,
        # Volume confirmation
        "vol_ratio_threshold": 1.0,
        # Use session (intraday) VWAP — set False for cumulative
        "use_session_vwap": True,
        # Lookback for VWAP std-dev bands
        "vwap_std_period": 20,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df* in-place."""
        p = self.params
        close = df["close"]

        # VWAP (session or cumulative)
        if p["use_session_vwap"]:
            try:
                df["vwap"] = vwap_session(df)
            except Exception:
                # Fallback to cumulative if index has no date info
                df["vwap"] = vwap(df)
        else:
            df["vwap"] = vwap(df)

        # Standard-deviation bands around VWAP
        vwap_diff = close - df["vwap"]
        vwap_std = vwap_diff.rolling(p["vwap_std_period"]).std()
        df["vwap_upper"] = df["vwap"] + p["vwap_deviation_mult"] * vwap_std
        df["vwap_lower"] = df["vwap"] - p["vwap_deviation_mult"] * vwap_std

        # EMA trend filter
        df["ema_trend"] = ema(close, p["ema_trend"])

        # RSI
        df["rsi"] = rsi(close, p["rsi_period"])

        # ATR
        df["atr"] = atr(df, p["atr_period"])

        # Volume ratio
        df["vol_ratio"] = volume_ratio(df)

        # VWAP cross / proximity flags
        df["above_vwap"] = close > df["vwap"]
        df["below_vwap"] = close < df["vwap"]

        # Pullback detection: price was above VWAP, dipped to it, now reclaiming
        df["vwap_bounce"] = (
            df["above_vwap"]
            & (~df["above_vwap"]).shift(1).fillna(False)  # was at/below VWAP prev bar
            & df["above_vwap"].shift(2).fillna(False)      # was above VWAP 2 bars ago
        )

        # Rejection detection: price was below VWAP, rallied to it, now failing
        df["vwap_reject"] = (
            df["below_vwap"]
            & (~df["below_vwap"]).shift(1).fillna(False)  # was at/above VWAP prev bar
            & df["below_vwap"].shift(2).fillna(False)      # was below VWAP 2 bars ago
        )

        return df

    def _confidence_score(self, row: pd.Series, direction: int) -> float:
        """Compute a 0–1 confidence score based on confluence conditions."""
        p = self.params
        checks: list[bool] = []

        if direction == Signal.BUY:
            checks.append(bool(row.get("vwap_bounce", False)))
            checks.append(row["close"] > row["ema_trend"])
            checks.append(row["rsi"] > 40 and row["rsi"] < 60)
            checks.append(row["vol_ratio"] >= p["vol_ratio_threshold"])
            checks.append(row["close"] <= row["vwap_upper"])  # not overextended
        else:
            checks.append(bool(row.get("vwap_reject", False)))
            checks.append(row["close"] < row["ema_trend"])
            checks.append(row["rsi"] > 40 and row["rsi"] < 60)
            checks.append(row["vol_ratio"] >= p["vol_ratio_threshold"])
            checks.append(row["close"] >= row["vwap_lower"])  # not overextended

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

        # 2. Initialise output columns
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        # 3. Entry conditions ------------------------------------------

        # Trend direction from EMA
        uptrend = df["close"] > df["ema_trend"]
        downtrend = df["close"] < df["ema_trend"]

        # BUY: VWAP bounce in an uptrend, volume confirmed
        buy_mask = (
            df["vwap_bounce"]
            & uptrend
            & (df["rsi"] < 65)   # not already overbought
            & (df["vol_ratio"] >= p["vol_ratio_threshold"])
        )

        # SELL: VWAP rejection in a downtrend, volume confirmed
        sell_mask = (
            df["vwap_reject"]
            & downtrend
            & (df["rsi"] > 35)   # not already oversold
            & (df["vol_ratio"] >= p["vol_ratio_threshold"])
        )

        # Additional entries from VWAP deviation bands (OB/OS extremes)
        # Buy near lower deviation band in uptrend
        buy_dev_mask = (
            (df["close"] <= df["vwap_lower"])
            & uptrend
            & (df["rsi"] < 40)
        )
        # Sell near upper deviation band in downtrend
        sell_dev_mask = (
            (df["close"] >= df["vwap_upper"])
            & downtrend
            & (df["rsi"] > 60)
        )

        buy_mask = buy_mask | buy_dev_mask
        sell_mask = sell_mask | sell_dev_mask

        # 4. Apply signals
        df.loc[buy_mask, "signal"] = Signal.BUY
        df.loc[sell_mask, "signal"] = Signal.SELL

        # 5. Stop-loss & take-profit
        atr_sl = df["atr"] * p["sl_atr_mult"]

        df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "close"] - atr_sl[buy_mask]
        df.loc[buy_mask, "take_profit"] = (
            df.loc[buy_mask, "close"] + atr_sl[buy_mask] * p["rr_ratio"]
        )

        df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "close"] + atr_sl[sell_mask]
        df.loc[sell_mask, "take_profit"] = (
            df.loc[sell_mask, "close"] - atr_sl[sell_mask] * p["rr_ratio"]
        )

        # 6. Confidence scores
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
