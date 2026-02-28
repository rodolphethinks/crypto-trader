"""
Strategy #8 — Bollinger Bands Mean Reversion / Breakout

Two modes of operation:

  **mean_reversion** (default)
    1. BUY  when price touches the lower Bollinger Band AND RSI < oversold
    2. SELL when price touches the upper Bollinger Band AND RSI > overbought
    3. Mean-reversion target: BB middle band
    4. EMA filter confirms overall trend direction
    5. ATR-based stop loss

  **breakout**
    1. Detect BB squeeze (width contraction below rolling percentile)
    2. After squeeze resolves, enter in the direction of expansion
    3. Confirmed by momentum from MFI and squeeze_momentum indicator
    4. Wider ATR stop, trend-following TP at 2× risk

Both modes use EMA as a trend filter and ATR for stop placement.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.volatility import bollinger_bands, atr, squeeze_momentum
from indicators.momentum import rsi, mfi
from indicators.trend import ema

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands strategy with mean-reversion and breakout modes."""

    name = "BollingerBands_MeanRev"
    description = (
        "Trades Bollinger Band touches with RSI confirmation in "
        "mean-reversion mode, or BB squeeze breakouts in breakout mode. "
        "EMA trend filter and ATR-based stops."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Bollinger Bands
        "bb_period": 20,
        "bb_std": 2.0,
        # RSI
        "rsi_period": 14,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        # Trend filter
        "ema_filter": 50,
        # ATR / stop-loss
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        # Mode: 'mean_reversion' or 'breakout'
        "mode": "mean_reversion",
        # Breakout-specific
        "squeeze_lookback": 120,
        "squeeze_pctl_threshold": 0.20,
        # Risk-reward for breakout mode
        "rr_ratio": 2.0,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df* in-place."""
        p = self.params
        close = df["close"]

        # Bollinger Bands
        bb = bollinger_bands(close, p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_width"] = bb["bb_width"]
        df["bb_pct"] = bb["bb_pct"]

        # RSI
        df["rsi"] = rsi(close, p["rsi_period"])

        # MFI (volume-weighted RSI — extra confirmation)
        df["mfi"] = mfi(df, p["rsi_period"])

        # EMA trend filter
        df["ema_trend"] = ema(close, p["ema_filter"])

        # ATR
        df["atr"] = atr(df, p["atr_period"])

        # Squeeze Momentum (used in breakout mode)
        sq = squeeze_momentum(df, bb_period=p["bb_period"], bb_std=p["bb_std"])
        df["squeeze_on"] = sq["squeeze_on"]
        df["squeeze_mom"] = sq["momentum"]

        # BB width percentile for squeeze detection
        df["bb_width_pctl"] = df["bb_width"].rolling(
            p["squeeze_lookback"]
        ).rank(pct=True)

        # Squeeze just released: was on, now off
        df["squeeze_release"] = df["squeeze_on"].shift(1).fillna(False) & ~df["squeeze_on"]

        return df

    def _confidence_score(self, row: pd.Series, direction: int) -> float:
        """Compute a 0–1 confidence score based on confluence conditions."""
        p = self.params
        checks: list[bool] = []

        if p["mode"] == "mean_reversion":
            if direction == Signal.BUY:
                checks.append(row["close"] <= row["bb_lower"])
                checks.append(row["rsi"] <= p["rsi_oversold"])
                checks.append(row["mfi"] < 40)
                checks.append(row["close"] > row["ema_trend"])  # uptrend bias
                checks.append(row["bb_pct"] < 0.05)
            else:
                checks.append(row["close"] >= row["bb_upper"])
                checks.append(row["rsi"] >= p["rsi_overbought"])
                checks.append(row["mfi"] > 60)
                checks.append(row["close"] < row["ema_trend"])  # downtrend bias
                checks.append(row["bb_pct"] > 0.95)
        else:  # breakout
            if direction == Signal.BUY:
                checks.append(bool(row.get("squeeze_release", False)))
                checks.append(row["squeeze_mom"] > 0)
                checks.append(row["close"] > row["bb_middle"])
                checks.append(row["close"] > row["ema_trend"])
                checks.append(row["rsi"] > 50)
            else:
                checks.append(bool(row.get("squeeze_release", False)))
                checks.append(row["squeeze_mom"] < 0)
                checks.append(row["close"] < row["bb_middle"])
                checks.append(row["close"] < row["ema_trend"])
                checks.append(row["rsi"] < 50)

        total = len(checks)
        return round(sum(checks) / total, 2) if total else 0.0

    # ------------------------------------------------------------------
    # Mean-reversion signals
    # ------------------------------------------------------------------

    def _mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params

        # BUY: price at/below lower BB + RSI oversold
        buy_bb = df["close"] <= df["bb_lower"]
        buy_rsi = df["rsi"] <= p["rsi_oversold"]
        buy_mask = buy_bb & buy_rsi

        # SELL: price at/above upper BB + RSI overbought
        sell_bb = df["close"] >= df["bb_upper"]
        sell_rsi = df["rsi"] >= p["rsi_overbought"]
        sell_mask = sell_bb & sell_rsi

        # Apply signals
        df.loc[buy_mask, "signal"] = Signal.BUY
        df.loc[sell_mask, "signal"] = Signal.SELL

        # Stop-loss / take-profit
        atr_sl = df["atr"] * p["sl_atr_mult"]

        df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "close"] - atr_sl[buy_mask]
        df.loc[buy_mask, "take_profit"] = df.loc[buy_mask, "bb_middle"]

        df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "close"] + atr_sl[sell_mask]
        df.loc[sell_mask, "take_profit"] = df.loc[sell_mask, "bb_middle"]

        return df

    # ------------------------------------------------------------------
    # Breakout signals
    # ------------------------------------------------------------------

    def _breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params

        # Squeeze must have just released
        release = df["squeeze_release"]

        # Bullish breakout: squeeze release + positive momentum + above EMA
        buy_mask = release & (df["squeeze_mom"] > 0) & (df["close"] > df["ema_trend"])

        # Bearish breakout: squeeze release + negative momentum + below EMA
        sell_mask = release & (df["squeeze_mom"] < 0) & (df["close"] < df["ema_trend"])

        df.loc[buy_mask, "signal"] = Signal.BUY
        df.loc[sell_mask, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]

        # BUY entries
        df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "close"] - atr_sl[buy_mask]
        df.loc[buy_mask, "take_profit"] = (
            df.loc[buy_mask, "close"] + atr_sl[buy_mask] * p["rr_ratio"]
        )

        # SELL entries
        df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "close"] + atr_sl[sell_mask]
        df.loc[sell_mask, "take_profit"] = (
            df.loc[sell_mask, "close"] - atr_sl[sell_mask] * p["rr_ratio"]
        )

        return df

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

        # 3. Generate signals based on mode
        if p["mode"] == "breakout":
            df = self._breakout_signals(df)
        else:
            df = self._mean_reversion_signals(df)

        # 4. Confidence scores
        signal_indices = df.index[df["signal"] != Signal.HOLD]
        for idx in signal_indices:
            row = df.loc[idx]
            direction = int(row["signal"])
            df.at[idx, "confidence"] = self._confidence_score(row, direction)

        logger.info(
            "%s [%s]: %d BUY, %d SELL signals on %d bars",
            self.name,
            p["mode"],
            (df["signal"] == Signal.BUY).sum(),
            (df["signal"] == Signal.SELL).sum(),
            len(df),
        )

        return df
