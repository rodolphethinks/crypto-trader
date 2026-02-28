"""
Strategy #3 — Classic Trend Following (EMA crossover + ADX filter)

A momentum-driven trend-following strategy built for crypto pairs
that can sustain directional moves.  The logic layers several
confirmations before issuing a signal:

  1. **EMA crossover** (fast / slow) provides the primary entry trigger.
  2. **ADX filter** ensures we only trade when a real trend is present
     (ADX > threshold).  DI+/DI- polarity must agree with direction.
  3. **MACD histogram** direction gives secondary momentum confirmation.
  4. **Supertrend** direction adds structural confirmation and provides
     an optional trailing-stop reference.
  5. **RSI guard** filters out overbought longs and oversold shorts to
     avoid chasing exhausted moves.
  6. **EMA 200 bias** restricts longs to above the 200-EMA zone and
     shorts to below, keeping trades aligned with the macro trend.

Risk management:
  - ATR-based stop loss (configurable multiplier).
  - Risk-reward ratio applied to the stop distance for take profit.
  - Optional trailing stop using the Supertrend level.

Designed for 5 m – 1 h timeframes on trending pairs.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.trend import ema, sma, adx, supertrend, macd
from indicators.momentum import rsi
from indicators.volatility import atr

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """EMA-crossover trend-following strategy with multi-layer confirmation."""

    name = "TrendFollowing_EMA_ADX"
    description = (
        "Classic trend-following using fast/slow EMA crossovers filtered "
        "by ADX strength, MACD histogram direction, Supertrend confirmation, "
        "and RSI extremes guard.  ATR-based stop loss with configurable "
        "risk-reward ratio take profit."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # EMA crossover
        "ema_fast": 9,
        "ema_slow": 21,
        "ema_trend": 200,
        # ADX trend filter
        "adx_period": 14,
        "adx_threshold": 25,
        # RSI guard
        "rsi_period": 14,
        "rsi_min": 30,
        "rsi_max": 70,
        # ATR / risk management
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "rr_ratio": 2.5,
        # Supertrend
        "supertrend_period": 10,
        "supertrend_multiplier": 3.0,
        "use_supertrend": True,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df* (returns copy)."""
        df = df.copy()
        p = self.params
        close = df["close"]

        # --- EMAs ---
        df["ema_fast"] = ema(close, p["ema_fast"])
        df["ema_slow"] = ema(close, p["ema_slow"])
        df["ema_trend"] = ema(close, p["ema_trend"])

        # --- ADX / DI ---
        adx_df = adx(df, period=p["adx_period"])
        df["adx"] = adx_df["adx"]
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]

        # --- MACD ---
        macd_df = macd(close)
        df["macd_line"] = macd_df["macd"]
        df["macd_signal"] = macd_df["signal"]
        df["macd_hist"] = macd_df["histogram"]

        # --- RSI ---
        df["rsi"] = rsi(close, p["rsi_period"])

        # --- ATR ---
        df["atr"] = atr(df, period=p["atr_period"])

        # --- Supertrend ---
        if p["use_supertrend"]:
            st_df = supertrend(
                df,
                period=p["supertrend_period"],
                multiplier=p["supertrend_multiplier"],
            )
            df["supertrend"] = st_df["supertrend"]
            df["st_direction"] = st_df["direction"]
        else:
            df["supertrend"] = np.nan
            df["st_direction"] = 0

        return df

    @staticmethod
    def _ema_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """Return +1 on bullish cross, -1 on bearish cross, else 0."""
        prev_fast = fast.shift(1)
        prev_slow = slow.shift(1)

        bullish = (prev_fast <= prev_slow) & (fast > slow)
        bearish = (prev_fast >= prev_slow) & (fast < slow)

        cross = pd.Series(Signal.HOLD, index=fast.index)
        cross[bullish] = Signal.BUY
        cross[bearish] = Signal.SELL
        return cross

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign a 0-1 confidence score to every bar based on how many
        confirmation layers agree with the raw EMA-crossover direction.

        Layers (each worth ~0.20):
          1. ADX above threshold
          2. DI polarity matches direction
          3. MACD histogram sign matches direction
          4. Supertrend direction matches
          5. Price vs EMA-trend alignment
        """
        p = self.params
        cross = self._ema_crossover(df["ema_fast"], df["ema_slow"])

        conf = pd.Series(0.0, index=df.index)

        # Use the *persisted* direction: forward-fill the last crossover
        direction = cross.replace(Signal.HOLD, np.nan).ffill().fillna(0).astype(int)

        # 1. ADX strength
        conf += np.where(df["adx"] >= p["adx_threshold"], 0.20, 0.0)

        # 2. DI polarity
        di_bull = (df["plus_di"] > df["minus_di"]).astype(int)
        di_bear = (df["minus_di"] > df["plus_di"]).astype(int)
        conf += np.where(
            ((direction == Signal.BUY) & di_bull.astype(bool))
            | ((direction == Signal.SELL) & di_bear.astype(bool)),
            0.20,
            0.0,
        )

        # 3. MACD histogram sign
        conf += np.where(
            ((direction == Signal.BUY) & (df["macd_hist"] > 0))
            | ((direction == Signal.SELL) & (df["macd_hist"] < 0)),
            0.20,
            0.0,
        )

        # 4. Supertrend direction
        if p["use_supertrend"]:
            conf += np.where(
                ((direction == Signal.BUY) & (df["st_direction"] == 1))
                | ((direction == Signal.SELL) & (df["st_direction"] == -1)),
                0.20,
                0.0,
            )

        # 5. Price vs EMA-trend (200)
        conf += np.where(
            ((direction == Signal.BUY) & (df["close"] > df["ema_trend"]))
            | ((direction == Signal.SELL) & (df["close"] < df["ema_trend"])),
            0.20,
            0.0,
        )

        return conf.clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse *df* and return it augmented with:
          - ``signal``      : 1 (buy) / -1 (sell) / 0 (hold)
          - ``stop_loss``   : ATR-based stop price
          - ``take_profit`` : risk-reward-based TP price
          - ``confidence``  : 0-1 score
        """
        p = self.params
        df = self._compute_indicators(df)

        # --- Primary trigger: EMA crossover ---
        cross = self._ema_crossover(df["ema_fast"], df["ema_slow"])

        # --- Filters (all must pass for a signal to fire) ---

        # 1. ADX > threshold  →  trend is strong enough
        adx_ok = df["adx"] >= p["adx_threshold"]

        # 2. DI polarity must agree with cross direction
        di_long = df["plus_di"] > df["minus_di"]
        di_short = df["minus_di"] > df["plus_di"]

        # 3. MACD histogram confirms momentum direction
        macd_long = df["macd_hist"] > 0
        macd_short = df["macd_hist"] < 0

        # 4. Supertrend confirmation (optional)
        if p["use_supertrend"]:
            st_long = df["st_direction"] == 1
            st_short = df["st_direction"] == -1
        else:
            st_long = pd.Series(True, index=df.index)
            st_short = pd.Series(True, index=df.index)

        # 5. RSI guard — avoid chasing exhausted moves
        rsi_ok_long = df["rsi"] < p["rsi_max"]
        rsi_ok_short = df["rsi"] > p["rsi_min"]

        # 6. Macro trend — price side of EMA-200
        trend_long = df["close"] > df["ema_trend"]
        trend_short = df["close"] < df["ema_trend"]

        # --- Combine ---
        buy_signal = (
            (cross == Signal.BUY)
            & adx_ok
            & di_long
            & macd_long
            & st_long
            & rsi_ok_long
            & trend_long
        )
        sell_signal = (
            (cross == Signal.SELL)
            & adx_ok
            & di_short
            & macd_short
            & st_short
            & rsi_ok_short
            & trend_short
        )

        df["signal"] = Signal.HOLD
        df.loc[buy_signal, "signal"] = Signal.BUY
        df.loc[sell_signal, "signal"] = Signal.SELL

        # --- Stop loss / Take profit (ATR-based) ---
        atr_val = df["atr"]
        sl_distance = atr_val * p["sl_atr_mult"]
        tp_distance = sl_distance * p["rr_ratio"]

        df["stop_loss"] = np.where(
            df["signal"] == Signal.BUY,
            df["close"] - sl_distance,
            np.where(
                df["signal"] == Signal.SELL,
                df["close"] + sl_distance,
                np.nan,
            ),
        )
        df["take_profit"] = np.where(
            df["signal"] == Signal.BUY,
            df["close"] + tp_distance,
            np.where(
                df["signal"] == Signal.SELL,
                df["close"] - tp_distance,
                np.nan,
            ),
        )

        # Optional trailing stop reference (Supertrend level)
        if p["use_supertrend"]:
            df["trailing_stop"] = np.where(
                df["signal"] != Signal.HOLD,
                df["supertrend"],
                np.nan,
            )
        else:
            df["trailing_stop"] = np.nan

        # --- Confidence ---
        df["confidence"] = self._compute_confidence(df)

        logger.info(
            "%s | signals generated — BUY: %d  SELL: %d  HOLD: %d",
            self.name,
            (df["signal"] == Signal.BUY).sum(),
            (df["signal"] == Signal.SELL).sum(),
            (df["signal"] == Signal.HOLD).sum(),
        )

        return df
