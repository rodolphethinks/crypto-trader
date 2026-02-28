"""
Strategy #4 — Breakout Trading with Volume Confirmation

A volatility-contraction / expansion strategy that catches powerful
moves emerging from tight consolidation zones.  The logic layers
several confirmations before issuing a signal:

  1. **Squeeze detection** — Bollinger Band width or the TTM Squeeze
     Momentum indicator identifies periods of low volatility
     (BB inside Keltner Channel).
  2. **Breakout identification** — Price breaks above/below the
     Donchian channel high/low (or Bollinger upper/lower band) once
     the squeeze releases.
  3. **Volume confirmation** — A volume spike (current volume /
     average volume > threshold) validates that real participation
     backs the breakout.
  4. **Retest entry (optional)** — Instead of entering on the raw
     breakout bar, waitfor a pullback to the breakout level before
     continuing in the breakout direction.
  5. **EMA trend filter** — Only take breakouts aligned with the
     longer-term trend (price vs EMA).
  6. **ATR-based stop loss** — Stop placed below/above the
     consolidation range using an ATR multiplier; take profit
     derived from the stop distance × risk-reward ratio.

Designed for 5 m – 1 h timeframes on volatile crypto pairs.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.volatility import (
    bollinger_bands,
    atr,
    donchian_channel,
    squeeze_momentum,
)
from indicators.volume import volume_ratio, obv
from indicators.momentum import rsi
from indicators.trend import ema

logger = logging.getLogger(__name__)


class BreakoutRetestStrategy(BaseStrategy):
    """Breakout-and-retest strategy with volume confirmation."""

    name = "Breakout_Retest_Volume"
    description = (
        "Detects volatility squeezes via Bollinger / Keltner squeeze, "
        "enters on breakout from Donchian channel with volume spike "
        "confirmation, and optionally waits for a retest of the "
        "breakout level.  EMA filter for trend bias; ATR-based risk "
        "management."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Bollinger Bands
        "bb_period": 20,
        "bb_std": 2.0,
        # Donchian Channel
        "donchian_period": 20,
        # Squeeze detection
        "squeeze_lookback": 20,
        # Volume
        "volume_threshold": 1.5,
        # ATR / risk
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        "rr_ratio": 2.0,
        # Trend filter
        "ema_filter": 50,
        # Retest mode
        "use_retest": True,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df* (in-place)."""
        p = self.params

        # -- Bollinger Bands ------------------------------------------
        bb = bollinger_bands(df["close"], period=p["bb_period"], std_dev=p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_width"] = bb["bb_width"]

        # -- Donchian Channel ------------------------------------------
        dc = donchian_channel(df, period=p["donchian_period"])
        df["dc_upper"] = dc["dc_upper"]
        df["dc_lower"] = dc["dc_lower"]
        df["dc_middle"] = dc["dc_middle"]

        # -- Squeeze Momentum (TTM) ------------------------------------
        sqz = squeeze_momentum(df,
                               bb_period=p["bb_period"],
                               bb_std=p["bb_std"])
        df["squeeze_on"] = sqz["squeeze_on"]
        df["squeeze_momentum"] = sqz["momentum"]

        # -- Volume ratio ----------------------------------------------
        df["vol_ratio"] = volume_ratio(df, period=p["squeeze_lookback"])

        # -- OBV -------------------------------------------------------
        df["obv"] = obv(df)

        # -- ATR -------------------------------------------------------
        df["atr"] = atr(df, period=p["atr_period"])

        # -- EMA trend filter ------------------------------------------
        df["ema_filter"] = ema(df["close"], period=p["ema_filter"])

        # -- RSI (auxiliary confidence weighting) ----------------------
        df["rsi"] = rsi(df["close"], period=14)

        return df

    # ------------------------------------------------------------------

    @staticmethod
    def _was_in_squeeze(squeeze_col: pd.Series, idx: int,
                        lookback: int) -> bool:
        """Return True if there was at least one squeeze bar inside the
        preceding *lookback* window (excluding the current bar)."""
        start = max(0, idx - lookback)
        return squeeze_col.iloc[start:idx].any()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse *df* and return it augmented with:
        - ``signal``     : 1 (buy) / -1 (sell) / 0 (hold)
        - ``stop_loss``  : suggested stop-loss price
        - ``take_profit``: suggested take-profit price
        - ``confidence`` : 0.0 – 1.0 conviction score
        """
        df = df.copy()
        df = self._compute_indicators(df)

        p = self.params
        vol_thresh = p["volume_threshold"]
        sl_mult = p["sl_atr_mult"]
        rr = p["rr_ratio"]
        lookback = p["squeeze_lookback"]
        use_retest = p["use_retest"]

        n = len(df)
        signals = np.zeros(n, dtype=int)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)
        confidences = np.full(n, 0.0)

        # Track breakout state for retest logic
        pending_direction = 0        # +1 long, -1 short, 0 none
        breakout_level = np.nan      # price level to retest
        breakout_atr = np.nan        # ATR at breakout bar
        bars_since_breakout = 0
        max_retest_bars = 10         # max bars to wait for retest

        for i in range(lookback, n):
            close = df["close"].iat[i]
            prev_close = df["close"].iat[i - 1]
            high_i = df["high"].iat[i]
            low_i = df["low"].iat[i]
            atr_i = df["atr"].iat[i]
            ema_val = df["ema_filter"].iat[i]
            vr = df["vol_ratio"].iat[i]
            dc_upper = df["dc_upper"].iat[i]
            dc_lower = df["dc_lower"].iat[i]
            bb_upper = df["bb_upper"].iat[i]
            bb_lower = df["bb_lower"].iat[i]
            squeeze_now = df["squeeze_on"].iat[i]
            sqz_mom = df["squeeze_momentum"].iat[i]
            rsi_val = df["rsi"].iat[i]

            # Previous Donchian levels (before this bar)
            prev_dc_upper = df["dc_upper"].iat[i - 1]
            prev_dc_lower = df["dc_lower"].iat[i - 1]

            # ---- Check for fresh breakout ----------------------------
            had_squeeze = self._was_in_squeeze(
                df["squeeze_on"], i, lookback
            )

            # Bullish breakout: close above previous Donchian upper
            bullish_break = (
                close > prev_dc_upper
                and prev_close <= prev_dc_upper
                and had_squeeze
            )

            # Bearish breakout: close below previous Donchian lower
            bearish_break = (
                close < prev_dc_lower
                and prev_close >= prev_dc_lower
                and had_squeeze
            )

            # ---- Volume confirmation ---------------------------------
            vol_ok = vr > vol_thresh

            # ---- EMA trend filter ------------------------------------
            bullish_trend = close > ema_val
            bearish_trend = close < ema_val

            # ---- Confidence scoring ----------------------------------
            def _calc_confidence(direction: int) -> float:
                """Return 0-1 confidence based on multiple factors."""
                conf = 0.5  # base

                # Volume strength
                if vr > vol_thresh * 2:
                    conf += 0.15
                elif vr > vol_thresh:
                    conf += 0.10

                # Squeeze momentum alignment
                if direction == Signal.BUY and sqz_mom > 0:
                    conf += 0.10
                elif direction == Signal.SELL and sqz_mom < 0:
                    conf += 0.10

                # OBV trend (rising for buys, falling for sells)
                if i >= 5:
                    obv_slope = df["obv"].iat[i] - df["obv"].iat[i - 5]
                    if (direction == Signal.BUY and obv_slope > 0) or \
                       (direction == Signal.SELL and obv_slope < 0):
                        conf += 0.10

                # RSI not extreme (avoid chasing)
                if direction == Signal.BUY and 40 < rsi_val < 70:
                    conf += 0.05
                elif direction == Signal.SELL and 30 < rsi_val < 60:
                    conf += 0.05

                return min(conf, 1.0)

            # ---- NON-RETEST mode: signal immediately -----------------
            if not use_retest:
                if bullish_break and vol_ok and bullish_trend:
                    signals[i] = Signal.BUY
                    sl = close - sl_mult * atr_i
                    stop_losses[i] = sl
                    take_profits[i] = close + rr * (close - sl)
                    confidences[i] = _calc_confidence(Signal.BUY)

                elif bearish_break and vol_ok and bearish_trend:
                    signals[i] = Signal.SELL
                    sl = close + sl_mult * atr_i
                    stop_losses[i] = sl
                    take_profits[i] = close - rr * (sl - close)
                    confidences[i] = _calc_confidence(Signal.SELL)

                continue  # next bar

            # ---- RETEST mode -----------------------------------------

            # 1. Register a new pending breakout
            if bullish_break and vol_ok and bullish_trend and pending_direction == 0:
                pending_direction = 1
                breakout_level = prev_dc_upper
                breakout_atr = atr_i
                bars_since_breakout = 0

            elif bearish_break and vol_ok and bearish_trend and pending_direction == 0:
                pending_direction = -1
                breakout_level = prev_dc_lower
                breakout_atr = atr_i
                bars_since_breakout = 0

            # 2. Manage pending retest
            if pending_direction != 0:
                bars_since_breakout += 1

                if pending_direction == 1:
                    # Price pulled back to breakout level (within 0.5 ATR)
                    retested = low_i <= breakout_level + 0.5 * breakout_atr
                    # and is now moving back up
                    resumed = close > breakout_level

                    if retested and resumed:
                        signals[i] = Signal.BUY
                        sl = breakout_level - sl_mult * breakout_atr
                        stop_losses[i] = sl
                        take_profits[i] = close + rr * (close - sl)
                        confidences[i] = _calc_confidence(Signal.BUY)
                        pending_direction = 0

                elif pending_direction == -1:
                    retested = high_i >= breakout_level - 0.5 * breakout_atr
                    resumed = close < breakout_level

                    if retested and resumed:
                        signals[i] = Signal.SELL
                        sl = breakout_level + sl_mult * breakout_atr
                        stop_losses[i] = sl
                        take_profits[i] = close - rr * (sl - close)
                        confidences[i] = _calc_confidence(Signal.SELL)
                        pending_direction = 0

                # Timeout — invalidate stale pending breakout
                if bars_since_breakout >= max_retest_bars:
                    pending_direction = 0

        # ---- Write results back to DataFrame -------------------------
        df["signal"] = signals
        df["stop_loss"] = stop_losses
        df["take_profit"] = take_profits
        df["confidence"] = confidences

        self._signals = df
        logger.info(
            "%s — generated %d buy / %d sell signals over %d bars",
            self.name,
            (signals == Signal.BUY).sum(),
            (signals == Signal.SELL).sum(),
            n,
        )
        return df
