"""
Strategy #5 — Scalping: Quick in-and-out trades on short timeframes (1m, 5m).

Uses fast EMA crossover, VWAP mean anchor, RSI confirmation,
Stochastic RSI entry timing, volume filtering, and session awareness
to generate high-frequency directional signals with tight risk management.
"""
import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal
from indicators.trend import ema
from indicators.momentum import rsi, stochastic, stoch_rsi
from indicators.volatility import atr, bollinger_bands
from indicators.volume import vwap, volume_ratio
from indicators.custom import session_indicator

logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """
    Fast scalping strategy for 1m / 5m charts.

    Entry logic (BUY example — SELL is mirrored):
      1. Fast EMA (3) crosses above Slow EMA (8)
      2. Price is below VWAP (buying into mean reversion pull)
      3. RSI is *not* overbought (< 70)
      4. Stochastic RSI %K crosses above %D while both are oversold (< 0.3)
      5. Volume ratio > threshold (1.2×)
      6. Current session is in the preferred session list

    Risk:
      - Stop-loss  = entry ∓ sl_atr_mult × ATR  (very tight, 0.5–1 ATR)
      - Take-profit = entry ± tp_atr_mult × ATR  (small, 1–1.5 ATR)

    Confidence is a 0-1 score built from the number of confirming factors.
    """

    name: str = "ScalpingStrategy"
    description: str = "Quick in-and-out scalp trades on 1m/5m timeframes"
    version: str = "1.0"

    default_params: Dict = {
        "ema_fast": 3,
        "ema_slow": 8,
        "rsi_period": 7,
        "stoch_period": 14,
        "atr_period": 10,
        "sl_atr_mult": 0.75,
        "tp_atr_mult": 1.0,
        "volume_threshold": 1.2,
        "bb_period": 15,
        "prefer_sessions": ["overlap", "london", "new_york"],
        # Stoch RSI smoothing
        "stoch_rsi_k_smooth": 3,
        "stoch_rsi_d_smooth": 3,
        # RSI thresholds
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        # Stoch RSI zone thresholds
        "stoch_rsi_oversold": 0.3,
        "stoch_rsi_overbought": 0.7,
    }

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse *df* and return a copy with added columns:
        ``signal`` (1 / -1 / 0), ``stop_loss``, ``take_profit``, ``confidence``.
        """
        df = df.copy()
        p = self.params

        # ---- Indicators ------------------------------------------------
        df["ema_fast"] = ema(df["close"], p["ema_fast"])
        df["ema_slow"] = ema(df["close"], p["ema_slow"])

        df["rsi"] = rsi(df["close"], p["rsi_period"])

        stoch_data = stoch_rsi(
            df["close"],
            rsi_period=p["rsi_period"],
            stoch_period=p["stoch_period"],
            k_smooth=p["stoch_rsi_k_smooth"],
            d_smooth=p["stoch_rsi_d_smooth"],
        )
        df["stoch_rsi_k"] = stoch_data["stoch_rsi_k"]
        df["stoch_rsi_d"] = stoch_data["stoch_rsi_d"]

        df["atr"] = atr(df, p["atr_period"])

        bb_data = bollinger_bands(df["close"], p["bb_period"])
        df["bb_upper"] = bb_data["bb_upper"]
        df["bb_lower"] = bb_data["bb_lower"]
        df["bb_middle"] = bb_data["bb_middle"]

        df["vwap"] = vwap(df)
        df["vol_ratio"] = volume_ratio(df)

        df["session"] = session_indicator(df)

        # ---- Derived conditions -----------------------------------------
        # EMA crossover flags (current bar)
        df["ema_cross_up"] = (
            (df["ema_fast"] > df["ema_slow"])
            & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        )
        df["ema_cross_down"] = (
            (df["ema_fast"] < df["ema_slow"])
            & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        )

        # EMA trend direction (non-crossover bars can still hold direction)
        df["ema_bullish"] = df["ema_fast"] > df["ema_slow"]
        df["ema_bearish"] = df["ema_fast"] < df["ema_slow"]

        # Stoch RSI crossovers
        df["stoch_cross_up"] = (
            (df["stoch_rsi_k"] > df["stoch_rsi_d"])
            & (df["stoch_rsi_k"].shift(1) <= df["stoch_rsi_d"].shift(1))
        )
        df["stoch_cross_down"] = (
            (df["stoch_rsi_k"] < df["stoch_rsi_d"])
            & (df["stoch_rsi_k"].shift(1) >= df["stoch_rsi_d"].shift(1))
        )

        # Session preference
        preferred = set(p["prefer_sessions"])
        df["session_ok"] = df["session"].isin(preferred)

        # ---- Signal generation ------------------------------------------
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        for i in range(1, len(df)):
            buy_score, sell_score = self._score_bar(df, i, p)

            if buy_score >= 3:
                df.iat[i, df.columns.get_loc("signal")] = Signal.BUY
                df.iat[i, df.columns.get_loc("stop_loss")] = (
                    df["close"].iat[i] - p["sl_atr_mult"] * df["atr"].iat[i]
                )
                df.iat[i, df.columns.get_loc("take_profit")] = (
                    df["close"].iat[i] + p["tp_atr_mult"] * df["atr"].iat[i]
                )
                # Confidence: proportion of possible confirming factors (max 6)
                df.iat[i, df.columns.get_loc("confidence")] = min(buy_score / 6.0, 1.0)

            elif sell_score >= 3:
                df.iat[i, df.columns.get_loc("signal")] = Signal.SELL
                df.iat[i, df.columns.get_loc("stop_loss")] = (
                    df["close"].iat[i] + p["sl_atr_mult"] * df["atr"].iat[i]
                )
                df.iat[i, df.columns.get_loc("take_profit")] = (
                    df["close"].iat[i] - p["tp_atr_mult"] * df["atr"].iat[i]
                )
                df.iat[i, df.columns.get_loc("confidence")] = min(sell_score / 6.0, 1.0)

        # Cache for get_signal_at
        self._signals = df
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_bar(df: pd.DataFrame, i: int, p: Dict) -> tuple:
        """
        Return (buy_score, sell_score) for row *i*.

        Each factor adds 1 point:
          1. EMA crossover / alignment
          2. VWAP anchor
          3. RSI confirmation
          4. Stoch RSI entry timing
          5. Volume above threshold
          6. Preferred session
        """
        buy = 0
        sell = 0

        # 1 — Fast EMA crossover / bullish alignment
        if df["ema_cross_up"].iat[i]:
            buy += 1
        elif df["ema_bullish"].iat[i]:
            buy += 0.5
        if df["ema_cross_down"].iat[i]:
            sell += 1
        elif df["ema_bearish"].iat[i]:
            sell += 0.5

        # 2 — VWAP anchor: buy below VWAP, sell above VWAP
        vwap_val = df["vwap"].iat[i]
        close_val = df["close"].iat[i]
        if not np.isnan(vwap_val):
            if close_val < vwap_val:
                buy += 1
            elif close_val > vwap_val:
                sell += 1

        # 3 — RSI confirmation
        rsi_val = df["rsi"].iat[i]
        if not np.isnan(rsi_val):
            if rsi_val < p["rsi_overbought"]:
                buy += 1
            if rsi_val > p["rsi_oversold"]:
                sell += 1

        # 4 — Stochastic RSI entry timing
        stk = df["stoch_rsi_k"].iat[i]
        std = df["stoch_rsi_d"].iat[i]
        if not (np.isnan(stk) or np.isnan(std)):
            # Oversold K crossing above D → buy
            if df["stoch_cross_up"].iat[i] and stk < p["stoch_rsi_oversold"]:
                buy += 1
            elif stk < p["stoch_rsi_oversold"]:
                buy += 0.5
            # Overbought K crossing below D → sell
            if df["stoch_cross_down"].iat[i] and stk > p["stoch_rsi_overbought"]:
                sell += 1
            elif stk > p["stoch_rsi_overbought"]:
                sell += 0.5

        # 5 — Volume above average
        vr = df["vol_ratio"].iat[i]
        if not np.isnan(vr) and vr > p["volume_threshold"]:
            buy += 1
            sell += 1

        # 6 — Preferred session
        if df["session_ok"].iat[i]:
            buy += 1
            sell += 1

        return buy, sell
