"""
Strategy #12 — Multi-Timeframe Analysis Strategy

Simulates multi-timeframe (MTF) analysis on a single DataFrame by resampling
the input data into higher timeframes using pandas ``resample``.

  • Higher TF  (HTF, e.g. 4 h) → overall TREND direction  (EMA, ADX, Supertrend)
  • Medium TF  (MTF, e.g. 1 h) → structural ZONES / bias   (Bollinger, RSI)
  • Lower TF   (LTF = raw data, e.g. 15 m) → precise ENTRIES (MACD, Stochastic)

BUY  : HTF trend up  + MTF zone bullish + LTF trigger fires
SELL : HTF trend down + MTF zone bearish + LTF trigger fires
"""
import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal
from indicators.trend import ema, macd, adx, supertrend
from indicators.momentum import rsi, stochastic
from indicators.volatility import atr, bollinger_bands
from indicators.volume import volume_ratio

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helper – resample OHLCV to a coarser bar size
# ------------------------------------------------------------------

def _resample_ohlcv(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame by grouping *multiplier* consecutive bars.

    If the index is a DatetimeIndex the native ``resample`` is used;
    otherwise we fall back to integer-based grouping so the strategy
    still works with non-datetime indices.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        # Infer the base frequency from the first two bars
        freq = df.index[1] - df.index[0]
        target_freq = freq * multiplier
        resampled = df.resample(target_freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])
        return resampled

    # Fallback for integer / RangeIndex
    groups = np.arange(len(df)) // multiplier
    resampled = df.groupby(groups).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return resampled


def _forward_fill_to_ltf(htf_series: pd.Series, ltf_index, multiplier: int) -> pd.Series:
    """
    Map a higher-timeframe Series back onto the lower-timeframe index
    via forward-fill so every LTF bar carries the latest HTF value.
    """
    if isinstance(ltf_index, pd.DatetimeIndex):
        return htf_series.reindex(ltf_index, method="ffill")

    # Integer-index fallback: repeat each HTF value *multiplier* times
    repeated = htf_series.values.repeat(multiplier)[:len(ltf_index)]
    return pd.Series(repeated, index=ltf_index[:len(repeated)])


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe confluence strategy.

    Higher TF determines trend, medium TF provides structural context,
    and the lower (actual) TF supplies precise entry triggers.  All
    three must align for a signal to fire.

    Risk:
      - Stop-loss   = entry ∓ sl_atr_mult × LTF ATR
      - Take-profit = entry ± rr_ratio × risk
    """

    name: str = "MultiTimeframeStrategy"
    description: str = "Multi-timeframe trend + zone + entry confluence strategy"
    version: str = "1.0"

    default_params: Dict = {
        # Timeframe multipliers relative to the input bar size
        "htf_multiplier": 16,   # e.g. 15 m × 16 ≈ 4 h
        "mtf_multiplier": 4,    # e.g. 15 m × 4  = 1 h
        # EMA for HTF trend
        "ema_fast": 9,
        "ema_slow": 21,
        # ADX for HTF trend strength
        "adx_period": 14,
        "adx_threshold": 25,
        # RSI (MTF zone + LTF filter)
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        # Bollinger Bands for MTF zone
        "bb_period": 20,
        "bb_std": 2.0,
        # Stochastic for LTF entry timing
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_smooth": 3,
        "stoch_oversold": 20,
        "stoch_overbought": 80,
        # MACD for LTF confirmation
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        # Supertrend for HTF
        "st_period": 10,
        "st_multiplier": 3.0,
        # Risk
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "rr_ratio": 2.5,
        # Volume filter
        "vol_ratio_threshold": 1.0,
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

        htf_mult = p["htf_multiplier"]
        mtf_mult = p["mtf_multiplier"]

        # ---- Resample to HTF & MTF -------------------------------------
        htf_df = _resample_ohlcv(df, htf_mult)
        mtf_df = _resample_ohlcv(df, mtf_mult)

        # ---- HTF indicators (trend) ------------------------------------
        htf_df["ema_fast"] = ema(htf_df["close"], p["ema_fast"])
        htf_df["ema_slow"] = ema(htf_df["close"], p["ema_slow"])

        adx_data = adx(htf_df, p["adx_period"])
        htf_df["adx"] = adx_data["adx"]
        htf_df["plus_di"] = adx_data["plus_di"]
        htf_df["minus_di"] = adx_data["minus_di"]

        st_data = supertrend(htf_df, p["st_period"], p["st_multiplier"])
        htf_df["st_direction"] = st_data["direction"]

        # Composite HTF trend: +1 up, -1 down, 0 neutral
        htf_df["trend"] = 0
        htf_up = (
            (htf_df["ema_fast"] > htf_df["ema_slow"])
            & (htf_df["adx"] >= p["adx_threshold"])
            & (htf_df["st_direction"] == 1)
        )
        htf_down = (
            (htf_df["ema_fast"] < htf_df["ema_slow"])
            & (htf_df["adx"] >= p["adx_threshold"])
            & (htf_df["st_direction"] == -1)
        )
        htf_df.loc[htf_up, "trend"] = 1
        htf_df.loc[htf_down, "trend"] = -1

        # ---- MTF indicators (zone / structure) --------------------------
        bb = bollinger_bands(mtf_df["close"], p["bb_period"], p["bb_std"])
        mtf_df["bb_upper"] = bb["bb_upper"]
        mtf_df["bb_lower"] = bb["bb_lower"]
        mtf_df["bb_middle"] = bb["bb_middle"]
        mtf_df["bb_pct"] = bb["bb_pct"]

        mtf_df["rsi"] = rsi(mtf_df["close"], p["rsi_period"])

        # MTF zone: +1 bullish, -1 bearish, 0 neutral
        mtf_df["zone"] = 0
        mtf_bullish = (
            (mtf_df["close"] > mtf_df["bb_middle"])
            & (mtf_df["rsi"] > 50)
            & (mtf_df["rsi"] < p["rsi_overbought"])
        )
        mtf_bearish = (
            (mtf_df["close"] < mtf_df["bb_middle"])
            & (mtf_df["rsi"] < 50)
            & (mtf_df["rsi"] > p["rsi_oversold"])
        )
        mtf_df.loc[mtf_bullish, "zone"] = 1
        mtf_df.loc[mtf_bearish, "zone"] = -1

        # ---- Map HTF / MTF back to LTF ---------------------------------
        df["htf_trend"] = _forward_fill_to_ltf(htf_df["trend"], df.index, htf_mult)
        df["mtf_zone"] = _forward_fill_to_ltf(mtf_df["zone"], df.index, mtf_mult)
        df["htf_adx"] = _forward_fill_to_ltf(htf_df["adx"], df.index, htf_mult)

        # Fill any remaining NaN from alignment to 0 / neutral
        df["htf_trend"] = df["htf_trend"].fillna(0).astype(int)
        df["mtf_zone"] = df["mtf_zone"].fillna(0).astype(int)
        df["htf_adx"] = df["htf_adx"].fillna(0)

        # ---- LTF indicators (entry triggers) ----------------------------
        macd_data = macd(df["close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])
        df["macd_line"] = macd_data["macd"]
        df["macd_signal_line"] = macd_data["signal"]
        df["macd_hist"] = macd_data["histogram"]

        stoch_data = stochastic(df, p["stoch_k"], p["stoch_d"], p["stoch_smooth"])
        df["stoch_k"] = stoch_data["stoch_k"]
        df["stoch_d"] = stoch_data["stoch_d"]

        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df, p["atr_period"])
        df["vol_ratio"] = volume_ratio(df)

        # MACD crossovers
        df["macd_cross_up"] = (
            (df["macd_line"] > df["macd_signal_line"])
            & (df["macd_line"].shift(1) <= df["macd_signal_line"].shift(1))
        )
        df["macd_cross_down"] = (
            (df["macd_line"] < df["macd_signal_line"])
            & (df["macd_line"].shift(1) >= df["macd_signal_line"].shift(1))
        )

        # Stochastic crossovers
        df["stoch_cross_up"] = (
            (df["stoch_k"] > df["stoch_d"])
            & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        )
        df["stoch_cross_down"] = (
            (df["stoch_k"] < df["stoch_d"])
            & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
        )

        # ---- Signal generation ------------------------------------------
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        for i in range(1, len(df)):
            buy_score = self._buy_score(df, i, p)
            sell_score = self._sell_score(df, i, p)

            if buy_score >= 4:
                risk = p["sl_atr_mult"] * df["atr"].iat[i]
                sl = df["close"].iat[i] - risk
                tp = df["close"].iat[i] + p["rr_ratio"] * risk
                conf = self._confidence(buy_score, df["htf_adx"].iat[i], p)

                df.iat[i, df.columns.get_loc("signal")] = Signal.BUY
                df.iat[i, df.columns.get_loc("stop_loss")] = sl
                df.iat[i, df.columns.get_loc("take_profit")] = tp
                df.iat[i, df.columns.get_loc("confidence")] = conf

            elif sell_score >= 4:
                risk = p["sl_atr_mult"] * df["atr"].iat[i]
                sl = df["close"].iat[i] + risk
                tp = df["close"].iat[i] - p["rr_ratio"] * risk
                conf = self._confidence(sell_score, df["htf_adx"].iat[i], p)

                df.iat[i, df.columns.get_loc("signal")] = Signal.SELL
                df.iat[i, df.columns.get_loc("stop_loss")] = sl
                df.iat[i, df.columns.get_loc("take_profit")] = tp
                df.iat[i, df.columns.get_loc("confidence")] = conf

        self._signals = df
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _buy_score(df: pd.DataFrame, i: int, p: Dict) -> float:
        """
        Return a buy score for row *i*.

        Each factor adds 1 point (max ~7):
          1. HTF trend is UP
          2. MTF zone is bullish
          3. MACD cross-up (or histogram > 0)
          4. Stochastic cross-up from oversold zone
          5. RSI between oversold and overbought
          6. Volume ratio above threshold
          7. ADX strong trend bonus
        """
        score = 0.0

        # 1 — HTF trend
        if df["htf_trend"].iat[i] == 1:
            score += 1.5  # primary weight

        # 2 — MTF zone
        if df["mtf_zone"].iat[i] == 1:
            score += 1.0

        # 3 — MACD
        if df["macd_cross_up"].iat[i]:
            score += 1.0
        elif df["macd_hist"].iat[i] > 0:
            score += 0.5

        # 4 — Stochastic from oversold
        if df["stoch_cross_up"].iat[i] and df["stoch_k"].iat[i] < p["stoch_overbought"]:
            score += 1.0
        elif df["stoch_k"].iat[i] < p["stoch_oversold"]:
            score += 0.5  # primed for reversal

        # 5 — RSI filter
        if p["rsi_oversold"] < df["rsi"].iat[i] < p["rsi_overbought"]:
            score += 0.5

        # 6 — Volume
        if df["vol_ratio"].iat[i] >= p["vol_ratio_threshold"]:
            score += 0.5

        # 7 — Bonus for strong HTF ADX
        if df["htf_adx"].iat[i] >= p["adx_threshold"] * 1.5:
            score += 0.5

        return score

    @staticmethod
    def _sell_score(df: pd.DataFrame, i: int, p: Dict) -> float:
        """Mirror of _buy_score for short signals."""
        score = 0.0

        if df["htf_trend"].iat[i] == -1:
            score += 1.5
        if df["mtf_zone"].iat[i] == -1:
            score += 1.0

        if df["macd_cross_down"].iat[i]:
            score += 1.0
        elif df["macd_hist"].iat[i] < 0:
            score += 0.5

        if df["stoch_cross_down"].iat[i] and df["stoch_k"].iat[i] > p["stoch_oversold"]:
            score += 1.0
        elif df["stoch_k"].iat[i] > p["stoch_overbought"]:
            score += 0.5

        if p["rsi_oversold"] < df["rsi"].iat[i] < p["rsi_overbought"]:
            score += 0.5

        if df["vol_ratio"].iat[i] >= p["vol_ratio_threshold"]:
            score += 0.5

        if df["htf_adx"].iat[i] >= p["adx_threshold"] * 1.5:
            score += 0.5

        return score

    @staticmethod
    def _confidence(score: float, htf_adx: float, p: Dict) -> float:
        """
        Combine the discrete score and HTF ADX into a 0-1 confidence.
        """
        base = min(score / 7.0, 1.0)
        # ADX bonus: stronger trend → more confidence
        adx_bonus = min((htf_adx - p["adx_threshold"]) / 50.0, 0.15) if htf_adx > p["adx_threshold"] else 0.0
        return min(base + adx_bonus, 1.0)
