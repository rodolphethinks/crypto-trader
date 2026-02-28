"""
Strategy #6 — Momentum / RSI-based trading strategy.

Uses RSI divergence detection, RSI momentum crossovers, ROC direction,
MFI volume-weighted confirmation, MACD histogram momentum, EMA trend filter,
and ATR-based stop/target placement.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.momentum import rsi, mfi, roc, awesome_oscillator, stoch_rsi
from indicators.trend import ema, macd
from indicators.volatility import atr
from indicators.volume import volume_ratio, obv

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """Momentum trading using RSI divergence and strength."""

    name = "MomentumStrategy"
    description = "RSI divergence & momentum with ROC/MFI/MACD confirmation"
    version = "1.0"

    default_params: Dict = {
        "rsi_period": 14,
        "roc_period": 12,
        "mfi_period": 14,
        "ema_trend": 50,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "rr_ratio": 2.0,
        "rsi_bull_threshold": 50,
        "rsi_bear_threshold": 50,
        "divergence_lookback": 20,
    }

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)

    # ------------------------------------------------------------------
    # Divergence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_swing_highs(series: pd.Series, order: int = 5) -> pd.Series:
        """Return a boolean mask where local swing highs occur."""
        highs = pd.Series(False, index=series.index)
        for i in range(order, len(series) - order):
            if all(series.iloc[i] >= series.iloc[i - j] for j in range(1, order + 1)) and \
               all(series.iloc[i] >= series.iloc[i + j] for j in range(1, order + 1)):
                highs.iloc[i] = True
        return highs

    @staticmethod
    def _find_swing_lows(series: pd.Series, order: int = 5) -> pd.Series:
        """Return a boolean mask where local swing lows occur."""
        lows = pd.Series(False, index=series.index)
        for i in range(order, len(series) - order):
            if all(series.iloc[i] <= series.iloc[i - j] for j in range(1, order + 1)) and \
               all(series.iloc[i] <= series.iloc[i + j] for j in range(1, order + 1)):
                lows.iloc[i] = True
        return lows

    def _detect_divergence(self, price: pd.Series, rsi_series: pd.Series,
                           lookback: int) -> pd.DataFrame:
        """
        Detect bullish and bearish RSI divergence.

        Bearish divergence: price makes a higher high but RSI makes a lower high.
        Bullish divergence: price makes a lower low but RSI makes a higher low.

        Returns DataFrame with columns: bullish_div, bearish_div (boolean).
        """
        bull_div = pd.Series(0, index=price.index, dtype=int)
        bear_div = pd.Series(0, index=price.index, dtype=int)

        swing_highs_price = self._find_swing_highs(price, order=3)
        swing_lows_price = self._find_swing_lows(price, order=3)
        swing_highs_rsi = self._find_swing_highs(rsi_series, order=3)
        swing_lows_rsi = self._find_swing_lows(rsi_series, order=3)

        # --- Bearish divergence (price HH, RSI LH) ---
        price_high_idx = price.index[swing_highs_price]
        rsi_high_idx = rsi_series.index[swing_highs_rsi]

        for i in range(1, len(price_high_idx)):
            curr = price_high_idx[i]
            prev = price_high_idx[i - 1]
            # Only consider within lookback window
            curr_pos = price.index.get_loc(curr)
            prev_pos = price.index.get_loc(prev)
            if curr_pos - prev_pos > lookback:
                continue
            if price.loc[curr] > price.loc[prev]:
                # Check if RSI made a lower high around these bars
                rsi_near_curr = rsi_high_idx[(rsi_high_idx >= prev) & (rsi_high_idx <= curr)]
                if len(rsi_near_curr) >= 2:
                    if rsi_series.loc[rsi_near_curr[-1]] < rsi_series.loc[rsi_near_curr[0]]:
                        bear_div.loc[curr] = 1

        # --- Bullish divergence (price LL, RSI HL) ---
        price_low_idx = price.index[swing_lows_price]
        rsi_low_idx = rsi_series.index[swing_lows_rsi]

        for i in range(1, len(price_low_idx)):
            curr = price_low_idx[i]
            prev = price_low_idx[i - 1]
            curr_pos = price.index.get_loc(curr)
            prev_pos = price.index.get_loc(prev)
            if curr_pos - prev_pos > lookback:
                continue
            if price.loc[curr] < price.loc[prev]:
                rsi_near_curr = rsi_low_idx[(rsi_low_idx >= prev) & (rsi_low_idx <= curr)]
                if len(rsi_near_curr) >= 2:
                    if rsi_series.loc[rsi_near_curr[-1]] > rsi_series.loc[rsi_near_curr[0]]:
                        bull_div.loc[curr] = 1

        return pd.DataFrame({"bullish_div": bull_div, "bearish_div": bear_div})

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse *df* and return it enriched with trading columns:
        ``signal`` (1 / -1 / 0), ``stop_loss``, ``take_profit``, ``confidence``.
        """
        p = self.params
        df = df.copy()

        # ---- Compute indicators ----
        df["rsi"] = rsi(df["close"], period=p["rsi_period"])
        df["rsi_prev"] = df["rsi"].shift(1)
        df["roc"] = roc(df["close"], period=p["roc_period"])
        df["mfi"] = mfi(df, period=p["mfi_period"])
        df["ema_trend"] = ema(df["close"], period=p["ema_trend"])
        df["atr"] = atr(df, period=p["atr_period"])

        macd_df = macd(df["close"])
        df["macd_hist"] = macd_df["histogram"]
        df["macd_hist_prev"] = df["macd_hist"].shift(1)

        df["vol_ratio"] = volume_ratio(df)
        df["obv"] = obv(df)
        df["obv_slope"] = df["obv"].diff(5)

        ao = awesome_oscillator(df)
        df["ao"] = ao

        srsi = stoch_rsi(df["close"], rsi_period=p["rsi_period"])
        df["stoch_rsi_k"] = srsi["stoch_rsi_k"]
        df["stoch_rsi_d"] = srsi["stoch_rsi_d"]

        # ---- Divergence detection ----
        div_df = self._detect_divergence(
            df["close"], df["rsi"], lookback=p["divergence_lookback"]
        )
        df["bullish_div"] = div_df["bullish_div"]
        df["bearish_div"] = div_df["bearish_div"]

        # ---- Initialise output columns ----
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        bull_thresh = p["rsi_bull_threshold"]
        bear_thresh = p["rsi_bear_threshold"]
        sl_mult = p["sl_atr_mult"]
        rr = p["rr_ratio"]

        for i in range(1, len(df)):
            rsi_val = df["rsi"].iat[i]
            rsi_prev = df["rsi_prev"].iat[i]
            roc_val = df["roc"].iat[i]
            mfi_val = df["mfi"].iat[i]
            macd_h = df["macd_hist"].iat[i]
            macd_h_prev = df["macd_hist_prev"].iat[i]
            close = df["close"].iat[i]
            ema_val = df["ema_trend"].iat[i]
            atr_val = df["atr"].iat[i]
            bull_div = df["bullish_div"].iat[i]
            bear_div = df["bearish_div"].iat[i]

            if np.isnan(rsi_val) or np.isnan(atr_val):
                continue

            # ---- BULLISH conditions ----
            rsi_bull_cross = (rsi_prev < bull_thresh) and (rsi_val >= bull_thresh)
            rsi_reversal_bull = (rsi_prev < 30) and (rsi_val >= 30)
            roc_positive = roc_val > 0 if not np.isnan(roc_val) else False
            mfi_confirm_bull = mfi_val > 40 if not np.isnan(mfi_val) else False
            macd_increasing = (macd_h > macd_h_prev) if not (np.isnan(macd_h) or np.isnan(macd_h_prev)) else False
            trend_bull = close > ema_val

            bull_score = sum([
                rsi_bull_cross or rsi_reversal_bull,
                roc_positive,
                mfi_confirm_bull,
                macd_increasing,
                trend_bull,
                bool(bull_div),
            ])

            # ---- BEARISH conditions ----
            rsi_bear_cross = (rsi_prev > bear_thresh) and (rsi_val <= bear_thresh)
            rsi_reversal_bear = (rsi_prev > 70) and (rsi_val <= 70)
            roc_negative = roc_val < 0 if not np.isnan(roc_val) else False
            mfi_confirm_bear = mfi_val < 60 if not np.isnan(mfi_val) else False
            macd_decreasing = (macd_h < macd_h_prev) if not (np.isnan(macd_h) or np.isnan(macd_h_prev)) else False
            trend_bear = close < ema_val

            bear_score = sum([
                rsi_bear_cross or rsi_reversal_bear,
                roc_negative,
                mfi_confirm_bear,
                macd_decreasing,
                trend_bear,
                bool(bear_div),
            ])

            # ---- Signal decision (require >= 4 / 6 confirmations) ----
            min_confirms = 4

            if bull_score >= min_confirms and bull_score > bear_score:
                df["signal"].iat[i] = Signal.BUY
                sl = close - sl_mult * atr_val
                tp = close + sl_mult * atr_val * rr
                df["stop_loss"].iat[i] = sl
                df["take_profit"].iat[i] = tp
                df["confidence"].iat[i] = round(min(bull_score / 6.0, 1.0), 2)

            elif bear_score >= min_confirms and bear_score > bull_score:
                df["signal"].iat[i] = Signal.SELL
                sl = close + sl_mult * atr_val
                tp = close - sl_mult * atr_val * rr
                df["stop_loss"].iat[i] = sl
                df["take_profit"].iat[i] = tp
                df["confidence"].iat[i] = round(min(bear_score / 6.0, 1.0), 2)

        self._signals = df
        logger.info(
            "MomentumStrategy signals generated — buys: %d, sells: %d",
            (df["signal"] == Signal.BUY).sum(),
            (df["signal"] == Signal.SELL).sum(),
        )
        return df
