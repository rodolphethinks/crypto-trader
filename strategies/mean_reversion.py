"""
Strategy #2 — Mean Reversion / Range Trading for USDCUSDT

Optimised for stablecoin pairs that fluctuate in very narrow ranges
(typically 0.9990 – 1.0010).  The strategy exploits small deviations
from the mean by:

  1. Confirming the pair is range-bound (range_detector + BB width filter)
  2. Entering BUY when price pierces the lower Bollinger Band with
     oversold RSI and a negative Z-score exceeding the threshold
  3. Entering SELL when price pierces the upper Bollinger Band with
     overbought RSI and a positive Z-score exceeding the threshold
  4. Targeting the Bollinger middle band (mean) as take-profit
  5. Using a tight ATR-based stop-loss just outside the range

Designed for frequent, small-profit trades with a high win rate.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.trend import sma, ema
from indicators.momentum import rsi, stoch_rsi
from indicators.volatility import bollinger_bands, atr
from indicators.custom import range_detector

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Mean-reversion strategy for stablecoin (USDCUSDT-like) pairs."""

    name = "MeanReversion_Range"
    description = (
        "Trades tiny price deviations in stablecoin pairs using Bollinger "
        "Bands, RSI, Z-score confirmation, and narrow-range detection.  "
        "Targets the mean for take-profit with tight ATR-based stops."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Bollinger Bands
        "bb_period": 20,
        "bb_std": 2.0,
        # RSI
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        # Z-score
        "zscore_threshold": 1.5,
        # ATR / stop-loss
        "atr_period": 14,
        "sl_atr_mult": 1.0,
        # Take-profit target — 'mean' sends TP to BB middle
        "tp_target": "mean",
        # Range detection
        "min_range_lookback": 50,
        "max_range_pct": 0.5,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore(series: pd.Series, period: int) -> pd.Series:
        """Rolling Z-score: how many σ the current value departs from mean."""
        mean = series.rolling(period).mean()
        std = series.rolling(period).std()
        return (series - mean) / std.replace(0, np.nan)

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

        # Stochastic RSI (supplementary confirmation)
        srsi = stoch_rsi(close, rsi_period=p["rsi_period"])
        df["stoch_rsi_k"] = srsi["stoch_rsi_k"]
        df["stoch_rsi_d"] = srsi["stoch_rsi_d"]

        # Simple & Exponential MAs (used for mean reference)
        df["sma"] = sma(close, p["bb_period"])
        df["ema"] = ema(close, p["bb_period"])

        # ATR
        df["atr"] = atr(df, p["atr_period"])

        # Z-score of close relative to its rolling mean
        df["zscore"] = self._zscore(close, p["bb_period"])

        # Range flag — True when price range is narrow
        df["in_range"] = range_detector(
            df,
            lookback=p["min_range_lookback"],
            threshold_pct=p["max_range_pct"],
        )

        # BB width percentile (low = narrow bands = favourable for mean reversion)
        df["bb_width_pctl"] = df["bb_width"].rolling(
            p["min_range_lookback"]
        ).rank(pct=True)

        return df

    def _confidence_score(self, row: pd.Series, direction: int) -> float:
        """
        Compute a 0–1 confidence score based on how many confluence
        conditions are met for the given *direction* (+1 buy / -1 sell).
        """
        p = self.params
        checks: list[bool] = []

        if direction == Signal.BUY:
            # Price at or below lower band
            checks.append(row["close"] <= row["bb_lower"])
            # RSI oversold
            checks.append(row["rsi"] <= p["rsi_oversold"])
            # Negative Z-score beyond threshold
            checks.append(row["zscore"] <= -p["zscore_threshold"])
            # Stoch RSI in oversold territory (< 0.2)
            checks.append(row["stoch_rsi_k"] < 0.2)
            # Narrow BB width (bottom 30 %)
            checks.append(row["bb_width_pctl"] < 0.30)
        else:
            # Price at or above upper band
            checks.append(row["close"] >= row["bb_upper"])
            # RSI overbought
            checks.append(row["rsi"] >= p["rsi_overbought"])
            # Positive Z-score beyond threshold
            checks.append(row["zscore"] >= p["zscore_threshold"])
            # Stoch RSI in overbought territory (> 0.8)
            checks.append(row["stoch_rsi_k"] > 0.8)
            # Narrow BB width
            checks.append(row["bb_width_pctl"] < 0.30)

        hit = sum(checks)
        total = len(checks)
        # Normalise but floor at 0.1 if any signal triggers at all
        return round(hit / total, 2) if total else 0.0

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

        # --- 1. Compute indicators ---------------------------------
        df = self._compute_indicators(df)

        # --- 2. Initialise output columns --------------------------
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        # --- 3. Vectorised pre-conditions --------------------------
        in_range = df["in_range"].fillna(False)

        # BUY conditions
        buy_price_cond = df["close"] <= df["bb_lower"]
        buy_rsi_cond = df["rsi"] <= p["rsi_oversold"]
        buy_zscore_cond = df["zscore"] <= -p["zscore_threshold"]
        buy_mask = in_range & buy_price_cond & buy_rsi_cond & buy_zscore_cond

        # SELL conditions
        sell_price_cond = df["close"] >= df["bb_upper"]
        sell_rsi_cond = df["rsi"] >= p["rsi_overbought"]
        sell_zscore_cond = df["zscore"] >= p["zscore_threshold"]
        sell_mask = in_range & sell_price_cond & sell_rsi_cond & sell_zscore_cond

        # --- 4. Apply signals --------------------------------------
        df.loc[buy_mask, "signal"] = Signal.BUY
        df.loc[sell_mask, "signal"] = Signal.SELL

        # --- 5. Stop-loss & take-profit ----------------------------
        atr_sl = df["atr"] * p["sl_atr_mult"]

        # BUY entries: SL below range low, TP at mean (BB middle)
        df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "bb_lower"] - atr_sl[buy_mask]
        df.loc[buy_mask, "take_profit"] = df.loc[buy_mask, "bb_middle"]

        # SELL entries: SL above range high, TP at mean (BB middle)
        df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "bb_upper"] + atr_sl[sell_mask]
        df.loc[sell_mask, "take_profit"] = df.loc[sell_mask, "bb_middle"]

        # Allow alternative TP modes in future
        if p["tp_target"] != "mean":
            logger.warning(
                "tp_target='%s' not implemented; defaulting to BB middle.",
                p["tp_target"],
            )

        # --- 6. Confidence scores ----------------------------------
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
