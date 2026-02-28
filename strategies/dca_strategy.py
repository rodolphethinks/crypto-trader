"""
Strategy #13 — Dollar-Cost Averaging (DCA) with Smart Timing

A long-term accumulation strategy that combines regular interval buying
with tactical improvements based on technical indicators.

Logic:
  1. Regular DCA: generate BUY signal at fixed intervals (every N candles).
  2. Smart DCA: increase buy confidence when price is at the lower Bollinger
     Band or RSI < 30 ("buy the dip").
  3. Reduce or skip buys when RSI > 70 or price is overextended above
     upper BB.
  4. Market-regime awareness: increase allocation in accumulation/ranging
     phases, decrease in distribution/trending-down phases.
  5. Confidence modulates position size (higher confidence = larger DCA buy).
  6. SELL signals only generated at major resistance or extreme RSI levels.
  7. Designed for long-term accumulation with tactical improvements.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.trend import ema, sma
from indicators.momentum import rsi
from indicators.volatility import atr, bollinger_bands
from indicators.custom import market_regime

logger = logging.getLogger(__name__)


class DCAStrategy(BaseStrategy):
    """Dollar-Cost Averaging with smart timing based on technical indicators."""

    name = "DCA_SmartTiming"
    description = (
        "Long-term accumulation strategy that buys at regular intervals, "
        "increases size on dips (RSI oversold / lower BB touch), skips or "
        "reduces when overbought, and adapts allocation to the market regime."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # DCA schedule
        "dca_interval": 24,           # buy every N candles
        # Bollinger Bands
        "bb_period": 20,
        "bb_std": 2.0,
        # RSI
        "rsi_period": 14,
        "dip_rsi_threshold": 30,      # RSI below = buy-the-dip
        "high_rsi_threshold": 70,     # RSI above = skip / sell zone
        # Confidence tuning
        "base_confidence": 0.5,
        "dip_boost": 0.3,             # extra confidence on dips
        # ATR stop
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        # Trend filter
        "ema_period": 50,
        # Sell threshold (extreme overbought RSI for exit)
        "extreme_rsi_sell": 80,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df*."""
        p = self.params
        close = df["close"]

        # Bollinger Bands
        bb = bollinger_bands(close, p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_pct"] = bb["bb_pct"]

        # RSI
        df["rsi"] = rsi(close, p["rsi_period"])

        # EMA trend filter
        df["ema_trend"] = ema(close, p["ema_period"])

        # ATR for stop-loss placement
        df["atr"] = atr(df, p["atr_period"])

        # Market regime classification
        df["regime"] = market_regime(df)

        return df

    def _regime_multiplier(self, regime: str) -> float:
        """
        Return a confidence multiplier based on current market regime.

        - accumulation / ranging  -> 1.2  (good time to DCA)
        - trending_up             -> 1.0  (normal DCA)
        - volatile                -> 0.8  (slightly reduce)
        - trending_down           -> 0.6  (significantly reduce)
        """
        return {
            "ranging": 1.2,
            "trending_up": 1.0,
            "volatile": 0.8,
            "trending_down": 0.6,
        }.get(regime, 1.0)

    def _dip_confidence(self, row: pd.Series) -> float:
        """
        Calculate the confidence for a DCA buy.

        Starts at base_confidence, then:
          +dip_boost  if RSI < dip_rsi_threshold
          +dip_boost  if close <= lower BB
          ×regime_multiplier
        Capped at 1.0.
        """
        p = self.params
        conf = p["base_confidence"]

        # Dip bonuses
        if row["rsi"] < p["dip_rsi_threshold"]:
            conf += p["dip_boost"]
        if row["close"] <= row["bb_lower"]:
            conf += p["dip_boost"]

        # Regime adjustment
        conf *= self._regime_multiplier(row["regime"])

        return round(min(conf, 1.0), 2)

    def _should_skip_buy(self, row: pd.Series) -> bool:
        """
        Skip the scheduled DCA buy if the market is overextended:
          - RSI above high_rsi_threshold AND price above upper BB
        """
        p = self.params
        overbought = row["rsi"] > p["high_rsi_threshold"]
        above_upper_bb = row["close"] > row["bb_upper"]
        return overbought and above_upper_bb

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return *df* with added columns:
          signal, stop_loss, take_profit, confidence
        """
        df = df.copy()
        df = self._compute_indicators(df)

        p = self.params

        # Initialise output columns
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        atr_sl = df["atr"] * p["sl_atr_mult"]

        for i in range(len(df)):
            row = df.iloc[i]

            # --- SELL at extreme overbought / upper BB resistance ----------
            if (
                row["rsi"] > p["extreme_rsi_sell"]
                and row["close"] >= row["bb_upper"]
            ):
                df.iloc[i, df.columns.get_loc("signal")] = Signal.SELL
                df.iloc[i, df.columns.get_loc("stop_loss")] = (
                    row["close"] + atr_sl.iloc[i]
                )
                df.iloc[i, df.columns.get_loc("take_profit")] = row["bb_middle"]
                df.iloc[i, df.columns.get_loc("confidence")] = round(
                    min(0.5 + (row["rsi"] - p["high_rsi_threshold"]) / 100, 1.0), 2
                )
                continue

            # --- Regular / Smart DCA buy -----------------------------------
            is_dca_candle = (i % p["dca_interval"] == 0) and i > 0

            # Also trigger a buy if price is at a deep dip regardless of interval
            deep_dip = (
                row["rsi"] < p["dip_rsi_threshold"]
                and row["close"] <= row["bb_lower"]
            )

            if is_dca_candle or deep_dip:
                # Skip if overextended
                if self._should_skip_buy(row):
                    continue

                conf = self._dip_confidence(row)

                df.iloc[i, df.columns.get_loc("signal")] = Signal.BUY
                df.iloc[i, df.columns.get_loc("stop_loss")] = (
                    row["close"] - atr_sl.iloc[i]
                )
                df.iloc[i, df.columns.get_loc("take_profit")] = row["bb_upper"]
                df.iloc[i, df.columns.get_loc("confidence")] = conf

        logger.info(
            "DCA SmartTiming signals generated — buys=%d, sells=%d",
            (df["signal"] == Signal.BUY).sum(),
            (df["signal"] == Signal.SELL).sum(),
        )

        self._signals = df
        return df
