"""
Strategy #14 — Statistical Arbitrage / Pairs Trading (spread-based)

Trades the spread between the close price and its SMA, treating the
single asset as its own "pair" (price vs. mean).

Logic:
  1. Spread  = close − SMA(close, sma_period)
  2. Z-score = (spread − rolling_mean(spread)) / rolling_std(spread)
  3. BUY  when Z-score < −entry_z  (price below mean → expect reversion up)
  4. SELL when Z-score >  entry_z  (price above mean → expect reversion down)
  5. Exit (counter signal) when |Z-score| < exit_z  (spread reverted)
  6. Confirm with range_detector — prefer signals in ranging markets
  7. Half-Kelly position sizing approximated via confidence score
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.trend import sma, ema
from indicators.volatility import bollinger_bands
from indicators.custom import range_detector

logger = logging.getLogger(__name__)


class PairsTradingStrategy(BaseStrategy):
    """Mean-reversion strategy based on Z-score of the price-vs-SMA spread."""

    name = "PairsTrading_Spread"
    description = (
        "Statistical-arbitrage style strategy that trades the spread between "
        "price and its SMA.  BUY when Z-score is extremely negative (expect "
        "reversion up), SELL when extremely positive.  Regime filter favours "
        "ranging markets."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Spread / mean
        "sma_period": 50,
        # Z-score
        "zscore_period": 20,
        "entry_z": 2.0,
        "exit_z": 0.5,
        # Bollinger Bands (visual aid + stop/TP reference)
        "bb_period": 20,
        "bb_std": 2.0,
        # Range detector
        "range_lookback": 50,
        "range_threshold_pct": 2.0,
        # ATR stop multiplier
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        # Historical hit-rate (used for half-Kelly confidence)
        "assumed_hit_rate": 0.55,
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all required indicator columns to *df*."""
        p = self.params
        close = df["close"]

        # SMA (the "pair" reference line)
        df["sma_ref"] = sma(close, p["sma_period"])

        # Spread & Z-score
        df["spread"] = close - df["sma_ref"]
        spread_mean = df["spread"].rolling(p["zscore_period"]).mean()
        spread_std = df["spread"].rolling(p["zscore_period"]).std()
        df["zscore"] = (df["spread"] - spread_mean) / spread_std

        # Bollinger Bands (for stop/TP levels and visual context)
        bb = bollinger_bands(close, p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]

        # EMA for short-term trend context
        df["ema_short"] = ema(close, 20)

        # Range detection (True when market is in a narrow range → ideal)
        df["in_range"] = range_detector(
            df, lookback=p["range_lookback"],
            threshold_pct=p["range_threshold_pct"],
        )

        # ATR for stop-loss
        from indicators.volatility import atr as atr_fn
        df["atr"] = atr_fn(df, p["atr_period"])

        return df

    def _half_kelly_confidence(self, zscore_abs: float) -> float:
        """
        Approximate a half-Kelly fraction as a 0–1 confidence score.

        Full Kelly = p − (1−p)/b ≈ 2p − 1 for even-money bets.
        Half Kelly = Kelly / 2.
        We scale by how extreme the Z-score is relative to entry_z.
        """
        p_hit = self.params["assumed_hit_rate"]
        entry_z = self.params["entry_z"]

        kelly = max(2 * p_hit - 1, 0.0)
        half_kelly = kelly / 2

        # Scale by Z-score magnitude (more extreme Z → closer to full half-Kelly)
        z_factor = min(zscore_abs / (entry_z * 2), 1.0)
        confidence = half_kelly + z_factor * (1.0 - half_kelly) * 0.5

        return round(min(confidence, 1.0), 2)

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

        zscore = df["zscore"]
        atr_sl = df["atr"] * p["sl_atr_mult"]

        # --- BUY: Z-score deeply negative (price far below mean) ----------
        buy_mask = (zscore < -p["entry_z"])
        # --- SELL: Z-score deeply positive (price far above mean) ----------
        sell_mask = (zscore > p["entry_z"])

        # --- EXIT / counter: Z-score reverted near zero --------------------
        # When in a long position context and Z reverts → generate SELL to close
        # When in a short context and Z reverts → generate BUY to close
        exit_long_mask = (
            (zscore.shift(1) < -p["exit_z"]) & (zscore >= -p["exit_z"])
        )
        exit_short_mask = (
            (zscore.shift(1) > p["exit_z"]) & (zscore <= p["exit_z"])
        )

        # Prefer signals in ranging markets (boost confidence, but still fire
        # in trending markets with reduced confidence)
        in_range = df["in_range"].fillna(False)

        # --- Apply BUY signals ---
        df.loc[buy_mask, "signal"] = Signal.BUY
        df.loc[buy_mask, "stop_loss"] = df.loc[buy_mask, "close"] - atr_sl[buy_mask]
        df.loc[buy_mask, "take_profit"] = df.loc[buy_mask, "bb_middle"]

        # --- Apply SELL signals ---
        df.loc[sell_mask, "signal"] = Signal.SELL
        df.loc[sell_mask, "stop_loss"] = df.loc[sell_mask, "close"] + atr_sl[sell_mask]
        df.loc[sell_mask, "take_profit"] = df.loc[sell_mask, "bb_middle"]

        # --- Apply exit signals (counter-trade to close) ---
        df.loc[exit_long_mask, "signal"] = Signal.SELL
        df.loc[exit_long_mask, "stop_loss"] = np.nan
        df.loc[exit_long_mask, "take_profit"] = np.nan

        df.loc[exit_short_mask, "signal"] = Signal.BUY
        df.loc[exit_short_mask, "stop_loss"] = np.nan
        df.loc[exit_short_mask, "take_profit"] = np.nan

        # --- Confidence scores ---
        # Entry signals: half-Kelly scaled by z-score magnitude
        entry_indices = buy_mask | sell_mask
        df.loc[entry_indices, "confidence"] = (
            zscore[entry_indices]
            .abs()
            .apply(self._half_kelly_confidence)
        )

        # Boost confidence by 0.1 if in a ranging market
        range_boost = in_range.astype(float) * 0.1
        df.loc[entry_indices, "confidence"] = (
            df.loc[entry_indices, "confidence"] + range_boost[entry_indices]
        ).clip(upper=1.0).round(2)

        # Exit signals get a flat moderate confidence
        exit_indices = exit_long_mask | exit_short_mask
        df.loc[exit_indices, "confidence"] = 0.4

        logger.info(
            "PairsTrading signals generated — buys=%d, sells=%d",
            (df["signal"] == Signal.BUY).sum(),
            (df["signal"] == Signal.SELL).sum(),
        )

        self._signals = df
        return df
