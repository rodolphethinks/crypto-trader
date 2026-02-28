"""
Strategy #7 — Grid Trading

Places buy and sell orders at regular price intervals within a
dynamically calculated range, profiting from oscillations in
ranging / sideways markets.

How it works:
  1. Define upper & lower bounds from Bollinger Bands (+/- N std) or
     from an ATR envelope around a moving average.
  2. Divide that range into *num_grids* equally-spaced levels.
  3. BUY  signal when price crosses **down** to a grid level that has
     no open virtual position.
  4. SELL signal when price crosses **up** to the grid level immediately
     above a filled buy, pairing entry with exit.
  5. A range_detector filter ensures the strategy only fires in
     sideways / low-volatility regimes.
  6. Per-step stop-loss is placed just outside the grid boundaries
     (below the lowest grid for longs, above the highest for shorts).

Best suited for ranging pairs — ideally stablecoin or low-vol majors.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.volatility import atr, bollinger_bands
from indicators.trend import sma
from indicators.custom import range_detector

logger = logging.getLogger(__name__)


class GridTradingStrategy(BaseStrategy):
    """Grid trading strategy – places orders at regular price intervals."""

    name = "GridTrading"
    description = (
        "Automatically builds a price grid from Bollinger Bands or ATR "
        "envelope and generates BUY signals when price drops to an un-filled "
        "grid level, SELL signals when price rises to the paired exit level.  "
        "Filtered by range_detector so it only trades in sideways markets."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Grid construction
        "num_grids": 10,
        "grid_method": "bb",          # 'bb' or 'atr'
        # Bollinger Bands params (used when grid_method == 'bb')
        "bb_period": 20,
        "bb_std": 2.5,
        # ATR params (used when grid_method == 'atr')
        "atr_period": 14,
        "atr_multiplier": 3.0,
        # Range filter
        "range_lookback": 100,
        "range_threshold_pct": 5.0,
        # Stop-loss: fraction of grid step placed beyond boundary
        "sl_grid_steps_beyond": 1.0,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid_levels(lower: float, upper: float,
                           num_grids: int) -> np.ndarray:
        """Return *num_grids + 1* equally-spaced price levels (inclusive)."""
        return np.linspace(lower, upper, num_grids + 1)

    @staticmethod
    def _nearest_grid_below(price: float,
                            levels: np.ndarray) -> Optional[float]:
        """Return the highest grid level that is <= *price*, or None."""
        candidates = levels[levels <= price]
        return float(candidates[-1]) if len(candidates) > 0 else None

    @staticmethod
    def _nearest_grid_above(price: float,
                            levels: np.ndarray) -> Optional[float]:
        """Return the lowest grid level that is >= *price*, or None."""
        candidates = levels[levels >= price]
        return float(candidates[0]) if len(candidates) > 0 else None

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach indicator columns needed for grid boundary computation."""
        p = self.params
        close = df["close"]

        # SMA — used as mid-line for ATR method and as reference
        df["sma"] = sma(close, p["bb_period"])

        # Bollinger Bands
        bb = bollinger_bands(close, p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_width"] = bb["bb_width"]

        # ATR
        df["atr"] = atr(df, p["atr_period"])

        # Range flag — True when the market is ranging
        df["in_range"] = range_detector(
            df,
            lookback=p["range_lookback"],
            threshold_pct=p["range_threshold_pct"],
        )

        return df

    # ------------------------------------------------------------------
    # Grid bounds at each bar
    # ------------------------------------------------------------------

    def _grid_bounds(self, row: pd.Series) -> tuple:
        """
        Return (lower, upper) grid boundaries for the current bar
        based on the chosen *grid_method*.
        """
        p = self.params

        if p["grid_method"] == "bb":
            lower = row["bb_lower"]
            upper = row["bb_upper"]
        elif p["grid_method"] == "atr":
            mid = row["sma"]
            atr_val = row["atr"]
            half_range = p["atr_multiplier"] * atr_val
            lower = mid - half_range
            upper = mid + half_range
        else:
            raise ValueError(
                f"Unknown grid_method: {p['grid_method']!r}.  "
                "Use 'bb' or 'atr'."
            )

        return lower, upper

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk through bars, maintain virtual grid positions, and emit
        BUY / SELL / HOLD signals.

        Returns the input DataFrame augmented with columns:
            signal, stop_loss, take_profit, confidence,
            grid_lower, grid_upper
        """
        df = df.copy()
        df = self._compute_indicators(df)

        p = self.params
        num_grids: int = p["num_grids"]

        n = len(df)
        signals = np.full(n, Signal.HOLD, dtype=int)
        stop_losses = np.full(n, np.nan)
        take_profits = np.full(n, np.nan)
        confidences = np.full(n, 0.0)
        grid_lowers = np.full(n, np.nan)
        grid_uppers = np.full(n, np.nan)

        # Track which grid levels currently have an open (bought) position.
        # Keys = grid-level index (int), Values = buy price.
        filled_levels: Dict[int, float] = {}

        for i in range(1, n):
            row = df.iloc[i]
            prev_close = df["close"].iloc[i - 1]
            close = row["close"]

            # --- skip if indicators not ready or market not ranging -------
            if pd.isna(row["bb_upper"]) or pd.isna(row["atr"]):
                continue
            if not row["in_range"]:
                continue

            # --- compute grid for this bar --------------------------------
            lower, upper = self._grid_bounds(row)
            if np.isnan(lower) or np.isnan(upper) or upper <= lower:
                continue

            grid_lowers[i] = lower
            grid_uppers[i] = upper

            levels = self._build_grid_levels(lower, upper, num_grids)
            step = (upper - lower) / num_grids

            # --- purge filled levels that are now outside the grid --------
            filled_levels = {
                k: v for k, v in filled_levels.items()
                if 0 <= k <= num_grids
            }

            # --- BUY logic: price crossed down to a grid level ------------
            for lvl_idx in range(num_grids + 1):
                lvl_price = levels[lvl_idx]

                # price crossed down through this level
                if prev_close > lvl_price >= close:
                    if lvl_idx not in filled_levels:
                        signals[i] = Signal.BUY
                        filled_levels[lvl_idx] = lvl_price

                        # Stop-loss: below the grid bottom by N steps
                        sl = lower - p["sl_grid_steps_beyond"] * step
                        stop_losses[i] = sl

                        # Take-profit: next grid level above
                        if lvl_idx + 1 <= num_grids:
                            take_profits[i] = levels[lvl_idx + 1]
                        else:
                            take_profits[i] = lvl_price + step

                        # Confidence increases the deeper inside the grid
                        depth = (upper - lvl_price) / (upper - lower) \
                            if (upper - lower) > 0 else 0.5
                        confidences[i] = np.clip(0.4 + 0.5 * depth, 0.3, 0.95)
                        break  # one signal per bar

            # --- SELL logic: price crossed up to a level with a fill ------
            if signals[i] == Signal.HOLD:
                for lvl_idx in sorted(filled_levels.keys(), reverse=True):
                    lvl_price = levels[lvl_idx] if lvl_idx <= num_grids \
                        else filled_levels[lvl_idx]

                    # Check the paired exit level (one step above the buy)
                    exit_idx = lvl_idx + 1
                    if exit_idx > num_grids:
                        exit_price = upper + step
                    else:
                        exit_price = levels[exit_idx]

                    if prev_close < exit_price <= close:
                        signals[i] = Signal.SELL
                        del filled_levels[lvl_idx]

                        # Stop-loss: above the grid top by N steps
                        sl = upper + p["sl_grid_steps_beyond"] * step
                        stop_losses[i] = sl

                        # Take-profit: the buy level (lock in the step profit)
                        take_profits[i] = lvl_price

                        # Confidence higher when exiting closer to the top
                        height = (exit_price - lower) / (upper - lower) \
                            if (upper - lower) > 0 else 0.5
                        confidences[i] = np.clip(0.4 + 0.5 * height, 0.3, 0.95)
                        break

        # --- Attach columns -----------------------------------------------
        df["signal"] = signals
        df["stop_loss"] = stop_losses
        df["take_profit"] = take_profits
        df["confidence"] = confidences
        df["grid_lower"] = grid_lowers
        df["grid_upper"] = grid_uppers

        logger.info(
            "GridTrading | signals generated: %d BUY, %d SELL, %d HOLD",
            (signals == Signal.BUY).sum(),
            (signals == Signal.SELL).sum(),
            (signals == Signal.HOLD).sum(),
        )

        return df
