"""
Strategy #11 — Ichimoku Cloud Strategy

Uses the full Ichimoku Kinko Hyo system (Tenkan-Sen / Kijun-Sen cross,
Kumo cloud position, Chikou Span confirmation) with RSI filtering and
ATR-based risk management.  Cloud edge-to-edge targets provide natural
take-profit levels.

BUY  : price above cloud, TK cross bullish, chikou confirms, RSI not overbought
SELL : price below cloud, TK cross bearish, chikou confirms, RSI not oversold
"""
import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal
from indicators.trend import ichimoku, ema
from indicators.momentum import rsi
from indicators.volatility import atr

logger = logging.getLogger(__name__)


class IchimokuCloudStrategy(BaseStrategy):
    """
    Ichimoku Cloud trend-following strategy.

    Entry logic (BUY — SELL is mirrored):
      1. Price is above the Kumo (both senkou_a and senkou_b)
      2. Tenkan-Sen > Kijun-Sen (TK cross / alignment)
      3. Chikou Span is above the price from 26 bars ago
      4. RSI is NOT overbought (< rsi_overbought)
      5. Optionally, cloud thickness gauges trend strength / confidence

    Risk:
      - Stop-loss   = entry ∓ sl_atr_mult × ATR
      - Take-profit = edge-to-edge cloud projection (capped by rr_ratio × risk)

    Confidence is a 0-1 score assembled from the number of confirming factors
    and the cloud thickness relative to price.
    """

    name: str = "IchimokuCloudStrategy"
    description: str = "Ichimoku Cloud trend strategy with TK cross and Chikou confirmation"
    version: str = "1.0"

    default_params: Dict = {
        "tenkan": 9,
        "kijun": 26,
        "senkou_b": 52,
        "displacement": 26,
        "rsi_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "rr_ratio": 2.0,
        # RSI thresholds
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        # Minimum cloud thickness (% of price) for a valid signal
        "min_cloud_pct": 0.0,
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
        ichi = ichimoku(
            df,
            tenkan=p["tenkan"],
            kijun=p["kijun"],
            senkou_b=p["senkou_b"],
            displacement=p["displacement"],
        )
        df["tenkan_sen"] = ichi["tenkan_sen"]
        df["kijun_sen"] = ichi["kijun_sen"]
        df["senkou_a"] = ichi["senkou_a"]
        df["senkou_b"] = ichi["senkou_b"]
        df["chikou"] = ichi["chikou"]

        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df, p["atr_period"])

        # Cloud boundaries (top / bottom regardless of which senkou is higher)
        df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
        df["cloud_bottom"] = df[["senkou_a", "senkou_b"]].min(axis=1)
        df["cloud_thickness"] = (df["cloud_top"] - df["cloud_bottom"]).abs()
        df["cloud_pct"] = df["cloud_thickness"] / df["close"]

        # Price vs cloud
        df["price_above_cloud"] = df["close"] > df["cloud_top"]
        df["price_below_cloud"] = df["close"] < df["cloud_bottom"]

        # TK cross / alignment
        df["tk_bullish"] = df["tenkan_sen"] > df["kijun_sen"]
        df["tk_bearish"] = df["tenkan_sen"] < df["kijun_sen"]
        df["tk_cross_up"] = (
            df["tk_bullish"] & (~df["tk_bullish"].shift(1).fillna(False))
        )
        df["tk_cross_down"] = (
            df["tk_bearish"] & (~df["tk_bearish"].shift(1).fillna(False))
        )

        # Chikou Span confirmation
        # Chikou is close shifted *back* by displacement, so to compare the
        # chikou with the price at the same historical position we compare
        # current close with close shifted forward by displacement (i.e. the
        # price that chikou would overlay).
        disp = p["displacement"]
        price_26_ago = df["close"].shift(disp)
        df["chikou_above"] = df["close"] > price_26_ago
        df["chikou_below"] = df["close"] < price_26_ago

        # ---- Signal generation ------------------------------------------
        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        for i in range(max(disp, p["senkou_b"]) + 1, len(df)):
            buy_score = self._buy_score(df, i, p)
            sell_score = self._sell_score(df, i, p)

            if buy_score >= 3:
                risk = p["sl_atr_mult"] * df["atr"].iat[i]
                sl = df["close"].iat[i] - risk

                # Edge-to-edge TP: distance from price to opposite cloud edge
                cloud_target = df["cloud_top"].iat[i] + df["cloud_thickness"].iat[i]
                rr_target = df["close"].iat[i] + p["rr_ratio"] * risk
                tp = min(cloud_target, rr_target) if cloud_target > df["close"].iat[i] else rr_target

                conf = self._confidence(buy_score, df["cloud_pct"].iat[i])

                df.iat[i, df.columns.get_loc("signal")] = Signal.BUY
                df.iat[i, df.columns.get_loc("stop_loss")] = sl
                df.iat[i, df.columns.get_loc("take_profit")] = tp
                df.iat[i, df.columns.get_loc("confidence")] = conf

            elif sell_score >= 3:
                risk = p["sl_atr_mult"] * df["atr"].iat[i]
                sl = df["close"].iat[i] + risk

                cloud_target = df["cloud_bottom"].iat[i] - df["cloud_thickness"].iat[i]
                rr_target = df["close"].iat[i] - p["rr_ratio"] * risk
                tp = max(cloud_target, rr_target) if cloud_target < df["close"].iat[i] else rr_target

                conf = self._confidence(sell_score, df["cloud_pct"].iat[i])

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
        Each factor adds 1 point (max 5):
          1. Price above cloud
          2. TK bullish (cross adds extra weight)
          3. Chikou above price 26 bars ago
          4. RSI not overbought
          5. Cloud thickness >= minimum
        """
        score = 0.0

        if df["price_above_cloud"].iat[i]:
            score += 1.0
        if df["tk_bullish"].iat[i]:
            score += 1.0
            if df["tk_cross_up"].iat[i]:
                score += 0.5  # bonus for fresh cross
        if df["chikou_above"].iat[i]:
            score += 1.0
        if df["rsi"].iat[i] < p["rsi_overbought"]:
            score += 1.0
        if df["cloud_pct"].iat[i] >= p["min_cloud_pct"]:
            score += 0.5

        return score

    @staticmethod
    def _sell_score(df: pd.DataFrame, i: int, p: Dict) -> float:
        """Mirror of _buy_score for short signals."""
        score = 0.0

        if df["price_below_cloud"].iat[i]:
            score += 1.0
        if df["tk_bearish"].iat[i]:
            score += 1.0
            if df["tk_cross_down"].iat[i]:
                score += 0.5
        if df["chikou_below"].iat[i]:
            score += 1.0
        if df["rsi"].iat[i] > p["rsi_oversold"]:
            score += 1.0
        if df["cloud_pct"].iat[i] >= p["min_cloud_pct"]:
            score += 0.5

        return score

    @staticmethod
    def _confidence(score: float, cloud_pct: float) -> float:
        """
        Combine the discrete score with cloud thickness into a 0-1 confidence.
        Thicker cloud ⇒ stronger trend ⇒ higher confidence.
        """
        base = min(score / 5.5, 1.0)  # 5.5 = max possible score
        # Boost slightly when cloud is thick (> 1 % of price)
        thickness_bonus = min(cloud_pct * 10, 0.15)  # up to +0.15
        return min(base + thickness_bonus, 1.0)
