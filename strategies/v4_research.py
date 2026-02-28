"""
V4 Research strategies — ensembles, optimized variants, and fixes.

1. Ensemble_Top3 — majority-vote of AdaptiveTrend + CrossTF_Momentum + Vol_Breakout
2. Ensemble_Weighted — composite score from KAMA + trend + squeeze signals
3. AdaptiveTrend_Tuned — volume-confirmed, tighter stops
4. CrossTF_Tuned — relaxed slope threshold, tighter RSI band
5. StatArb_Relaxed — relaxed Hurst filter for crypto
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from strategies.base import BaseStrategy, Signal
from strategies.alt_alpha import (
    AdaptiveTrendStrategy, CrossTFMomentumStrategy, VolatilityBreakoutStrategy,
    _atr, _rsi, _adx, _kaufman_ama, _hurst_exponent,
)

logger = logging.getLogger(__name__)


class EnsembleTop3Strategy(BaseStrategy):
    """
    Majority-vote ensemble of the 3 best walk-forward validated families.
    Runs AdaptiveTrend + CrossTF_Momentum + Vol_Breakout independently,
    takes trade when ≥2 agree within a lookback window.
    """
    name = "Ensemble_Top3"
    description = "Majority-vote of AdaptiveTrend + CrossTF + VolBreakout"
    version = "1.0"

    default_params = {
        "agreement_window": 3,
        "min_agree": 2,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 3.5,
        "atr_period": 14,
        "cooldown": 5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n = len(df)

        # Run sub-strategies with cooldown=1 so they generate all possible signals
        sig_at = AdaptiveTrendStrategy(params={"cooldown": 1}).generate_signals(df)["signal"]
        sig_ct = CrossTFMomentumStrategy(params={"cooldown": 1}).generate_signals(df)["signal"]
        sig_vb = VolatilityBreakoutStrategy(params={"cooldown": 1}).generate_signals(df)["signal"]

        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        atr = _atr(df, p["atr_period"])
        last_signal_bar = -p["cooldown"] - 1
        w = p["agreement_window"]

        for i in range(max(210, p["atr_period"] + 5), n):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            # Count votes in window
            buy_votes = 0
            sell_votes = 0
            for j in range(max(0, i - w + 1), i + 1):
                for sig in [sig_at, sig_ct, sig_vb]:
                    if sig.iloc[j] == Signal.BUY:
                        buy_votes += 1
                    elif sig.iloc[j] == Signal.SELL:
                        sell_votes += 1

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            if buy_votes >= p["min_agree"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(buy_votes / 3.0, 1.0)
                last_signal_bar = i
            elif sell_votes >= p["min_agree"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(sell_votes / 3.0, 1.0)
                last_signal_bar = i

        return signals


class EnsembleWeightedStrategy(BaseStrategy):
    """
    Composite-score ensemble using continuous indicator strengths.
    Combines KAMA direction + multi-lookback trend + squeeze state
    into a single [-1, +1] score; trades when score exceeds threshold.
    """
    name = "Ensemble_Weighted"
    description = "Weighted composite of KAMA + trend + squeeze signals"
    version = "1.0"

    default_params = {
        "er_period": 10,
        "fast_sc": 2,
        "slow_sc": 30,
        "trend_ema": 200,
        "medium_ema": 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_mult": 1.5,
        "buy_threshold": 0.55,
        "sell_threshold": -0.55,
        "adx_period": 14,
        "adx_min": 15,
        "rsi_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.5,
        "atr_period": 14,
        "cooldown": 4,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n = len(df)
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        # Component 1: KAMA direction [-1, +1]
        kama = _kaufman_ama(df["close"], p["er_period"], p["fast_sc"], p["slow_sc"])
        kama_diff = (df["close"] - kama) / kama
        kama_score = kama_diff.clip(-0.05, 0.05) / 0.05

        # Component 2: Multi-lookback trend [-1, +1]
        ema_long = df["close"].ewm(span=p["trend_ema"], adjust=False).mean()
        ema_med = df["close"].ewm(span=p["medium_ema"], adjust=False).mean()
        trend_score = pd.Series(0.0, index=df.index)
        trend_score += (df["close"] > ema_long).astype(float) * 0.5 - 0.25
        trend_score += (df["close"] > ema_med).astype(float) * 0.5 - 0.25
        ema_f = df["close"].ewm(span=p["macd_fast"], adjust=False).mean()
        ema_s = df["close"].ewm(span=p["macd_slow"], adjust=False).mean()
        macd_hist = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=p["macd_signal"], adjust=False).mean()
        macd_norm = macd_hist / df["close"] * 1000
        trend_score += macd_norm.clip(-0.5, 0.5)

        # Component 3: Squeeze breakout [-1, +1]
        sma_bb = df["close"].rolling(p["bb_period"]).mean()
        std_bb = df["close"].rolling(p["bb_period"]).std()
        bb_upper = sma_bb + p["bb_std"] * std_bb
        bb_lower = sma_bb - p["bb_std"] * std_bb
        ema_kc = df["close"].ewm(span=p["kc_period"], adjust=False).mean()
        atr_kc = _atr(df, p["kc_period"])
        kc_upper = ema_kc + p["kc_mult"] * atr_kc
        kc_lower = ema_kc - p["kc_mult"] * atr_kc
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_score = pd.Series(0.0, index=df.index)
        for i in range(1, n):
            if not squeeze.iloc[i] and squeeze.iloc[i-1]:
                squeeze_score.iloc[i] = 1.0 if df["close"].iloc[i] > sma_bb.iloc[i] else -1.0
            elif abs(squeeze_score.iloc[i-1]) > 0.01:
                squeeze_score.iloc[i] = squeeze_score.iloc[i-1] * 0.7

        # Composite: KAMA=0.4, Trend=0.35, Squeeze=0.25
        composite = 0.4 * kama_score + 0.35 * trend_score + 0.25 * squeeze_score

        adx = _adx(df, p["adx_period"])
        atr = _atr(df, p["atr_period"])
        rsi = _rsi(df["close"], p["rsi_period"])

        last_signal_bar = -p["cooldown"] - 1

        for i in range(max(210, p["atr_period"] + 5), n):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if np.isnan(adx.iloc[i]) or adx.iloc[i] < p["adx_min"]:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]
            score = composite.iloc[i]

            if score > p["buy_threshold"] and rsi.iloc[i] < 75:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(score), 1.0)
                last_signal_bar = i
            elif score < p["sell_threshold"] and rsi.iloc[i] > 25:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(score), 1.0)
                last_signal_bar = i

        return signals


class AdaptiveTrendTunedStrategy(BaseStrategy):
    """
    Optimized AdaptiveTrend with volume confirmation, tighter stops,
    and slightly faster KAMA parameters.
    """
    name = "AdaptiveTrend_Tuned"
    description = "Kaufman AMA + volume filter + tighter parameters"
    version = "1.0"

    default_params = {
        "er_period": 8,
        "fast_sc": 2,
        "slow_sc": 25,
        "adx_period": 14,
        "adx_min": 18,
        "rsi_period": 14,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "vol_avg_period": 20,
        "vol_min_ratio": 0.8,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 3.5,
        "atr_period": 14,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        kama = _kaufman_ama(df["close"], p["er_period"], p["fast_sc"], p["slow_sc"])
        adx = _adx(df, p["adx_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])
        vol_avg = df["volume"].rolling(p["vol_avg_period"]).mean()

        last_signal_bar = -p["cooldown"] - 1
        start = max(p["er_period"], p["adx_period"], p["atr_period"], p["vol_avg_period"]) + 5

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if np.isnan(adx.iloc[i]):
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Volume filter
            if vol_avg.iloc[i] > 0 and df["volume"].iloc[i] < vol_avg.iloc[i] * p["vol_min_ratio"]:
                continue

            if (close > kama.iloc[i]
                and df["close"].iloc[i-1] <= kama.iloc[i-1]
                and adx.iloc[i] > p["adx_min"]
                and rsi.iloc[i] < p["rsi_overbought"]
                and rsi.iloc[i] > p["rsi_oversold"]):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(adx.iloc[i] / 40, 1.0)
                last_signal_bar = i
            elif (close < kama.iloc[i]
                  and df["close"].iloc[i-1] >= kama.iloc[i-1]
                  and adx.iloc[i] > p["adx_min"]
                  and rsi.iloc[i] > p["rsi_oversold"]
                  and rsi.iloc[i] < p["rsi_overbought"]):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(adx.iloc[i] / 40, 1.0)
                last_signal_bar = i

        return signals


class CrossTFTunedStrategy(BaseStrategy):
    """
    CrossTF_Momentum with relaxed parameters:
    - Lower slope threshold for more signals
    - Wider RSI buy zone
    - Shorter trend EMA for faster adaptation
    """
    name = "CrossTF_Tuned"
    description = "CrossTF Momentum with relaxed slope + wider RSI zone"
    version = "1.0"

    default_params = {
        "trend_ema": 150,         # 150 vs 200 — faster adaptation
        "medium_slope_period": 40, # 40 vs 50 — more responsive
        "medium_slope_min": 0.0005, # half the original threshold
        "rsi_period": 14,
        "rsi_buy_zone": 40,       # 40 vs 45 — wider buy zone
        "rsi_sell_zone": 60,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 4.0,
        "atr_period": 14,
        "cooldown": 4,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        ema_long = df["close"].ewm(span=p["trend_ema"], adjust=False).mean()
        trend_up = df["close"] > ema_long
        trend_down = df["close"] < ema_long

        ema_med = df["close"].ewm(span=p["medium_slope_period"], adjust=False).mean()
        slope = (ema_med - ema_med.shift(5)) / ema_med.shift(5)

        rsi = _rsi(df["close"], p["rsi_period"])
        ema_f = df["close"].ewm(span=p["macd_fast"], adjust=False).mean()
        ema_s = df["close"].ewm(span=p["macd_slow"], adjust=False).mean()
        macd = ema_f - ema_s
        macd_sig = macd.ewm(span=p["macd_signal"], adjust=False).mean()
        macd_hist = macd - macd_sig

        atr = _atr(df, p["atr_period"])
        last_signal_bar = -p["cooldown"] - 1
        start = p["trend_ema"] + 5

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            if (trend_up.iloc[i]
                and slope.iloc[i] > p["medium_slope_min"]
                and rsi.iloc[i] > p["rsi_buy_zone"] and rsi.iloc[i] < 68
                and macd_hist.iloc[i] > 0 and macd_hist.iloc[i-1] <= 0):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                last_signal_bar = i
            elif (trend_down.iloc[i]
                  and slope.iloc[i] < -p["medium_slope_min"]
                  and rsi.iloc[i] < p["rsi_sell_zone"] and rsi.iloc[i] > 32
                  and macd_hist.iloc[i] < 0 and macd_hist.iloc[i-1] >= 0):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                last_signal_bar = i

        return signals


class StatArbRelaxedStrategy(BaseStrategy):
    """
    StatArb with relaxed Hurst filter (0.55 vs 0.45) for crypto markets.
    Crypto rarely has H < 0.45, so the original produced 0 trades.
    """
    name = "StatArb_Relaxed"
    description = "Z-score mean reversion with relaxed Hurst filter"
    version = "1.0"

    default_params = {
        "lookback": 40,
        "entry_z": 1.8,
        "hurst_period": 80,
        "hurst_max": 0.55,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 2.5,
        "atr_period": 14,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        sma = df["close"].rolling(p["lookback"]).mean()
        std = df["close"].rolling(p["lookback"]).std()
        z_score = (df["close"] - sma) / std.replace(0, np.nan)

        hurst = df["close"].rolling(p["hurst_period"]).apply(
            _hurst_exponent, raw=True
        )

        atr = _atr(df, p["atr_period"])
        last_signal_bar = -p["cooldown"] - 1
        start = max(p["lookback"], p["hurst_period"]) + 1

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            z = z_score.iloc[i]
            h = hurst.iloc[i]
            if np.isnan(z) or np.isnan(h):
                continue
            if h > p["hurst_max"]:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            if z < -p["entry_z"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = sma.iloc[i]
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(z) / 4.0, 1.0)
                last_signal_bar = i
            elif z > p["entry_z"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = sma.iloc[i]
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(z) / 4.0, 1.0)
                last_signal_bar = i

        return signals
