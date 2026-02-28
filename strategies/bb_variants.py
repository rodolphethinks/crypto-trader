"""
Bollinger Bands Strategy Variants — from simple to complex.

Eight levels of increasing sophistication:

  Level 1 – BB_Naive:        Pure band touch, no filters
  Level 2 – BB_RSI:          Band touch + RSI oversold/overbought
  Level 3 – BB_Trend:        Band touch + RSI + EMA trend filter
  Level 4 – BB_Volume:       Band touch + RSI + volume spike
  Level 5 – BB_MACD:         Band touch + MACD histogram confirmation
  Level 6 – BB_Double:       Dual-band (1.5σ / 2.5σ) entry/exit
  Level 7 – BB_Squeeze:      Keltner squeeze → breakout entry
  Level 8 – BB_MultiConf:    BB + RSI + Stoch + CMF + ADX (max confluence)
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.volatility import bollinger_bands, atr, squeeze_momentum, keltner_channel
from indicators.momentum import rsi, mfi, stochastic, stoch_rsi
from indicators.trend import ema, macd, adx
from indicators.volume import volume_ratio, cmf, obv

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — Naive: pure band touch
# ═══════════════════════════════════════════════════════════════════════════════

class BB_Naive(BaseStrategy):
    """Simplest BB: buy at lower band, sell at upper band. No filters."""

    name = "BB_Naive"
    description = "Buy when close <= lower BB, sell when close >= upper BB. ATR stop."
    version = "1.0"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "sl_atr_mult": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        buy = df["close"] <= df["bb_lower"]
        sell = df["close"] >= df["bb_upper"]

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "bb_middle"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "bb_middle"]

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — RSI: band touch + RSI confirmation
# ═══════════════════════════════════════════════════════════════════════════════

class BB_RSI(BaseStrategy):
    """BB band touch confirmed by RSI oversold/overbought."""

    name = "BB_RSI"
    description = "Buy at lower BB when RSI oversold; sell at upper BB when RSI overbought."
    version = "1.0"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "atr_period": 14,
        "sl_atr_mult": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        buy = (df["close"] <= df["bb_lower"]) & (df["rsi"] <= p["rsi_oversold"])
        sell = (df["close"] >= df["bb_upper"]) & (df["rsi"] >= p["rsi_overbought"])

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "bb_middle"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "bb_middle"]

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — Trend: band touch + RSI + EMA filter
# ═══════════════════════════════════════════════════════════════════════════════

class BB_Trend(BaseStrategy):
    """BB + RSI + EMA: buy on lower-band pullback in uptrend, sell on upper-band rally in downtrend."""

    name = "BB_Trend"
    description = "Band proximity + RSI + EMA trend filter. Trades pullbacks within the trend."
    version = "1.1"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 40,        # loosened — only need mild oversold
        "rsi_overbought": 60,
        "ema_period": 50,
        "bb_pct_buy": 0.15,        # buy when bb_%b < 15% (near lower band)
        "bb_pct_sell": 0.85,        # sell when bb_%b > 85% (near upper band)
        "atr_period": 14,
        "sl_atr_mult": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_pct"] = bb["bb_pct"]
        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["ema_trend"] = ema(df["close"], p["ema_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        # Buy: near lower band + RSI softly oversold + price above EMA (uptrend pullback)
        buy = ((df["bb_pct"] <= p["bb_pct_buy"]) &
               (df["rsi"] <= p["rsi_oversold"]) &
               (df["close"] > df["ema_trend"]))

        # Sell: near upper band + RSI softly overbought + price below EMA (downtrend rally)
        sell = ((df["bb_pct"] >= p["bb_pct_sell"]) &
                (df["rsi"] >= p["rsi_overbought"]) &
                (df["close"] < df["ema_trend"]))

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "bb_middle"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "bb_middle"]

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 4 — Volume: band touch + RSI + volume spike
# ═══════════════════════════════════════════════════════════════════════════════

class BB_Volume(BaseStrategy):
    """BB + RSI + volume above average confirmation."""

    name = "BB_Volume"
    description = "Band touch + RSI + above-average volume. High-conviction reversals."
    version = "1.0"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "vol_ratio_threshold": 1.2,   # volume must be 1.2x average
        "vol_avg_period": 20,
        "atr_period": 14,
        "sl_atr_mult": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["vol_ratio"] = volume_ratio(df, p["vol_avg_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        vol_ok = df["vol_ratio"] >= p["vol_ratio_threshold"]

        buy = ((df["close"] <= df["bb_lower"]) &
               (df["rsi"] <= p["rsi_oversold"]) &
               vol_ok)

        sell = ((df["close"] >= df["bb_upper"]) &
                (df["rsi"] >= p["rsi_overbought"]) &
                vol_ok)

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "bb_middle"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "bb_middle"]

        # Confidence based on volume strength
        for idx in df.index[df["signal"] != Signal.HOLD]:
            base = 0.5
            if df.at[idx, "vol_ratio"] > 2.0:
                base += 0.3
            elif df.at[idx, "vol_ratio"] > 1.5:
                base += 0.2
            else:
                base += 0.1
            df.at[idx, "confidence"] = min(base, 1.0)

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 5 — MACD: band touch + MACD histogram confirms direction
# ═══════════════════════════════════════════════════════════════════════════════

class BB_MACD(BaseStrategy):
    """BB proximity + MACD histogram turning: momentum must shift toward entry direction."""

    name = "BB_MACD"
    description = "Near BB band + MACD histogram turning + mild RSI. Momentum-aligned entries."
    version = "1.1"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "bb_pct_buy": 0.15,        # near lower band
        "bb_pct_sell": 0.85,        # near upper band
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_oversold": 40,
        "rsi_overbought": 60,
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        "rr_ratio": 2.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_pct"] = bb["bb_pct"]

        macd_df = macd(df["close"], p["macd_fast"], p["macd_slow"], p["macd_signal"])
        df["macd_hist"] = macd_df["histogram"]
        df["macd_hist_rising"] = df["macd_hist"] > df["macd_hist"].shift(1)
        df["macd_hist_falling"] = df["macd_hist"] < df["macd_hist"].shift(1)

        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        # Buy: near lower band + RSI mildly oversold + MACD hist turning up
        buy = ((df["bb_pct"] <= p["bb_pct_buy"]) &
               (df["rsi"] <= p["rsi_oversold"]) &
               df["macd_hist_rising"])

        # Sell: near upper band + RSI mildly overbought + MACD hist turning down
        sell = ((df["bb_pct"] >= p["bb_pct_sell"]) &
                (df["rsi"] >= p["rsi_overbought"]) &
                df["macd_hist_falling"])

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "close"] + atr_sl[buy] * p["rr_ratio"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "close"] - atr_sl[sell] * p["rr_ratio"]

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 6 — Double Band: inner (1.5σ) + outer (2.5σ)
# ═══════════════════════════════════════════════════════════════════════════════

class BB_Double(BaseStrategy):
    """Dual Bollinger Bands: entry at outer band (2.5σ), TP at inner band (1.5σ)."""

    name = "BB_Double"
    description = "Entry at extreme outer band (2.5σ), take-profit at inner band (1.5σ)."
    version = "1.0"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_inner_std": 1.5,
        "bb_outer_std": 2.5,
        "rsi_period": 14,
        "rsi_oversold": 25,       # stricter for extreme entries
        "rsi_overbought": 75,
        "atr_period": 14,
        "sl_atr_mult": 2.0,       # wider stop for extreme entries
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb_inner = bollinger_bands(df["close"], p["bb_period"], p["bb_inner_std"])
        bb_outer = bollinger_bands(df["close"], p["bb_period"], p["bb_outer_std"])

        df["bb_outer_upper"] = bb_outer["bb_upper"]
        df["bb_outer_lower"] = bb_outer["bb_lower"]
        df["bb_inner_upper"] = bb_inner["bb_upper"]
        df["bb_inner_lower"] = bb_inner["bb_lower"]
        df["bb_middle"] = bb_inner["bb_middle"]  # same for both

        df["rsi"] = rsi(df["close"], p["rsi_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        # Buy: close at or below outer lower band + RSI extremely oversold
        buy = ((df["close"] <= df["bb_outer_lower"]) &
               (df["rsi"] <= p["rsi_oversold"]))

        # Sell: close at or above outer upper band + RSI extremely overbought
        sell = ((df["close"] >= df["bb_outer_upper"]) &
                (df["rsi"] >= p["rsi_overbought"]))

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        # TP at inner band (closer target, higher probability)
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "bb_inner_lower"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "bb_inner_upper"]

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 7 — Squeeze Breakout: Keltner squeeze → momentum breakout
# ═══════════════════════════════════════════════════════════════════════════════

class BB_Squeeze(BaseStrategy):
    """BB squeeze detection (inside Keltner) → breakout entry on release."""

    name = "BB_Squeeze"
    description = "Detects BB squeeze inside Keltner Channel, enters on breakout with momentum."
    version = "1.0"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_atr_period": 14,
        "kc_mult": 1.5,
        "ema_period": 50,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "rr_ratio": 2.5,
        "min_squeeze_bars": 3,   # require squeeze for at least N bars
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]

        kc = keltner_channel(df, p["kc_period"], p["kc_atr_period"], p["kc_mult"])
        df["kc_upper"] = kc["kc_upper"]
        df["kc_lower"] = kc["kc_lower"]

        sq = squeeze_momentum(df, bb_period=p["bb_period"], bb_std=p["bb_std"],
                               kc_period=p["kc_period"], kc_atr_period=p["kc_atr_period"],
                               kc_mult=p["kc_mult"])
        df["squeeze_on"] = sq["squeeze_on"]
        df["squeeze_mom"] = sq["momentum"]

        # Count consecutive squeeze bars
        squeeze_count = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df["squeeze_on"].iloc[i]:
                squeeze_count.iloc[i] = squeeze_count.iloc[i - 1] + 1
            else:
                squeeze_count.iloc[i] = 0
        df["squeeze_count"] = squeeze_count

        # Squeeze release: was in squeeze (enough bars), now released
        was_squeezed = df["squeeze_count"].shift(1) >= p["min_squeeze_bars"]
        now_released = ~df["squeeze_on"]
        df["squeeze_release"] = was_squeezed & now_released

        df["ema_trend"] = ema(df["close"], p["ema_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.5

        # Bullish breakout: squeeze release + positive momentum + above EMA
        buy = (df["squeeze_release"] &
               (df["squeeze_mom"] > 0) &
               (df["close"] > df["ema_trend"]))

        # Bearish breakout: squeeze release + negative momentum + below EMA
        sell = (df["squeeze_release"] &
                (df["squeeze_mom"] < 0) &
                (df["close"] < df["ema_trend"]))

        df.loc[buy, "signal"] = Signal.BUY
        df.loc[sell, "signal"] = Signal.SELL

        atr_sl = df["atr"] * p["sl_atr_mult"]
        df.loc[buy, "stop_loss"] = df.loc[buy, "close"] - atr_sl[buy]
        df.loc[buy, "take_profit"] = df.loc[buy, "close"] + atr_sl[buy] * p["rr_ratio"]
        df.loc[sell, "stop_loss"] = df.loc[sell, "close"] + atr_sl[sell]
        df.loc[sell, "take_profit"] = df.loc[sell, "close"] - atr_sl[sell] * p["rr_ratio"]

        # Confidence based on squeeze strength
        for idx in df.index[df["signal"] != Signal.HOLD]:
            sq_bars = squeeze_count.shift(1).get(idx, 0)
            mom = abs(df.at[idx, "squeeze_mom"])
            conf = 0.5 + min(0.3, sq_bars * 0.02) + min(0.2, mom / df.at[idx, "close"] * 50)
            df.at[idx, "confidence"] = min(round(conf, 2), 1.0)

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 8 — Multi-Confluence: BB + RSI + Stoch + CMF + ADX
# ═══════════════════════════════════════════════════════════════════════════════

class BB_MultiConf(BaseStrategy):
    """Maximum confluence: BB + RSI + Stochastic + CMF + ADX. Highest quality signals."""

    name = "BB_MultiConf"
    description = "Multi-indicator confluence: BB + RSI + Stoch + CMF + ADX. Strict entries."
    version = "1.0"

    default_params: Dict[str, Any] = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_smooth": 3,
        "stoch_oversold": 20,
        "stoch_overbought": 80,
        "cmf_period": 20,
        "adx_period": 14,
        "adx_threshold": 20,     # ADX > 20 = trending
        "ema_period": 50,
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        "min_confluence": 4,     # need at least 4 of 5 conditions
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # Compute all indicators
        bb = bollinger_bands(df["close"], p["bb_period"], p["bb_std"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_middle"] = bb["bb_middle"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_pct"] = bb["bb_pct"]

        df["rsi"] = rsi(df["close"], p["rsi_period"])

        stoch_df = stochastic(df, p["stoch_k"], p["stoch_d"], p["stoch_smooth"])
        df["stoch_k"] = stoch_df["stoch_k"]
        df["stoch_d"] = stoch_df["stoch_d"]

        df["cmf"] = cmf(df, p["cmf_period"])

        adx_df = adx(df, p["adx_period"])
        df["adx"] = adx_df["adx"]
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]

        df["ema_trend"] = ema(df["close"], p["ema_period"])
        df["atr"] = atr(df, p["atr_period"])

        df["signal"] = Signal.HOLD
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["confidence"] = 0.0

        min_conf = p["min_confluence"]

        for i in range(max(p["bb_period"], p["adx_period"], p["ema_period"]) + 5, len(df)):
            row = df.iloc[i]

            # Skip if ADX too low (choppy market)
            if pd.isna(row["adx"]) or row["adx"] < p["adx_threshold"]:
                continue

            # ── BUY confluence ──
            buy_checks = [
                row["close"] <= row["bb_lower"],                    # 1. At lower band
                row["rsi"] <= p["rsi_oversold"],                    # 2. RSI oversold
                row["stoch_k"] <= p["stoch_oversold"],              # 3. Stoch oversold
                row["cmf"] > 0,                                     # 4. Money flowing in
                row["plus_di"] > row["minus_di"],                   # 5. +DI > -DI
            ]

            if sum(buy_checks) >= min_conf:
                idx = df.index[i]
                df.at[idx, "signal"] = Signal.BUY
                atr_val = row["atr"] * p["sl_atr_mult"]
                df.at[idx, "stop_loss"] = row["close"] - atr_val
                df.at[idx, "take_profit"] = row["bb_middle"]
                df.at[idx, "confidence"] = round(sum(buy_checks) / len(buy_checks), 2)
                continue

            # ── SELL confluence ──
            sell_checks = [
                row["close"] >= row["bb_upper"],                    # 1. At upper band
                row["rsi"] >= p["rsi_overbought"],                  # 2. RSI overbought
                row["stoch_k"] >= p["stoch_overbought"],            # 3. Stoch overbought
                row["cmf"] < 0,                                     # 4. Money flowing out
                row["minus_di"] > row["plus_di"],                   # 5. -DI > +DI
            ]

            if sum(sell_checks) >= min_conf:
                idx = df.index[i]
                df.at[idx, "signal"] = Signal.SELL
                atr_val = row["atr"] * p["sl_atr_mult"]
                df.at[idx, "stop_loss"] = row["close"] + atr_val
                df.at[idx, "take_profit"] = row["bb_middle"]
                df.at[idx, "confidence"] = round(sum(sell_checks) / len(sell_checks), 2)

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — all BB variants for sweep
# ═══════════════════════════════════════════════════════════════════════════════

def get_bb_variants() -> Dict[str, type]:
    """Return all BB strategy variants keyed by name."""
    return {
        "BB_Naive": BB_Naive,
        "BB_RSI": BB_RSI,
        "BB_Trend": BB_Trend,
        "BB_Volume": BB_Volume,
        "BB_MACD": BB_MACD,
        "BB_Double": BB_Double,
        "BB_Squeeze": BB_Squeeze,
        "BB_MultiConf": BB_MultiConf,
    }
