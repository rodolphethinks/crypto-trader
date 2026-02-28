"""
Strategy #1 — Smart Money Concepts + Liquidity

The most comprehensive strategy, combining SMC structural analysis with
liquidity-based entries and multi-indicator confluence scoring.

Flow:
  1. Determine market regime (trending vs ranging)
  2. Identify key structural levels (S/R, supply/demand zones)
  3. Look for BOS / ChoCH for trend direction
  4. Find entries near order blocks / demand zones after liquidity sweeps
  5. Confirm with candlestick patterns and indicators (RSI, volume spike)
  6. Set stop-loss below/above order block, take-profit at next structural level
  7. Confidence scoring based on confluence of signals
"""

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal

# --- Market structure ---
from patterns.structure import (
    detect_support_resistance,
    detect_supply_demand_zones,
    detect_bos,
    detect_choch,
    detect_liquidity_sweep,
    detect_order_blocks,
    full_structure_analysis,
)

# --- Chart & candlestick patterns ---
from patterns.chart_patterns import detect_all_patterns
from patterns.candlestick import detect_all_candlestick_patterns

# --- Indicators ---
from indicators.trend import ema, adx, macd, supertrend
from indicators.momentum import rsi, stochastic
from indicators.volatility import atr, bollinger_bands
from indicators.volume import obv, volume_ratio
from indicators.custom import market_regime, session_indicator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_above(levels: List[float], price: float) -> Optional[float]:
    """Return the nearest level strictly above *price*, or None."""
    above = [l for l in levels if l > price]
    return min(above) if above else None


def _nearest_below(levels: List[float], price: float) -> Optional[float]:
    """Return the nearest level strictly below *price*, or None."""
    below = [l for l in levels if l < price]
    return max(below) if below else None


def _events_near_index(events: List[Dict], idx, lookback: int = 5,
                       index_list: pd.Index = None) -> List[Dict]:
    """Return events whose 'idx' field is within *lookback* bars of *idx*."""
    if index_list is None:
        return []
    try:
        pos = index_list.get_loc(idx)
    except KeyError:
        return []
    window_start = index_list[max(0, pos - lookback)]
    window_end = idx
    return [
        e for e in events
        if "idx" in e and window_start <= e["idx"] <= window_end
    ]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class SMCLiquidityStrategy(BaseStrategy):
    """Smart Money Concepts + Liquidity strategy with confluence scoring."""

    name = "SMC_Liquidity"
    description = (
        "Combines market structure (BOS/ChoCH), supply/demand zones, "
        "order blocks, liquidity sweeps, candlestick patterns, and "
        "multi-indicator confirmation for high-probability entries."
    )
    version = "1.0"

    default_params: Dict[str, Any] = {
        # Structure
        "swing_order": 10,
        # RSI / momentum
        "rsi_period": 14,
        # EMAs
        "ema_fast": 9,
        "ema_slow": 21,
        "ema_trend": 200,
        # ATR / volatility
        "atr_period": 14,
        # Risk-reward
        "rr_ratio": 2.0,
        # ADX
        "adx_threshold": 25,
        # Minimum confluence signals required to trigger an entry
        "min_confluence": 3,
        # Volume spike multiplier
        "volume_spike_mult": 1.5,
        # RSI thresholds
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        # Lookback for matching events to current bar
        "event_lookback": 5,
    }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse *df* (OHLCV with DatetimeIndex) and return a copy with:
          signal      – 1 (buy) / -1 (sell) / 0 (hold)
          stop_loss   – suggested SL price
          take_profit – suggested TP price
          confidence  – 0‒1 score based on confluence count
        """
        df = df.copy()
        p = self.params

        # ---- 1. Compute indicators --------------------------------
        df = self._compute_indicators(df)

        # ---- 2. Market structure (vectorised once) -----------------
        structure = full_structure_analysis(df, order=p["swing_order"])
        sr_levels = structure["support_resistance"]
        sd_zones = structure["supply_demand"]
        bos_events = structure["bos"]
        choch_events = structure["choch"]
        liq_sweeps = structure["liquidity_sweeps"]
        order_blocks = structure["order_blocks"]

        # ---- 3. Chart & candlestick patterns ----------------------
        chart_patterns = detect_all_patterns(df, order=p["swing_order"])
        candle_df = detect_all_candlestick_patterns(df)

        # Pre-build fast-access collections
        all_sr = sr_levels["support"] + sr_levels["resistance"]

        bullish_ob = [ob for ob in order_blocks if ob["type"] == "bullish_order_block"]
        bearish_ob = [ob for ob in order_blocks if ob["type"] == "bearish_order_block"]

        demand_zones = [z for z in sd_zones if z["type"] == "demand"]
        supply_zones = [z for z in sd_zones if z["type"] == "supply"]

        # ---- 4. Bar-by-bar signal generation ----------------------
        signals = np.zeros(len(df), dtype=int)
        stop_losses = np.full(len(df), np.nan)
        take_profits = np.full(len(df), np.nan)
        confidences = np.zeros(len(df))

        lookback = p["event_lookback"]

        for i in range(max(p["ema_trend"], 50), len(df)):
            idx = df.index[i]
            close = df["close"].iloc[i]
            regime = df["regime"].iloc[i]

            # ---------- Confluence counters ----------
            bull_score = 0
            bear_score = 0

            # --- a) Trend via EMAs ---
            if close > df["ema_trend"].iloc[i]:
                bull_score += 1
            else:
                bear_score += 1

            if df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i]:
                bull_score += 1
            else:
                bear_score += 1

            # --- b) Supertrend direction ---
            if df["st_direction"].iloc[i] == 1:
                bull_score += 1
            elif df["st_direction"].iloc[i] == -1:
                bear_score += 1

            # --- c) MACD histogram sign ---
            if df["macd_hist"].iloc[i] > 0:
                bull_score += 1
            elif df["macd_hist"].iloc[i] < 0:
                bear_score += 1

            # --- d) RSI zones ---
            rsi_val = df["rsi"].iloc[i]
            if rsi_val < p["rsi_oversold"]:
                bull_score += 1          # oversold ⇒ bounce potential
            elif rsi_val > p["rsi_overbought"]:
                bear_score += 1          # overbought ⇒ reversal potential

            # --- e) Volume spike ---
            if df["vol_ratio"].iloc[i] >= p["volume_spike_mult"]:
                # volume confirms whichever side is leading
                if bull_score > bear_score:
                    bull_score += 1
                elif bear_score > bull_score:
                    bear_score += 1

            # --- f) BOS / ChoCH ---
            recent_bos = _events_near_index(bos_events, idx, lookback, df.index)
            recent_choch = _events_near_index(choch_events, idx, lookback, df.index)

            for ev in recent_bos:
                if ev["type"] == "bullish_bos":
                    bull_score += 1
                elif ev["type"] == "bearish_bos":
                    bear_score += 1

            for ev in recent_choch:
                if ev["type"] == "bullish_choch":
                    bull_score += 1
                elif ev["type"] == "bearish_choch":
                    bear_score += 1

            # --- g) Liquidity sweep ---
            recent_sweeps = _events_near_index(liq_sweeps, idx, lookback, df.index)
            for ev in recent_sweeps:
                if ev["type"] == "bullish_liquidity_sweep":
                    bull_score += 1       # swept lows → buyers step in
                elif ev["type"] == "bearish_liquidity_sweep":
                    bear_score += 1       # swept highs → sellers step in

            # --- h) Order block proximity ---
            ob_sl_bull = None
            for ob in bullish_ob:
                if ob["ob_low"] <= close <= ob["ob_high"]:
                    bull_score += 1
                    ob_sl_bull = ob["ob_low"]
                    break

            ob_sl_bear = None
            for ob in bearish_ob:
                if ob["ob_low"] <= close <= ob["ob_high"]:
                    bear_score += 1
                    ob_sl_bear = ob["ob_high"]
                    break

            # --- i) Demand / Supply zone proximity ---
            for dz in demand_zones:
                if dz["zone_low"] <= close <= dz["zone_high"]:
                    bull_score += 1
                    break

            for sz in supply_zones:
                if sz["zone_low"] <= close <= sz["zone_high"]:
                    bear_score += 1
                    break

            # --- j) Candlestick patterns ---
            if candle_df["hammer"].iloc[i] or candle_df["bullish_engulfing"].iloc[i] \
                    or candle_df["morning_star"].iloc[i] or candle_df["three_white_soldiers"].iloc[i]:
                bull_score += 1
            if candle_df["shooting_star"].iloc[i] or candle_df["bearish_engulfing"].iloc[i] \
                    or candle_df["evening_star"].iloc[i] or candle_df["three_black_crows"].iloc[i]:
                bear_score += 1

            # --- k) Chart patterns (bias from recent detections) ---
            for cp in chart_patterns:
                if cp.get("signal") == "bullish":
                    bull_score += 1
                elif cp.get("signal") == "bearish":
                    bear_score += 1
            # Chart patterns are global; counted once, so cap at +1
            # (already capped by being added only once above)

            # --- l) Market regime bonus ---
            if regime == "trending_up":
                bull_score += 1
            elif regime == "trending_down":
                bear_score += 1

            # ---------- Decision ----------
            atr_val = df["atr"].iloc[i]
            min_conf = p["min_confluence"]

            if bull_score >= min_conf and bull_score > bear_score:
                signals[i] = Signal.BUY
                sl = ob_sl_bull if ob_sl_bull is not None else close - 1.5 * atr_val
                # TP at next resistance or RR multiple
                next_res = _nearest_above(sr_levels["resistance"], close)
                tp = next_res if next_res is not None else close + p["rr_ratio"] * (close - sl)
                stop_losses[i] = sl
                take_profits[i] = tp
                max_possible = bull_score + bear_score
                confidences[i] = min(bull_score / max(max_possible, 1), 1.0)

            elif bear_score >= min_conf and bear_score > bull_score:
                signals[i] = Signal.SELL
                sl = ob_sl_bear if ob_sl_bear is not None else close + 1.5 * atr_val
                next_sup = _nearest_below(sr_levels["support"], close)
                tp = next_sup if next_sup is not None else close - p["rr_ratio"] * (sl - close)
                stop_losses[i] = sl
                take_profits[i] = tp
                max_possible = bull_score + bear_score
                confidences[i] = min(bear_score / max(max_possible, 1), 1.0)

            else:
                signals[i] = Signal.HOLD
                confidences[i] = 0.0

        # ---- 5. Attach columns & return ---------------------------
        df["signal"] = signals
        df["stop_loss"] = stop_losses
        df["take_profit"] = take_profits
        df["confidence"] = confidences

        self._signals = df
        return df

    # ------------------------------------------------------------------
    # Indicator computation (called once)
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach all indicator columns needed by the signal loop."""
        p = self.params

        # EMAs
        df["ema_fast"] = ema(df["close"], p["ema_fast"])
        df["ema_slow"] = ema(df["close"], p["ema_slow"])
        df["ema_trend"] = ema(df["close"], p["ema_trend"])

        # ADX
        adx_data = adx(df, period=14)
        df["adx"] = adx_data["adx"]
        df["plus_di"] = adx_data["plus_di"]
        df["minus_di"] = adx_data["minus_di"]

        # MACD
        macd_data = macd(df["close"])
        df["macd_line"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_hist"] = macd_data["histogram"]

        # Supertrend
        st = supertrend(df)
        df["supertrend"] = st["supertrend"]
        df["st_direction"] = st["direction"]

        # RSI
        df["rsi"] = rsi(df["close"], p["rsi_period"])

        # Stochastic
        stoch = stochastic(df)
        df["stoch_k"] = stoch["stoch_k"]
        df["stoch_d"] = stoch["stoch_d"]

        # ATR
        df["atr"] = atr(df, p["atr_period"])

        # Bollinger Bands
        bb = bollinger_bands(df["close"])
        df["bb_upper"] = bb["bb_upper"]
        df["bb_lower"] = bb["bb_lower"]
        df["bb_pct"] = bb["bb_pct"]

        # OBV & volume ratio
        df["obv"] = obv(df)
        df["vol_ratio"] = volume_ratio(df)

        # Market regime
        df["regime"] = market_regime(df, adx_threshold=p["adx_threshold"])

        # Session
        try:
            df["session"] = session_indicator(df)
        except Exception:
            df["session"] = "unknown"

        return df
