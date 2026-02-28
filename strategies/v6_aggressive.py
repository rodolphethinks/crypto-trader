"""
V6 Aggressive strategies — designed for higher returns with controlled risk.

Key differences from V1-V5:
- Trailing stops instead of fixed TP (ride trends longer)
- Dynamic position sizing based on signal confidence
- Faster regime adaptation
- More aggressive risk parameters (2-3% risk per trade)

Strategies:
1. TrendRider — Supertrend + KAMA + trailing stop (hold winners longer)
2. MomentumAccelerator — catches momentum acceleration with volume confirm
3. RegimeMomentumV2 — improved regime switching with faster adaptation
4. BreakoutAccumulator — Wyckoff-style accumulation/distribution detection
5. DynamicKellyTrader — Full Kelly criterion for edge maximization
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from strategies.base import BaseStrategy, Signal
from strategies.alt_alpha import _atr, _rsi, _adx, _kaufman_ama

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Calculate Supertrend indicator."""
    atr = _atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 = up, -1 = down
    
    for i in range(period, len(df)):
        if np.isnan(upper_band.iloc[i]):
            continue
            
        # Adjust bands
        if i > period:
            if not np.isnan(lower_band.iloc[i-1]):
                if lower_band.iloc[i] < lower_band.iloc[i-1] and df["close"].iloc[i-1] > lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
            if not np.isnan(upper_band.iloc[i-1]):
                if upper_band.iloc[i] > upper_band.iloc[i-1] and df["close"].iloc[i-1] < upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
        
        # Determine direction
        if i > period:
            prev_st = supertrend.iloc[i-1]
            if not np.isnan(prev_st):
                if prev_st == upper_band.iloc[i-1]:
                    direction.iloc[i] = -1 if df["close"].iloc[i] > upper_band.iloc[i] else -1 if df["close"].iloc[i] < upper_band.iloc[i] else direction.iloc[i-1]
                else:
                    direction.iloc[i] = 1 if df["close"].iloc[i] > lower_band.iloc[i] else -1
            
        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
    
    return supertrend, direction


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _momentum_acceleration(close: pd.Series, period: int = 14) -> pd.Series:
    """Rate of change of momentum (second derivative of price)."""
    mom = close.pct_change(period)
    return mom.diff()


def _volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def _vwap_deviation(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Price deviation from rolling VWAP."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()
    return (df["close"] - vwap) / vwap


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R oscillator."""
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


class TrendRiderStrategy(BaseStrategy):
    """
    Rides trends using Supertrend + KAMA + ADX with trailing stop.
    
    KEY DIFFERENCE: Uses trailing stop (ATR-based) instead of fixed TP.
    This lets profits run in strong trends while protecting capital.
    
    Entry: Supertrend flip + KAMA confirmation + ADX > threshold
    Exit: Trailing stop or Supertrend reversal
    """
    name = "TrendRider"
    description = "Supertrend + KAMA trend rider with trailing stops"
    version = "6.0"

    default_params = {
        "st_period": 10,
        "st_multiplier": 3.0,
        "er_period": 10,
        "fast_sc": 2,
        "slow_sc": 30,
        "adx_period": 14,
        "adx_min": 22,
        "rsi_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 2.5,       # Initial stop
        "trailing_atr_mult": 2.0,  # Trailing stop tightens
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
        
        # Supertrend
        st, st_dir = _supertrend(df, p["st_period"], p["st_multiplier"])

        last_signal_bar = -p["cooldown"] - 1
        start = max(p["st_period"], p["er_period"], p["adx_period"], p["atr_period"]) + 10

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if np.isnan(adx.iloc[i]):
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Supertrend flipped UP + KAMA bullish + ADX trending
            if (st_dir.iloc[i] == 1 and st_dir.iloc[i-1] == -1
                and close > kama.iloc[i]
                and adx.iloc[i] > p["adx_min"]
                and rsi.iloc[i] < 75):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                # No fixed TP — use large target to let trailing stop work
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["sl_atr_mult"] * a * 3
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(adx.iloc[i] / 50, 1.0)
                last_signal_bar = i

            elif (st_dir.iloc[i] == -1 and st_dir.iloc[i-1] == 1
                  and close < kama.iloc[i]
                  and adx.iloc[i] > p["adx_min"]
                  and rsi.iloc[i] > 25):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["sl_atr_mult"] * a * 3
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(adx.iloc[i] / 50, 1.0)
                last_signal_bar = i

        return signals


class MomentumAcceleratorStrategy(BaseStrategy):
    """
    Detects momentum ACCELERATION — not just momentum, but the speeding up
    of momentum. Combined with volume surge for confirmation.
    
    Entry: Momentum acceleration + volume > 1.5x average + trend alignment
    Exit: Momentum deceleration or stop loss
    """
    name = "MomAccelerator"
    description = "Catches momentum acceleration with volume confirmation"
    version = "6.0"

    default_params = {
        "mom_period": 10,
        "accel_threshold": 0.002,  # Minimum acceleration
        "vol_mult": 1.3,           # Volume must be 1.3x average
        "vol_period": 20,
        "trend_ema": 50,
        "rsi_period": 14,
        "adx_period": 14,
        "adx_min": 18,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        # Momentum and its acceleration
        mom = df["close"].pct_change(p["mom_period"])
        accel = mom.diff()

        # Volume analysis
        vol_avg = df["volume"].rolling(p["vol_period"]).mean()
        vol_ratio = df["volume"] / vol_avg.replace(0, np.nan)

        # Trend
        ema_trend = _ema(df["close"], p["trend_ema"])
        adx = _adx(df, p["adx_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])

        last_signal_bar = -p["cooldown"] - 1
        start = max(p["mom_period"], p["vol_period"], p["trend_ema"], p["atr_period"]) + 5

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # BUY: Positive acceleration + volume surge + above trend
            if (accel.iloc[i] > p["accel_threshold"]
                and mom.iloc[i] > 0
                and vol_ratio.iloc[i] > p["vol_mult"]
                and close > ema_trend.iloc[i]
                and not np.isnan(adx.iloc[i]) and adx.iloc[i] > p["adx_min"]
                and rsi.iloc[i] < 78
                and rsi.iloc[i] > 40):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                conf = min(accel.iloc[i] / (p["accel_threshold"] * 3), 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_signal_bar = i

            # SELL: Negative acceleration + volume surge + below trend
            elif (accel.iloc[i] < -p["accel_threshold"]
                  and mom.iloc[i] < 0
                  and vol_ratio.iloc[i] > p["vol_mult"]
                  and close < ema_trend.iloc[i]
                  and not np.isnan(adx.iloc[i]) and adx.iloc[i] > p["adx_min"]
                  and rsi.iloc[i] > 22
                  and rsi.iloc[i] < 60):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                conf = min(abs(accel.iloc[i]) / (p["accel_threshold"] * 3), 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_signal_bar = i

        return signals


class RegimeMomentumV2Strategy(BaseStrategy):
    """
    Improved regime detection with faster adaptation.
    
    Uses 3 regime states:
    - TREND: ride momentum aggressively (wider TP, trailing stop)
    - RANGE: mean reversion with tight targets
    - VOLATILE: momentum but smaller size, tighter stops
    
    Key improvement over V5 RegimeMomHybrid:
    - Faster regime switching (20-bar lookback vs 50)
    - Adaptive stop distances based on recent volatility
    - Volume-weighted momentum signals
    """
    name = "RegimeMomV2"
    description = "Fast-adapt regime detection + momentum/reversion"
    version = "6.0"

    default_params = {
        # Regime detection
        "regime_lookback": 25,
        "adx_period": 14,
        "adx_trend_threshold": 25,
        "adx_range_threshold": 18,
        "vol_lookback": 20,
        # Trend-following params
        "fast_ema": 9,
        "slow_ema": 21,
        "rsi_period": 14,
        # Mean reversion params
        "bb_period": 20,
        "bb_std": 2.0,
        # Risk
        "atr_period": 14,
        "trend_sl_mult": 2.0,
        "trend_tp_mult": 5.0,    # Wide TP in trends
        "range_sl_mult": 1.5,
        "range_tp_mult": 2.0,    # Tight TP in ranges
        "vol_sl_mult": 1.5,
        "vol_tp_mult": 3.0,
        "cooldown": 2,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        # Calculate indicators
        adx = _adx(df, p["adx_period"])
        atr = _atr(df, p["atr_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        
        ema_fast = _ema(df["close"], p["fast_ema"])
        ema_slow = _ema(df["close"], p["slow_ema"])
        
        sma = df["close"].rolling(p["bb_period"]).mean()
        std = df["close"].rolling(p["bb_period"]).std()
        bb_upper = sma + p["bb_std"] * std
        bb_lower = sma - p["bb_std"] * std
        
        # Volatility regime: compare recent vol to longer-term
        recent_vol = df["close"].pct_change().rolling(p["vol_lookback"]).std()
        long_vol = df["close"].pct_change().rolling(p["vol_lookback"] * 3).std()
        vol_ratio = recent_vol / long_vol.replace(0, np.nan)
        
        # OBV for volume confirmation
        obv = _obv(df["close"], df["volume"])
        obv_ema = _ema(obv, 20)
        
        last_signal_bar = -p["cooldown"] - 1
        start = max(p["regime_lookback"], p["adx_period"], p["vol_lookback"] * 3) + 10

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if np.isnan(adx.iloc[i]):
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]
            
            # Determine regime
            is_volatile = not np.isnan(vol_ratio.iloc[i]) and vol_ratio.iloc[i] > 1.3
            is_trending = adx.iloc[i] > p["adx_trend_threshold"]
            is_ranging = adx.iloc[i] < p["adx_range_threshold"]
            
            sig = Signal.HOLD
            sl_mult = p["trend_sl_mult"]
            tp_mult = p["trend_tp_mult"]
            conf = 0.5

            if is_trending and not is_volatile:
                # TREND MODE — aggressive momentum
                sl_mult = p["trend_sl_mult"]
                tp_mult = p["trend_tp_mult"]
                
                if (ema_fast.iloc[i] > ema_slow.iloc[i]
                    and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]
                    and rsi.iloc[i] > 45 and rsi.iloc[i] < 75
                    and obv.iloc[i] > obv_ema.iloc[i]):
                    sig = Signal.BUY
                    conf = min(adx.iloc[i] / 40, 1.0)
                elif (ema_fast.iloc[i] < ema_slow.iloc[i]
                      and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]
                      and rsi.iloc[i] < 55 and rsi.iloc[i] > 25
                      and obv.iloc[i] < obv_ema.iloc[i]):
                    sig = Signal.SELL
                    conf = min(adx.iloc[i] / 40, 1.0)

            elif is_ranging:
                # RANGE MODE — mean reversion
                sl_mult = p["range_sl_mult"]
                tp_mult = p["range_tp_mult"]
                
                if close < bb_lower.iloc[i] and rsi.iloc[i] < 30:
                    sig = Signal.BUY
                    conf = min((bb_lower.iloc[i] - close) / (a * 2), 1.0)
                elif close > bb_upper.iloc[i] and rsi.iloc[i] > 70:
                    sig = Signal.SELL
                    conf = min((close - bb_upper.iloc[i]) / (a * 2), 1.0)

            elif is_volatile:
                # VOLATILE MODE — momentum with tight stops
                sl_mult = p["vol_sl_mult"]
                tp_mult = p["vol_tp_mult"]
                
                # Only trade strong momentum in volatile regime
                mom_5 = (close - df["close"].iloc[i-5]) / df["close"].iloc[i-5]
                vol_confirm = df["volume"].iloc[i] > df["volume"].rolling(20).mean().iloc[i] * 1.5
                
                if mom_5 > 0.03 and vol_confirm and rsi.iloc[i] < 80:
                    sig = Signal.BUY
                    conf = 0.6
                elif mom_5 < -0.03 and vol_confirm and rsi.iloc[i] > 20:
                    sig = Signal.SELL
                    conf = 0.6

            if sig != Signal.HOLD:
                signals.iloc[i, signals.columns.get_loc("signal")] = sig
                if sig == Signal.BUY:
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - sl_mult * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = close + tp_mult * a
                else:
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + sl_mult * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = close - tp_mult * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_signal_bar = i

        return signals


class BreakoutAccumulatorStrategy(BaseStrategy):
    """
    Wyckoff-style accumulation/distribution detection.
    
    Looks for:
    1. Consolidation period (low volatility, narrowing range)
    2. Volume accumulation (OBV rising while price flat)
    3. Breakout with volume confirmation
    
    This catches the 'spring' move after accumulation.
    """
    name = "BreakoutAccum"
    description = "Wyckoff accumulation/distribution breakout"
    version = "6.0"

    default_params = {
        "consolidation_period": 20,
        "consolidation_max_range": 0.04,  # Max 4% range during consolidation
        "vol_divergence_bars": 10,
        "breakout_vol_mult": 1.8,
        "rsi_period": 14,
        "adx_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 4.0,
        "cooldown": 5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        atr = _atr(df, p["atr_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        obv = _obv(df["close"], df["volume"])
        vol_avg = df["volume"].rolling(p["consolidation_period"]).mean()

        last_signal_bar = -p["cooldown"] - 1
        start = p["consolidation_period"] + p["vol_divergence_bars"] + 5

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Check consolidation: price range over period
            period_high = df["high"].iloc[i-p["consolidation_period"]:i].max()
            period_low = df["low"].iloc[i-p["consolidation_period"]:i].min()
            price_range = (period_high - period_low) / period_low

            if price_range > p["consolidation_max_range"]:
                continue  # Not in consolidation

            # Check OBV direction during consolidation
            obv_start = obv.iloc[i - p["vol_divergence_bars"]]
            obv_now = obv.iloc[i]
            obv_rising = obv_now > obv_start
            obv_falling = obv_now < obv_start

            # Check for breakout with volume
            vol_surge = df["volume"].iloc[i] > vol_avg.iloc[i] * p["breakout_vol_mult"]

            if not vol_surge:
                continue

            # BULLISH BREAKOUT: OBV accumulating + price breaks above range
            if (obv_rising and close > period_high
                and rsi.iloc[i] > 50 and rsi.iloc[i] < 80):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = period_low  # Below range
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                conf = min(df["volume"].iloc[i] / (vol_avg.iloc[i] * 2), 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_signal_bar = i

            # BEARISH BREAKOUT: OBV distributing + price breaks below range
            elif (obv_falling and close < period_low
                  and rsi.iloc[i] < 50 and rsi.iloc[i] > 20):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = period_high  # Above range
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                conf = min(df["volume"].iloc[i] / (vol_avg.iloc[i] * 2), 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_signal_bar = i

        return signals


class DynamicKellyStrategy(BaseStrategy):
    """
    Position sizing via Kelly Criterion applied within strategy logic.
    
    Instead of fixed risk %, calculates optimal bet size based on
    rolling win rate and win/loss ratio. Uses half-Kelly for safety.
    
    The actual strategy is an adaptive trend follower (proven edge)
    but with Kelly-optimized signals that encode sizing in confidence.
    
    Confidence field is used to encode Kelly fraction:
    - Higher confidence = bigger position (Kelly says bet more)
    - Lower confidence = smaller position (low edge detected)
    """
    name = "DynamicKelly"
    description = "Kelly-criterion sized adaptive trend follower"
    version = "6.0"

    default_params = {
        "er_period": 10,
        "fast_sc": 2,
        "slow_sc": 30,
        "adx_period": 14,
        "adx_min": 18,
        "rsi_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "kelly_lookback": 30,    # Bars to compute rolling Kelly
        "kelly_fraction": 0.5,   # Half-Kelly for safety
        "min_kelly": 0.05,       # Don't trade if Kelly < 5%
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n = len(df)
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        kama = _kaufman_ama(df["close"], p["er_period"], p["fast_sc"], p["slow_sc"])
        adx = _adx(df, p["adx_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])

        # Precompute potential trade outcomes for Kelly calculation
        # Use forward returns as proxy for trade outcomes
        fwd_returns = df["close"].pct_change(5).shift(-5)  # 5-bar forward return
        
        last_signal_bar = -p["cooldown"] - 1
        start = max(p["er_period"], p["adx_period"], p["atr_period"], p["kelly_lookback"]) + 10

        # Track recent signals for Kelly calculation
        recent_outcomes = []

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Determine base signal (KAMA crossover + ADX)
            base_signal = Signal.HOLD
            if (close > kama.iloc[i] and df["close"].iloc[i-1] <= kama.iloc[i-1]
                and not np.isnan(adx.iloc[i]) and adx.iloc[i] > p["adx_min"]
                and rsi.iloc[i] < 75):
                base_signal = Signal.BUY
            elif (close < kama.iloc[i] and df["close"].iloc[i-1] >= kama.iloc[i-1]
                  and not np.isnan(adx.iloc[i]) and adx.iloc[i] > p["adx_min"]
                  and rsi.iloc[i] > 25):
                base_signal = Signal.SELL

            if base_signal == Signal.HOLD:
                continue

            # Calculate rolling Kelly fraction from past signal outcomes
            # Look at recent bars where similar conditions occurred
            lookback_start = max(0, i - p["kelly_lookback"] * 10)
            wins = 0
            losses = 0
            win_sum = 0
            loss_sum = 0
            
            for j in range(lookback_start, i):
                if np.isnan(fwd_returns.iloc[j]):
                    continue
                # Check if similar signal condition existed
                if base_signal == Signal.BUY and df["close"].iloc[j] > kama.iloc[j]:
                    ret = fwd_returns.iloc[j]
                    if ret > 0:
                        wins += 1
                        win_sum += ret
                    else:
                        losses += 1
                        loss_sum += abs(ret)
                elif base_signal == Signal.SELL and df["close"].iloc[j] < kama.iloc[j]:
                    ret = -fwd_returns.iloc[j]  # Invert for short
                    if ret > 0:
                        wins += 1
                        win_sum += ret
                    else:
                        losses += 1
                        loss_sum += abs(ret)

            total = wins + losses
            if total < 10:
                # Not enough data, use conservative sizing
                kelly = 0.1
            else:
                win_rate = wins / total
                avg_win = win_sum / max(wins, 1)
                avg_loss = loss_sum / max(losses, 1)
                win_loss_ratio = avg_win / max(avg_loss, 0.001)
                
                # Kelly formula: f = (p * b - q) / b
                # p = win_rate, q = 1-p, b = win_loss_ratio
                kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / max(win_loss_ratio, 0.001)
                kelly = max(0, kelly) * p["kelly_fraction"]  # Half-Kelly

            if kelly < p["min_kelly"]:
                continue  # Not enough edge

            # Generate signal with Kelly-encoded confidence
            signals.iloc[i, signals.columns.get_loc("signal")] = base_signal
            if base_signal == Signal.BUY:
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
            else:
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
            
            # Encode Kelly as confidence (will be used for position sizing)
            signals.iloc[i, signals.columns.get_loc("confidence")] = min(kelly, 1.0)
            last_signal_bar = i

        return signals


class MultiEdgeCompositeStrategy(BaseStrategy):
    """
    Runs 5 independent sub-strategies and combines their signals
    into a single composite signal. Uses confidence-weighted voting.
    
    This is a "strategy of strategies" that diversifies alpha sources
    within a single execution.
    
    Sub-strategies:
    - Adaptive trend (KAMA)  
    - EMA crossover momentum
    - Bollinger Band reversion
    - Volatility breakout
    - Volume-price divergence
    """
    name = "MultiEdge"
    description = "5-strategy composite with confidence-weighted voting"
    version = "6.0"

    default_params = {
        # Component weights
        "w_trend": 0.30,
        "w_momentum": 0.25,
        "w_reversion": 0.15,
        "w_breakout": 0.15,
        "w_volume": 0.15,
        # Thresholds  
        "buy_threshold": 0.35,
        "sell_threshold": -0.35,
        # Indicators
        "kama_er": 10,
        "kama_fast": 2,
        "kama_slow": 30,
        "ema_fast": 9,
        "ema_slow": 21,
        "bb_period": 20,
        "bb_std": 2.0,
        "squeeze_bb_std": 2.0,
        "squeeze_kc_mult": 1.5,
        "obv_ema": 20,
        "adx_period": 14,
        "adx_min": 15,
        "rsi_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        n = len(df)
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        # Precompute all indicators
        kama = _kaufman_ama(df["close"], p["kama_er"], p["kama_fast"], p["kama_slow"])
        ema_fast = _ema(df["close"], p["ema_fast"])
        ema_slow = _ema(df["close"], p["ema_slow"])
        rsi = _rsi(df["close"], p["rsi_period"])
        adx = _adx(df, p["adx_period"])
        atr = _atr(df, p["atr_period"])
        
        sma_bb = df["close"].rolling(p["bb_period"]).mean()
        std_bb = df["close"].rolling(p["bb_period"]).std()
        bb_upper = sma_bb + p["bb_std"] * std_bb
        bb_lower = sma_bb - p["bb_std"] * std_bb
        
        # Squeeze detection
        ema_kc = _ema(df["close"], p["bb_period"])
        atr_kc = _atr(df, p["bb_period"])
        kc_upper = ema_kc + p["squeeze_kc_mult"] * atr_kc
        kc_lower = ema_kc - p["squeeze_kc_mult"] * atr_kc
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        obv = _obv(df["close"], df["volume"])
        obv_ema = _ema(obv, p["obv_ema"])

        last_signal_bar = -p["cooldown"] - 1
        start = max(200, p["atr_period"] + 5)

        for i in range(start, n):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Component 1: Trend (KAMA direction)
            trend_score = 0.0
            if close > kama.iloc[i]:
                trend_score = min((close - kama.iloc[i]) / (a * 2), 1.0)
            elif close < kama.iloc[i]:
                trend_score = max(-(kama.iloc[i] - close) / (a * 2), -1.0)

            # Component 2: Momentum (EMA cross)
            mom_score = 0.0
            ema_diff = (ema_fast.iloc[i] - ema_slow.iloc[i]) / ema_slow.iloc[i] * 100
            mom_score = np.clip(ema_diff * 10, -1, 1)

            # Component 3: Reversion (BB position)
            rev_score = 0.0
            bb_range = bb_upper.iloc[i] - bb_lower.iloc[i]
            if bb_range > 0:
                bb_pos = (close - bb_lower.iloc[i]) / bb_range
                rev_score = -(bb_pos - 0.5) * 2  # -1 at top, +1 at bottom

            # Component 4: Breakout (squeeze release)
            brk_score = 0.0
            if i > 0 and not squeeze.iloc[i] and squeeze.iloc[i-1]:
                brk_score = 1.0 if close > sma_bb.iloc[i] else -1.0

            # Component 5: Volume (OBV trend)
            vol_score = 0.0
            if obv.iloc[i] > obv_ema.iloc[i]:
                vol_score = 0.5
            elif obv.iloc[i] < obv_ema.iloc[i]:
                vol_score = -0.5

            # Composite score
            composite = (
                p["w_trend"] * trend_score +
                p["w_momentum"] * mom_score +
                p["w_reversion"] * rev_score +
                p["w_breakout"] * brk_score +
                p["w_volume"] * vol_score
            )

            if composite > p["buy_threshold"] and rsi.iloc[i] < 78:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(composite), 1.0)
                last_signal_bar = i
            elif composite < p["sell_threshold"] and rsi.iloc[i] > 22:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(composite), 1.0)
                last_signal_bar = i

        return signals


class CrossPairLeaderStrategy(BaseStrategy):
    """
    BTC leads altcoins — trade the lag.
    
    Monitors BTC momentum and OBV, then trades altcoins that haven't
    yet caught up. Requires the DataFrame to have a 'btc_close' column
    (injected by the runner).
    
    If btc_close is not available, falls back to self-referencing momentum.
    """
    name = "CrossPairLead"
    description = "BTC momentum leads altcoin moves"
    version = "6.0"

    default_params = {
        "btc_mom_period": 5,
        "btc_mom_threshold": 0.02,    # 2% BTC move
        "alt_lag_bars": 3,            # Altcoin catches up in ~3 bars
        "rsi_period": 14,
        "adx_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.5,
        "cooldown": 4,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        has_btc = "btc_close" in df.columns
        
        if has_btc:
            btc_mom = df["btc_close"].pct_change(p["btc_mom_period"])
        else:
            # Fallback: use own momentum of higher period
            btc_mom = df["close"].pct_change(p["btc_mom_period"] * 2)

        alt_mom = df["close"].pct_change(p["btc_mom_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])
        adx = _adx(df, p["adx_period"])

        last_signal_bar = -p["cooldown"] - 1
        start = max(p["btc_mom_period"] * 2, p["atr_period"], p["adx_period"]) + 10

        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Check BTC momentum from alt_lag_bars ago
            lag_idx = i - p["alt_lag_bars"]
            if lag_idx < 0 or np.isnan(btc_mom.iloc[lag_idx]):
                continue

            btc_m = btc_mom.iloc[lag_idx]
            alt_m = alt_mom.iloc[i]

            # BTC surged but alt hasn't caught up yet
            if (btc_m > p["btc_mom_threshold"] 
                and alt_m < btc_m * 0.5   # Alt lagging
                and rsi.iloc[i] < 70
                and not np.isnan(adx.iloc[i]) and adx.iloc[i] > 15):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(btc_m / 0.05, 1.0)
                last_signal_bar = i

            elif (btc_m < -p["btc_mom_threshold"]
                  and alt_m > btc_m * 0.5
                  and rsi.iloc[i] > 30
                  and not np.isnan(adx.iloc[i]) and adx.iloc[i] > 15):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(btc_m) / 0.05, 1.0)
                last_signal_bar = i

        return signals
