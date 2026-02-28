"""
Alternative alpha strategies — approaches beyond simple technical indicators.

1. RegimeAdaptive — switches sub-strategy based on detected market regime
2. CrossTFMomentum — confirms short-timeframe signals with higher-TF trend
3. VolatilityBreakout — enters on realized vol expansion from compressed states
4. StatArb — z-score mean reversion on normalized price series
5. AdaptiveTrend — dynamic EMA with Kaufman's Adaptive Moving Average
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from strategies.base import BaseStrategy, Signal
from indicators.regime import detect_trend_regime, detect_volatility_regime, MarketRegime, VolRegime

logger = logging.getLogger(__name__)


class RegimeAdaptiveStrategy(BaseStrategy):
    """
    Meta-strategy that adapts to the current market regime:
    - TRENDING: use momentum (ride the trend)
    - RANGING: use mean reversion (fade extremes)
    - HIGH_VOL: tighter stops, smaller targets
    - LOW_VOL: breakout mode (expect expansion)
    """
    name = "RegimeAdaptive"
    description = "Switches between momentum/reversion based on regime detection"
    version = "1.0"

    default_params = {
        # Regime detection
        "regime_lookback": 50,
        # Momentum params (for trending)
        "mom_fast_ema": 8,
        "mom_slow_ema": 21,
        "mom_rsi_period": 14,
        "mom_rsi_threshold": 50,
        # Mean reversion params (for ranging)
        "mr_bb_period": 20,
        "mr_bb_std": 2.0,
        "mr_rsi_period": 14,
        "mr_rsi_oversold": 30,
        "mr_rsi_overbought": 70,
        # Risk
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
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

        # Detect regimes
        trend_regime = detect_trend_regime(df["close"], p["regime_lookback"])
        vol_regime = detect_volatility_regime(df["close"])

        # Compute indicators for both sub-strategies
        ema_fast = df["close"].ewm(span=p["mom_fast_ema"], adjust=False).mean()
        ema_slow = df["close"].ewm(span=p["mom_slow_ema"], adjust=False).mean()
        rsi = _rsi(df["close"], p["mom_rsi_period"])

        sma = df["close"].rolling(p["mr_bb_period"]).mean()
        std = df["close"].rolling(p["mr_bb_period"]).std()
        bb_upper = sma + p["mr_bb_std"] * std
        bb_lower = sma - p["mr_bb_std"] * std
        rsi_mr = _rsi(df["close"], p["mr_rsi_period"])

        atr = _atr(df, p["atr_period"])

        last_signal_bar = -p["cooldown"] - 1

        for i in range(p["regime_lookback"] + 5, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]
            regime = trend_regime.iloc[i]
            vol_r = vol_regime.iloc[i]

            # Adjust SL/TP based on volatility regime
            sl_mult = p["sl_atr_mult"] * (0.8 if vol_r == VolRegime.HIGH else 1.0)
            tp_mult = p["tp_atr_mult"] * (0.7 if vol_r == VolRegime.HIGH else 1.0)

            sig = Signal.HOLD

            if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
                # MOMENTUM MODE
                if regime == MarketRegime.TRENDING_UP:
                    # Buy on EMA crossover + RSI above 50
                    if ema_fast.iloc[i] > ema_slow.iloc[i] and rsi.iloc[i] > p["mom_rsi_threshold"]:
                        # Only if just crossed (within 2 bars)
                        if ema_fast.iloc[i-2] <= ema_slow.iloc[i-2]:
                            sig = Signal.BUY
                elif regime == MarketRegime.TRENDING_DOWN:
                    if ema_fast.iloc[i] < ema_slow.iloc[i] and rsi.iloc[i] < (100 - p["mom_rsi_threshold"]):
                        if ema_fast.iloc[i-2] >= ema_slow.iloc[i-2]:
                            sig = Signal.SELL
            else:
                # MEAN REVERSION MODE (ranging or low-vol)
                if close < bb_lower.iloc[i] and rsi_mr.iloc[i] < p["mr_rsi_oversold"]:
                    sig = Signal.BUY
                elif close > bb_upper.iloc[i] and rsi_mr.iloc[i] > p["mr_rsi_overbought"]:
                    sig = Signal.SELL

            if sig != Signal.HOLD:
                signals.iloc[i, signals.columns.get_loc("signal")] = sig
                if sig == Signal.BUY:
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - sl_mult * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = close + tp_mult * a
                else:
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + sl_mult * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = close - tp_mult * a
                last_signal_bar = i

        return signals


class CrossTFMomentumStrategy(BaseStrategy):
    """
    Multi-timeframe momentum within a single timeframe:
    Uses different lookback periods to simulate higher-TF alignment.
    
    - Long-term trend filter (200-bar EMA)
    - Medium-term momentum (50-bar slope)
    - Short-term entry trigger (RSI + MACD crossover)
    
    Only trade in the direction of all three.
    """
    name = "CrossTF_Momentum"
    description = "Multi-lookback momentum alignment (simulated multi-TF)"
    version = "1.0"

    default_params = {
        "trend_ema": 200,
        "medium_slope_period": 50,
        "medium_slope_min": 0.001,
        "rsi_period": 14,
        "rsi_buy_zone": 45,      # RSI between 45-65 = healthy pullback
        "rsi_sell_zone": 55,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "atr_period": 14,
        "cooldown": 5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        # Long-term trend
        ema_long = df["close"].ewm(span=p["trend_ema"], adjust=False).mean()
        trend_up = df["close"] > ema_long
        trend_down = df["close"] < ema_long

        # Medium-term slope
        ema_med = df["close"].ewm(span=p["medium_slope_period"], adjust=False).mean()
        slope = (ema_med - ema_med.shift(5)) / ema_med.shift(5)

        # Short-term
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

            # BUY: all three timeframes bullish
            if (trend_up.iloc[i] 
                and slope.iloc[i] > p["medium_slope_min"]
                and rsi.iloc[i] > p["rsi_buy_zone"] and rsi.iloc[i] < 65
                and macd_hist.iloc[i] > 0 and macd_hist.iloc[i-1] <= 0):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                last_signal_bar = i

            # SELL: all three bearish
            elif (trend_down.iloc[i]
                  and slope.iloc[i] < -p["medium_slope_min"]
                  and rsi.iloc[i] < p["rsi_sell_zone"] and rsi.iloc[i] > 35
                  and macd_hist.iloc[i] < 0 and macd_hist.iloc[i-1] >= 0):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                last_signal_bar = i

        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Enter when volatility expands from a compressed (squeeze) state.
    
    Squeeze = Bollinger Bands inside Keltner Channels.
    Breakout = first bar that exits the squeeze with volume surge.
    Direction = determined by close relative to midline.
    """
    name = "Vol_Breakout"
    description = "Volatility expansion breakout from squeeze state"
    version = "1.0"

    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_mult": 1.5,
        "squeeze_min_bars": 5,    # min bars in squeeze before valid breakout
        "vol_surge_mult": 1.5,
        "vol_avg_period": 20,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.0,
        "atr_period": 14,
        "cooldown": 5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        # Bollinger Bands
        sma = df["close"].rolling(p["bb_period"]).mean()
        std = df["close"].rolling(p["bb_period"]).std()
        bb_upper = sma + p["bb_std"] * std
        bb_lower = sma - p["bb_std"] * std

        # Keltner Channels
        ema = df["close"].ewm(span=p["kc_period"], adjust=False).mean()
        atr_kc = _atr(df, p["kc_period"])
        kc_upper = ema + p["kc_mult"] * atr_kc
        kc_lower = ema - p["kc_mult"] * atr_kc

        # Squeeze: BB inside KC
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Count consecutive squeeze bars
        squeeze_count = _squeeze_duration(squeeze)

        vol_avg = df["volume"].rolling(p["vol_avg_period"]).mean()
        atr = _atr(df, p["atr_period"])

        # Momentum direction (using MACD histogram slope)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        mom = ema12 - ema26

        last_signal_bar = -p["cooldown"] - 1

        for i in range(max(p["bb_period"], p["kc_period"]) + 5, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            # Squeeze just ended (was in squeeze, now out)
            if not squeeze.iloc[i] and squeeze.iloc[i-1]:
                # Was it a valid squeeze (long enough)?
                if squeeze_count.iloc[i-1] >= p["squeeze_min_bars"]:
                    vol_ok = df["volume"].iloc[i] > vol_avg.iloc[i] * p["vol_surge_mult"]
                    
                    close = df["close"].iloc[i]
                    a = atr.iloc[i]

                    if mom.iloc[i] > 0 and vol_ok:
                        signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                        signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                        signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                        last_signal_bar = i
                    elif mom.iloc[i] < 0 and vol_ok:
                        signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                        signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                        signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                        last_signal_bar = i

        return signals


class StatArbStrategy(BaseStrategy):
    """
    Statistical arbitrage / z-score mean reversion.
    
    Normalizes price as a z-score from rolling mean, trades extremes:
    - Buy when z < -entry_z (oversold)
    - Sell when z > +entry_z (overbought)
    - Exit at z = 0 (mean)
    
    Uses Hurst exponent to confirm mean-reverting character.
    """
    name = "StatArb"
    description = "Z-score mean reversion with Hurst exponent filter"
    version = "1.0"

    default_params = {
        "lookback": 50,
        "entry_z": 2.0,
        "exit_z": 0.0,
        "hurst_period": 100,
        "hurst_max": 0.45,       # only trade when Hurst < 0.45 (mean-reverting)
        "sl_atr_mult": 2.5,
        "tp_atr_mult": 2.0,
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

        # Z-score
        sma = df["close"].rolling(p["lookback"]).mean()
        std = df["close"].rolling(p["lookback"]).std()
        z_score = (df["close"] - sma) / std.replace(0, np.nan)

        # Hurst exponent (rolling)
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

            # Only trade if market shows mean-reverting character
            if h > p["hurst_max"]:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            if z < -p["entry_z"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = sma.iloc[i]  # target = mean
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(z) / 4.0, 1.0)
                last_signal_bar = i

            elif z > p["entry_z"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = sma.iloc[i]
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(z) / 4.0, 1.0)
                last_signal_bar = i

        return signals


class AdaptiveTrendStrategy(BaseStrategy):
    """
    Kaufman Adaptive Moving Average (KAMA) strategy.
    
    KAMA adjusts its speed based on market noise:
    - Trending: fast (like EMA 2)
    - Noisy: slow (like EMA 30)
    
    Trade when price crosses KAMA with ADX confirmation.
    """
    name = "AdaptiveTrend"
    description = "Kaufman AMA with ADX filter for adaptive trend following"
    version = "1.0"

    default_params = {
        "er_period": 10,          # Efficiency Ratio period
        "fast_sc": 2,             # fast smoothing constant (SC = 2/(N+1))
        "slow_sc": 30,            # slow smoothing constant
        "adx_period": 14,
        "adx_min": 20,            # min ADX to confirm trend
        "rsi_period": 14,
        "sl_atr_mult": 2.0,
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

        last_signal_bar = -p["cooldown"] - 1

        start = max(p["er_period"], p["adx_period"], p["atr_period"]) + 5
        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue
            if np.isnan(adx.iloc[i]):
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]

            # Price crosses above KAMA + ADX confirms trend + RSI not overbought
            if (close > kama.iloc[i] 
                and df["close"].iloc[i-1] <= kama.iloc[i-1]
                and adx.iloc[i] > p["adx_min"]
                and rsi.iloc[i] < 70):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                last_signal_bar = i

            elif (close < kama.iloc[i]
                  and df["close"].iloc[i-1] >= kama.iloc[i-1]
                  and adx.iloc[i] > p["adx_min"]
                  and rsi.iloc[i] > 30):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                last_signal_bar = i

        return signals


# ── Helpers ──────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift(1)).abs(),
        "lc": (df["low"] - df["close"].shift(1)).abs(),
    }).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr = _atr(df, period)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _squeeze_duration(squeeze: pd.Series) -> pd.Series:
    """Count consecutive True bars."""
    groups = (squeeze != squeeze.shift(1)).cumsum()
    return squeeze.groupby(groups).cumcount() + 1


def _hurst_exponent(series: np.ndarray) -> float:
    """Estimate Hurst exponent using R/S method."""
    if len(series) < 20:
        return 0.5
    try:
        n = len(series)
        mean = np.mean(series)
        deviations = series - mean
        cumulative = np.cumsum(deviations)
        r = np.max(cumulative) - np.min(cumulative)
        s = np.std(series, ddof=1)
        if s == 0 or r == 0:
            return 0.5
        return np.log(r / s) / np.log(n)
    except Exception:
        return 0.5


def _kaufman_ama(close: pd.Series, er_period: int = 10,
                 fast_sc: int = 2, slow_sc: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    fast_alpha = 2 / (fast_sc + 1)
    slow_alpha = 2 / (slow_sc + 1)

    # Efficiency Ratio
    direction = (close - close.shift(er_period)).abs()
    volatility = close.diff().abs().rolling(er_period).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)

    # Smoothing Constant
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    # KAMA
    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[:er_period] = close.iloc[:er_period]
    
    for i in range(er_period, len(close)):
        if np.isnan(kama.iloc[i-1]):
            kama.iloc[i] = close.iloc[i]
        else:
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

    return kama
