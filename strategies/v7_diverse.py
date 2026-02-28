"""
V7 Strategies — Fundamentally new approaches for diversification.

1. OrderFlowMomentum  — Volume-weighted momentum with OBV divergence
2. TrendPulse         — Multi-period trend alignment with acceleration
3. VolatilityCapture  — Bollinger Band squeeze breakout with ATR sizing  
4. MeanReversionRSI   — RSI extremes with volume confirmation
5. AdaptiveChannel    — Self-adjusting Keltner channel breakout
6. MomentumSwitch     — Fast/slow regime detection with strategy switching
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, Signal


# ── Helper functions ─────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = h.diff().where(lambda x: (x > 0) & (x > -l.diff()), 0.0)
    minus_dm = (-l.diff()).where(lambda x: (x > 0) & (x > h.diff()), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    return dx.rolling(period).mean()

def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff())
    return (sign * df["volume"]).cumsum()

def _bb(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower

def _keltner(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0):
    mid = _ema(df["close"], ema_period)
    atr = _atr(df, atr_period)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return mid, upper, lower


# ──────────────────────────────────────────────────────────────────────────────

class OrderFlowMomentumStrategy(BaseStrategy):
    """
    Combines momentum direction with order flow analysis.
    
    Uses OBV divergence from price to detect accumulation/distribution,
    then enters when momentum aligns with order flow.
    Volume surge confirmation required.
    """
    name = "OrderFlowMom"
    description = "Volume-weighted momentum with OBV divergence"
    version = "7.0"
    
    default_params = {
        "mom_period": 10,
        "obv_ema_period": 20,
        "vol_surge_mult": 1.5,
        "rsi_period": 14,
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

        mom = df["close"].pct_change(p["mom_period"])
        obv = _obv(df)
        obv_ema = _ema(obv, p["obv_ema_period"])
        obv_mom = obv.pct_change(p["mom_period"])
        vol_avg = _sma(df["volume"], 20)
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])

        last_bar = -p["cooldown"] - 1
        start = max(p["mom_period"], p["obv_ema_period"], p["atr_period"]) + 5

        for i in range(start, len(df)):
            if i - last_bar < p["cooldown"]:
                continue
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            c = df["close"].iloc[i]
            vol_surge = df["volume"].iloc[i] > vol_avg.iloc[i] * p["vol_surge_mult"]
            
            # Price momentum up + OBV accumulating + volume surge
            if (mom.iloc[i] > 0 and obv.iloc[i] > obv_ema.iloc[i]
                and obv_mom.iloc[i] > 0 and vol_surge
                and rsi.iloc[i] > 40 and rsi.iloc[i] < 75):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(mom.iloc[i] * 10, 1.0)
                last_bar = i
            
            # Price momentum down + OBV distributing + volume surge
            elif (mom.iloc[i] < 0 and obv.iloc[i] < obv_ema.iloc[i]
                  and obv_mom.iloc[i] < 0 and vol_surge
                  and rsi.iloc[i] > 25 and rsi.iloc[i] < 60):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = min(abs(mom.iloc[i]) * 10, 1.0)
                last_bar = i

        return signals


class TrendPulseStrategy(BaseStrategy):
    """
    Uses multi-period trend alignment with acceleration detection.
    
    Combines 3 EMA timeframes (fast/medium/slow) to detect aligned trends,
    then enters on acceleration (rate of change of trend strength).
    """
    name = "TrendPulse"
    description = "Multi-period trend alignment with acceleration"
    version = "7.0"
    
    default_params = {
        "fast_ema": 8,
        "med_ema": 21,
        "slow_ema": 55,
        "accel_period": 5,
        "accel_threshold": 0.001,
        "adx_period": 14,
        "adx_min": 20,
        "atr_period": 14,
        "sl_atr_mult": 1.8,
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

        fast = _ema(df["close"], p["fast_ema"])
        med = _ema(df["close"], p["med_ema"])
        slow = _ema(df["close"], p["slow_ema"])
        adx = _adx(df, p["adx_period"])
        atr = _atr(df, p["atr_period"])
        
        # Trend strength: distance between EMAs
        trend_strength = (fast - slow) / slow
        # Acceleration: rate of change of trend strength
        accel = trend_strength.diff(p["accel_period"])

        last_bar = -p["cooldown"] - 1
        start = max(p["slow_ema"], p["adx_period"], p["atr_period"]) + p["accel_period"] + 5

        for i in range(start, len(df)):
            if i - last_bar < p["cooldown"]:
                continue
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            c = df["close"].iloc[i]
            
            # Full alignment: fast > med > slow with accelerating trend
            aligned_up = (fast.iloc[i] > med.iloc[i] > slow.iloc[i])
            aligned_down = (fast.iloc[i] < med.iloc[i] < slow.iloc[i])
            
            strong_trend = not np.isnan(adx.iloc[i]) and adx.iloc[i] > p["adx_min"]
            accel_val = accel.iloc[i] if not np.isnan(accel.iloc[i]) else 0
            
            if aligned_up and strong_trend and accel_val > p["accel_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                conf = min(accel_val / 0.005, 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_bar = i
            
            elif aligned_down and strong_trend and accel_val < -p["accel_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                conf = min(abs(accel_val) / 0.005, 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_bar = i

        return signals


class VolatilityCaptureStrategy(BaseStrategy):
    """
    Captures breakouts from low-volatility squeeze states.
    
    Detects Bollinger Band squeezes (bands inside Keltner channels),
    then enters on expansion direction with momentum confirmation.
    """
    name = "VolCapture"
    description = "Squeeze breakout with momentum confirmation"
    version = "7.0"
    
    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "keltner_ema": 20,
        "keltner_atr": 14,
        "keltner_mult": 1.5,
        "rsi_period": 14,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.5,
        "squeeze_bars": 3,  # Min bars in squeeze before breakout
        "cooldown": 5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        bb_mid, bb_upper, bb_lower = _bb(df["close"], p["bb_period"], p["bb_std"])
        k_mid, k_upper, k_lower = _keltner(df, p["keltner_ema"], p["keltner_atr"], p["keltner_mult"])
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])
        
        # Squeeze: BB inside Keltner
        squeeze = (bb_lower > k_lower) & (bb_upper < k_upper)
        
        # Count consecutive squeeze bars
        squeeze_count = pd.Series(0, index=df.index, dtype=int)
        cnt = 0
        for i in range(len(squeeze)):
            if squeeze.iloc[i]:
                cnt += 1
            else:
                cnt = 0
            squeeze_count.iloc[i] = cnt

        mom = df["close"].pct_change(5)  # Short momentum
        last_bar = -p["cooldown"] - 1
        start = max(p["bb_period"], p["keltner_ema"], p["keltner_atr"], p["atr_period"]) + 10

        for i in range(start, len(df)):
            if i - last_bar < p["cooldown"]:
                continue
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            c = df["close"].iloc[i]
            
            # Squeeze release: was in squeeze, now breaking out
            was_squeezed = squeeze_count.iloc[i-1] >= p["squeeze_bars"] if i > 0 else False
            now_released = not squeeze.iloc[i]
            
            if not (was_squeezed and now_released):
                continue
            
            # Determine direction from momentum and price relative to BB midline
            if (c > bb_mid.iloc[i] and mom.iloc[i] > 0 
                and rsi.iloc[i] > 45 and rsi.iloc[i] < 80):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = 0.7
                last_bar = i
            
            elif (c < bb_mid.iloc[i] and mom.iloc[i] < 0
                  and rsi.iloc[i] > 20 and rsi.iloc[i] < 55):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = 0.7
                last_bar = i

        return signals


class MeanReversionRSIStrategy(BaseStrategy):
    """
    Mean reversion using RSI extremes with volume confirmation.
    
    Enters on RSI oversold/overbought conditions with divergence
    detection and volume confirmation. Targets reversion to mean.
    """
    name = "MeanRevRSI"
    description = "RSI extreme reversion with volume"
    version = "7.0"
    
    default_params = {
        "rsi_period": 14,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "sma_period": 50,
        "atr_period": 14,
        "sl_atr_mult": 2.5,
        "tp_atr_mult": 3.0,
        "vol_confirm_mult": 1.3,
        "cooldown": 4,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        rsi = _rsi(df["close"], p["rsi_period"])
        sma = _sma(df["close"], p["sma_period"])
        atr = _atr(df, p["atr_period"])
        vol_avg = _sma(df["volume"], 20)

        last_bar = -p["cooldown"] - 1
        start = max(p["rsi_period"], p["sma_period"], p["atr_period"]) + 5

        for i in range(start, len(df)):
            if i - last_bar < p["cooldown"]:
                continue
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            c = df["close"].iloc[i]
            r = rsi.iloc[i]
            if np.isnan(r):
                continue
            
            vol_high = df["volume"].iloc[i] > vol_avg.iloc[i] * p["vol_confirm_mult"]
            
            # Oversold: RSI < threshold, price below SMA, high volume (capitulation)
            if (r < p["rsi_oversold"] and c < sma.iloc[i] and vol_high):
                # Check for RSI divergence (price lower low, RSI higher low)
                divergence = False
                for j in range(max(0, i-20), i):
                    if rsi.iloc[j] < p["rsi_oversold"] and df["close"].iloc[j] > c and rsi.iloc[j] < r:
                        divergence = True
                        break
                
                if divergence or r < p["rsi_oversold"] - 5:  # Extra oversold OR divergence
                    signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                    conf = min((p["rsi_oversold"] - r) / 20, 1.0)
                    signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                    last_bar = i
            
            # Overbought: RSI > threshold, price above SMA, high volume
            elif (r > p["rsi_overbought"] and c > sma.iloc[i] and vol_high):
                divergence = False
                for j in range(max(0, i-20), i):
                    if rsi.iloc[j] > p["rsi_overbought"] and df["close"].iloc[j] < c and rsi.iloc[j] > r:
                        divergence = True
                        break
                
                if divergence or r > p["rsi_overbought"] + 5:
                    signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                    conf = min((r - p["rsi_overbought"]) / 20, 1.0)
                    signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                    last_bar = i

        return signals


class AdaptiveChannelStrategy(BaseStrategy):
    """
    Self-adjusting Keltner channel breakout using volatility feedback.
    
    Channel width adapts based on recent regime (volatility state).
    Uses tighter channels in low-vol (more signals) and wider in high-vol
    (fewer but higher-conviction signals).
    """
    name = "AdaptChan"
    description = "Adaptive Keltner channel breakout"
    version = "7.0"
    
    default_params = {
        "ema_period": 20,
        "atr_period": 14,
        "base_mult": 2.0,
        "vol_lookback": 50,
        "rsi_period": 14,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 3.5,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        mid = _ema(df["close"], p["ema_period"])
        atr = _atr(df, p["atr_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        
        # Adaptive multiplier based on volatility percentile
        atr_pctile = atr.rolling(p["vol_lookback"]).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        # Lower vol = tighter channels (more signals), higher vol = wider
        adaptive_mult = p["base_mult"] * (0.5 + atr_pctile)
        
        last_bar = -p["cooldown"] - 1
        start = max(p["ema_period"], p["atr_period"], p["vol_lookback"]) + 5

        for i in range(start, len(df)):
            if i - last_bar < p["cooldown"]:
                continue
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            mult = adaptive_mult.iloc[i]
            if np.isnan(mult):
                mult = p["base_mult"]
            
            c = df["close"].iloc[i]
            upper = mid.iloc[i] + mult * a
            lower = mid.iloc[i] - mult * a
            
            # Breakout above adaptive channel
            if (c > upper and df["close"].iloc[i-1] <= mid.iloc[i-1] + mult * atr.iloc[i-1]
                and rsi.iloc[i] > 50 and rsi.iloc[i] < 80):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                conf = min((c - upper) / (a * 2), 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = max(conf, 0.3)
                last_bar = i
            
            # Breakdown below adaptive channel
            elif (c < lower and df["close"].iloc[i-1] >= mid.iloc[i-1] - mult * atr.iloc[i-1]
                  and rsi.iloc[i] > 20 and rsi.iloc[i] < 50):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                conf = min((lower - c) / (a * 2), 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = max(conf, 0.3)
                last_bar = i

        return signals


class MomentumSwitchStrategy(BaseStrategy):
    """
    Switches between fast and slow momentum strategies based on regime.
    
    In high-volatility regimes: uses fast momentum (short lookback)
    In low-volatility regimes: uses slow momentum (long lookback)
    Adapts dynamically without requiring pre-classification.
    """
    name = "MomSwitch"
    description = "Regime-adaptive momentum switching"
    version = "7.0"
    
    default_params = {
        "fast_period": 5,
        "slow_period": 20,
        "vol_period": 20,
        "vol_threshold_pctile": 50,
        "atr_period": 14,
        "adx_period": 14,
        "adx_min": 15,
        "rsi_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.5,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        fast_mom = df["close"].pct_change(p["fast_period"])
        slow_mom = df["close"].pct_change(p["slow_period"])
        
        # Volatility regime
        returns = df["close"].pct_change()
        vol = returns.rolling(p["vol_period"]).std()
        vol_pctile = vol.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        adx = _adx(df, p["adx_period"])
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])
        
        fast_ema = _ema(df["close"], p["fast_period"])
        slow_ema = _ema(df["close"], p["slow_period"])

        last_bar = -p["cooldown"] - 1
        start = max(p["slow_period"], p["atr_period"], p["adx_period"], 100) + 10

        for i in range(start, len(df)):
            if i - last_bar < p["cooldown"]:
                continue
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            c = df["close"].iloc[i]
            vp = vol_pctile.iloc[i] if not np.isnan(vol_pctile.iloc[i]) else 0.5
            
            # Select regime
            high_vol = vp > (p["vol_threshold_pctile"] / 100)
            
            if high_vol:
                # Fast momentum mode
                mom = fast_mom.iloc[i]
                threshold = 0.01  # 1% move
                ema_ref = fast_ema
            else:
                # Slow momentum mode
                mom = slow_mom.iloc[i]
                threshold = 0.02  # 2% move
                ema_ref = slow_ema
            
            if np.isnan(mom) or np.isnan(adx.iloc[i]):
                continue
            
            strong_trend = adx.iloc[i] > p["adx_min"]
            
            if (mom > threshold and c > ema_ref.iloc[i] and strong_trend
                and rsi.iloc[i] > 40 and rsi.iloc[i] < 78):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                conf = min(abs(mom) / 0.05, 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_bar = i
            
            elif (mom < -threshold and c < ema_ref.iloc[i] and strong_trend
                  and rsi.iloc[i] > 22 and rsi.iloc[i] < 60):
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                conf = min(abs(mom) / 0.05, 1.0)
                signals.iloc[i, signals.columns.get_loc("confidence")] = conf
                last_bar = i

        return signals


# ── Strategy Registry ────────────────────────────────────────────────────────

def get_v7_strategies():
    return {
        "OrderFlowMom": OrderFlowMomentumStrategy,
        "TrendPulse": TrendPulseStrategy,
        "VolCapture": VolatilityCaptureStrategy,
        "MeanRevRSI": MeanReversionRSIStrategy,
        "AdaptChan": AdaptiveChannelStrategy,
        "MomSwitch": MomentumSwitchStrategy,
    }
