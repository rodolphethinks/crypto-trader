"""
High-frequency strategies for 1m/5m timeframes.

These target many trades per day for statistical significance:
1. MicroMomentum — short-burst momentum after volume spikes
2. MeanReversionHF — Bollinger reversion on sub-hourly data
3. OrderFlowImbalance — buy/sell volume imbalance signals
4. BreakoutMicro — breakout of micro-consolidation ranges
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MicroMomentumStrategy(BaseStrategy):
    """
    Momentum bursts on 1m/5m: enter after a volume spike + directional bar,
    ride the continuation, tight trailing stop.
    """
    name = "MicroMomentum"
    description = "Short-burst momentum after volume spikes (1m/5m)"
    version = "1.0"

    default_params = {
        "vol_spike_mult": 2.5,      # volume must be N× average
        "vol_avg_period": 20,        # lookback for average volume
        "momentum_period": 3,        # consecutive directional bars required
        "min_body_pct": 0.003,       # minimum body size (0.3%)
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "atr_period": 14,
        "cooldown": 3,               # min bars between trades
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        vol_avg = df["volume"].rolling(p["vol_avg_period"]).mean()
        atr = _atr(df, p["atr_period"])
        body = (df["close"] - df["open"]) / df["open"]
        direction = np.sign(body)

        # Count consecutive same-direction bars
        consec = _consecutive_count(direction)

        last_signal_bar = -p["cooldown"] - 1

        for i in range(p["vol_avg_period"] + p["momentum_period"], len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue

            vol_ok = df["volume"].iloc[i] > vol_avg.iloc[i] * p["vol_spike_mult"]
            body_ok = abs(body.iloc[i]) > p["min_body_pct"]
            momentum_ok = consec.iloc[i] >= p["momentum_period"]
            
            if vol_ok and body_ok and momentum_ok and atr.iloc[i] > 0:
                close = df["close"].iloc[i]
                a = atr.iloc[i]

                if direction.iloc[i] > 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                elif direction.iloc[i] < 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a

                last_signal_bar = i

        return signals


class MeanReversionHFStrategy(BaseStrategy):
    """
    Sub-hourly mean reversion: fade extreme moves from Bollinger Bands
    with RSI confirmation, targeting snap-back to mean.
    """
    name = "MeanReversion_HF"
    description = "Bollinger mean reversion on 1m/5m with RSI filter"
    version = "1.0"

    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 1.0,          # target = SMA (mean reversion)
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

        sma = df["close"].rolling(p["bb_period"]).mean()
        std = df["close"].rolling(p["bb_period"]).std()
        upper = sma + p["bb_std"] * std
        lower = sma - p["bb_std"] * std
        rsi = _rsi(df["close"], p["rsi_period"])
        atr = _atr(df, p["atr_period"])

        last_signal_bar = -p["cooldown"] - 1

        for i in range(max(p["bb_period"], p["rsi_period"]) + 1, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]
            mean = sma.iloc[i]

            # Oversold: price below lower band + RSI oversold
            if close < lower.iloc[i] and rsi.iloc[i] < p["rsi_oversold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = mean  # target = SMA
                last_signal_bar = i

            # Overbought: price above upper band + RSI overbought
            elif close > upper.iloc[i] and rsi.iloc[i] > p["rsi_overbought"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = mean
                last_signal_bar = i

        return signals


class OrderFlowImbalanceStrategy(BaseStrategy):
    """
    Infer order flow from candle data:
    - Buying pressure = close - low  (buyers pushed price up from low)
    - Selling pressure = high - close (sellers pushed price down from high)
    
    Trade when imbalance is extreme + volume confirms.
    """
    name = "OrderFlow_Imbalance"
    description = "Volume-weighted buy/sell pressure imbalance"
    version = "1.0"

    default_params = {
        "imbalance_period": 10,
        "imbalance_threshold": 0.7,  # >70% buy pressure => bullish
        "vol_confirm_mult": 1.5,     # volume must be above avg
        "vol_avg_period": 20,
        "sl_atr_mult": 1.5,
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

        # Buy/sell pressure per bar
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        buy_pressure = (df["close"] - df["low"]) / hl_range
        sell_pressure = (df["high"] - df["close"]) / hl_range

        # Volume-weighted rolling imbalance
        vw_buy = (buy_pressure * df["volume"]).rolling(p["imbalance_period"]).sum()
        vw_sell = (sell_pressure * df["volume"]).rolling(p["imbalance_period"]).sum()
        total = vw_buy + vw_sell
        buy_ratio = vw_buy / total.replace(0, np.nan)

        vol_avg = df["volume"].rolling(p["vol_avg_period"]).mean()
        atr = _atr(df, p["atr_period"])

        last_signal_bar = -p["cooldown"] - 1

        start = max(p["imbalance_period"], p["vol_avg_period"], p["atr_period"]) + 1
        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            vol_ok = df["volume"].iloc[i] > vol_avg.iloc[i] * p["vol_confirm_mult"]
            close = df["close"].iloc[i]
            a = atr.iloc[i]
            br = buy_ratio.iloc[i]

            if np.isnan(br):
                continue

            if br > p["imbalance_threshold"] and vol_ok:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = br
                last_signal_bar = i

            elif br < (1 - p["imbalance_threshold"]) and vol_ok:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = 1 - br
                last_signal_bar = i

        return signals


class BreakoutMicroStrategy(BaseStrategy):
    """
    Micro-consolidation breakout: detect tight price ranges (squeeze),
    trade the breakout direction with volume confirmation.
    """
    name = "Breakout_Micro"
    description = "Breakout from micro-consolidation on 1m/5m"
    version = "1.0"

    default_params = {
        "range_period": 10,           # bars to define consolidation
        "squeeze_threshold": 0.003,   # range < 0.3% = squeeze
        "vol_breakout_mult": 2.0,     # breakout bar volume vs avg
        "vol_avg_period": 20,
        "sl_atr_mult": 1.0,
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

        rolling_high = df["high"].rolling(p["range_period"]).max()
        rolling_low = df["low"].rolling(p["range_period"]).min()
        range_pct = (rolling_high - rolling_low) / df["close"]

        vol_avg = df["volume"].rolling(p["vol_avg_period"]).mean()
        atr = _atr(df, p["atr_period"])

        # Was in squeeze on previous bar?
        was_squeeze = range_pct.shift(1) < p["squeeze_threshold"]

        last_signal_bar = -p["cooldown"] - 1

        start = max(p["range_period"], p["vol_avg_period"], p["atr_period"]) + 1
        for i in range(start, len(df)):
            if i - last_signal_bar < p["cooldown"]:
                continue
            if np.isnan(atr.iloc[i]) or atr.iloc[i] <= 0:
                continue

            if not was_squeeze.iloc[i]:
                continue

            vol_ok = df["volume"].iloc[i] > vol_avg.iloc[i] * p["vol_breakout_mult"]
            if not vol_ok:
                continue

            close = df["close"].iloc[i]
            a = atr.iloc[i]
            prev_high = rolling_high.iloc[i - 1]
            prev_low = rolling_low.iloc[i - 1]

            # Breakout above consolidation
            if close > prev_high:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = close - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = close + p["tp_atr_mult"] * a
                last_signal_bar = i

            # Breakdown below consolidation
            elif close < prev_low:
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


def _consecutive_count(direction: pd.Series) -> pd.Series:
    """Count consecutive bars in the same direction."""
    groups = (direction != direction.shift(1)).cumsum()
    return direction.groupby(groups).cumcount() + 1
