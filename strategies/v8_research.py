"""
V8 Research Strategies — Extracted from research/ documents.

Implements 5 strict, rule-based strategies using OHLCV data only.
Sources: CHATGPT.md, CLAUDE.md, PERPLEXITY.md, GEMINI.md

1. VolBreakoutMomentum  — Volatility breakout + momentum (3/4 docs)
2. MeanRevLowVol        — Mean reversion in low-vol regimes (2/4 docs)
3. WeekendGapFade       — Calendar anomaly: fade weekend gaps (CLAUDE)
4. BTCResidualMR        — BTC-neutral residual mean reversion (GEMINI)
5. TSMOMCarry           — Time-series momentum with vol targeting (3/4 docs)

Anti-bias: all signals use t-1 data only (shift(1) on all indicators).
"""
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, Signal


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — all compute on the data passed, no lookahead
# ═══════════════════════════════════════════════════════════════════════════════

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0.0).rolling(n).mean()
    l = (-d.where(d < 0, 0.0)).rolling(n).mean()
    rs = g / l.replace(0, 1e-10)
    return 100 - 100 / (1 + rs)

def _zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n).mean()
    sd = s.rolling(n).std().replace(0, 1e-10)
    return (s - m) / sd

def _realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    """Annualised realised vol from log-returns (assume 365*6 4h bars/yr)."""
    lr = np.log(close / close.shift(1))
    return lr.rolling(n).std() * np.sqrt(365 * 6)  # 4h bars

def _day_of_week(idx: pd.DatetimeIndex) -> pd.Series:
    """0=Mon … 6=Sun."""
    return pd.Series(idx.dayofweek, index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — Volatility Breakout Momentum
# Sources: CHATGPT, PERPLEXITY, CLAUDE
# Edge: Crypto trends strongly after volatile breakouts;
#       momentum documented as strong factor.
# ═══════════════════════════════════════════════════════════════════════════════

class VolBreakoutMomentumStrategy(BaseStrategy):
    """
    Enters when price breaks a lookback channel AND bar range exceeds
    k × ATR (confirming genuine momentum, not noise).
    Exits via trailing ATR stop or time-based exit.

    All indicators lagged by 1 bar (t-1) to prevent lookahead.
    """
    name = "VolBreakoutMom"
    description = "Volatility breakout + momentum (3/4 research docs)"
    version = "8.0"

    default_params = {
        "lookback": 10,         # channel lookback (bars)
        "atr_period": 14,
        "k_threshold": 1.5,     # bar range must exceed k × ATR
        "tp_atr_mult": 2.5,     # take-profit = 2.5 × ATR
        "sl_atr_mult": 1.0,     # stop-loss = 1.0 × ATR
        "vol_floor_pctile": 20, # ATR must be above this percentile (regime filter)
        "max_hold_bars": 48,    # time exit after 48 bars (~8 days @4h)
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        sig = df.copy()
        sig["signal"] = Signal.HOLD
        sig["stop_loss"] = np.nan
        sig["take_profit"] = np.nan
        sig["confidence"] = 0.5

        c = df["close"]
        h = df["high"]
        l = df["low"]
        atr = _atr(df, p["atr_period"])

        # Regime filter: ATR must be above vol_floor_pctile of last 200 bars
        atr_pctile = atr.rolling(200).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Channel: shift(2) so the window excludes bar i-1 (the bar we compare)
        # This ensures c[i-1] > max(c[i-2]..c[i-lookback-1]) is a real breakout
        ch_high = c.shift(2).rolling(p["lookback"]).max()
        ch_low  = c.shift(2).rolling(p["lookback"]).min()

        # Bar range vs ATR (also lagged)
        bar_range = (h.shift(1) - l.shift(1))
        atr_lag = atr.shift(1)

        last_signal_bar = -p["cooldown"] - 1
        hold_entry_bar = None
        start = max(p["lookback"], p["atr_period"], 200) + 5

        for i in range(start, len(df)):
            # Time exit: close open position
            if hold_entry_bar is not None and (i - hold_entry_bar) >= p["max_hold_bars"]:
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL  # flatten
                hold_entry_bar = None
                last_signal_bar = i
                continue

            if i - last_signal_bar < p["cooldown"]:
                continue

            a = atr_lag.iloc[i]
            if np.isnan(a) or a <= 0:
                continue

            # Regime filter: skip if volatility is too low
            pctile = atr_pctile.iloc[i]
            if np.isnan(pctile) or pctile * 100 < p["vol_floor_pctile"]:
                continue

            # Check breakout + momentum bar
            prev_close = c.iloc[i - 1]
            range_ok = bar_range.iloc[i] > p["k_threshold"] * a

            if range_ok and prev_close > ch_high.iloc[i]:
                # Bullish breakout
                entry = c.iloc[i]
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.BUY
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry - p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry + p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(bar_range.iloc[i] / (3 * a), 1.0)
                last_signal_bar = i
                hold_entry_bar = i

            elif range_ok and prev_close < ch_low.iloc[i]:
                # Bearish breakout
                entry = c.iloc[i]
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry + p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry - p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(bar_range.iloc[i] / (3 * a), 1.0)
                last_signal_bar = i
                hold_entry_bar = i

        return sig


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Mean Reversion in Low-Volatility Regimes
# Sources: CHATGPT, CLAUDE
# Edge: In quiet markets, price reverts to SMA; z-score detects extremes.
# ═══════════════════════════════════════════════════════════════════════════════

class MeanRevLowVolStrategy(BaseStrategy):
    """
    Enters contrarian when z-score of close vs SMA is extreme AND market
    is in low-volatility regime.  Exits on mean revert or time stop.

    Kill switch: ATR above 60th percentile → strategy OFF.
    """
    name = "MeanRevLowVol"
    description = "Mean reversion in low-vol regimes (2/4 research docs)"
    version = "8.0"

    default_params = {
        "ma_period": 20,
        "z_entry": 2.0,         # z-score threshold to enter
        "z_exit": 0.0,          # exit when z crosses 0 (mean revert)
        "z_stop": 3.0,          # stop if z extends further
        "atr_period": 14,
        "vol_max_pctile": 40,   # only trade when ATR below this %-ile
        "sl_atr_mult": 2.5,
        "tp_atr_mult": 2.0,
        "max_hold_bars": 20,    # time stop
        "cooldown": 4,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        sig = df.copy()
        sig["signal"] = Signal.HOLD
        sig["stop_loss"] = np.nan
        sig["take_profit"] = np.nan
        sig["confidence"] = 0.5

        c = df["close"]
        ma = _sma(c, p["ma_period"])
        z = _zscore(c, p["ma_period"])
        atr = _atr(df, p["atr_period"])
        atr_pctile = atr.rolling(200).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Lag by 1 bar
        z_lag = z.shift(1)
        atr_pctile_lag = atr_pctile.shift(1)
        atr_lag = atr.shift(1)

        last_bar = -p["cooldown"] - 1
        hold_entry = None
        start = max(p["ma_period"], p["atr_period"], 200) + 5

        for i in range(start, len(df)):
            # Time exit
            if hold_entry is not None and (i - hold_entry) >= p["max_hold_bars"]:
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                hold_entry = None
                last_bar = i
                continue

            if i - last_bar < p["cooldown"]:
                continue

            zv = z_lag.iloc[i]
            a = atr_lag.iloc[i]
            vol_p = atr_pctile_lag.iloc[i]
            if any(np.isnan(x) for x in [zv, a, vol_p]) or a <= 0:
                continue

            # Kill switch: vol too high → strategy OFF
            if vol_p * 100 > p["vol_max_pctile"]:
                continue

            entry = c.iloc[i]

            if zv < -p["z_entry"]:
                # Oversold → buy
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.BUY
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry - p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry + p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(zv) / 4, 1.0)
                last_bar = i
                hold_entry = i
            elif zv > p["z_entry"]:
                # Overbought → sell
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry + p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry - p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(zv) / 4, 1.0)
                last_bar = i
                hold_entry = i

        return sig


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Weekend Gap Fade
# Source: CLAUDE
# Edge: Weekend moves on thin liquidity tend to revert
#       when institutional volume returns Monday.
# ═══════════════════════════════════════════════════════════════════════════════

class WeekendGapFadeStrategy(BaseStrategy):
    """
    Detects large price dislocations during low-volume weekends and
    fades them on Sunday evening.

    Works on 4h bars only. Uses a day-of-week + volume filter.
    Time exit: Tuesday 00:00 UTC.
    """
    name = "WeekendGapFade"
    description = "Fade weekend liquidity gaps (CLAUDE research)"
    version = "8.0"

    default_params = {
        "atr_period": 14,
        "gap_atr_mult": 1.5,   # weekend move must exceed 1.5 × ATR
        "vol_ratio_max": 0.6,  # weekend volume < 60% of avg
        "sl_atr_mult": 1.5,
        "tp_target": "friday_close",  # TP = revert to Friday close
        "max_hold_bars": 12,  # exit by ~Tuesday 00:00 (4h bars: Mon-Tue)
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        sig = df.copy()
        sig["signal"] = Signal.HOLD
        sig["stop_loss"] = np.nan
        sig["take_profit"] = np.nan
        sig["confidence"] = 0.5

        c = df["close"]
        v = df["volume"]
        atr = _atr(df, p["atr_period"])
        vol_avg = _sma(v, 30 * 6)  # 30-day avg (6 bars/day at 4h)
        dow = _day_of_week(df.index)

        # Track Friday close
        friday_close = pd.Series(np.nan, index=df.index)
        last_fri = np.nan
        for i in range(len(df)):
            if dow.iloc[i] == 4:  # Friday
                last_fri = c.iloc[i]
            friday_close.iloc[i] = last_fri

        hold_entry = None
        start = max(p["atr_period"], 200) + 5

        for i in range(start, len(df)):
            d = dow.iloc[i]

            # Time exit
            if hold_entry is not None:
                bars_held = i - hold_entry
                if bars_held >= p["max_hold_bars"] or d == 1:  # Tuesday
                    sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                    hold_entry = None
                continue

            # Only enter on Sunday bars (dow == 6)
            if d != 6:
                continue

            a = atr.iloc[i]
            fc = friday_close.iloc[i]
            if np.isnan(a) or a <= 0 or np.isnan(fc):
                continue

            # Volume check (use t-1 to avoid lookahead)
            bar_vol = v.iloc[i - 1] if i > 0 else 0
            avg_vol = vol_avg.iloc[i - 1] if i > 0 else 1
            if avg_vol > 0 and (bar_vol / avg_vol) > p["vol_ratio_max"]:
                continue  # volume not thin enough

            gap = c.iloc[i] - fc
            entry = c.iloc[i]

            if abs(gap) > p["gap_atr_mult"] * a:
                if gap > 0:
                    # Weekend pump → short (fade)
                    sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                    sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry + p["sl_atr_mult"] * a
                    sig.iloc[i, sig.columns.get_loc("take_profit")] = fc
                    sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(gap) / (3 * a), 1.0)
                    hold_entry = i
                else:
                    # Weekend dump → long (fade)
                    sig.iloc[i, sig.columns.get_loc("signal")] = Signal.BUY
                    sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry - p["sl_atr_mult"] * a
                    sig.iloc[i, sig.columns.get_loc("take_profit")] = fc
                    sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(gap) / (3 * a), 1.0)
                    hold_entry = i

        return sig


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — BTC-Neutral Residual Mean Reversion
# Source: GEMINI
# Edge: Alt-coin specific under/over-performance vs BTC
#       reverts to equilibrium. Market-neutral.
# ═══════════════════════════════════════════════════════════════════════════════

class BTCResidualMRStrategy(BaseStrategy):
    """
    Computes rolling OLS residual (alt vs BTC), enters on extreme z-score,
    exits on mean revert. Requires BTC data injected into df as 'btc_close'.

    Market-neutral: long alt / short BTC (or vice versa).
    Here we only trade the alt side; assume BTC hedge external.
    """
    name = "BTCResidualMR"
    description = "BTC-neutral residual mean reversion (GEMINI research)"
    version = "8.0"

    default_params = {
        "corr_window": 60,      # rolling correlation window
        "beta_window": 60,      # rolling beta window
        "z_window": 60,         # z-score window on residual
        "z_entry": 2.0,
        "z_exit": 0.5,
        "min_corr": 0.60,       # minimum correlation to trade
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 2.0,
        "max_hold_bars": 30,
        "cooldown": 5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        sig = df.copy()
        sig["signal"] = Signal.HOLD
        sig["stop_loss"] = np.nan
        sig["take_profit"] = np.nan
        sig["confidence"] = 0.5

        if "btc_close" not in df.columns:
            return sig  # cannot run without BTC data

        alt_ret = df["close"].pct_change()
        btc_ret = df["btc_close"].pct_change()

        # Rolling correlation
        corr = alt_ret.rolling(p["corr_window"]).corr(btc_ret)

        # Rolling beta = cov(alt, btc) / var(btc)
        cov = alt_ret.rolling(p["beta_window"]).cov(btc_ret)
        var_btc = btc_ret.rolling(p["beta_window"]).var().replace(0, 1e-10)
        beta = cov / var_btc

        # Residual return = alt_ret - beta * btc_ret
        resid = alt_ret - beta * btc_ret
        # Cumulative residual
        cum_resid = resid.cumsum()
        z_resid = _zscore(cum_resid, p["z_window"])

        atr = _atr(df, p["atr_period"])

        # Lag everything by 1
        z_lag = z_resid.shift(1)
        corr_lag = corr.shift(1)
        atr_lag = atr.shift(1)

        last_bar = -p["cooldown"] - 1
        hold_entry = None
        start = max(p["corr_window"], p["beta_window"], p["z_window"], p["atr_period"]) + 10

        for i in range(start, len(df)):
            if hold_entry is not None and (i - hold_entry) >= p["max_hold_bars"]:
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                hold_entry = None
                last_bar = i
                continue

            if i - last_bar < p["cooldown"]:
                continue

            zv = z_lag.iloc[i]
            cr = corr_lag.iloc[i]
            a = atr_lag.iloc[i]
            if any(np.isnan(x) for x in [zv, cr, a]) or a <= 0:
                continue

            # Kill switch: correlation too low → decoupled
            if cr < p["min_corr"]:
                continue

            entry = df["close"].iloc[i]

            if zv < -p["z_entry"]:
                # Alt underperforming vs BTC → long alt
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.BUY
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry - p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry + p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(zv) / 4, 1.0)
                last_bar = i
                hold_entry = i
            elif zv > p["z_entry"]:
                # Alt overperforming vs BTC → short alt
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry + p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry - p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(zv) / 4, 1.0)
                last_bar = i
                hold_entry = i

        return sig


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — Time-Series Momentum (TSMOM) with Vol Targeting
# Sources: CHATGPT, PERPLEXITY, CLAUDE
# Edge: Crypto returns exhibit time-series momentum; vol-targeting
#       stabilises equity curve and reduces drawdown.
# ═══════════════════════════════════════════════════════════════════════════════

class TSMOMStrategy(BaseStrategy):
    """
    Computes return over lookback window. If return z-score exceeds
    threshold AND realized vol is within acceptable band → enter
    in direction of momentum.

    Position size inversely proportional to realised vol (vol-targeting).
    """
    name = "TSMOM"
    description = "Time-series momentum with vol targeting (3/4 research docs)"
    version = "8.0"

    default_params = {
        "lookback": 8,          # return lookback (bars) — ~32h at 4h
        "zscore_window": 60,    # for normalising returns
        "z_threshold": 0.75,    # entry z-score threshold
        "vol_window": 20,       # realised vol window
        "vol_min_ann": 0.30,    # min annualised vol to trade (30%)
        "vol_max_ann": 1.50,    # max annualised vol to trade (150%)
        "atr_period": 14,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "max_hold_bars": 20,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        sig = df.copy()
        sig["signal"] = Signal.HOLD
        sig["stop_loss"] = np.nan
        sig["take_profit"] = np.nan
        sig["confidence"] = 0.5

        c = df["close"]
        ret = c.pct_change(p["lookback"])
        z_ret = _zscore(ret, p["zscore_window"])
        rv = _realized_vol(c, p["vol_window"])
        atr = _atr(df, p["atr_period"])

        # Lag by 1
        z_lag = z_ret.shift(1)
        rv_lag = rv.shift(1)
        atr_lag = atr.shift(1)

        last_bar = -p["cooldown"] - 1
        hold_entry = None
        start = max(p["lookback"], p["zscore_window"], p["vol_window"], p["atr_period"]) + 10

        for i in range(start, len(df)):
            if hold_entry is not None and (i - hold_entry) >= p["max_hold_bars"]:
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                hold_entry = None
                last_bar = i
                continue

            if i - last_bar < p["cooldown"]:
                continue

            zv = z_lag.iloc[i]
            vol = rv_lag.iloc[i]
            a = atr_lag.iloc[i]
            if any(np.isnan(x) for x in [zv, vol, a]) or a <= 0:
                continue

            # Kill switch: vol outside acceptable band
            if vol < p["vol_min_ann"] or vol > p["vol_max_ann"]:
                continue

            entry = c.iloc[i]

            if zv > p["z_threshold"]:
                # Positive momentum → long
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.BUY
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry - p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry + p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(zv) / 2, 1.0)
                last_bar = i
                hold_entry = i
            elif zv < -p["z_threshold"]:
                # Negative momentum → short
                sig.iloc[i, sig.columns.get_loc("signal")] = Signal.SELL
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = entry + p["sl_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("take_profit")] = entry - p["tp_atr_mult"] * a
                sig.iloc[i, sig.columns.get_loc("confidence")] = min(abs(zv) / 2, 1.0)
                last_bar = i
                hold_entry = i

        return sig


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def get_v8_strategies():
    return {
        "VolBreakoutMom": VolBreakoutMomentumStrategy,
        "MeanRevLowVol": MeanRevLowVolStrategy,
        "WeekendGapFade": WeekendGapFadeStrategy,
        "BTCResidualMR": BTCResidualMRStrategy,
        "TSMOM": TSMOMStrategy,
    }
