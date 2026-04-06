"""
Regime-filtered long-only backtest pipeline.

Implements the 3-layer trading system:
  1. BTC Market Regime Filter (master switch)
  2. Relative Strength Selection (top assets only)
  3. Strategy Execution (only when layers 1+2 align)

Tests all V6/V7/V8 strategies with and without regime filter.

Usage:
    python scripts/run_regime_filtered.py
"""
import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from data.fetcher import DataFetcher
from strategies.base import Signal

from strategies.v6_aggressive import (
    MomentumAcceleratorStrategy,
    MultiEdgeCompositeStrategy,
    RegimeMomentumV2Strategy,
    CrossPairLeaderStrategy,
)
from strategies.v7_diverse import (
    AdaptiveChannelStrategy,
    VolatilityCaptureStrategy,
)
from strategies.v8_research import TSMOMStrategy, VolBreakoutMomentumStrategy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

fetcher = DataFetcher()

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL = "4h"
START = "2024-01-01"
END = "2026-04-01"
INITIAL_CAPITAL = 10_000.0
COMMISSION = 0.0
SLIPPAGE = 0.0005
RISK_PCT = 5.0
POS_PCT = 100.0
MAX_DD = 35.0
N_FOLDS = 5

ALL_SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "AVAXUSDC",
               "DOGEUSDC", "NEARUSDC", "ADAUSDC", "BNBUSDC"]

# Configs to test: (name, class, symbol, params)
CONFIGS = [
    # V6/V7 (default params)
    ("AdaptChan", AdaptiveChannelStrategy, "NEARUSDC", None),
    ("AdaptChan", AdaptiveChannelStrategy, "SOLUSDC", None),
    ("VolCapture", VolatilityCaptureStrategy, "AVAXUSDC", None),
    ("VolCapture", VolatilityCaptureStrategy, "NEARUSDC", None),
    ("RegimeMomV2", RegimeMomentumV2Strategy, "XRPUSDC", None),
    ("MultiEdge", MultiEdgeCompositeStrategy, "XRPUSDC", None),
    ("MultiEdge", MultiEdgeCompositeStrategy, "DOGEUSDC", None),
    ("MomAccel", MomentumAcceleratorStrategy, "AVAXUSDC", None),
    ("CrossPair", CrossPairLeaderStrategy, "BTCUSDC", None),
    ("CrossPair", CrossPairLeaderStrategy, "AVAXUSDC", None),
    # V8 fine-tuned best
    ("VolBreakoutMom", VolBreakoutMomentumStrategy, "XRPUSDC",
     {"lookback": 18, "k_threshold": 1.0, "tp_atr_mult": 3.0, "sl_atr_mult": 0.8}),
    ("VolBreakoutMom", VolBreakoutMomentumStrategy, "DOGEUSDC",
     {"lookback": 12, "k_threshold": 1.5, "tp_atr_mult": 3.5, "sl_atr_mult": 1.2}),
    ("TSMOM", TSMOMStrategy, "ADAUSDC",
     {"lookback": 10, "z_threshold": 1.1, "vol_min_ann": 0.3}),
    ("TSMOM", TSMOMStrategy, "SOLUSDC",
     {"lookback": 12, "z_threshold": 1.1, "vol_min_ann": 0.35}),
    ("TSMOM", TSMOMStrategy, "XRPUSDC",
     {"lookback": 10, "z_threshold": 1.0, "vol_min_ann": 0.25}),
]


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1: BTC REGIME FILTER
# ═════════════════════════════════════════════════════════════════════════════

def compute_btc_regime(btc_df: pd.DataFrame, sma_period: int = 50) -> pd.Series:
    """
    Compute BTC regime for each bar: True = RISK-ON, False = RISK-OFF.

    RISK-ON requires 2+ of these 4 conditions:
      1. BTC above SMA(50)
      2. BTC trend UP (higher highs & higher lows over last 20 bars)
      3. Volatility expanding (ATR rising over 10 bars)
      4. Breakouts succeeding (close near high of range, not fading)

    All indicators use t-1 data (shift) to prevent lookahead.
    """
    c = btc_df["close"]
    h = btc_df["high"]
    l = btc_df["low"]

    # Condition 1: Price above SMA
    sma = c.rolling(sma_period).mean()
    above_sma = (c.shift(1) > sma.shift(1))

    # Condition 2: Higher highs & higher lows (structural uptrend)
    # Compare recent 10-bar high/low vs previous 10-bar high/low
    hh_recent = h.shift(1).rolling(10).max()
    hh_prior = h.shift(11).rolling(10).max()
    ll_recent = l.shift(1).rolling(10).min()
    ll_prior = l.shift(11).rolling(10).min()
    hh_hl = (hh_recent > hh_prior) & (ll_recent > ll_prior)

    # Condition 3: Volatility expanding (ATR increasing)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_slope = (atr.shift(1) - atr.shift(11)) / atr.shift(11)
    vol_expanding = atr_slope > 0.05  # ATR grew >5% over 10 bars

    # Condition 4: Breakouts succeeding (close near range high, not fading)
    range_high = h.shift(1).rolling(20).max()
    range_low = l.shift(1).rolling(20).min()
    range_pos = (c.shift(1) - range_low) / (range_high - range_low + 1e-10)
    breakout_ok = range_pos > 0.7  # close in top 30% of range

    # RISK-ON = 2+ conditions true
    score = above_sma.astype(int) + hh_hl.astype(int) + vol_expanding.astype(int) + breakout_ok.astype(int)
    risk_on = score >= 2

    return risk_on


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2: RELATIVE STRENGTH SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_relative_strength(data: dict, bar_idx: int, sma_period: int = 50) -> list:
    """
    At a given bar index, rank all symbols by relative strength.
    Returns list of (symbol, score) sorted descending.

    Score = weighted combo of:
      - 7-day return (42 bars at 4h)
      - 14-day return (84 bars at 4h)
      - Volume expansion (recent vs avg)
      - Above SMA filter (must pass)

    All use t-1 data.
    """
    scores = []
    for sym, df in data.items():
        if sym == "BTCUSDC":
            continue  # BTC is the regime, not traded via RS
        if bar_idx < 100 or bar_idx >= len(df):
            continue

        c = df["close"]
        v = df["volume"]
        i = bar_idx

        # Must be above SMA (filter, not scored)
        sma = c.rolling(sma_period).mean()
        if c.iloc[i-1] <= sma.iloc[i-1]:
            continue

        # 7-day return
        ret_7d = (c.iloc[i-1] / c.iloc[max(i-43, 0)] - 1) * 100

        # 14-day return
        ret_14d = (c.iloc[i-1] / c.iloc[max(i-85, 0)] - 1) * 100

        # Volume expansion: recent 7-day avg vs 30-day avg
        vol_recent = v.iloc[max(i-42, 0):i].mean()
        vol_long = v.iloc[max(i-180, 0):i].mean()
        vol_ratio = vol_recent / vol_long if vol_long > 0 else 1.0

        # Higher highs check (structural uptrend)
        hh = df["high"].iloc[max(i-42, 0):i].max() > df["high"].iloc[max(i-84, 0):max(i-42, 0)].max()
        hh_bonus = 5.0 if hh else 0.0

        # Composite score (weighted)
        score = ret_7d * 0.35 + ret_14d * 0.30 + (vol_ratio - 1) * 20 * 0.15 + hh_bonus * 0.20

        scores.append((sym, round(score, 2)))

    scores.sort(key=lambda x: -x[1])
    return scores


def compute_rs_rankings(data: dict, btc_df_len: int) -> dict:
    """
    Pre-compute relative strength rankings for every bar.
    Returns {bar_idx: [(sym, score), ...]}
    """
    rankings = {}
    for i in range(100, btc_df_len):
        rankings[i] = compute_relative_strength(data, i)
    return rankings


# ═════════════════════════════════════════════════════════════════════════════
# LONG-ONLY BACKTEST WITH REGIME FILTER
# ═════════════════════════════════════════════════════════════════════════════

def backtest_regime(strat_cls, df, regime_series, params=None,
                    capital=INITIAL_CAPITAL, risk_pct=RISK_PCT,
                    pos_pct=POS_PCT, max_dd=MAX_DD,
                    commission=COMMISSION, slippage=SLIPPAGE):
    """
    Long-only backtest with regime filter.
    regime_series: bool Series aligned to df index. True = RISK-ON (allow trades).
    When RISK-OFF: existing position is NOT force-closed (SL/TP still apply),
    but NO new positions are opened.
    """
    strat = strat_cls(params=params)
    sig_df = strat.generate_signals(df.copy())

    cap = capital
    peak = cap
    pos = None
    trades = []
    equity_curve = []
    weekly_returns = []
    week_start_cap = cap
    bar_count = 0
    bars_per_week = 42
    regime_bars_on = 0
    regime_bars_off = 0

    for i in range(len(sig_df)):
        c_price = df["close"].iloc[i]
        lo = df["low"].iloc[i]
        hi = df["high"].iloc[i]
        sig_val = sig_df["signal"].iloc[i]
        sl_v = sig_df["stop_loss"].iloc[i]
        tp_v = sig_df["take_profit"].iloc[i]
        if pd.isna(sl_v): sl_v = 0
        if pd.isna(tp_v): tp_v = 0
        ts = df.index[i] if hasattr(df.index[i], 'isoformat') else str(df.index[i])

        # Regime check
        is_risk_on = True
        if regime_series is not None and i < len(regime_series):
            idx = df.index[i]
            if idx in regime_series.index:
                is_risk_on = bool(regime_series.loc[idx])
            else:
                is_risk_on = False

        if is_risk_on:
            regime_bars_on += 1
        else:
            regime_bars_off += 1

        # Weekly tracking
        bar_count += 1
        if bar_count >= bars_per_week:
            wr = (cap - week_start_cap) / week_start_cap * 100 if week_start_cap > 0 else 0
            weekly_returns.append(wr)
            week_start_cap = max(cap, 1.0)
            bar_count = 0

        # Exit logic (always active — respect SL/TP even in RISK-OFF)
        if pos:
            ex, reason = None, None
            if pos["sl"] > 0 and lo <= pos["sl"]:
                ex, reason = pos["sl"], "stop_loss"
            elif pos["tp"] > 0 and hi >= pos["tp"]:
                ex, reason = pos["tp"], "take_profit"
            elif sig_val == Signal.SELL:
                ex, reason = c_price, "signal"
            if ex:
                actual_ex = ex * (1 - slippage)
                pnl = (actual_ex - pos["entry"]) * pos["qty"]
                pnl -= actual_ex * pos["qty"] * commission
                cap += pnl
                cap = max(cap, 0.0)
                if cap > peak:
                    peak = cap
                trades.append({
                    "side": "BUY", "entry_price": pos["entry"],
                    "exit_price": actual_ex, "qty": pos["qty"],
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl / (pos["entry"] * pos["qty"]) * 100, 4),
                    "entry_time": pos["entry_time"], "exit_time": str(ts),
                    "exit_reason": reason,
                })
                pos = None

        # Entry logic — ONLY when RISK-ON
        if pos is None and sig_val == Signal.BUY and cap > 1.0 and is_risk_on:
            entry_p = c_price * (1 + slippage)
            if sl_v > 0:
                risk_amt = cap * (risk_pct / 100)
                pr = abs(entry_p - sl_v)
                qty = risk_amt / pr if pr > 0 else 0
                max_q = (cap * pos_pct / 100) / entry_p
                qty = min(qty, max_q)
            else:
                qty = (cap * pos_pct / 100) / entry_p
            cap -= entry_p * qty * commission
            if qty > 0:
                dd = (peak - cap) / peak * 100 if peak > 0 else 0
                if dd < max_dd:
                    pos = {"entry": entry_p, "qty": qty,
                           "sl": sl_v, "tp": tp_v, "entry_time": str(ts)}

        # MTM equity
        mtm = cap
        if pos:
            mtm += (c_price - pos["entry"]) * pos["qty"]
        equity_curve.append(mtm)

    # Close remaining
    if pos:
        c_price = df["close"].iloc[-1]
        pnl = (c_price * (1 - slippage) - pos["entry"]) * pos["qty"]
        pnl -= c_price * pos["qty"] * commission
        cap += pnl
        cap = max(cap, 0.0)
        trades.append({
            "side": "BUY", "entry_price": pos["entry"],
            "exit_price": c_price, "qty": pos["qty"],
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl / (pos["entry"] * pos["qty"]) * 100, 4),
            "entry_time": pos["entry_time"],
            "exit_time": str(df.index[-1]),
            "exit_reason": "end_of_data",
        })

    total_ret = (cap - capital) / capital * 100
    n_weeks = len(df) / bars_per_week
    weekly_avg = total_ret / max(n_weeks, 1)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    sharpe = 0
    if len(weekly_returns) > 1:
        wm = np.mean(weekly_returns)
        ws = np.std(weekly_returns)
        sharpe = wm / ws * np.sqrt(52) if ws > 0 else 0

    eq = pd.Series(equity_curve)
    pk = eq.expanding().max()
    dd_series = (eq - pk) / pk * 100
    max_dd_val = abs(dd_series.min()) if len(dd_series) > 0 else 0

    total_bars = regime_bars_on + regime_bars_off
    risk_on_pct = regime_bars_on / total_bars * 100 if total_bars > 0 else 0

    return {
        "final_capital": round(cap, 2),
        "total_return_pct": round(total_ret, 4),
        "weekly_return_pct": round(weekly_avg, 4),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_pct": round(max_dd_val, 2),
        "sharpe_annual": round(sharpe, 4),
        "n_weeks": round(n_weeks, 1),
        "risk_on_pct": round(risk_on_pct, 1),
        "trades": trades,
    }


def walk_forward_regime(strat_cls, df, btc_regime, params, n_folds=N_FOLDS):
    """Walk-forward with regime filter."""
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        s = f * fold_size
        e = min(s + fold_size, n)
        df_fold = df.iloc[s:e].copy()
        # Align regime to fold's date range
        fold_regime = btc_regime.reindex(df_fold.index) if btc_regime is not None else None
        r = backtest_regime(strat_cls, df_fold, fold_regime, params=params)
        folds.append({
            "fold": f + 1,
            "return_pct": r["total_return_pct"],
            "weekly_pct": r["weekly_return_pct"],
            "n_trades": r["n_trades"],
            "win_rate": r["win_rate_pct"],
            "max_dd": r["max_drawdown_pct"],
            "risk_on_pct": r["risk_on_pct"],
        })
    return folds


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 110)
    print("  REGIME-FILTERED LONG-ONLY BACKTEST PIPELINE")
    print(f"  3-Layer System: BTC Regime → Relative Strength → Strategy Execution")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}  |  Risk: {RISK_PCT}%  |  Period: {START} → {END}")
    print("=" * 110)

    # ── Fetch data ───────────────────────────────────────────────────────────
    print(f"\n  [Phase 1] Fetching data for {len(ALL_SYMBOLS)} symbols...")
    data = {}
    for sym in ALL_SYMBOLS:
        df = fetcher.fetch_klines(sym, INTERVAL, START, END)
        if not df.empty:
            data[sym] = df
            print(f"    {sym}: {len(df)} bars")
        time.sleep(0.3)

    btc_df = data["BTCUSDC"]
    btc_close = btc_df["close"].rename("btc_close")

    # ── Layer 1: Compute BTC regime ──────────────────────────────────────────
    print(f"\n  [Phase 2] Computing BTC regime filter...")
    btc_regime = compute_btc_regime(btc_df)
    risk_on_total = btc_regime.sum()
    risk_on_pct = risk_on_total / len(btc_regime) * 100
    print(f"    BTC RISK-ON bars: {risk_on_total}/{len(btc_regime)} ({risk_on_pct:.1f}%)")
    print(f"    BTC RISK-OFF bars: {len(btc_regime) - risk_on_total} ({100 - risk_on_pct:.1f}%)")

    # Show regime transitions
    changes = btc_regime.astype(int).diff().ne(0).sum()
    print(f"    Regime transitions: {changes}")

    # ── Layer 2: Compute relative strength (pre-compute) ────────────────────
    print(f"\n  [Phase 3] Computing relative strength rankings...")
    rs_rankings = compute_rs_rankings(data, len(btc_df))
    # Show a sample ranking at bar 2000
    sample_bar = min(2000, len(btc_df) - 1)
    if sample_bar in rs_rankings:
        sample = rs_rankings[sample_bar][:5]
        ts = btc_df.index[sample_bar]
        print(f"    Sample ranking at {ts}:")
        for sym, sc in sample:
            print(f"      {sym:<12} score: {sc:+.1f}")

    # ── Phase 4: Test ALL configs — with & without regime filter ─────────────
    print(f"\n" + "=" * 110)
    print(f"  [Phase 4] BACKTEST: {len(CONFIGS)} configs × 2 modes (raw vs regime-filtered)")
    print("=" * 110)

    results = []
    for name, scls, sym, params in CONFIGS:
        if sym not in data:
            continue

        df = data[sym].copy()
        # Inject BTC data for CrossPair
        if name == "CrossPair" and sym != "BTCUSDC":
            df = df.join(btc_close, how="left")
            df["btc_close"] = df["btc_close"].ffill()

        # --- A: Raw (no regime filter) ---
        r_raw = backtest_regime(scls, df, None, params=params)
        folds_raw = walk_forward_regime(scls, df, None, params)
        pos_raw = sum(1 for f in folds_raw if f["return_pct"] > 0)
        avg_wkly_raw = np.mean([f["weekly_pct"] for f in folds_raw])

        # --- B: With BTC regime filter ---
        r_regime = backtest_regime(scls, df, btc_regime, params=params)
        folds_regime = walk_forward_regime(scls, df, btc_regime, params)
        pos_regime = sum(1 for f in folds_regime if f["return_pct"] > 0)
        avg_wkly_regime = np.mean([f["weekly_pct"] for f in folds_regime])

        # --- C: With regime + relative strength filter ---
        # Only allow entry when symbol is in top 3 of RS ranking
        # We implement this as an additional filter on the regime series
        rs_regime = btc_regime.copy()
        for bar_i in range(len(btc_df)):
            if not btc_regime.iloc[bar_i]:
                continue
            ranking = rs_rankings.get(bar_i, [])
            top_syms = [s for s, _ in ranking[:3]]
            if sym not in top_syms and sym != "BTCUSDC":
                rs_regime.iloc[bar_i] = False

        r_full = backtest_regime(scls, df, rs_regime, params=params)
        folds_full = walk_forward_regime(scls, df, rs_regime, params)
        pos_full = sum(1 for f in folds_full if f["return_pct"] > 0)
        avg_wkly_full = np.mean([f["weekly_pct"] for f in folds_full])
        avg_dd_full = np.mean([f["max_dd"] for f in folds_full])

        label = f"{name} {sym}"
        fold_raw_s = f"{pos_raw}/5"
        fold_reg_s = f"{pos_regime}/5"
        fold_full_s = f"{pos_full}/5"

        imp_regime = avg_wkly_regime - avg_wkly_raw
        imp_full = avg_wkly_full - avg_wkly_raw

        print(f"\n  {label:<30}")
        print(f"    Raw (no filter):    full:{r_raw['weekly_return_pct']:+.2f}%/wk  "
              f"WF:{fold_raw_s} avg:{avg_wkly_raw:+.2f}%/wk  DD:{r_raw['max_drawdown_pct']:.1f}%  "
              f"trades:{r_raw['n_trades']}")
        print(f"    + BTC Regime:       full:{r_regime['weekly_return_pct']:+.2f}%/wk  "
              f"WF:{fold_reg_s} avg:{avg_wkly_regime:+.2f}%/wk  DD:{r_regime['max_drawdown_pct']:.1f}%  "
              f"trades:{r_regime['n_trades']}  risk_on:{r_regime['risk_on_pct']:.0f}%  "
              f"(Δ{imp_regime:+.2f}%)")
        print(f"    + Regime + RS:      full:{r_full['weekly_return_pct']:+.2f}%/wk  "
              f"WF:{fold_full_s} avg:{avg_wkly_full:+.2f}%/wk  DD:{r_full['max_drawdown_pct']:.1f}%  "
              f"trades:{r_full['n_trades']}  risk_on:{r_full['risk_on_pct']:.0f}%  "
              f"(Δ{imp_full:+.2f}%)")

        results.append({
            "strategy": name, "symbol": sym,
            # Raw
            "raw_full_wkly": r_raw["weekly_return_pct"],
            "raw_wf_avg": round(avg_wkly_raw, 4),
            "raw_wf_folds": pos_raw,
            "raw_dd": r_raw["max_drawdown_pct"],
            "raw_trades": r_raw["n_trades"],
            "raw_sharpe": r_raw["sharpe_annual"],
            # Regime only
            "regime_full_wkly": r_regime["weekly_return_pct"],
            "regime_wf_avg": round(avg_wkly_regime, 4),
            "regime_wf_folds": pos_regime,
            "regime_dd": r_regime["max_drawdown_pct"],
            "regime_trades": r_regime["n_trades"],
            "regime_sharpe": r_regime["sharpe_annual"],
            "regime_risk_on_pct": r_regime["risk_on_pct"],
            # Full (regime + RS)
            "full_full_wkly": r_full["weekly_return_pct"],
            "full_wf_avg": round(avg_wkly_full, 4),
            "full_wf_folds": pos_full,
            "full_dd": r_full["max_drawdown_pct"],
            "full_trades": r_full["n_trades"],
            "full_sharpe": r_full["sharpe_annual"],
            "full_risk_on_pct": r_full["risk_on_pct"],
            # Deltas
            "regime_improvement": round(avg_wkly_regime - avg_wkly_raw, 4),
            "full_improvement": round(avg_wkly_full - avg_wkly_raw, 4),
            # WF detail (full system)
            "full_folds": folds_full,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    rdf = pd.DataFrame([{k: v for k, v in r.items() if k != "full_folds"} for r in results])
    rdf.to_csv(RESULTS_DIR / "regime_filtered_results.csv", index=False)

    print("\n" + "=" * 110)
    print("  SUMMARY: REGIME FILTER IMPACT")
    print("=" * 110)

    # Sort by full system WF avg
    rdf_s = rdf.sort_values("full_wf_avg", ascending=False)

    print(f"\n  {'Config':<30} {'Raw WF':>8} {'+ Regime':>9} {'+ RS':>9} {'Δ':>6} {'Folds':>6} {'DD':>6}")
    print(f"  {'─' * 80}")
    for _, row in rdf_s.iterrows():
        label = f"{row['strategy']} {row['symbol']}"
        delta = row["full_improvement"]
        ds = f"{delta:+.2f}" if delta != 0 else "  0.00"
        flag = ""
        if row["full_wf_folds"] == 5 and row["full_wf_avg"] >= 1.0:
            flag = " <<< PRODUCTION"
        elif row["full_wf_folds"] == 5:
            flag = " <<< 5/5"
        elif row["full_wf_folds"] >= 4 and row["full_wf_avg"] >= 0.8:
            flag = " <<< STRONG"
        print(f"  {label:<30} {row['raw_wf_avg']:>+7.2f}% {row['regime_wf_avg']:>+8.2f}% "
              f"{row['full_wf_avg']:>+8.2f}% {ds}% {row['full_wf_folds']:>4}/5 "
              f"{row['full_dd']:>5.1f}%{flag}")

    # Count improvements
    improved_regime = (rdf["regime_improvement"] > 0).sum()
    improved_full = (rdf["full_improvement"] > 0).sum()
    five_five_raw = (rdf["raw_wf_folds"] == 5).sum()
    five_five_full = (rdf["full_wf_folds"] == 5).sum()
    target_raw = ((rdf["raw_wf_folds"] == 5) & (rdf["raw_wf_avg"] >= 1.0)).sum()
    target_full = ((rdf["full_wf_folds"] == 5) & (rdf["full_wf_avg"] >= 1.0)).sum()

    print(f"\n  Regime filter improved: {improved_regime}/{len(rdf)} configs")
    print(f"  Full system improved: {improved_full}/{len(rdf)} configs")
    print(f"  5/5 folds: {five_five_raw} (raw) → {five_five_full} (filtered)")
    print(f"  5/5 + ≥1%/wk: {target_raw} (raw) → {target_full} (filtered)")

    # Regime filter stats
    avg_dd_raw = rdf["raw_dd"].mean()
    avg_dd_full = rdf["full_dd"].mean()
    avg_trades_raw = rdf["raw_trades"].mean()
    avg_trades_full = rdf["full_trades"].mean()
    print(f"\n  Average DD: {avg_dd_raw:.1f}% (raw) → {avg_dd_full:.1f}% (filtered)")
    print(f"  Average trades: {avg_trades_raw:.0f} (raw) → {avg_trades_full:.0f} (filtered)")

    # Per-strategy JSON for best full-system configs
    for r in results:
        if r["full_wf_folds"] >= 4:
            fname = f"regime_{r['strategy']}_{r['symbol']}.json"
            with open(RESULTS_DIR / fname, "w") as f:
                json.dump({
                    "strategy": r["strategy"],
                    "symbol": r["symbol"],
                    "mode": "regime_filtered_longonly",
                    "raw_wf_avg": r["raw_wf_avg"],
                    "regime_wf_avg": r["regime_wf_avg"],
                    "full_wf_avg": r["full_wf_avg"],
                    "full_wf_folds": r["full_wf_folds"],
                    "full_dd": r["full_dd"],
                    "full_sharpe": r["full_sharpe"],
                    "risk_on_pct": r["full_risk_on_pct"],
                    "folds": r["full_folds"],
                }, f, indent=2, default=str)

    print("\n" + "=" * 110)
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 110)


if __name__ == "__main__":
    main()
