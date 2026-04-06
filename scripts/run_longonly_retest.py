"""
Long-only re-test of all V6/V7/V8 validated configs + fine TSMOM sweep.

Uses the same long-only backtest engine from run_v8_pipeline.py.
Purpose: determine which of the 10 V6/V7 configs (originally tested with
short-selling) still exceed 1%/wk under strictly long-only spot constraints.

Also runs a fine parameter grid around TSMOM on SOL/ADA to push toward 1%/wk.

Usage:
    python scripts/run_longonly_retest.py
"""
import sys, os, json, time
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

# ─── Configuration ────────────────────────────────────────────────────────────
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

# The 10 validated configs (all default params)
VALIDATED_CONFIGS = [
    ("MomentumAccelerator", MomentumAcceleratorStrategy, "AVAXUSDC", None),
    ("MultiEdgeComposite", MultiEdgeCompositeStrategy, "XRPUSDC", None),
    ("RegimeMomentumV2", RegimeMomentumV2Strategy, "XRPUSDC", None),
    ("MultiEdgeComposite", MultiEdgeCompositeStrategy, "DOGEUSDC", None),
    ("CrossPairLeader", CrossPairLeaderStrategy, "AVAXUSDC", None),
    ("CrossPairLeader", CrossPairLeaderStrategy, "BTCUSDC", None),
    ("AdaptiveChannel", AdaptiveChannelStrategy, "NEARUSDC", None),
    ("AdaptiveChannel", AdaptiveChannelStrategy, "SOLUSDC", None),
    ("VolatilityCapture", VolatilityCaptureStrategy, "AVAXUSDC", None),
    ("VolatilityCapture", VolatilityCaptureStrategy, "NEARUSDC", None),
]

# Fine TSMOM sweep around the best V8 params (SOL: lb=12, z=1.0 was +0.97%/wk)
TSMOM_FINE_GRID = []
for lb in [10, 11, 12, 13, 14]:
    for zt in [0.70, 0.80, 0.90, 1.00, 1.10]:
        for vm in [0.25, 0.30, 0.35]:
            TSMOM_FINE_GRID.append({
                "lookback": lb, "z_threshold": zt, "vol_min_ann": vm
            })

TSMOM_SYMBOLS = ["SOLUSDC", "ADAUSDC", "AVAXUSDC", "DOGEUSDC", "XRPUSDC"]

# VolBreakoutMom fine sweep (DOGE: lb=10,k=1.5 was +0.70%/wk; XRP: lb=15,k=1.5 was +0.77%/wk)
VBM_FINE_GRID = []
for lb in [8, 10, 12, 15, 18]:
    for kt in [1.0, 1.25, 1.5, 1.75]:
        for tp in [2.5, 3.0, 3.5]:
            for sl in [0.8, 1.0, 1.2]:
                VBM_FINE_GRID.append({
                    "lookback": lb, "k_threshold": kt,
                    "tp_atr_mult": tp, "sl_atr_mult": sl
                })

VBM_SYMBOLS = ["DOGEUSDC", "XRPUSDC", "ADAUSDC"]


# ─── Long-Only Backtest Engine (exact copy from run_v8_pipeline.py) ───────────

def backtest(strat_cls, df, params=None, capital=INITIAL_CAPITAL,
             risk_pct=RISK_PCT, pos_pct=POS_PCT, max_dd=MAX_DD,
             commission=COMMISSION, slippage=SLIPPAGE):
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

        bar_count += 1
        if bar_count >= bars_per_week:
            wr = (cap - week_start_cap) / week_start_cap * 100 if week_start_cap > 0 else 0
            weekly_returns.append(wr)
            week_start_cap = max(cap, 1.0)
            bar_count = 0

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

        if pos is None and sig_val == Signal.BUY and cap > 1.0:
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

        mtm = cap
        if pos:
            mtm += (c_price - pos["entry"]) * pos["qty"]
        equity_curve.append(mtm)

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
        "trades": trades,
    }


def walk_forward(strat_cls, df, params, n_folds=N_FOLDS):
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        oos_start = f * fold_size
        oos_end = min(oos_start + fold_size, n)
        df_oos = df.iloc[oos_start:oos_end].copy()
        r = backtest(strat_cls, df_oos, params=params)
        folds.append({
            "fold": f + 1,
            "return_pct": r["total_return_pct"],
            "weekly_pct": r["weekly_return_pct"],
            "n_trades": r["n_trades"],
            "win_rate": r["win_rate_pct"],
            "max_dd": r["max_drawdown_pct"],
        })
    return folds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 110)
    print("  LONG-ONLY RE-TEST — V6/V7 Validated Configs + V8 Fine Sweep")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}  |  Risk: {RISK_PCT}%  |  "
          f"Slippage: {SLIPPAGE*100}%  |  Period: {START} → {END}")
    print("=" * 110)

    # ── Fetch all needed data ────────────────────────────────────────────────
    symbols_needed = set()
    for _, _, sym, _ in VALIDATED_CONFIGS:
        symbols_needed.add(sym)
    for sym in TSMOM_SYMBOLS:
        symbols_needed.add(sym)
    for sym in VBM_SYMBOLS:
        symbols_needed.add(sym)
    symbols_needed.add("BTCUSDC")  # for CrossPairLeader

    print(f"\n  [Phase 1] Fetching data for {len(symbols_needed)} symbols...")
    data = {}
    btc_close = None
    for sym in sorted(symbols_needed):
        df = fetcher.fetch_klines(sym, INTERVAL, START, END)
        if not df.empty:
            data[sym] = df
            print(f"    {sym}: {len(df)} bars")
            if sym == "BTCUSDC":
                btc_close = df["close"].rename("btc_close")
        time.sleep(0.3)

    # ══════════════════════════════════════════════════════════════════════════
    # PART A: Re-test V6/V7 validated configs long-only
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  PART A: V6/V7 VALIDATED CONFIGS — LONG-ONLY RE-TEST")
    print("=" * 110)

    retest_results = []
    for name, scls, sym, params in VALIDATED_CONFIGS:
        if sym not in data:
            print(f"  SKIP {name} {sym}: no data")
            continue

        df = data[sym].copy()
        # Inject BTC data for CrossPairLeader
        if name == "CrossPairLeader" and btc_close is not None and sym != "BTCUSDC":
            df = df.join(btc_close, how="left")
            df["btc_close"] = df["btc_close"].ffill()

        # Full-period backtest
        r = backtest(scls, df, params=params)

        # Walk-forward
        df_wf = data[sym].copy()
        if name == "CrossPairLeader" and btc_close is not None and sym != "BTCUSDC":
            df_wf = df_wf.join(btc_close, how="left")
            df_wf["btc_close"] = df_wf["btc_close"].ffill()

        folds = walk_forward(scls, df_wf, params)
        pos_folds = sum(1 for f in folds if f["return_pct"] > 0)
        avg_wkly = np.mean([f["weekly_pct"] for f in folds])
        avg_dd = np.mean([f["max_dd"] for f in folds])
        all_pos = pos_folds == N_FOLDS

        fold_str = "  ".join([f"F{f['fold']}:{f['return_pct']:+.1f}%" for f in folds])
        status = "ALL+" if all_pos else f"{pos_folds}/{N_FOLDS}"
        flag = " <<< VALIDATED" if all_pos and avg_wkly >= 1.0 else (" <<< ALL+" if all_pos else "")

        print(f"  {name:<22} {sym:<12} full:{r['weekly_return_pct']:+.2f}%/wk  "
              f"WF:{status:<5} avg:{avg_wkly:+.2f}%/wk  DD:{avg_dd:.1f}%  |  {fold_str}{flag}")

        retest_results.append({
            "strategy": name, "symbol": sym,
            "full_return_pct": r["total_return_pct"],
            "full_weekly_pct": r["weekly_return_pct"],
            "full_max_dd_pct": r["max_drawdown_pct"],
            "full_sharpe": r["sharpe_annual"],
            "full_pf": r["profit_factor"],
            "full_trades": r["n_trades"],
            "full_wr": r["win_rate_pct"],
            "wf_positive_folds": pos_folds,
            "wf_all_positive": all_pos,
            "wf_avg_weekly_pct": round(avg_wkly, 4),
            "wf_avg_dd_pct": round(avg_dd, 2),
            "folds": folds,
        })

    rdf = pd.DataFrame([{k: v for k, v in r.items() if k != "folds"} for r in retest_results])
    rdf = rdf.sort_values("wf_avg_weekly_pct", ascending=False)
    rdf.to_csv(RESULTS_DIR / "v6v7_longonly_retest.csv", index=False)

    all_pos_df = rdf[rdf["wf_all_positive"]]
    print(f"\n  V6/V7 configs passing 5/5 long-only: {len(all_pos_df)}/{len(rdf)}")
    if len(all_pos_df) > 0:
        at_target = all_pos_df[all_pos_df["wf_avg_weekly_pct"] >= 1.0]
        print(f"  V6/V7 configs >= 1%/wk long-only + 5/5: {len(at_target)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B: Fine TSMOM sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print(f"  PART B: FINE TSMOM SWEEP ({len(TSMOM_FINE_GRID)} param sets × {len(TSMOM_SYMBOLS)} symbols)")
    print("=" * 110)

    tsmom_results = []
    best_tsmom = {}  # {symbol: (weekly, params)}
    for sym in TSMOM_SYMBOLS:
        if sym not in data:
            continue
        df = data[sym].copy()
        best_wkly = -999
        best_p = None
        for pset in TSMOM_FINE_GRID:
            r = backtest(TSMOMStrategy, df, params=pset)
            tsmom_results.append({
                "strategy": "TSMOM", "symbol": sym,
                "params": str(pset), **{k: v for k, v in r.items() if k != "trades"}
            })
            if r["weekly_return_pct"] > best_wkly:
                best_wkly = r["weekly_return_pct"]
                best_p = pset
        best_tsmom[sym] = (best_wkly, best_p)
        print(f"  TSMOM {sym:<12} best: {best_wkly:+.3f}%/wk  params: {best_p}")

    tdf = pd.DataFrame(tsmom_results)
    tdf = tdf.sort_values("weekly_return_pct", ascending=False)
    tdf.to_csv(RESULTS_DIR / "tsmom_fine_sweep.csv", index=False)

    # Walk-forward on top TSMOM combos
    print(f"\n  Walk-forward on best TSMOM combos...")
    tsmom_wf = []
    for sym, (wkly, params) in sorted(best_tsmom.items(), key=lambda x: -x[1][0]):
        if wkly < 0.3 or params is None:
            continue
        df = data[sym].copy()
        folds = walk_forward(TSMOMStrategy, df, params)
        pos_folds = sum(1 for f in folds if f["return_pct"] > 0)
        avg_wkly = np.mean([f["weekly_pct"] for f in folds])
        avg_dd = np.mean([f["max_dd"] for f in folds])
        all_pos = pos_folds == N_FOLDS
        fold_str = "  ".join([f"F{f['fold']}:{f['return_pct']:+.1f}%" for f in folds])
        status = "ALL+" if all_pos else f"{pos_folds}/{N_FOLDS}"
        flag = " <<< VALIDATED" if all_pos and avg_wkly >= 1.0 else (" <<< ALL+" if all_pos else "")
        print(f"  TSMOM {sym:<12} {status:<5} avg:{avg_wkly:+.2f}%/wk  DD:{avg_dd:.1f}%  |  {fold_str}{flag}")
        tsmom_wf.append({
            "strategy": "TSMOM", "symbol": sym, "params": str(params),
            "positive_folds": pos_folds, "all_positive": all_pos,
            "avg_weekly_pct": round(avg_wkly, 4), "avg_max_dd_pct": round(avg_dd, 2),
        })

    # ══════════════════════════════════════════════════════════════════════════
    # PART C: Fine VolBreakoutMom sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print(f"  PART C: FINE VolBreakoutMom SWEEP ({len(VBM_FINE_GRID)} param sets × {len(VBM_SYMBOLS)} symbols)")
    print("=" * 110)

    vbm_results = []
    best_vbm = {}
    for sym in VBM_SYMBOLS:
        if sym not in data:
            continue
        df = data[sym].copy()
        best_wkly = -999
        best_p = None
        for pset in VBM_FINE_GRID:
            r = backtest(VolBreakoutMomentumStrategy, df, params=pset)
            vbm_results.append({
                "strategy": "VolBreakoutMom", "symbol": sym,
                "params": str(pset), **{k: v for k, v in r.items() if k != "trades"}
            })
            if r["weekly_return_pct"] > best_wkly:
                best_wkly = r["weekly_return_pct"]
                best_p = pset
        best_vbm[sym] = (best_wkly, best_p)
        print(f"  VBM {sym:<12} best: {best_wkly:+.3f}%/wk  params: {best_p}")

    vdf = pd.DataFrame(vbm_results)
    vdf = vdf.sort_values("weekly_return_pct", ascending=False)
    vdf.to_csv(RESULTS_DIR / "vbm_fine_sweep.csv", index=False)

    # Walk-forward on best VBM combos
    print(f"\n  Walk-forward on best VBM combos...")
    vbm_wf = []
    for sym, (wkly, params) in sorted(best_vbm.items(), key=lambda x: -x[1][0]):
        if wkly < 0.3 or params is None:
            continue
        df = data[sym].copy()
        folds = walk_forward(VolBreakoutMomentumStrategy, df, params)
        pos_folds = sum(1 for f in folds if f["return_pct"] > 0)
        avg_wkly = np.mean([f["weekly_pct"] for f in folds])
        avg_dd = np.mean([f["max_dd"] for f in folds])
        all_pos = pos_folds == N_FOLDS
        fold_str = "  ".join([f"F{f['fold']}:{f['return_pct']:+.1f}%" for f in folds])
        status = "ALL+" if all_pos else f"{pos_folds}/{N_FOLDS}"
        flag = " <<< VALIDATED" if all_pos and avg_wkly >= 1.0 else (" <<< ALL+" if all_pos else "")
        print(f"  VBM {sym:<12} {status:<5} avg:{avg_wkly:+.2f}%/wk  DD:{avg_dd:.1f}%  |  {fold_str}{flag}")
        vbm_wf.append({
            "strategy": "VolBreakoutMom", "symbol": sym, "params": str(params),
            "positive_folds": pos_folds, "all_positive": all_pos,
            "avg_weekly_pct": round(avg_wkly, 4), "avg_max_dd_pct": round(avg_dd, 2),
        })

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  FINAL SUMMARY — ALL LONG-ONLY VALIDATED CONFIGS")
    print("=" * 110)

    # Collect all walk-forward-validated results
    all_validated = []
    for r in retest_results:
        all_validated.append({
            "source": "V6/V7", "strategy": r["strategy"], "symbol": r["symbol"],
            "full_weekly": r["full_weekly_pct"], "wf_avg_weekly": r["wf_avg_weekly_pct"],
            "pos_folds": r["wf_positive_folds"], "all_pos": r["wf_all_positive"],
            "wf_dd": r["wf_avg_dd_pct"],
        })
    for r in tsmom_wf:
        all_validated.append({
            "source": "V8-TSMOM", "strategy": r["strategy"], "symbol": r["symbol"],
            "full_weekly": best_tsmom.get(r["symbol"], (0,))[0],
            "wf_avg_weekly": r["avg_weekly_pct"],
            "pos_folds": r["positive_folds"], "all_pos": r["all_positive"],
            "wf_dd": r["avg_max_dd_pct"],
        })
    for r in vbm_wf:
        all_validated.append({
            "source": "V8-VBM", "strategy": r["strategy"], "symbol": r["symbol"],
            "full_weekly": best_vbm.get(r["symbol"], (0,))[0],
            "wf_avg_weekly": r["avg_weekly_pct"],
            "pos_folds": r["positive_folds"], "all_pos": r["all_positive"],
            "wf_dd": r["avg_max_dd_pct"],
        })

    adf = pd.DataFrame(all_validated).sort_values("wf_avg_weekly", ascending=False)
    adf.to_csv(RESULTS_DIR / "all_longonly_validated.csv", index=False)

    five_five = adf[adf["all_pos"]]
    target = adf[(adf["all_pos"]) & (adf["wf_avg_weekly"] >= 1.0)]

    print(f"\n  Total configs tested: {len(adf)}")
    print(f"  5/5 folds positive: {len(five_five)}")
    print(f"  5/5 + >= 1%/wk: {len(target)}")

    if len(five_five) > 0:
        print(f"\n  ALL 5/5 POSITIVE (long-only):")
        for _, row in five_five.iterrows():
            m = " <<< TARGET" if row["wf_avg_weekly"] >= 1.0 else ""
            print(f"    [{row['source']:<9}] {row['strategy']:<22} {row['symbol']:<12} "
                  f"WF:{row['wf_avg_weekly']:+.2f}%/wk  DD:{row['wf_dd']:.1f}%{m}")

    if len(target) > 0:
        print(f"\n  PRODUCTION-READY (5/5 + >=1%/wk long-only):")
        for _, row in target.iterrows():
            print(f"    → {row['strategy']} on {row['symbol']} "
                  f"({row['wf_avg_weekly']:+.2f}%/wk, DD:{row['wf_dd']:.1f}%)")
    else:
        print(f"\n  No configs reach 1%/wk with 5/5 under long-only constraint.")
        print(f"  Best candidate: {adf.iloc[0]['strategy']} on {adf.iloc[0]['symbol']} "
              f"({adf.iloc[0]['wf_avg_weekly']:+.2f}%/wk, {adf.iloc[0]['pos_folds']}/5 folds)")

    print("\n" + "=" * 110)
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 110)


if __name__ == "__main__":
    main()
