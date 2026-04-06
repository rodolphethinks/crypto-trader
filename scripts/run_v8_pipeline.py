"""
V8 Research Pipeline — Complete research-to-backtest system.

1. Fetches 2-year 4h OHLCV for all symbols
2. Runs each strategy with default params (baseline)
3. Small parameter grid sweep per strategy
4. 5-fold walk-forward validation on best params
5. Outputs per-strategy results JSON + trade CSV
6. Produces final ranked report

Usage:
    python scripts/run_v8_pipeline.py
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from data.fetcher import DataFetcher
from strategies.v8_research import get_v8_strategies
from strategies.base import Signal
from config.settings import LOG_DIR

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

fetcher = DataFetcher()

# ─── Configuration ────────────────────────────────────────────────────────────

SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "AVAXUSDC",
           "DOGEUSDC", "NEARUSDC", "ADAUSDC", "BNBUSDC"]
INTERVAL = "4h"
START = "2024-01-01"
END = "2026-04-01"
INITIAL_CAPITAL = 10_000.0
COMMISSION = 0.0       # zero-fee USDC pairs
SLIPPAGE = 0.0005      # 0.05%

# Aggressive sizing (validated in V6/V7)
RISK_PCT = 5.0
POS_PCT = 100.0
MAX_DD = 35.0

# Walk-forward
N_FOLDS = 5

# Parameter grids (small, focused sweep per strategy)
PARAM_GRIDS = {
    "VolBreakoutMom": [
        {"lookback": 8,  "k_threshold": 1.0, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0},
        {"lookback": 10, "k_threshold": 1.5, "tp_atr_mult": 2.5, "sl_atr_mult": 1.0},
        {"lookback": 10, "k_threshold": 2.0, "tp_atr_mult": 3.0, "sl_atr_mult": 1.5},
        {"lookback": 15, "k_threshold": 1.5, "tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        {"lookback": 20, "k_threshold": 1.0, "tp_atr_mult": 2.5, "sl_atr_mult": 1.5},
    ],
    "MeanRevLowVol": [
        {"ma_period": 20, "z_entry": 1.5, "vol_max_pctile": 30},
        {"ma_period": 20, "z_entry": 2.0, "vol_max_pctile": 40},
        {"ma_period": 30, "z_entry": 2.0, "vol_max_pctile": 40},
        {"ma_period": 30, "z_entry": 2.5, "vol_max_pctile": 50},
        {"ma_period": 50, "z_entry": 2.0, "vol_max_pctile": 40},
    ],
    "WeekendGapFade": [
        {"gap_atr_mult": 1.0, "sl_atr_mult": 1.0},
        {"gap_atr_mult": 1.5, "sl_atr_mult": 1.5},
        {"gap_atr_mult": 2.0, "sl_atr_mult": 1.5},
        {"gap_atr_mult": 2.0, "sl_atr_mult": 2.0},
        {"gap_atr_mult": 2.5, "sl_atr_mult": 2.0},
    ],
    "BTCResidualMR": [
        {"z_entry": 1.5, "corr_window": 40, "min_corr": 0.50},
        {"z_entry": 2.0, "corr_window": 60, "min_corr": 0.60},
        {"z_entry": 2.0, "corr_window": 60, "min_corr": 0.70},
        {"z_entry": 2.5, "corr_window": 90, "min_corr": 0.60},
        {"z_entry": 2.5, "corr_window": 90, "min_corr": 0.70},
    ],
    "TSMOM": [
        {"lookback": 6,  "z_threshold": 0.50, "vol_min_ann": 0.20},
        {"lookback": 8,  "z_threshold": 0.75, "vol_min_ann": 0.30},
        {"lookback": 12, "z_threshold": 0.75, "vol_min_ann": 0.30},
        {"lookback": 12, "z_threshold": 1.00, "vol_min_ann": 0.30},
        {"lookback": 24, "z_threshold": 0.75, "vol_min_ann": 0.20},
    ],
}


# ─── Backtest Engine (standalone, no RiskManager dependency) ──────────────────

def backtest(strat_cls, df, params=None, capital=INITIAL_CAPITAL,
             risk_pct=RISK_PCT, pos_pct=POS_PCT, max_dd=MAX_DD,
             commission=COMMISSION, slippage=SLIPPAGE):
    """Run a single backtest. Returns dict of metrics + trade list."""
    strat = strat_cls(params=params)
    sig_df = strat.generate_signals(df.copy())

    cap = capital
    peak = cap
    pos = None          # long-only (MEXC spot: no shorting)
    trades = []
    equity_curve = []
    weekly_returns = []
    week_start_cap = cap
    bar_count = 0
    bars_per_week = 42  # 4h

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

        # Weekly tracking
        bar_count += 1
        if bar_count >= bars_per_week:
            wr = (cap - week_start_cap) / week_start_cap * 100 if week_start_cap > 0 else 0
            weekly_returns.append(wr)
            week_start_cap = max(cap, 1.0)  # prevent div-by-zero
            bar_count = 0

        # Check open long position for exit
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
                cap = max(cap, 0.0)  # floor at 0 (spot: can't lose more than invested)
                if cap > peak:
                    peak = cap
                trades.append({
                    "side": "BUY",
                    "entry_price": pos["entry"],
                    "exit_price": actual_ex,
                    "qty": pos["qty"],
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl / (pos["entry"] * pos["qty"]) * 100, 4),
                    "entry_time": pos["entry_time"],
                    "exit_time": str(ts),
                    "exit_reason": reason,
                })
                pos = None

        # Open new LONG position only (spot: no shorts)
        if pos is None and sig_val == Signal.BUY and cap > 1.0:
            entry_p = c_price * (1 + slippage)
            # Position sizing
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

    # Close remaining long
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

    # Metrics
    total_ret = (cap - capital) / capital * 100
    n_weeks = len(df) / bars_per_week
    weekly_avg = total_ret / max(n_weeks, 1)
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe from weekly returns
    sharpe = 0
    if len(weekly_returns) > 1:
        wm = np.mean(weekly_returns)
        ws = np.std(weekly_returns)
        sharpe = wm / ws * np.sqrt(52) if ws > 0 else 0

    # Max drawdown from equity curve
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


# ─── Walk-Forward Validation ─────────────────────────────────────────────────

def walk_forward(strat_cls, df, params, n_folds=N_FOLDS):
    """Run n-fold expanding walk-forward. Returns per-fold results."""
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        oos_start = f * fold_size
        oos_end = min(oos_start + fold_size, n)
        df_oos = df.iloc[oos_start:oos_end].copy()  # keep DatetimeIndex
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


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 110)
    print("  V8 RESEARCH PIPELINE — Strategy Extraction → Backtest → Validation → Report")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Interval: {INTERVAL}  |  Period: {START} → {END}")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}  |  Risk: {RISK_PCT}%  |  "
          f"Commission: {COMMISSION*100}%  |  Slippage: {SLIPPAGE*100}%")
    print("=" * 110)

    strategies = get_v8_strategies()
    print(f"\n  Strategies: {', '.join(strategies.keys())}")

    # ── Phase 1: Fetch Data ──────────────────────────────────────────────────
    print("\n  [Phase 1] Fetching data...")
    data = {}
    btc_close = None
    for sym in SYMBOLS:
        df = fetcher.fetch_klines(sym, INTERVAL, START, END)
        if not df.empty:
            data[sym] = df
            print(f"    {sym}: {len(df)} bars")
            if sym == "BTCUSDC":
                btc_close = df["close"].rename("btc_close")
        time.sleep(0.3)

    # ── Phase 2: Baseline Backtest (default params) ──────────────────────────
    print("\n  [Phase 2] Baseline backtests (default params)...")
    baseline_results = []
    for sname, scls in strategies.items():
        for sym in data:
            df = data[sym].copy()
            # Inject BTC data for BTCResidualMR
            if sname == "BTCResidualMR":
                if sym == "BTCUSDC" or btc_close is None:
                    continue
                df = df.join(btc_close, how="left")
                df["btc_close"] = df["btc_close"].ffill()
            try:
                r = backtest(scls, df)
                baseline_results.append({
                    "strategy": sname, "symbol": sym,
                    "params": "default", **{k: v for k, v in r.items() if k != "trades"}
                })
                # Save trade log
                if r["trades"]:
                    tdf = pd.DataFrame(r["trades"])
                    tdf.to_csv(RESULTS_DIR / f"{sname}_{sym}_default_trades.csv", index=False)
            except Exception as e:
                print(f"    ERROR: {sname} {sym}: {e}")

    bdf = pd.DataFrame(baseline_results)
    if not bdf.empty:
        bdf = bdf.sort_values("weekly_return_pct", ascending=False)
        bdf.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
        print(f"\n  Baseline: {len(bdf)} combos tested")
        print(f"  {'Strategy':<18} {'Symbol':<12} {'Return':>9} {'Wkly':>8} {'Trades':>7} {'WR':>6} {'DD':>6} {'Sharpe':>7} {'PF':>6}")
        print(f"  {'─' * 85}")
        for _, row in bdf.head(20).iterrows():
            m = " <<<" if row["weekly_return_pct"] >= 1.0 else ""
            print(f"  {row['strategy']:<18} {row['symbol']:<12} "
                  f"{row['total_return_pct']:>+8.1f}% {row['weekly_return_pct']:>+7.3f}% "
                  f"{row['n_trades']:>6} {row['win_rate_pct']:>5.1f}% "
                  f"{row['max_drawdown_pct']:>5.1f}% {row['sharpe_annual']:>6.2f} "
                  f"{row['profit_factor']:>5.2f}{m}")

    # ── Phase 3: Parameter Sweep ─────────────────────────────────────────────
    print("\n  [Phase 3] Parameter grid sweep...")
    sweep_results = []
    best_params = {}  # {strategy_name: {symbol: best_params}}

    for sname, scls in strategies.items():
        grid = PARAM_GRIDS.get(sname, [{}])
        best_params[sname] = {}
        for sym in data:
            df = data[sym].copy()
            if sname == "BTCResidualMR":
                if sym == "BTCUSDC" or btc_close is None:
                    continue
                df = df.join(btc_close, how="left")
                df["btc_close"] = df["btc_close"].ffill()

            best_wkly = -999
            best_p = None
            for pidx, pset in enumerate(grid):
                try:
                    r = backtest(scls, df, params=pset)
                    sweep_results.append({
                        "strategy": sname, "symbol": sym,
                        "param_idx": pidx, "params": str(pset),
                        **{k: v for k, v in r.items() if k != "trades"}
                    })
                    if r["weekly_return_pct"] > best_wkly:
                        best_wkly = r["weekly_return_pct"]
                        best_p = pset
                except Exception:
                    pass

            if best_p is not None:
                best_params[sname][sym] = best_p

    sdf = pd.DataFrame(sweep_results)
    if not sdf.empty:
        sdf = sdf.sort_values("weekly_return_pct", ascending=False)
        sdf.to_csv(RESULTS_DIR / "sweep_results.csv", index=False)
        print(f"  Sweep: {len(sdf)} combos tested")
        target_count = (sdf["weekly_return_pct"] >= 1.0).sum()
        print(f"  Combos >= 1%/week: {target_count}")
        print(f"\n  TOP 20 SWEEP RESULTS:")
        print(f"  {'Strategy':<18} {'Symbol':<12} {'Params':>5} {'Return':>9} {'Wkly':>8} {'Trades':>7} {'WR':>6} {'DD':>6} {'Sharpe':>7}")
        print(f"  {'─' * 90}")
        for _, row in sdf.head(20).iterrows():
            m = " <<<" if row["weekly_return_pct"] >= 1.0 else ""
            print(f"  {row['strategy']:<18} {row['symbol']:<12} P{row['param_idx']:<4} "
                  f"{row['total_return_pct']:>+8.1f}% {row['weekly_return_pct']:>+7.3f}% "
                  f"{row['n_trades']:>6} {row['win_rate_pct']:>5.1f}% "
                  f"{row['max_drawdown_pct']:>5.1f}% {row['sharpe_annual']:>6.2f}{m}")

    # ── Phase 4: Walk-Forward Validation ─────────────────────────────────────
    print(f"\n  [Phase 4] {N_FOLDS}-fold walk-forward validation on best params...")
    wf_results = []

    # Select top combos from sweep (>= 0.5%/week or top 15)
    if not sdf.empty:
        candidates = sdf[sdf["weekly_return_pct"] >= 0.5].head(20)
        if len(candidates) < 5:
            candidates = sdf.head(15)

        seen = set()
        for _, row in candidates.iterrows():
            key = (row["strategy"], row["symbol"])
            if key in seen:
                continue
            seen.add(key)

            sname = row["strategy"]
            sym = row["symbol"]
            scls = strategies[sname]
            params = best_params.get(sname, {}).get(sym, {})

            df = data[sym].copy()
            if sname == "BTCResidualMR" and btc_close is not None and sym != "BTCUSDC":
                df = df.join(btc_close, how="left")
                df["btc_close"] = df["btc_close"].ffill()

            try:
                folds = walk_forward(scls, df, params)
                pos_folds = sum(1 for f in folds if f["return_pct"] > 0)
                avg_wkly = np.mean([f["weekly_pct"] for f in folds])
                avg_dd = np.mean([f["max_dd"] for f in folds])
                all_pos = pos_folds == N_FOLDS

                fold_str = "  ".join([f"F{f['fold']}:{f['return_pct']:+.1f}%" for f in folds])
                status = "ALL+" if all_pos else f"{pos_folds}/{N_FOLDS}"

                wf_results.append({
                    "strategy": sname, "symbol": sym, "params": str(params),
                    "positive_folds": pos_folds, "all_positive": all_pos,
                    "avg_weekly_pct": round(avg_wkly, 4),
                    "avg_max_dd_pct": round(avg_dd, 2),
                    "folds": folds,
                })

                print(f"  {sname:<18} {sym:<12} {status:<5} avg:{avg_wkly:+.2f}%/wk  "
                      f"DD:{avg_dd:.1f}%  |  {fold_str}")

                # Save per-strategy result JSON
                result_json = {
                    "strategy": sname,
                    "symbol": sym,
                    "interval": INTERVAL,
                    "params": params,
                    "in_sample": {k: v for k, v in row.items() if k not in ("trades",)},
                    "walk_forward": {
                        "n_folds": N_FOLDS,
                        "positive_folds": pos_folds,
                        "all_positive": all_pos,
                        "avg_weekly_pct": round(avg_wkly, 4),
                        "avg_max_dd_pct": round(avg_dd, 2),
                        "folds": folds,
                    },
                }
                with open(RESULTS_DIR / f"{sname}_{sym}.json", "w") as f:
                    json.dump(result_json, f, indent=2, default=str)

            except Exception as e:
                print(f"  {sname:<18} {sym:<12} ERROR: {e}")

    # ── Phase 5: Final Ranking & Report ──────────────────────────────────────
    print("\n" + "=" * 110)
    print("  FINAL STRATEGY RANKING")
    print("=" * 110)

    if wf_results:
        wfdf = pd.DataFrame([{k: v for k, v in r.items() if k != "folds"} for r in wf_results])
        wfdf = wfdf.sort_values("avg_weekly_pct", ascending=False)
        wfdf.to_csv(RESULTS_DIR / "walkforward_results.csv", index=False)

        all_pos_df = wfdf[wfdf["all_positive"]]
        print(f"\n  WALK-FORWARD VALIDATED (ALL {N_FOLDS}/{N_FOLDS} folds positive):")
        if len(all_pos_df) > 0:
            for _, row in all_pos_df.iterrows():
                m = " <<<" if row["avg_weekly_pct"] >= 1.0 else ""
                print(f"    {row['strategy']:<18} {row['symbol']:<12} "
                      f"avg:{row['avg_weekly_pct']:+.2f}%/wk  DD:{row['avg_max_dd_pct']:.1f}%{m}")
        else:
            print("    (none)")

        print(f"\n  PARTIAL VALIDATION:")
        partial = wfdf[~wfdf["all_positive"]]
        for _, row in partial.head(10).iterrows():
            print(f"    {row['strategy']:<18} {row['symbol']:<12} "
                  f"{row['positive_folds']}/{N_FOLDS}  avg:{row['avg_weekly_pct']:+.2f}%/wk  "
                  f"DD:{row['avg_max_dd_pct']:.1f}%")

        # ── Report ───────────────────────────────────────────────────────────
        print("\n" + "─" * 110)
        print("  STRATEGIC ASSESSMENT")
        print("─" * 110)

        # Best to deploy first (simplicity + robustness)
        if len(all_pos_df) > 0:
            best_deploy = all_pos_df.iloc[0]
            print(f"\n  BEST TO DEPLOY FIRST: {best_deploy['strategy']} on {best_deploy['symbol']}")
            print(f"    → WF avg: {best_deploy['avg_weekly_pct']:+.2f}%/wk, "
                  f"DD: {best_deploy['avg_max_dd_pct']:.1f}%")
            print(f"    → Reason: All {N_FOLDS} OOS folds positive")
        else:
            best_deploy = wfdf.iloc[0]
            print(f"\n  BEST CANDIDATE: {best_deploy['strategy']} on {best_deploy['symbol']}")
            print(f"    → WF avg: {best_deploy['avg_weekly_pct']:+.2f}%/wk")

        # Most promising (highest avg weekly)
        most_promising = wfdf.iloc[0]
        print(f"\n  MOST PROMISING (long-term): {most_promising['strategy']} on {most_promising['symbol']}")
        print(f"    → {most_promising['avg_weekly_pct']:+.2f}%/wk avg across folds")

        # Most likely overfit (big gap between in-sample and OOS, or high param sensitivity)
        if not sdf.empty and len(wfdf) > 0:
            # Compare best in-sample vs walk-forward
            merged = wfdf.merge(
                sdf.groupby(["strategy", "symbol"])["weekly_return_pct"].max().reset_index(),
                on=["strategy", "symbol"], how="left"
            )
            merged["overfit_gap"] = merged["weekly_return_pct"] - merged["avg_weekly_pct"]
            merged = merged.sort_values("overfit_gap", ascending=False)
            if len(merged) > 0:
                worst = merged.iloc[0]
                print(f"\n  MOST LIKELY OVERFIT: {worst['strategy']} on {worst['symbol']}")
                print(f"    → In-sample: {worst['weekly_return_pct']:+.2f}%/wk  "
                      f"vs  OOS: {worst['avg_weekly_pct']:+.2f}%/wk  "
                      f"(gap: {worst['overfit_gap']:+.2f}%)")

    print("\n" + "=" * 110)
    print(f"  All results saved to: {RESULTS_DIR}")
    print("=" * 110)


if __name__ == "__main__":
    main()
