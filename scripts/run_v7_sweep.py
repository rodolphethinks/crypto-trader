"""
Wide 4h sweep for V7 diverse strategies — Test all V7 strategies on all pairs
with aggressive sizing (same pipeline as V6 wide sweep).
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd, numpy as np
from data.fetcher import DataFetcher
from strategies.v7_diverse import get_v7_strategies
from strategies.base import Signal
from config.settings import LOG_DIR

fetcher = DataFetcher()

V7_STRATS = get_v7_strategies()
STRATEGIES = [(name, cls) for name, cls in V7_STRATS.items()]

PAIRS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "BNBUSDC",
    "ADAUSDC", "DOGEUSDC", "AVAXUSDC", "NEARUSDC",
]

RISK_PCT = 5.0
POS_PCT = 100.0
MAX_DD = 35.0
INITIAL_CAPITAL = 10000.0


def backtest_aggressive(strategy_cls, df, risk_pct, pos_pct, max_dd):
    strat = strategy_cls()
    sig_df = strat.generate_signals(df.copy())
    
    cap = INITIAL_CAPITAL; peak = cap; pos = None
    trades = 0; wins = 0; max_drawdown = 0
    weekly_returns = []
    week_start_cap = cap
    bars_per_week = 42
    bar_count = 0
    
    for i in range(len(sig_df)):
        c = df["close"].iloc[i]
        lo = df["low"].iloc[i]
        hi = df["high"].iloc[i]
        sig = sig_df["signal"].iloc[i]
        sl_val = sig_df["stop_loss"].iloc[i] if "stop_loss" in sig_df else 0
        tp_val = sig_df["take_profit"].iloc[i] if "take_profit" in sig_df else 0
        if pd.isna(sl_val): sl_val = 0
        if pd.isna(tp_val): tp_val = 0
        
        bar_count += 1
        if bar_count >= bars_per_week:
            wr = (cap - week_start_cap) / week_start_cap * 100 if week_start_cap > 0 else 0
            weekly_returns.append(wr)
            week_start_cap = cap
            bar_count = 0
        
        if pos:
            ex, reason = None, None
            if pos["side"] == "BUY":
                if pos["sl"] > 0 and lo <= pos["sl"]: ex, reason = pos["sl"], "SL"
                elif pos["tp"] > 0 and hi >= pos["tp"]: ex, reason = pos["tp"], "TP"
                elif sig == Signal.SELL: ex, reason = c, "REV"
            else:
                if pos["sl"] > 0 and hi >= pos["sl"]: ex, reason = pos["sl"], "SL"
                elif pos["tp"] > 0 and lo <= pos["tp"]: ex, reason = pos["tp"], "TP"
                elif sig == Signal.BUY: ex, reason = c, "REV"
            if ex:
                pnl = (ex - pos["e"]) * pos["q"] if pos["side"] == "BUY" else (pos["e"] - ex) * pos["q"]
                cap += pnl; trades += 1
                if pnl > 0: wins += 1
                if cap > peak: peak = cap
                dd = (peak - cap) / peak * 100
                if dd > max_drawdown: max_drawdown = dd
                pos = None
        
        if pos is None and sig in (Signal.BUY, Signal.SELL):
            side = "BUY" if sig == Signal.BUY else "SELL"
            if sl_val > 0:
                risk_amt = cap * (risk_pct / 100)
                pr = abs(c - sl_val)
                qty = risk_amt / pr if pr > 0 else 0
                max_q = (cap * pos_pct / 100) / c
                qty = min(qty, max_q)
            else:
                qty = (cap * pos_pct / 100) / c
            if qty > 0:
                dd = (peak - cap) / peak * 100 if peak > 0 else 0
                if dd < max_dd:
                    pos = {"side": side, "e": c, "q": qty, "sl": sl_val, "tp": tp_val}
    
    if pos:
        c = df["close"].iloc[-1]
        pnl = (c - pos["e"]) * pos["q"] if pos["side"] == "BUY" else (pos["e"] - c) * pos["q"]
        cap += pnl; trades += 1

    total_ret = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_weeks = len(df) / 42
    weekly_avg = total_ret / max(n_weeks, 1)
    wr_pct = wins / trades * 100 if trades > 0 else 0
    
    if len(weekly_returns) > 1:
        w_mean = np.mean(weekly_returns)
        w_std = np.std(weekly_returns)
        sharpe = w_mean / w_std * np.sqrt(52) if w_std > 0 else 0
    else:
        sharpe = 0
    
    return {
        "final": cap, "return": total_ret, "weekly": weekly_avg,
        "trades": trades, "win_rate": wr_pct, "max_dd": max_drawdown,
        "sharpe": sharpe, "n_weeks": n_weeks,
    }


def main():
    print("\n" + "=" * 100)
    print("  V7 WIDE 4H SWEEP — All V7 Strategies × All Pairs × Aggressive Sizing")
    print(f"  Risk: {RISK_PCT}% per trade | Position: {POS_PCT}% | Max DD: {MAX_DD}%")
    print(f"  Strategies: {', '.join(V7_STRATS.keys())}")
    print("=" * 100)
    
    # Pre-fetch all pair data
    print("\n  Fetching data for all pairs...")
    data_cache = {}
    for pair in PAIRS:
        df = fetcher.fetch_klines(pair, "4h", "2024-01-01", "2026-03-01")
        if not df.empty:
            data_cache[pair] = df
            print(f"    {pair}: {len(df)} bars")
        time.sleep(0.3)
    
    results = []
    total = len(STRATEGIES) * len(data_cache)
    done = 0
    
    print(f"\n  Running {total} combinations...\n")
    
    for strat_name, strat_cls in STRATEGIES:
        for pair in data_cache:
            done += 1
            df = data_cache[pair].copy()
            
            try:
                r = backtest_aggressive(strat_cls, df, RISK_PCT, POS_PCT, MAX_DD)
                results.append({
                    "Strategy": strat_name, "Pair": pair,
                    "Return": r["return"], "Weekly": r["weekly"],
                    "Trades": r["trades"], "WinRate": r["win_rate"],
                    "MaxDD": r["max_dd"], "Sharpe": r["sharpe"], "Final": r["final"],
                })
                
                marker = " <<<" if r["weekly"] >= 1.0 else ""
                if done % 10 == 0 or r["weekly"] >= 1.0:
                    print(f"  [{done:3d}/{total}] {strat_name:<14} {pair:<12} "
                          f"Ret: {r['return']:+8.1f}%  Wkly: {r['weekly']:+6.3f}%  "
                          f"Trades: {r['trades']:3d}  WR: {r['win_rate']:5.1f}%  "
                          f"DD: {r['max_dd']:5.1f}%{marker}")
            except Exception as e:
                print(f"  [{done:3d}/{total}] {strat_name:<14} {pair:<12} ERROR: {e}")
    
    df_results = pd.DataFrame(results).sort_values("Weekly", ascending=False)
    df_results.to_csv(LOG_DIR / "v7_wide_4h_sweep.csv", index=False)
    
    print("\n" + "=" * 100)
    print("  TOP 30 RESULTS (sorted by Weekly Return)")
    print("=" * 100)
    print(f"  {'Strategy':<14} {'Pair':<12} {'Return':>9} {'Weekly':>8} {'Trades':>7} {'WR':>6} {'MaxDD':>6} {'Sharpe':>7}")
    print(f"  {'─' * 80}")
    
    target_count = 0
    for _, row in df_results.head(30).iterrows():
        marker = " <<<" if row["Weekly"] >= 1.0 else ""
        if row["Weekly"] >= 1.0: target_count += 1
        print(f"  {row['Strategy']:<14} {row['Pair']:<12} {row['Return']:>+8.1f}% "
              f"{row['Weekly']:>+7.3f}% {row['Trades']:>6.0f} {row['WinRate']:>5.1f}% "
              f"{row['MaxDD']:>5.1f}% {row['Sharpe']:>6.2f}{marker}")
    
    print(f"\n  TOTAL CONFIGS MEETING 1%/WEEK TARGET: {target_count}")
    print(f"  Results saved to logs/v7_wide_4h_sweep.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
