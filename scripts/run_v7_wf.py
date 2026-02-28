"""
5-Fold Walk-Forward Validation for V7 wide sweep winners.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd, numpy as np
from data.fetcher import DataFetcher
from strategies.v7_diverse import (
    OrderFlowMomentumStrategy,
    TrendPulseStrategy,
    VolatilityCaptureStrategy,
    MeanReversionRSIStrategy,
    AdaptiveChannelStrategy,
    MomentumSwitchStrategy,
)
from strategies.base import Signal
from config.settings import LOG_DIR

fetcher = DataFetcher()

# Top 8 from V7 wide sweep
WINNERS = [
    ("AdaptChan",  "NEARUSDC",  AdaptiveChannelStrategy),
    ("MomSwitch",  "AVAXUSDC",  MomentumSwitchStrategy),
    ("VolCapture", "AVAXUSDC",  VolatilityCaptureStrategy),
    ("AdaptChan",  "SOLUSDC",   AdaptiveChannelStrategy),
    ("MomSwitch",  "NEARUSDC",  MomentumSwitchStrategy),
    ("TrendPulse", "ADAUSDC",   TrendPulseStrategy),
    ("VolCapture", "NEARUSDC",  VolatilityCaptureStrategy),
    ("VolCapture", "SOLUSDC",   VolatilityCaptureStrategy),
]

RISK_PCT = 5.0
POS_PCT = 100.0
MAX_DD = 35.0
INITIAL_CAPITAL = 10000.0
N_FOLDS = 5


def backtest(strat_cls, df, risk_pct=5.0, pos_pct=100.0, max_dd=35.0):
    strat = strat_cls()
    sig_df = strat.generate_signals(df.copy())
    
    cap = INITIAL_CAPITAL; peak = cap; pos = None
    trades = 0; wins = 0; max_drawdown = 0
    
    for i in range(len(sig_df)):
        c = df["close"].iloc[i]
        lo = df["low"].iloc[i]
        hi = df["high"].iloc[i]
        sig = sig_df["signal"].iloc[i]
        sl_val = sig_df["stop_loss"].iloc[i] if "stop_loss" in sig_df else 0
        tp_val = sig_df["take_profit"].iloc[i] if "take_profit" in sig_df else 0
        if pd.isna(sl_val): sl_val = 0
        if pd.isna(tp_val): tp_val = 0
        
        if pos:
            ex = None
            if pos["side"] == "BUY":
                if pos["sl"] > 0 and lo <= pos["sl"]: ex = pos["sl"]
                elif pos["tp"] > 0 and hi >= pos["tp"]: ex = pos["tp"]
                elif sig == Signal.SELL: ex = c
            else:
                if pos["sl"] > 0 and hi >= pos["sl"]: ex = pos["sl"]
                elif pos["tp"] > 0 and lo <= pos["tp"]: ex = pos["tp"]
                elif sig == Signal.BUY: ex = c
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
    weekly = total_ret / max(n_weeks, 1)
    wr = wins / trades * 100 if trades > 0 else 0
    return total_ret, weekly, trades, wr, max_drawdown


def main():
    print("\n" + "=" * 100)
    print("  V7 5-FOLD WALK-FORWARD VALIDATION — Top 8 Configs")
    print(f"  Risk: {RISK_PCT}% | Position: {POS_PCT}% | Max DD: {MAX_DD}%")
    print("=" * 100)
    
    # Fetch data
    pairs_needed = list(set(p for _, p, _ in WINNERS))
    data_cache = {}
    for pair in pairs_needed:
        df = fetcher.fetch_klines(pair, "4h", "2024-01-01", "2026-03-01")
        if not df.empty:
            data_cache[pair] = df
            print(f"  {pair}: {len(df)} bars")
        time.sleep(0.3)
    
    print(f"\n  Running {N_FOLDS}-fold walk-forward...\n")
    
    results = []
    
    for strat_name, pair, strat_cls in WINNERS:
        if pair not in data_cache:
            print(f"  {strat_name} {pair}: NO DATA")
            continue
        
        df_full = data_cache[pair]
        n = len(df_full)
        fold_size = n // N_FOLDS
        
        fold_results = []
        
        for fold in range(N_FOLDS):
            # OOS = fold block
            oos_start = fold * fold_size
            oos_end = min(oos_start + fold_size, n)
            
            df_oos = df_full.iloc[oos_start:oos_end].copy().reset_index(drop=True)
            
            ret, wkly, trades, wr, dd = backtest(strat_cls, df_oos, RISK_PCT, POS_PCT, MAX_DD)
            fold_results.append({
                "fold": fold + 1, "ret": ret, "weekly": wkly,
                "trades": trades, "wr": wr, "dd": dd
            })
        
        positive_folds = sum(1 for f in fold_results if f["ret"] > 0)
        avg_weekly = np.mean([f["weekly"] for f in fold_results])
        avg_dd = np.mean([f["dd"] for f in fold_results])
        all_positive = positive_folds == N_FOLDS
        meets_target = all(f["weekly"] >= 1.0 for f in fold_results)
        
        status = "ALL+" if all_positive else f"{positive_folds}/{N_FOLDS}"
        target = "1%+/wk" if meets_target else "mixed"
        
        fold_strs = [f"F{f['fold']}:{f['ret']:+.1f}%" for f in fold_results]
        
        print(f"  {strat_name:<14} {pair:<12} {status:<5} avg:{avg_weekly:+.2f}%/wk  "
              f"DD:{avg_dd:.1f}%  {target}  |  {', '.join(fold_strs)}")
        
        results.append({
            "Strategy": strat_name, "Pair": pair,
            "PositiveFolds": positive_folds, "AllPositive": all_positive,
            "AvgWeekly": avg_weekly, "AvgDD": avg_dd,
            "MeetsTarget": meets_target,
            "Folds": fold_results,
        })
    
    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    
    all_positive_configs = [r for r in results if r["AllPositive"]]
    target_met = [r for r in results if r["MeetsTarget"]]
    
    print(f"  ALL 5/5 folds positive: {len(all_positive_configs)}/{len(results)}")
    print(f"  ALL folds >= 1%/week:   {len(target_met)}/{len(results)}")
    
    if all_positive_configs:
        print(f"\n  VALIDATED CONFIGS (all folds positive):")
        for r in sorted(all_positive_configs, key=lambda x: -x["AvgWeekly"]):
            print(f"    {r['Strategy']:<14} {r['Pair']:<12} avg: {r['AvgWeekly']:+.2f}%/wk  DD: {r['AvgDD']:.1f}%")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
