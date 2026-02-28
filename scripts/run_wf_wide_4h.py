"""
Walk-forward validation for the wide 4h sweep winners.
Tests 7 configs that exceeded 1%/week on full 2-year data.
Uses 5-fold time-series cross-validation.
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd, numpy as np
from data.fetcher import DataFetcher
from strategies.v6_aggressive import (
    MomentumAcceleratorStrategy,
    RegimeMomentumV2Strategy,
    MultiEdgeCompositeStrategy,
    CrossPairLeaderStrategy,
)
from strategies.base import Signal
from config.settings import LOG_DIR

fetcher = DataFetcher()

CONFIGS = [
    {"name": "MultiEdge XRP 4h", "cls": MultiEdgeCompositeStrategy,
     "symbol": "XRPUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": False},
    {"name": "MomAccel AVAX 4h", "cls": MomentumAcceleratorStrategy,
     "symbol": "AVAXUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": False},
    {"name": "RegimeMomV2 XRP 4h", "cls": RegimeMomentumV2Strategy,
     "symbol": "XRPUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": False},
    {"name": "MultiEdge DOGE 4h", "cls": MultiEdgeCompositeStrategy,
     "symbol": "DOGEUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": False},
    {"name": "CrossPair BTC 4h", "cls": CrossPairLeaderStrategy,
     "symbol": "BTCUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": False},
    {"name": "MultiEdge SOL 4h", "cls": MultiEdgeCompositeStrategy,
     "symbol": "SOLUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": False},
    {"name": "CrossPair AVAX 4h", "cls": CrossPairLeaderStrategy,
     "symbol": "AVAXUSDC", "risk": 5.0, "pos": 100.0, "needs_btc": True},
]

N_FOLDS = 5
INITIAL_CAPITAL = 10000.0
MAX_DD = 35.0


def backtest_fold(strategy_cls, df, risk_pct, pos_pct, max_dd):
    """Backtest one fold, return metrics."""
    strat = strategy_cls()
    sig_df = strat.generate_signals(df.copy())
    
    cap = INITIAL_CAPITAL; peak = cap; pos = None
    trades = 0; wins = 0; max_drawdown = 0
    
    for i in range(len(sig_df)):
        c = df["close"].iloc[i]
        lo = df["low"].iloc[i]
        hi = df["high"].iloc[i]
        sig = sig_df["signal"].iloc[i]
        sl = sig_df["stop_loss"].iloc[i] if "stop_loss" in sig_df else 0
        tp = sig_df["take_profit"].iloc[i] if "take_profit" in sig_df else 0
        if pd.isna(sl): sl = 0
        if pd.isna(tp): tp = 0
        
        if pos:
            ex, reason = None, None
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
            if sl > 0:
                risk_amt = cap * (risk_pct / 100)
                pr = abs(c - sl)
                qty = risk_amt / pr if pr > 0 else 0
                max_q = (cap * pos_pct / 100) / c
                qty = min(qty, max_q)
            else:
                qty = (cap * pos_pct / 100) / c
            if qty > 0:
                dd = (peak - cap) / peak * 100 if peak > 0 else 0
                if dd < max_dd:
                    pos = {"side": side, "e": c, "q": qty, "sl": sl, "tp": tp}
    
    if pos:
        c = df["close"].iloc[-1]
        pnl = (c - pos["e"]) * pos["q"] if pos["side"] == "BUY" else (pos["e"] - c) * pos["q"]
        cap += pnl; trades += 1
    
    ret = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_weeks = len(df) / 42
    weekly = ret / max(n_weeks, 1)
    
    return ret, weekly, trades, max_drawdown


def main():
    print("\n" + "=" * 100)
    print("  WALK-FORWARD VALIDATION — Top 7 Wide Sweep Winners")
    print(f"  {N_FOLDS}-fold time-series cross-validation | 5% risk | 100% position")
    print("=" * 100)
    
    # Fetch BTC data for injection
    btc_4h = fetcher.fetch_klines("BTCUSDC", "4h", "2024-01-01", "2026-03-01")
    btc_close = btc_4h["close"].rename("btc_close") if not btc_4h.empty else None
    
    all_results = []
    
    for cfg in CONFIGS:
        print(f"\n  {cfg['name']}")
        df = fetcher.fetch_klines(cfg["symbol"], "4h", "2024-01-01", "2026-03-01")
        if df.empty:
            print("    NO DATA")
            continue
        
        if cfg["needs_btc"] and btc_close is not None:
            df = df.join(btc_close, how="left")
            df["btc_close"] = df["btc_close"].ffill()
        
        n = len(df)
        fold_size = n // N_FOLDS
        fold_returns = []
        fold_weeklies = []
        all_positive = True
        
        for fold in range(N_FOLDS):
            # OOS test window (no peeking into future)
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)
            
            # Need some warmup bars before test starts
            warmup = min(200, test_start)
            data_start = test_start - warmup
            
            fold_df = df.iloc[data_start:test_end].copy()
            fold_df = fold_df.reset_index(drop=True)
            
            if len(fold_df) < 100:
                fold_returns.append(0)
                fold_weeklies.append(0)
                continue
            
            ret, weekly, trades, max_dd = backtest_fold(
                cfg["cls"], fold_df, cfg["risk"], cfg["pos"], MAX_DD
            )
            fold_returns.append(ret)
            fold_weeklies.append(weekly)
            
            if ret <= 0:
                all_positive = False
        
        n_positive = sum(1 for r in fold_returns if r > 0)
        avg_return = np.mean(fold_returns)
        avg_weekly = np.mean(fold_weeklies)
        
        status = "ALL+" if all_positive else f"{n_positive}/{N_FOLDS}"
        marker = "<<< TARGET" if avg_weekly >= 1.0 else ""
        
        print(f"    [{status}]  Avg Return: {avg_return:+.2f}%  Avg Weekly: {avg_weekly:+.3f}%  "
              f"Folds: {[f'{r:+.1f}%' for r in fold_returns]}  {marker}")
        
        all_results.append({
            "Config": cfg["name"],
            "Folds_Positive": f"{n_positive}/{N_FOLDS}",
            "All_Positive": all_positive,
            "Avg_Return": avg_return,
            "Avg_Weekly": avg_weekly,
            "Fold_Returns": fold_returns,
            "Meets_Target": avg_weekly >= 1.0,
        })
    
    # Summary
    print("\n" + "=" * 100)
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 100)
    
    # Sort by avg weekly
    all_results.sort(key=lambda x: x["Avg_Weekly"], reverse=True)
    
    for r in all_results:
        marker = "<<< TARGET" if r["Meets_Target"] else ""
        ap = "ALL+" if r["All_Positive"] else r["Folds_Positive"]
        print(f"  {r['Config']:<25} [{ap}]  Weekly: {r['Avg_Weekly']:+.3f}%  "
              f"Return: {r['Avg_Return']:+.2f}%  {marker}")
    
    validated = [r for r in all_results if r["Meets_Target"] and r["Avg_Weekly"] > 0]
    print(f"\n  VALIDATED configs meeting 1%/wk OOS: {len(validated)}")
    
    # Save
    rows = []
    for r in all_results:
        rows.append({
            "Config": r["Config"],
            "Folds_Positive": r["Folds_Positive"],
            "All_Positive": r["All_Positive"],
            "Avg_Return": r["Avg_Return"],
            "Avg_Weekly": r["Avg_Weekly"],
            "Fold_0": r["Fold_Returns"][0] if len(r["Fold_Returns"]) > 0 else 0,
            "Fold_1": r["Fold_Returns"][1] if len(r["Fold_Returns"]) > 1 else 0,
            "Fold_2": r["Fold_Returns"][2] if len(r["Fold_Returns"]) > 2 else 0,
            "Fold_3": r["Fold_Returns"][3] if len(r["Fold_Returns"]) > 3 else 0,
            "Fold_4": r["Fold_Returns"][4] if len(r["Fold_Returns"]) > 4 else 0,
            "Meets_Target": r["Meets_Target"],
        })
    pd.DataFrame(rows).to_csv(LOG_DIR / "wf_wide_4h.csv", index=False)
    print(f"  Results saved to logs/wf_wide_4h.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
