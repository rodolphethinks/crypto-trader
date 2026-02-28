"""
Independent verification backtest of the top validated configs.
Tests the exact same strategies with aggressive sizing on recent data
to confirm the walk-forward results are reproducible.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.fetcher import DataFetcher
from strategies.v6_aggressive import (
    CrossPairLeaderStrategy,
    MomentumAcceleratorStrategy,
    MultiEdgeCompositeStrategy,
)
from strategies.base import Signal
from config.settings import LOG_DIR

fetcher = DataFetcher()

# ── Configs to verify ──────────────────────────────────────────────────────
CONFIGS = [
    {"name": "CrossPair AVAX 1d R10 P100", "cls": CrossPairLeaderStrategy,
     "symbol": "AVAXUSDC", "interval": "1d", "risk_pct": 10.0, "pos_pct": 100.0,
     "needs_btc": True},
    {"name": "CrossPair AVAX 1d R10 P75", "cls": CrossPairLeaderStrategy,
     "symbol": "AVAXUSDC", "interval": "1d", "risk_pct": 10.0, "pos_pct": 75.0,
     "needs_btc": True},
    {"name": "MomAccel AVAX 4h R5 P100", "cls": MomentumAcceleratorStrategy,
     "symbol": "AVAXUSDC", "interval": "4h", "risk_pct": 5.0, "pos_pct": 100.0,
     "needs_btc": False},
    {"name": "MultiEdge DOGE 4h R5 P100", "cls": MultiEdgeCompositeStrategy,
     "symbol": "DOGEUSDC", "interval": "4h", "risk_pct": 5.0, "pos_pct": 100.0,
     "needs_btc": False},
]


def run_manual_backtest(cfg, df, initial_capital=10000):
    """
    Simple backtest engine that matches the aggressive sizing logic.
    Process bar-by-bar, tracking one position at a time.
    """
    strategy = cfg["cls"]()
    sig_df = strategy.generate_signals(df.copy())
    
    capital = initial_capital
    peak = capital
    position = None  # {side, entry, qty, sl, tp}
    trades = []
    equity_curve = []
    max_dd = 0
    
    for i in range(len(sig_df)):
        close = df["close"].iloc[i]
        low = df["low"].iloc[i]
        high = df["high"].iloc[i]
        sig = sig_df["signal"].iloc[i]
        
        # Check existing position
        if position:
            exit_price = None
            reason = None
            
            if position["side"] == "BUY":
                if position["sl"] > 0 and low <= position["sl"]:
                    exit_price = position["sl"]
                    reason = "SL"
                elif position["tp"] > 0 and high >= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP"
                elif sig == Signal.SELL:
                    exit_price = close
                    reason = "REV"
            else:
                if position["sl"] > 0 and high >= position["sl"]:
                    exit_price = position["sl"]
                    reason = "SL"
                elif position["tp"] > 0 and low <= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP"
                elif sig == Signal.BUY:
                    exit_price = close
                    reason = "REV"
            
            if exit_price:
                if position["side"] == "BUY":
                    pnl = (exit_price - position["entry"]) * position["qty"]
                else:
                    pnl = (position["entry"] - exit_price) * position["qty"]
                
                capital += pnl
                if capital > peak:
                    peak = capital
                dd = (peak - capital) / peak * 100
                if dd > max_dd:
                    max_dd = dd
                
                trades.append({
                    "entry": position["entry"],
                    "exit": exit_price,
                    "side": position["side"],
                    "pnl": pnl,
                    "pnl_pct": pnl / position["capital_at_entry"] * 100,
                    "reason": reason,
                    "time": str(df.index[i]),
                })
                position = None
        
        # Check for new signal (no position)
        if position is None and sig in (Signal.BUY, Signal.SELL):
            sl = sig_df["stop_loss"].iloc[i] if "stop_loss" in sig_df else 0
            tp = sig_df["take_profit"].iloc[i] if "take_profit" in sig_df else 0
            if pd.isna(sl): sl = 0
            if pd.isna(tp): tp = 0
            
            side = "BUY" if sig == Signal.BUY else "SELL"
            entry = close
            
            # Aggressive position sizing
            if sl > 0:
                risk_amount = capital * (cfg["risk_pct"] / 100)
                price_risk = abs(entry - sl)
                if price_risk > 0:
                    qty = risk_amount / price_risk
                else:
                    qty = 0
                max_qty = (capital * cfg["pos_pct"] / 100) / entry
                qty = min(qty, max_qty)
            else:
                qty = (capital * cfg["pos_pct"] / 100) / entry
            
            if qty > 0:
                # Drawdown check
                dd = (peak - capital) / peak * 100 if peak > 0 else 0
                if dd < 30:  # 30% max drawdown
                    position = {
                        "side": side,
                        "entry": entry,
                        "qty": qty,
                        "sl": sl,
                        "tp": tp,
                        "capital_at_entry": capital,
                    }
        
        equity_curve.append(capital)
    
    # Close any remaining position at last bar
    if position:
        close = df["close"].iloc[-1]
        if position["side"] == "BUY":
            pnl = (close - position["entry"]) * position["qty"]
        else:
            pnl = (position["entry"] - close) * position["qty"]
        capital += pnl
        trades.append({
            "entry": position["entry"],
            "exit": close,
            "side": position["side"],
            "pnl": pnl,
            "pnl_pct": pnl / position["capital_at_entry"] * 100,
            "reason": "END",
            "time": str(df.index[-1]),
        })
    
    total_return = (capital - initial_capital) / initial_capital * 100
    n_weeks = len(df) * {"1d": 1/7, "4h": 1/42, "1h": 1/168}.get(cfg["interval"], 1/7)
    weekly_return = total_return / max(n_weeks, 1)
    
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] <= 0)
    win_rate = wins / len(trades) * 100 if trades else 0
    
    return {
        "final_capital": capital,
        "total_return": total_return,
        "weekly_return": weekly_return,
        "n_weeks": n_weeks,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "equity_curve": equity_curve,
        "trade_details": trades,
    }


def main():
    print("\n" + "=" * 90)
    print("  INDEPENDENT VERIFICATION BACKTEST")
    print("  Testing validated configs on full available data")
    print("=" * 90)
    
    # Fetch BTC data once for cross-pair strategies
    btc_data = {}
    for interval in ["1d"]:
        df_btc = fetcher.fetch_klines("BTCUSDC", interval, "2024-01-01", "2026-03-01")
        if not df_btc.empty:
            btc_data[interval] = df_btc
            print(f"\n  BTC {interval}: {len(df_btc)} bars fetched")
    
    results = []
    
    for cfg in CONFIGS:
        print(f"\n  Testing: {cfg['name']}")
        print(f"  {'─' * 60}")
        
        # Fetch data
        df = fetcher.fetch_klines(cfg["symbol"], cfg["interval"], "2024-01-01", "2026-03-01")
        if df.empty:
            print(f"    NO DATA")
            continue
        print(f"    Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        # Inject BTC data if needed
        if cfg["needs_btc"] and cfg["interval"] in btc_data:
            btc_close = btc_data[cfg["interval"]]["close"].rename("btc_close")
            df = df.join(btc_close, how="left")
            df["btc_close"] = df["btc_close"].ffill()
        
        # Run backtest
        result = run_manual_backtest(cfg, df)
        result["name"] = cfg["name"]
        results.append(result)
        
        print(f"    Capital: $10,000 → ${result['final_capital']:,.2f}")
        print(f"    Return: {result['total_return']:+.2f}% over {result['n_weeks']:.1f} weeks")
        print(f"    Weekly: {result['weekly_return']:+.3f}%")
        print(f"    Trades: {result['trades']} (W:{result['wins']} L:{result['losses']} WR:{result['win_rate']:.1f}%)")
        print(f"    Max DD: {result['max_drawdown']:.2f}%")
        
        if result['weekly_return'] >= 1.0:
            print(f"    >>> MEETS 1%/WEEK TARGET <<<")
        
        # Print individual trades
        if result["trade_details"]:
            print(f"\n    Trade Log:")
            for t in result["trade_details"]:
                icon = "W" if t["pnl"] > 0 else "L"
                print(f"      [{icon}] {t['side']} @ ${t['entry']:.4f} → ${t['exit']:.4f} "
                      f"PnL: ${t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%) [{t['reason']}] "
                      f"{t['time']}")
    
    # Summary
    print("\n" + "=" * 90)
    print("  VERIFICATION SUMMARY")
    print("=" * 90)
    print(f"  {'Config':<38} {'Return':>8} {'Weekly':>8} {'Trades':>7} {'WR':>6} {'MaxDD':>6}")
    print(f"  {'─' * 75}")
    
    for r in results:
        marker = "<<<" if r["weekly_return"] >= 1.0 else ""
        print(f"  {r['name']:<38} {r['total_return']:>+7.2f}% {r['weekly_return']:>+7.3f}% "
              f"{r['trades']:>6}  {r['win_rate']:>5.1f}% {r['max_drawdown']:>5.1f}% {marker}")
    
    print("=" * 90)
    
    # Save detailed results
    rows = []
    for r in results:
        rows.append({
            "Config": r["name"],
            "Final_Capital": r["final_capital"],
            "Total_Return_Pct": r["total_return"],
            "Weekly_Return_Pct": r["weekly_return"],
            "Weeks": r["n_weeks"],
            "Trades": r["trades"],
            "Wins": r["wins"],
            "Losses": r["losses"],
            "Win_Rate": r["win_rate"],
            "Max_Drawdown": r["max_drawdown"],
        })
    pd.DataFrame(rows).to_csv(LOG_DIR / "verification_v6.csv", index=False)
    print(f"\n  Results saved to logs/verification_v6.csv")


if __name__ == "__main__":
    main()
