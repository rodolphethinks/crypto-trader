"""
V6 Aggressive Simulation — test realistic path to 1%/week.

The main V6 pipeline showed ~1% total return. The bottleneck is:
1. MAX_POSITION_SIZE_PCT = 2% (each trade uses only ~$200 of $10k)
2. Capital fragmentation across 20 components  
3. Too few trades (5-80/year per strategy)

This script tests:
- Full capital per strategy (no fragmentation)
- Aggressive position sizing (25-75% of capital per trade)
- Best strategies individually with proper sizing
- Concentrated portfolio (top 3-5 only)
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import compute_metrics
from backtesting.runner import get_v6_strategies, get_v3_strategies, get_v4_strategies
from risk.manager import RiskManager
from data.fetcher import DataFetcher
from config.settings import LOG_DIR

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

START_DATE = "2023-06-01"
CAPITAL = 10000


def run_aggressive_single(strategy_class, symbol, interval, risk_pct, max_pos_pct):
    """Run a single strategy with aggressive position sizing."""
    fetcher = DataFetcher()
    df = fetcher.fetch_klines_cached(symbol, interval, START_DATE)
    if df.empty:
        return None

    strategy = strategy_class()
    rm = RiskManager(
        max_position_pct=max_pos_pct,
        max_open_positions=1,
        max_drawdown_pct=30.0,
        risk_per_trade_pct=risk_pct,
    )
    engine = BacktestEngine(
        initial_capital=CAPITAL,
        commission_pct=0.0,
        risk_manager=rm,
    )
    result = engine.run(strategy, df, symbol, interval)
    metrics = compute_metrics(result)
    
    # Calculate weekly returns
    if result.equity_curve is not None and len(result.equity_curve) > 7:
        weekly = result.equity_curve.resample("W").last().pct_change().dropna()
        avg_weekly = weekly.mean() * 100
        weeks_positive = (weekly > 0).mean() * 100
        weeks_over_1pct = (weekly >= 0.01).mean() * 100
    else:
        avg_weekly = 0
        weeks_positive = 0
        weeks_over_1pct = 0
    
    return {
        "strategy": strategy.name,
        "symbol": symbol,
        "interval": interval,
        "risk_pct": risk_pct,
        "max_pos_pct": max_pos_pct,
        "return_pct": metrics["total_return_pct"],
        "sharpe": metrics["sharpe_ratio"],
        "trades": metrics["total_trades"],
        "win_rate": metrics["win_rate"],
        "max_dd": metrics["max_drawdown_pct"],
        "profit_factor": metrics["profit_factor"],
        "avg_weekly_pct": avg_weekly,
        "weeks_positive_pct": weeks_positive,
        "weeks_over_1pct": weeks_over_1pct,
        "final_equity": metrics["final_equity"],
    }


def main():
    print("\n" + "=" * 90)
    print("  V6 AGGRESSIVE SIZING SIMULATION")
    print("  Testing: What returns are achievable with proper position sizing?")
    print("=" * 90)

    # Load V6 walk-forward validated strategies
    wf = pd.read_csv(LOG_DIR / "walk_forward_v6.csv")
    # Also load V3/V4 WF results
    validated = []
    
    v6_strats = get_v6_strategies()
    for _, row in wf.iterrows():
        if row.get("oos_mean_return", 0) > 0 and row["strategy"] in v6_strats:
            validated.append({
                "class": v6_strats[row["strategy"]],
                "symbol": row["symbol"],
                "interval": row["interval"],
                "oos_mean": row["oos_mean_return"],
                "name": f"{row['strategy']}_{row['symbol']}_{row['interval']}",
            })
    
    # Also V3/V4
    for wf_file, get_fn, label in [
        ("walk_forward_v3.csv", get_v3_strategies, "V3"),
        ("walk_forward_v4.csv", get_v4_strategies, "V4"),
    ]:
        path = LOG_DIR / wf_file
        if path.exists():
            try:
                wf_prev = pd.read_csv(path)
                strats = get_fn()
                for _, row in wf_prev.iterrows():
                    if row.get("oos_mean_return", 0) > 0.05 and row["strategy"] in strats:
                        validated.append({
                            "class": strats[row["strategy"]],
                            "symbol": row["symbol"],
                            "interval": row["interval"],
                            "oos_mean": row["oos_mean_return"],
                            "name": f"{label}_{row['strategy']}_{row['symbol']}_{row['interval']}",
                        })
            except Exception as e:
                print(f"  Warning: could not load {wf_file}: {e}")

    validated.sort(key=lambda x: x["oos_mean"], reverse=True)
    print(f"\n  Total validated strategies: {len(validated)}")

    # Test 1: Individual strategies with aggressive sizing
    print(f"\n{'='*90}")
    print("  TEST 1: Individual strategies with aggressive position sizing (FULL CAPITAL)")
    print(f"{'='*90}")
    
    risk_levels = [5, 10, 15, 20]  # % risk per trade
    max_pos_levels = [50, 75, 100]  # % of capital per position

    # Test top 10 validated strategies
    top10 = validated[:10]
    results_all = []
    
    for v in top10:
        print(f"\n  --- {v['name']} (OOS: {v['oos_mean']:+.4f}%) ---")
        for risk_pct in risk_levels:
            for max_pos in max_pos_levels:
                result = run_aggressive_single(
                    v["class"], v["symbol"], v["interval"],
                    risk_pct, max_pos,
                )
                if result:
                    results_all.append(result)
                    marker = " <<<" if result["avg_weekly_pct"] >= 1.0 else ""
                    print(f"    Risk {risk_pct:2d}% MaxPos {max_pos:3d}%: "
                          f"Return {result['return_pct']:+8.2f}%  "
                          f"MaxDD {result['max_dd']:6.2f}%  "
                          f"AvgWeek {result['avg_weekly_pct']:+.3f}%  "
                          f"Trades {result['trades']:4d}  "
                          f"WR {result['win_rate']:5.1f}%  "
                          f"Final ${result['final_equity']:,.0f}{marker}")

    # Save all results
    df_results = pd.DataFrame(results_all)
    df_results.to_csv(LOG_DIR / "aggressive_sizing_v6.csv", index=False)

    # Summary: Best weekly performers
    print(f"\n{'='*90}")
    print("  BEST CONFIGURATIONS BY AVERAGE WEEKLY RETURN")
    print(f"{'='*90}")
    
    if not df_results.empty:
        top_weekly = df_results.nlargest(20, "avg_weekly_pct")
        for _, r in top_weekly.iterrows():
            marker = " *** TARGET MET ***" if r["avg_weekly_pct"] >= 1.0 else ""
            print(f"  {r['strategy']:20s} {r['symbol']:10s} {r['interval']:3s} "
                  f"Risk {r['risk_pct']:2.0f}% Pos {r['max_pos_pct']:3.0f}% | "
                  f"AvgWeek {r['avg_weekly_pct']:+.3f}% | "
                  f"Return {r['return_pct']:+8.2f}% | "
                  f"MaxDD {r['max_dd']:6.2f}% | "
                  f"Final ${r['final_equity']:,.0f}{marker}")

    # Test 2: Concentrated portfolio (top 3) with aggressive sizing
    print(f"\n{'='*90}")
    print("  TEST 2: Concentrated Portfolio (top 3 strategies, sequential capital)")
    print(f"{'='*90}")
    
    top3 = validated[:3]
    for risk_pct in [10, 15, 20]:
        max_pos = 75
        total_return = 0
        total_trades = 0
        all_weekly = []
        max_dd_portfolio = 0
        
        for v in top3:
            result = run_aggressive_single(v["class"], v["symbol"], v["interval"], risk_pct, max_pos)
            if result:
                total_return += result["return_pct"] / 3  # Equal weight
                total_trades += result["trades"]
                max_dd_portfolio = max(max_dd_portfolio, result["max_dd"])
        
        print(f"  Risk {risk_pct:2d}% MaxPos {max_pos}%: "
              f"Avg Return {total_return:+.2f}%  "
              f"Total Trades {total_trades:4d}  "
              f"Max Component DD {max_dd_portfolio:.2f}%")

    print(f"\n{'='*90}")
    print("  DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
