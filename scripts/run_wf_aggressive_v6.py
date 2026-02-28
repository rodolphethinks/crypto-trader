"""
Walk-forward validation of the aggressive sizing configurations
that met 1%/week target. This is CRITICAL - we need to confirm
the results hold out-of-sample.

Tests the top configurations from aggressive_sizing_v6.csv
with proper train/test splits.
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from backtesting.runner import get_v6_strategies, get_v3_strategies, get_v4_strategies
from risk.manager import RiskManager
from data.fetcher import DataFetcher
from config.settings import LOG_DIR

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

START_DATE = "2023-06-01"
CAPITAL = 10000


def walk_forward_aggressive(strategy_class, symbol, interval, risk_pct, max_pos_pct, n_folds=4):
    """
    Walk-forward validation with aggressive sizing.
    Uses 4 folds (more than the 3 in basic WF) for robustness.
    """
    fetcher = DataFetcher()
    df = fetcher.fetch_klines_cached(symbol, interval, START_DATE)
    if df.empty:
        return None

    n = len(df)
    fold_size = n // (n_folds + 1)
    if fold_size < 20:
        return None

    oos_returns = []
    oos_weekly_avgs = []
    oos_max_dds = []
    oos_trades = []
    oos_win_rates = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 15:
            continue

        test_df = df.iloc[test_start:test_end].copy()
        
        strategy = strategy_class()
        rm = RiskManager(
            max_position_pct=max_pos_pct,
            max_open_positions=1,
            max_drawdown_pct=35.0,
            risk_per_trade_pct=risk_pct,
        )
        engine = BacktestEngine(
            initial_capital=CAPITAL,
            commission_pct=0.0,
            risk_manager=rm,
        )
        result = engine.run(strategy, test_df, symbol, interval)
        m = compute_metrics(result)
        
        oos_returns.append(m["total_return_pct"])
        oos_max_dds.append(m["max_drawdown_pct"])
        oos_trades.append(m["total_trades"])
        oos_win_rates.append(m["win_rate"])
        
        # Weekly returns for this fold
        if result.equity_curve is not None and len(result.equity_curve) > 7:
            weekly = result.equity_curve.resample("W").last().pct_change().dropna()
            oos_weekly_avgs.append(weekly.mean() * 100)
        else:
            oos_weekly_avgs.append(0)

    if not oos_returns:
        return None

    return {
        "oos_returns": oos_returns,
        "oos_mean_return": np.mean(oos_returns),
        "oos_weekly_means": oos_weekly_avgs,
        "oos_mean_weekly": np.mean(oos_weekly_avgs),
        "oos_max_dds": oos_max_dds,
        "oos_mean_dd": np.mean(oos_max_dds),
        "oos_trades": oos_trades,
        "oos_win_rates": oos_win_rates,
        "folds_positive": sum(1 for r in oos_returns if r > 0),
        "total_folds": len(oos_returns),
        "all_positive": all(r > 0 for r in oos_returns),
    }


def main():
    print("\n" + "=" * 100)
    print("  WALK-FORWARD VALIDATION OF AGGRESSIVE CONFIGURATIONS")
    print("  Testing top strategies that met 1%/week in full backtest")
    print("=" * 100)

    # Top configs to validate
    v6 = get_v6_strategies()
    
    configs = [
        # (strategy_class, symbol, interval, risk_pct, max_pos_pct, label)
        (v6["MomAccelerator"], "AVAXUSDC", "4h", 5, 100, "MomAccel AVAX 4h R5 P100"),
        (v6["MomAccelerator"], "AVAXUSDC", "4h", 5, 75, "MomAccel AVAX 4h R5 P75"),
        (v6["MomAccelerator"], "AVAXUSDC", "4h", 5, 50, "MomAccel AVAX 4h R5 P50"),
        (v6["MomAccelerator"], "AVAXUSDC", "4h", 10, 75, "MomAccel AVAX 4h R10 P75"),
        (v6["CrossPairLead"], "AVAXUSDC", "1d", 20, 100, "CrossPair AVAX 1d R20 P100"),
        (v6["CrossPairLead"], "AVAXUSDC", "1d", 15, 100, "CrossPair AVAX 1d R15 P100"),
        (v6["CrossPairLead"], "AVAXUSDC", "1d", 10, 100, "CrossPair AVAX 1d R10 P100"),
        (v6["CrossPairLead"], "AVAXUSDC", "1d", 15, 75, "CrossPair AVAX 1d R15 P75"),
        (v6["CrossPairLead"], "AVAXUSDC", "1d", 10, 75, "CrossPair AVAX 1d R10 P75"),
        (v6["CrossPairLead"], "SUIUSDC", "1d", 20, 75, "CrossPair SUI 1d R20 P75"),
        (v6["CrossPairLead"], "SUIUSDC", "1d", 15, 75, "CrossPair SUI 1d R15 P75"),
        (v6["CrossPairLead"], "SUIUSDC", "1d", 10, 75, "CrossPair SUI 1d R10 P75"),
        (v6["MultiEdge"], "DOGEUSDC", "4h", 5, 100, "MultiEdge DOGE 4h R5 P100"),
        (v6["MultiEdge"], "DOGEUSDC", "4h", 5, 75, "MultiEdge DOGE 4h R5 P75"),
        (v6["MomAccelerator"], "LINKUSDC", "4h", 5, 100, "MomAccel LINK 4h R5 P100"),
        (v6["MomAccelerator"], "LINKUSDC", "1d", 5, 100, "MomAccel LINK 1d R5 P100"),
        (v6["CrossPairLead"], "ETHUSDC", "1d", 10, 100, "CrossPair ETH 1d R10 P100"),
        (v6["CrossPairLead"], "BTCUSDC", "1d", 10, 100, "CrossPair BTC 1d R10 P100"),
        (v6["MomAccelerator"], "BNBUSDC", "1d", 10, 100, "MomAccel BNB 1d R10 P100"),
        (v6["RegimeMomV2"], "AVAXUSDC", "4h", 5, 100, "RegimeMomV2 AVAX 4h R5 P100"),
    ]

    results = []
    print(f"\n  Testing {len(configs)} configurations with 4-fold walk-forward...\n")

    for strat_class, symbol, interval, risk_pct, max_pos, label in configs:
        wf = walk_forward_aggressive(strat_class, symbol, interval, risk_pct, max_pos, n_folds=4)
        
        if wf is None:
            print(f"  {label:40s}  FAILED (insufficient data)")
            continue

        marker = "PASSED" if wf["all_positive"] else f"{wf['folds_positive']}/{wf['total_folds']}"
        target = " *** 1%/wk OOS ***" if wf["oos_mean_weekly"] >= 1.0 else ""
        
        result_row = {
            "label": label,
            "oos_mean_return": wf["oos_mean_return"],
            "oos_mean_weekly": wf["oos_mean_weekly"],
            "oos_mean_dd": wf["oos_mean_dd"],
            "folds_positive": wf["folds_positive"],
            "total_folds": wf["total_folds"],
            "all_positive": wf["all_positive"],
            "oos_returns": wf["oos_returns"],
            "oos_weekly_means": wf["oos_weekly_means"],
            "oos_trades": wf["oos_trades"],
        }
        results.append(result_row)

        print(f"  {label:40s}  {marker:8s}  "
              f"OOS Return: {wf['oos_mean_return']:+7.2f}%  "
              f"OOS Weekly: {wf['oos_mean_weekly']:+6.3f}%  "
              f"OOS MaxDD: {wf['oos_mean_dd']:5.2f}%  "
              f"Folds: {wf['oos_returns']}{target}")

    # Summary
    print(f"\n{'='*100}")
    print("  SUMMARY — WALK-FORWARD VALIDATED RESULTS")
    print(f"{'='*100}")
    
    passed = [r for r in results if r["all_positive"]]
    print(f"\n  All-positive (all folds profitable): {len(passed)}/{len(results)}")
    
    for r in sorted(results, key=lambda x: x["oos_mean_weekly"], reverse=True):
        status = "ALL+" if r["all_positive"] else f"{r['folds_positive']}/{r['total_folds']}"
        marker = " <<<< TARGET" if r["oos_mean_weekly"] >= 1.0 else ""
        print(f"  {r['label']:40s}  [{status:5s}]  "
              f"Weekly: {r['oos_mean_weekly']:+6.3f}%  "
              f"Return: {r['oos_mean_return']:+7.2f}%  "
              f"MaxDD: {r['oos_mean_dd']:5.2f}%{marker}")

    # Evaluate honest weekly capability 
    if results:
        best = max(results, key=lambda x: x["oos_mean_weekly"])
        print(f"\n  BEST OOS WEEKLY: {best['label']}")
        print(f"    Average weekly return: {best['oos_mean_weekly']:+.3f}%")
        print(f"    Average fold return: {best['oos_mean_return']:+.2f}%")
        print(f"    Per-fold weekly means: {[f'{w:+.3f}%' for w in best['oos_weekly_means']]}")
        
        # Projected annual from OOS
        annual_from_weekly = ((1 + best['oos_mean_weekly']/100) ** 52 - 1) * 100
        print(f"    Projected annual (compound): {annual_from_weekly:+.1f}%")
        print(f"    On $10k: ${CAPITAL * (1 + annual_from_weekly/100):,.0f}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(LOG_DIR / "wf_aggressive_v6.csv", index=False)
    print(f"\n  Results saved to logs/wf_aggressive_v6.csv")


if __name__ == "__main__":
    main()
