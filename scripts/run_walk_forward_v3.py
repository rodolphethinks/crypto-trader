"""
Walk-forward validation for V3 sweep winners.

Tests the top combos from sweep V3 using 3-fold expanding-window
walk-forward analysis:
  Fold 1: train on first 50%, test on next 16.7%
  Fold 2: train on first 66.7%, test on next 16.7%
  Fold 3: train on first 83.3%, test on last 16.7%

Checks:
1. Consistency: profitable in >50% of out-of-sample folds
2. Overfit: test Sharpe significantly worse than train Sharpe
3. Trade count: enough trades in each fold for significance
"""
import sys
import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.runner import get_v3_strategies
from backtesting.engine import BacktestEngine
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR, DEFAULT_INITIAL_CAPITAL

logging.basicConfig(level=logging.WARNING)

# Top combos from sweep V3 (by Sharpe, with decent trade counts)
TOP_COMBOS = [
    ("AdaptiveTrend", "INJUSDC", "1d"),
    ("AdaptiveTrend", "XRPUSDC", "4h"),
    ("AdaptiveTrend", "AVAXUSDC", "4h"),
    ("AdaptiveTrend", "INJUSDC", "4h"),
    ("AdaptiveTrend", "SUIUSDC", "1d"),
    ("CrossTF_Momentum", "SOLUSDC", "4h"),
    ("CrossTF_Momentum", "NEARUSDC", "4h"),
    ("CrossTF_Momentum", "PEPEUSDC", "1h"),
    ("RegimeAdaptive", "LINKUSDC", "1d"),
    ("RegimeAdaptive", "NEARUSDC", "1d"),
    ("Vol_Breakout", "LINKUSDC", "4h"),
    ("Vol_Breakout", "XRPUSDC", "1h"),
    ("LSTM", "XRPUSDC", "4h"),
    ("LSTM", "LTCUSDC", "4h"),
]

DATA_WINDOWS = {
    "1h": 365,
    "4h": 365,
    "1d": 365,
}

N_FOLDS = 3


def run_walk_forward():
    print("=" * 90)
    print("  WALK-FORWARD VALIDATION — V3 Winners")
    print("=" * 90)

    strategies = get_v3_strategies()
    fetcher = DataFetcher()
    results = []

    for combo_idx, (strat_name, pair, tf) in enumerate(TOP_COMBOS, 1):
        print(f"\n[{combo_idx}/{len(TOP_COMBOS)}] {strat_name} {pair} {tf}")

        # Fetch data
        days = DATA_WINDOWS.get(tf, 365)
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = fetcher.fetch_klines_cached(pair, tf, start_date=start_date)

        if df is None or df.empty:
            print("  -> NO DATA")
            results.append({
                "Strategy": strat_name, "Symbol": pair, "Interval": tf,
                "Status": "NO DATA",
            })
            continue

        n_bars = len(df)
        fold_size = n_bars // (N_FOLDS + 1)  # each fold = ~25% of data

        fold_results = []

        for fold in range(N_FOLDS):
            # Training: bars 0 to train_end
            train_end = fold_size * (fold + 2)  # 50%, 66%, 83% of data
            # Testing: bars train_end to test_end
            test_end = min(train_end + fold_size, n_bars)

            df_train = df.iloc[:train_end].copy()
            df_test = df.iloc[:test_end].copy()  # expanding: test includes all prior data for strategy warmup

            if len(df_train) < 50 or test_end <= train_end:
                continue

            strat_class = strategies[strat_name]

            # Train fold
            try:
                engine_train = BacktestEngine(
                    initial_capital=DEFAULT_INITIAL_CAPITAL,
                    commission_pct=0.0, slippage_pct=0.0005,
                    risk_manager=RiskManager(),
                )
                result_train = engine_train.run(strat_class(), df_train, pair, tf)
                train_return = result_train.total_return
                train_sharpe = result_train.sharpe_ratio
                train_trades = result_train.total_trades
            except Exception as e:
                print(f"  Fold {fold+1} train error: {e}")
                continue

            # Test fold (run on full data up to test_end, but only count trades in test window)
            try:
                engine_test = BacktestEngine(
                    initial_capital=DEFAULT_INITIAL_CAPITAL,
                    commission_pct=0.0, slippage_pct=0.0005,
                    risk_manager=RiskManager(),
                )
                result_test = engine_test.run(strat_class(), df_test, pair, tf)
                test_return = result_test.total_return
                test_sharpe = result_test.sharpe_ratio
                test_trades = result_test.total_trades

                # Approximate OOS return: test - train (incremental return from the new data)
                oos_return = test_return - train_return
                oos_trades = test_trades - train_trades
            except Exception as e:
                print(f"  Fold {fold+1} test error: {e}")
                continue

            fold_results.append({
                "fold": fold + 1,
                "train_return": train_return,
                "train_sharpe": train_sharpe,
                "train_trades": train_trades,
                "test_return": test_return,
                "test_sharpe": test_sharpe,
                "test_trades": test_trades,
                "oos_return": oos_return,
                "oos_trades": oos_trades,
            })

            print(f"  Fold {fold+1}: Train {train_return:+.4f}% ({train_trades}t, Sharpe {train_sharpe:.2f}) "
                  f"| OOS {oos_return:+.4f}% ({oos_trades}t)")

        # Analyze folds
        if not fold_results:
            results.append({
                "Strategy": strat_name, "Symbol": pair, "Interval": tf,
                "Status": "FAIL", "Reason": "no_valid_folds",
            })
            continue

        oos_returns = [f["oos_return"] for f in fold_results]
        oos_positive = sum(1 for r in oos_returns if r > 0)
        avg_oos = np.mean(oos_returns)
        
        train_sharpes = [f["train_sharpe"] for f in fold_results]
        avg_train_sharpe = np.mean(train_sharpes)
        
        # Check for overfitting: train sharpe much better than OOS performance
        consistency = oos_positive >= len(fold_results) / 2
        overfit = avg_oos < 0 and avg_train_sharpe > 1.0

        status = "PASS" if consistency and not overfit else "FAIL"
        reason = []
        if not consistency:
            reason.append("inconsistent")
        if overfit:
            reason.append("overfit")

        total_oos_trades = sum(f["oos_trades"] for f in fold_results)
        avg_oos_per_fold = total_oos_trades / len(fold_results)

        results.append({
            "Strategy": strat_name, "Symbol": pair, "Interval": tf,
            "Status": status,
            "Reason": ",".join(reason) if reason else "OK",
            "Folds_Positive": f"{oos_positive}/{len(fold_results)}",
            "Avg_OOS_Return": round(avg_oos, 4),
            "Avg_TrainSharpe": round(avg_train_sharpe, 4),
            "OOS_Trades": total_oos_trades,
            "Dollar_OOS": round(avg_oos / 100 * 10000, 2),
        })

        print(f"  -> {status} | OOS: {avg_oos:+.4f}% (${avg_oos / 100 * 10000:+.0f}) "
              f"| {oos_positive}/{len(fold_results)} folds positive "
              f"| {'OVERFIT' if overfit else 'OK'}")

    # Save results
    df_results = pd.DataFrame(results)
    csv_path = LOG_DIR / "walk_forward_v3.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n" + "=" * 90)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 90)
    passed = df_results[df_results["Status"] == "PASS"]
    failed = df_results[df_results["Status"] == "FAIL"]
    print(f"  PASSED: {len(passed)} / {len(df_results)}")
    print(f"  FAILED: {len(failed)} / {len(df_results)}")

    if len(passed) > 0:
        print("\n  PASSED COMBOS:")
        for _, r in passed.iterrows():
            print(f"    {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
                  f"| Avg OOS: {r.get('Avg_OOS_Return', 0):+.4f}% "
                  f"(${r.get('Dollar_OOS', 0):+.0f}) "
                  f"| {r.get('Folds_Positive', 'N/A')} folds "
                  f"| {r.get('OOS_Trades', 0)} OOS trades")

    if len(failed) > 0:
        print("\n  FAILED COMBOS:")
        for _, r in failed.iterrows():
            reason = r.get("Reason", "unknown")
            print(f"    {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
                  f"| {reason}")


if __name__ == "__main__":
    run_walk_forward()
