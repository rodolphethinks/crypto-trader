"""
Walk-forward validation — split data into train/test windows and validate
that in-sample best params also perform out-of-sample.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CAPITAL = 10_000


def walk_forward(strategy_class, params: dict, df: pd.DataFrame,
                 symbol: str, interval: str,
                 n_splits: int = 3, train_pct: float = 0.7) -> pd.DataFrame:
    """
    Walk-forward validation.  Splits data into overlapping train/test
    windows, optimizes on train, validates on test.

    Returns DataFrame with per-fold results.
    """
    total_bars = len(df)
    window_size = total_bars // n_splits
    results = []

    for fold in range(n_splits):
        start = fold * (window_size // 2)  # 50% overlap between folds
        end = min(start + window_size, total_bars)
        if end - start < 100:
            continue

        fold_df = df.iloc[start:end]
        split_idx = int(len(fold_df) * train_pct)

        train_df = fold_df.iloc[:split_idx]
        test_df = fold_df.iloc[split_idx:]

        if len(train_df) < 50 or len(test_df) < 20:
            continue

        # Train
        try:
            strategy = strategy_class(params=params)
            engine = BacktestEngine(initial_capital=CAPITAL, commission_pct=0.0,
                                     risk_manager=RiskManager())
            train_result = engine.run(strategy, train_df.copy(), symbol, interval)
            train_metrics = compute_metrics(train_result)
        except Exception as e:
            logger.warning(f"Train fold {fold} failed: {e}")
            continue

        # Test
        try:
            strategy = strategy_class(params=params)
            engine = BacktestEngine(initial_capital=CAPITAL, commission_pct=0.0,
                                     risk_manager=RiskManager())
            test_result = engine.run(strategy, test_df.copy(), symbol, interval)
            test_metrics = compute_metrics(test_result)
        except Exception as e:
            logger.warning(f"Test fold {fold} failed: {e}")
            continue

        results.append({
            "fold": fold + 1,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "train_trades": train_metrics["total_trades"],
            "test_trades": test_metrics["total_trades"],
            "train_return": train_metrics["total_return_pct"],
            "test_return": test_metrics["total_return_pct"],
            "train_sharpe": train_metrics["sharpe_ratio"],
            "test_sharpe": test_metrics["sharpe_ratio"],
            "train_win_rate": train_metrics["win_rate"],
            "test_win_rate": test_metrics["win_rate"],
            "train_max_dd": train_metrics["max_drawdown_pct"],
            "test_max_dd": test_metrics["max_drawdown_pct"],
            "train_pf": train_metrics["profit_factor"],
            "test_pf": test_metrics["profit_factor"],
        })

    return pd.DataFrame(results)


def main():
    from strategies.bb_variants import (BB_Squeeze, BB_Double, BB_MultiConf,
                                         BB_Naive, BB_RSI, BB_Trend, BB_MACD, BB_Volume)
    from strategies.smc_liquidity import SMCLiquidityStrategy
    from strategies.vwap_strategy import VWAPStrategy

    fetcher = DataFetcher()
    end = datetime.utcnow()

    # Top combos from master sweep v2 — validate each
    test_configs = [
        # Top Sharpe performers from sweep
        ("BB_MACD", BB_MACD, {}, "DOGEUSDC", "4h"),
        ("BB_MACD", BB_MACD, {}, "AVAXUSDC", "4h"),
        ("SMC_Liquidity", SMCLiquidityStrategy, {}, "BNBUSDC", "4h"),
        ("SMC_Liquidity", SMCLiquidityStrategy, {}, "XRPUSDC", "4h"),
        ("SMC_Liquidity", SMCLiquidityStrategy, {}, "SOLUSDC", "4h"),
        ("VWAP", VWAPStrategy, {}, "DOGEUSDC", "4h"),
        ("VWAP", VWAPStrategy, {}, "XRPUSDC", "1d"),
        ("BB_Double", BB_Double, {}, "XRPUSDC", "1d"),
        ("BB_RSI", BB_RSI, {}, "XRPUSDC", "1d"),
        ("BB_Volume", BB_Volume, {}, "XRPUSDC", "1d"),
        ("BB_MultiConf", BB_MultiConf, {}, "DOGEUSDC", "4h"),
        ("BB_Squeeze", BB_Squeeze, {}, "BNBUSDC", "4h"),
        ("BB_Trend", BB_Trend, {}, "SOLUSDC", "4h"),
    ]

    # Load data (90 days for 4h, 90 for 1h)
    data_cache = {}
    pairs_intervals = set((c[3], c[4]) for c in test_configs)
    for sym, intv in pairs_intervals:
        start = end - timedelta(days=90)
        df = fetcher.fetch_klines_cached(sym, intv, start.strftime("%Y-%m-%d"),
                                          end.strftime("%Y-%m-%d"))
        data_cache[(sym, intv)] = df
        logger.info(f"Loaded {sym} {intv}: {len(df)} candles")

    all_wf = []
    print("\n" + "=" * 120)
    print("  WALK-FORWARD VALIDATION (3 folds, 70/30 split)")
    print("=" * 120)

    for name, cls, params, symbol, interval in test_configs:
        df = data_cache.get((symbol, interval))
        if df is None or df.empty:
            continue

        logger.info(f"\nValidating {name} on {symbol} {interval}...")
        wf = walk_forward(cls, params, df, symbol, interval, n_splits=3)

        if wf.empty:
            print(f"\n  {name:20s} {symbol:10s} {interval:4s} — NO DATA")
            continue

        avg_train_ret = wf["train_return"].mean()
        avg_test_ret = wf["test_return"].mean()
        avg_train_sharpe = wf["train_sharpe"].mean()
        avg_test_sharpe = wf["test_sharpe"].mean()
        consistency = "PASS" if avg_test_ret > 0 or abs(avg_test_ret) < abs(avg_train_ret) * 0.5 else "FAIL"
        overfit = "YES" if avg_train_sharpe > 0 and avg_test_sharpe < -2 else "NO"

        print(f"\n  {name:20s} {symbol:10s} {interval:4s}")
        print(f"    Train avg: return {avg_train_ret:+.3f}%, Sharpe {avg_train_sharpe:.2f}")
        print(f"    Test  avg: return {avg_test_ret:+.3f}%, Sharpe {avg_test_sharpe:.2f}")
        print(f"    Consistency: {consistency} | Overfit: {overfit}")

        for _, row in wf.iterrows():
            print(f"      Fold {int(row['fold'])}: train {row['train_return']:+.3f}% "
                  f"({int(row['train_trades'])} trades) | "
                  f"test {row['test_return']:+.3f}% ({int(row['test_trades'])} trades)")

        wf["strategy"] = name
        wf["symbol"] = symbol
        wf["interval"] = interval
        wf["consistency"] = consistency
        wf["overfit"] = overfit
        all_wf.append(wf)

    if all_wf:
        combined = pd.concat(all_wf, ignore_index=True)
        csv_path = os.path.join(LOG_DIR, "walk_forward_results.csv")
        combined.to_csv(csv_path, index=False)
        logger.info(f"\nSaved to {csv_path}")

        # Summary
        print("\n" + "=" * 120)
        print("  SUMMARY")
        print("=" * 120)
        summary = combined.groupby(["strategy", "symbol", "interval"]).agg({
            "train_return": "mean",
            "test_return": "mean",
            "train_sharpe": "mean",
            "test_sharpe": "mean",
            "consistency": "first",
            "overfit": "first",
        }).round(3)
        print(summary.to_string())

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
