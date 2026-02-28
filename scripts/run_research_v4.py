"""
Master Research V4 — comprehensive pipeline:

1. Grid search optimization on the 3 walk-forward validated winners
   (AdaptiveTrend, CrossTF_Momentum, Vol_Breakout) on their best pairs
2. Sweep of 5 new V4 strategies across 20 pairs (original 15 + 5 new)
3. Walk-forward validation on all winners
4. Final ranked results

Timeframes: 1h, 4h, 1d (only TFs that showed promise in V3)
Data: 365 days
"""
import sys
import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from itertools import product as iterproduct

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.runner import get_v3_strategies, get_v4_strategies
from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR, DEFAULT_INITIAL_CAPITAL

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CAPITAL = 10_000

# Extended pair list (original 15 + 5 new)
PAIRS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "DOGEUSDC",
    "BNBUSDC", "AVAXUSDC", "LINKUSDC", "ADAUSDC", "SUIUSDC",
    "PEPEUSDC", "NEARUSDC", "LTCUSDC", "INJUSDC", "APTUSDC",
    # New pairs
    "DOTUSDC", "MATICUSDC", "ATOMUSDC", "FILUSDC", "ARBUSDC",
]

TIMEFRAMES = ["1h", "4h", "1d"]

DATA_WINDOWS = {"1h": 365, "4h": 365, "1d": 365}


def fetch_data(fetcher, pair, tf):
    """Fetch data with cache awareness."""
    days = DATA_WINDOWS[tf]
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        df = fetcher.fetch_klines_cached(pair, tf, start_date=start)
        if df is not None and not df.empty:
            # Check if we need to re-fetch for longer window
            data_days = (df.index[-1] - df.index[0]).days
            if data_days < days - 30:
                df = fetcher.fetch_klines(pair, tf, start_date=start)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch {pair} {tf}: {e}")
        return None


def run_single_backtest(strategy_class, params, df, pair, tf):
    """Run one backtest, return metrics dict or None."""
    try:
        strategy = strategy_class(params=params) if params else strategy_class()
        engine = BacktestEngine(
            initial_capital=CAPITAL,
            commission_pct=0.0,
            slippage_pct=0.0005,
            risk_manager=RiskManager(),
        )
        result = engine.run(strategy, df.copy(), pair, tf)
        metrics = compute_metrics(result)
        return {
            "Strategy": strategy.name,
            "Symbol": pair,
            "Interval": tf,
            "Trades": metrics["total_trades"],
            "Win Rate %": round(metrics["win_rate"], 1),
            "Return %": round(metrics["total_return_pct"], 4),
            "Max DD %": round(metrics["max_drawdown_pct"], 2),
            "Sharpe": round(metrics["sharpe_ratio"], 2),
            "Profit Factor": round(metrics["profit_factor"], 2),
            "Avg Trade $": round(metrics.get("avg_trade_pnl", 0), 2),
            "Params": str(params) if params else "default",
        }
    except Exception as e:
        logger.warning(f"Error: {strategy_class.__name__} {pair} {tf}: {e}")
        return None


def walk_forward_validate(strategy_class, params, df, pair, tf, n_folds=3):
    """Run walk-forward validation, return dict with results."""
    n = len(df)
    fold_size = n // (n_folds + 1)
    fold_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_end = min(train_end + fold_size, n)
        if train_end >= n or test_end <= train_end:
            continue

        df_train = df.iloc[:train_end].copy()
        df_test = df.iloc[:test_end].copy()

        try:
            strat = strategy_class(params=params) if params else strategy_class()
            eng_train = BacktestEngine(CAPITAL, 0.0, 0.0005, RiskManager())
            res_train = eng_train.run(strat, df_train, pair, tf)
            train_ret = res_train.total_return
            train_trades = res_train.total_trades

            strat2 = strategy_class(params=params) if params else strategy_class()
            eng_test = BacktestEngine(CAPITAL, 0.0, 0.0005, RiskManager())
            res_test = eng_test.run(strat2, df_test, pair, tf)
            test_ret = res_test.total_return
            test_trades = res_test.total_trades

            fold_results.append({
                "oos_return": test_ret - train_ret,
                "oos_trades": test_trades - train_trades,
            })
        except Exception:
            continue

    if not fold_results:
        return {"status": "FAIL", "reason": "no_folds", "oos_return": 0, "oos_trades": 0}

    oos_returns = [f["oos_return"] for f in fold_results]
    oos_positive = sum(1 for r in oos_returns if r > 0)
    avg_oos = np.mean(oos_returns)
    total_oos_trades = sum(f["oos_trades"] for f in fold_results)

    passed = oos_positive >= len(fold_results) / 2
    return {
        "status": "PASS" if passed else "FAIL",
        "reason": "OK" if passed else "inconsistent",
        "oos_return": round(avg_oos, 4),
        "oos_trades": total_oos_trades,
        "folds_positive": f"{oos_positive}/{len(fold_results)}",
    }


def phase1_grid_search(fetcher, data_cache):
    """Grid search on validated V3 winners."""
    logger.info("\n" + "=" * 80)
    logger.info("  PHASE 1: Grid Search Optimization on V3 Winners")
    logger.info("=" * 80)

    v3 = get_v3_strategies()
    results = []

    # AdaptiveTrend grid — top 3 pairs from V3 walk-forward
    at_grid = {
        "er_period": [6, 8, 10, 12],
        "slow_sc": [20, 25, 30, 35],
        "adx_min": [15, 18, 20, 25],
        "sl_atr_mult": [1.5, 1.8, 2.0, 2.5],
        "tp_atr_mult": [2.5, 3.0, 3.5, 4.0],
        "cooldown": [2, 3, 4],
    }
    at_pairs = [("XRPUSDC", "4h"), ("INJUSDC", "4h"), ("AVAXUSDC", "4h")]

    for pair, tf in at_pairs:
        df = data_cache.get((pair, tf))
        if df is None or df.empty:
            continue
        logger.info(f"  Grid search: AdaptiveTrend on {pair} {tf} ({len(at_grid)} params)")
        
        # Selective grid: don't try all combos, use smart sampling
        keys = list(at_grid.keys())
        values = list(at_grid.values())
        all_combos = list(iterproduct(*values))
        # Limit to 500 random combos if too many
        if len(all_combos) > 500:
            np.random.seed(42)
            indices = np.random.choice(len(all_combos), 500, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos

        logger.info(f"    Testing {len(combos)} combinations...")
        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            params["fast_sc"] = 2  # keep fixed
            params["rsi_period"] = 14
            params["atr_period"] = 14
            r = run_single_backtest(v3["AdaptiveTrend"], params, df, pair, tf)
            if r and r["Trades"] >= 5:
                r["Phase"] = "GridSearch"
                results.append(r)
            if (idx + 1) % 100 == 0:
                logger.info(f"    {idx+1}/{len(combos)} done")

    # CrossTF_Momentum grid — top 2 pairs
    ct_grid = {
        "trend_ema": [120, 150, 180, 200],
        "medium_slope_min": [0.0003, 0.0005, 0.001, 0.0015],
        "rsi_buy_zone": [38, 42, 45, 48],
        "sl_atr_mult": [1.5, 1.8, 2.0],
        "tp_atr_mult": [3.0, 3.5, 4.0, 5.0],
        "cooldown": [3, 4, 5],
    }
    ct_pairs = [("SOLUSDC", "4h"), ("NEARUSDC", "4h")]

    for pair, tf in ct_pairs:
        df = data_cache.get((pair, tf))
        if df is None or df.empty:
            continue
        logger.info(f"  Grid search: CrossTF_Momentum on {pair} {tf}")

        keys = list(ct_grid.keys())
        values = list(ct_grid.values())
        all_combos = list(iterproduct(*values))
        if len(all_combos) > 500:
            np.random.seed(42)
            indices = np.random.choice(len(all_combos), 500, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos

        logger.info(f"    Testing {len(combos)} combinations...")
        for idx, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            params["medium_slope_period"] = 50
            params["rsi_sell_zone"] = 100 - params["rsi_buy_zone"]
            params["macd_fast"] = 12
            params["macd_slow"] = 26
            params["macd_signal"] = 9
            params["rsi_period"] = 14
            params["atr_period"] = 14
            r = run_single_backtest(v3["CrossTF_Momentum"], params, df, pair, tf)
            if r and r["Trades"] >= 5:
                r["Phase"] = "GridSearch"
                results.append(r)
            if (idx + 1) % 100 == 0:
                logger.info(f"    {idx+1}/{len(combos)} done")

    return results


def phase2_v4_sweep(fetcher, data_cache):
    """Sweep V4 research strategies across all pairs/TFs."""
    logger.info("\n" + "=" * 80)
    logger.info("  PHASE 2: V4 Strategy Sweep (ensembles + tuned + fixes)")
    logger.info("=" * 80)

    v4 = get_v4_strategies()
    results = []
    total = len(v4) * len(PAIRS) * len(TIMEFRAMES)
    done = 0

    for strat_name, strat_class in v4.items():
        for pair in PAIRS:
            for tf in TIMEFRAMES:
                done += 1
                df = data_cache.get((pair, tf))
                if df is None or df.empty:
                    continue

                r = run_single_backtest(strat_class, None, df, pair, tf)
                if r:
                    r["Phase"] = "V4Sweep"
                    results.append(r)

                if done % 20 == 0:
                    logger.info(f"  Phase 2 progress: {done}/{total}")

    return results


def phase3_validate(fetcher, data_cache, all_results):
    """Walk-forward validate the top combos from phases 1 & 2."""
    logger.info("\n" + "=" * 80)
    logger.info("  PHASE 3: Walk-Forward Validation")
    logger.info("=" * 80)

    df_res = pd.DataFrame(all_results)
    if df_res.empty:
        return []

    # Filter: profitable, min 5 trades, Sharpe > 0.5
    candidates = df_res[
        (df_res["Return %"] > 0) &
        (df_res["Trades"] >= 5) &
        (df_res["Sharpe"] > 0.5)
    ].sort_values("Sharpe", ascending=False).head(30)

    logger.info(f"  Validating top {len(candidates)} candidates")

    v3 = get_v3_strategies()
    v4 = get_v4_strategies()
    all_strats = {**v3, **v4}

    wf_results = []
    for idx, (_, row) in enumerate(candidates.iterrows()):
        strat_name = row["Strategy"]
        pair = row["Symbol"]
        tf = row["Interval"]
        params_str = row.get("Params", "default")

        logger.info(f"  [{idx+1}/{len(candidates)}] {strat_name} {pair} {tf}")

        strat_class = all_strats.get(strat_name)
        if strat_class is None:
            continue

        df = data_cache.get((pair, tf))
        if df is None or df.empty:
            continue

        # Parse params if not default
        params = None
        if params_str != "default":
            try:
                params = eval(params_str)
            except:
                params = None

        wf = walk_forward_validate(strat_class, params, df, pair, tf)
        wf["Strategy"] = strat_name
        wf["Symbol"] = pair
        wf["Interval"] = tf
        wf["Sweep_Return"] = row["Return %"]
        wf["Sweep_Sharpe"] = row["Sharpe"]
        wf["Sweep_Trades"] = row["Trades"]
        wf["Params"] = params_str
        wf_results.append(wf)

        status = wf["status"]
        oos = wf["oos_return"]
        logger.info(f"    -> {status} | OOS: {oos:+.4f}%")

    return wf_results


def main():
    start_time = time.time()
    fetcher = DataFetcher()

    # Pre-fetch all data
    logger.info("Pre-fetching data for all pairs and timeframes...")
    data_cache = {}
    for pair in PAIRS:
        for tf in TIMEFRAMES:
            df = fetch_data(fetcher, pair, tf)
            if df is not None and not df.empty:
                data_cache[(pair, tf)] = df
                logger.info(f"  {pair} {tf}: {len(df)} bars")
            else:
                logger.warning(f"  {pair} {tf}: NO DATA")
            time.sleep(0.3)  # Rate limiting

    logger.info(f"Data cached: {len(data_cache)} pair/TF combos")

    # Phase 1: Grid search
    grid_results = phase1_grid_search(fetcher, data_cache)
    logger.info(f"Phase 1 complete: {len(grid_results)} grid search results")

    # Phase 2: V4 sweep
    sweep_results = phase2_v4_sweep(fetcher, data_cache)
    logger.info(f"Phase 2 complete: {len(sweep_results)} sweep results")

    # Also re-test V3 winners on new pairs (5 new pairs)
    logger.info("\n  Re-testing V3 winners on new pairs...")
    v3_strats = get_v3_strategies()
    v3_on_new = []
    new_pairs = ["DOTUSDC", "MATICUSDC", "ATOMUSDC", "FILUSDC", "ARBUSDC"]
    best_v3 = ["AdaptiveTrend", "CrossTF_Momentum", "Vol_Breakout"]
    for sn in best_v3:
        for pair in new_pairs:
            for tf in TIMEFRAMES:
                df = data_cache.get((pair, tf))
                if df is None or df.empty:
                    continue
                r = run_single_backtest(v3_strats[sn], None, df, pair, tf)
                if r:
                    r["Phase"] = "V3_NewPairs"
                    v3_on_new.append(r)

    logger.info(f"V3 on new pairs: {len(v3_on_new)} results")

    # Combine all results
    all_results = grid_results + sweep_results + v3_on_new

    # Save combined sweep results
    df_all = pd.DataFrame(all_results)
    sweep_csv = LOG_DIR / "master_sweep_v4.csv"
    df_all.to_csv(sweep_csv, index=False)
    logger.info(f"\nAll results saved to {sweep_csv}")

    # Phase 3: Walk-forward validation
    wf_results = phase3_validate(fetcher, data_cache, all_results)

    # Save WF results
    df_wf = pd.DataFrame(wf_results)
    wf_csv = LOG_DIR / "walk_forward_v4.csv"
    df_wf.to_csv(wf_csv, index=False)

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 90)
    print("  V4 RESEARCH COMPLETE")
    print("=" * 90)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Grid search results:    {len(grid_results)}")
    print(f"  V4 sweep results:       {len(sweep_results)}")
    print(f"  V3 on new pairs:        {len(v3_on_new)}")
    print(f"  Total combos tested:    {len(all_results)}")
    print(f"  Walk-forward validated:  {len(wf_results)}")

    if not df_wf.empty:
        passed = df_wf[df_wf["status"] == "PASS"]
        print(f"\n  Walk-forward PASSED: {len(passed)} / {len(df_wf)}")

        if len(passed) > 0:
            print(f"\n  TOP VALIDATED WINNERS:")
            for _, r in passed.sort_values("oos_return", ascending=False).head(15).iterrows():
                ann = r["oos_return"] * (365 / 91)
                dollar = ann / 100 * CAPITAL
                print(f"    {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
                      f"| OOS {r['oos_return']:+.4f}% (~${dollar:+.0f}/yr) "
                      f"| Sweep {r['Sweep_Return']:+.2f}% Sharpe {r['Sweep_Sharpe']:.2f} "
                      f"| {r.get('folds_positive', '?')} folds")

    print(f"\n  Results: {sweep_csv}")
    print(f"  Walk-forward: {wf_csv}")


if __name__ == "__main__":
    main()
