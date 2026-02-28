"""
Master sweep v2 — incorporates:
  1. New risk model (1% per-trade, no daily halt)
  2. Fixed BB_Trend & BB_MACD (looser thresholds)
  3. Additional pairs (BNBUSDC, AVAXUSDC, LINKUSDC)
  4. 1d timeframe
  5. Ensemble strategies (majority vote combos)
  6. Full recap
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product

from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR

from strategies.bb_variants import get_bb_variants
from backtesting.runner import get_all_strategies
from strategies.ensemble import EnsembleStrategy

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC",
    "DOGEUSDC", "ADAUSDC", "USDCUSDT",
    # New pairs
    "BNBUSDC", "AVAXUSDC", "LINKUSDC",
]

INTERVALS = ["15m", "1h", "4h", "1d"]

DAYS = 90
CAPITAL = 10_000
SKIP_STRATEGIES = ["pairs_trading", "multi_timeframe"]


def fetch_all_data(fetcher, symbols, intervals):
    """Pre-fetch all data into cache."""
    data_cache = {}
    end = datetime.utcnow()

    for symbol in symbols:
        for interval in intervals:
            days = 30 if interval == "15m" else DAYS
            start = end - timedelta(days=days)
            try:
                df = fetcher.fetch_klines_cached(symbol, interval,
                                                  start.strftime("%Y-%m-%d"),
                                                  end.strftime("%Y-%m-%d"))
                data_cache[(symbol, interval)] = df
                if not df.empty:
                    logger.info(f"  {symbol} {interval}: {len(df)} candles")
                else:
                    logger.warning(f"  {symbol} {interval}: no data")
            except Exception as e:
                logger.warning(f"  {symbol} {interval}: fetch failed: {e}")
                data_cache[(symbol, interval)] = pd.DataFrame()

    return data_cache


def run_single(strategy, df, symbol, interval):
    """Run a single backtest with new risk model."""
    engine = BacktestEngine(
        initial_capital=CAPITAL,
        commission_pct=0.0,
        risk_manager=RiskManager(risk_per_trade_pct=1.0, max_drawdown_pct=20.0),
    )
    result = engine.run(strategy, df.copy(), symbol, interval)
    return compute_metrics(result)


def build_ensembles():
    """Build ensemble strategies from top performers."""
    from strategies.bb_variants import BB_Squeeze, BB_Double, BB_MultiConf
    from strategies.smc_liquidity import SMCLiquidityStrategy
    from strategies.vwap_strategy import VWAPStrategy

    ensembles = {}

    # Ensemble 1: SMC + VWAP + BB_Squeeze (top 3 from original sweep)
    ensembles["Ens_SMC_VWAP_Squeeze"] = EnsembleStrategy(
        strategies=[SMCLiquidityStrategy(), VWAPStrategy(), BB_Squeeze()],
        weights=[2.0, 1.5, 1.0],
        params={"mode": "weighted", "threshold": 0.4},
    )

    # Ensemble 2: BB_Double + BB_MultiConf + BB_Squeeze (top 3 BB variants)
    ensembles["Ens_BB_Top3"] = EnsembleStrategy(
        strategies=[BB_Double(), BB_MultiConf(), BB_Squeeze()],
        params={"mode": "any"},
    )

    # Ensemble 3: SMC + BB_Double (best return + best risk-adjusted)
    ensembles["Ens_SMC_BBDouble"] = EnsembleStrategy(
        strategies=[SMCLiquidityStrategy(), BB_Double()],
        params={"mode": "any"},
    )

    return ensembles


def main():
    fetcher = DataFetcher()

    # ── 1. Fetch data ──────────────────────────────────────────────────────
    logger.info("Fetching data for all symbols and intervals...")
    data_cache = fetch_all_data(fetcher, SYMBOLS, INTERVALS)

    # ── 2. Build strategy roster ───────────────────────────────────────────
    base_strats = get_all_strategies()
    bb_strats = get_bb_variants()
    ensemble_strats = build_ensembles()

    # Merge all (skip problematic ones)
    all_strats = {}
    for name, cls in base_strats.items():
        if name not in SKIP_STRATEGIES:
            all_strats[name] = cls
    for name, cls in bb_strats.items():
        all_strats[name] = cls
    # Ensembles are instances, not classes
    ensemble_names = list(ensemble_strats.keys())

    total_combos = (len(all_strats) + len(ensemble_strats)) * len(SYMBOLS) * len(INTERVALS)
    logger.info(f"\nTotal combos: ({len(all_strats)} class + {len(ensemble_strats)} ensemble) "
                f"x {len(SYMBOLS)} pairs x {len(INTERVALS)} TFs = ~{total_combos}")

    # ── 3. Run all backtests ───────────────────────────────────────────────
    results = []
    done = 0
    errors = 0

    # Class-based strategies
    for strat_name in all_strats:
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                done += 1
                df = data_cache.get((symbol, interval))
                if df is None or df.empty:
                    continue
                try:
                    strategy = all_strats[strat_name]()
                    metrics = run_single(strategy, df, symbol, interval)
                    results.append({
                        "Strategy": strat_name,
                        "Symbol": symbol,
                        "Interval": interval,
                        "Trades": metrics["total_trades"],
                        "Win Rate %": metrics["win_rate"],
                        "Return %": metrics["total_return_pct"],
                        "Final $": metrics["final_equity"],
                        "Max DD %": metrics["max_drawdown_pct"],
                        "Sharpe": metrics["sharpe_ratio"],
                        "Sortino": metrics["sortino_ratio"],
                        "Profit Factor": metrics["profit_factor"],
                        "Avg Trade %": metrics["avg_trade_pnl_pct"],
                    })
                except Exception as e:
                    errors += 1
                if done % 100 == 0:
                    logger.info(f"  Progress: {done}/{total_combos} ({errors} errors)")

    # Ensemble strategies (instances)
    for ens_name, ens_instance in ensemble_strats.items():
        for symbol in SYMBOLS:
            for interval in INTERVALS:
                done += 1
                df = data_cache.get((symbol, interval))
                if df is None or df.empty:
                    continue
                try:
                    metrics = run_single(ens_instance, df, symbol, interval)
                    results.append({
                        "Strategy": ens_name,
                        "Symbol": symbol,
                        "Interval": interval,
                        "Trades": metrics["total_trades"],
                        "Win Rate %": metrics["win_rate"],
                        "Return %": metrics["total_return_pct"],
                        "Final $": metrics["final_equity"],
                        "Max DD %": metrics["max_drawdown_pct"],
                        "Sharpe": metrics["sharpe_ratio"],
                        "Sortino": metrics["sortino_ratio"],
                        "Profit Factor": metrics["profit_factor"],
                        "Avg Trade %": metrics["avg_trade_pnl_pct"],
                    })
                except Exception as e:
                    errors += 1
                if done % 100 == 0:
                    logger.info(f"  Progress: {done}/{total_combos}")

    logger.info(f"\nCompleted: {done} backtests, {errors} errors")

    if not results:
        logger.error("No results!")
        return

    df_r = pd.DataFrame(results)

    # Save
    csv_path = os.path.join(LOG_DIR, "master_sweep_v2.csv")
    df_r.to_csv(csv_path, index=False)
    logger.info(f"Saved to {csv_path}")

    # ── 4. Display ─────────────────────────────────────────────────────────
    W = 130
    print("\n" + "=" * W)
    print("  MASTER SWEEP v2 — New Risk Model (1% per trade, no daily halt)")
    n_strats = len(all_strats) + len(ensemble_strats)
    print(f"  {n_strats} strategies x {len(SYMBOLS)} pairs x {len(INTERVALS)} TFs")
    print(f"  Capital: ${CAPITAL:,} | Risk: 1%/trade | Max DD: 20%")
    print("=" * W)

    # Top 30 by Sharpe
    print("\n--- TOP 30 BY SHARPE ---")
    top = df_r.nlargest(30, "Sharpe")
    print(top[["Strategy","Symbol","Interval","Trades","Win Rate %",
               "Return %","Max DD %","Sharpe","Profit Factor"]].to_string(index=False))

    # Top 30 by Return
    print("\n--- TOP 30 BY RETURN % ---")
    top_ret = df_r.nlargest(30, "Return %")
    print(top_ret[["Strategy","Symbol","Interval","Trades","Win Rate %",
                    "Return %","Max DD %","Sharpe","Profit Factor"]].to_string(index=False))

    # Bottom 10
    print("\n--- BOTTOM 10 ---")
    bottom = df_r.nsmallest(10, "Return %")
    print(bottom[["Strategy","Symbol","Interval","Trades","Win Rate %",
                   "Return %","Max DD %","Sharpe"]].to_string(index=False))

    # By Strategy
    print("\n--- AVERAGE BY STRATEGY (sorted by Sharpe) ---")
    strat_agg = df_r.groupby("Strategy").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
        "Max DD %": "mean",
        "Profit Factor": "mean",
    }).round(3).sort_values("Sharpe", ascending=False)
    print(strat_agg.to_string())

    # By Symbol
    print("\n--- AVERAGE BY SYMBOL ---")
    sym_agg = df_r.groupby("Symbol").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
    }).round(3).sort_values("Return %", ascending=False)
    print(sym_agg.to_string())

    # By Timeframe
    print("\n--- AVERAGE BY TIMEFRAME ---")
    tf_agg = df_r.groupby("Interval").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
    }).round(3).sort_values("Return %", ascending=False)
    print(tf_agg.to_string())

    # Ensemble comparison
    print("\n--- ENSEMBLE STRATEGIES vs TOP INDIVIDUALS ---")
    ens_names = list(ensemble_strats.keys())
    top_individuals = ["smc_liquidity", "vwap", "BB_Squeeze", "BB_Double", "BB_MultiConf"]
    compare_names = ens_names + top_individuals
    compare = df_r[df_r["Strategy"].isin(compare_names)]
    compare_agg = compare.groupby("Strategy").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
        "Max DD %": "mean",
        "Profit Factor": "mean",
    }).round(3).sort_values("Sharpe", ascending=False)
    print(compare_agg.to_string())

    # Summary
    profitable = df_r[df_r["Return %"] > 0]
    print(f"\n--- SUMMARY ---")
    print(f"  Total backtests:   {len(df_r)}")
    print(f"  Profitable:        {len(profitable)} ({len(profitable)/len(df_r)*100:.1f}%)")
    print(f"  Avg Return:        {df_r['Return %'].mean():+.4f}%")
    print(f"  Avg Sharpe:        {df_r['Sharpe'].mean():.3f}")
    if not top.empty:
        b = top.iloc[0]
        print(f"  Best combo:        {b['Strategy']} on {b['Symbol']} {b['Interval']} "
              f"(Sharpe {b['Sharpe']:.3f})")
    if not bottom.empty:
        w = bottom.iloc[0]
        print(f"  Worst combo:       {w['Strategy']} on {w['Symbol']} {w['Interval']} "
              f"(Return {w['Return %']:+.3f}%)")
    print("=" * W)


if __name__ == "__main__":
    main()
