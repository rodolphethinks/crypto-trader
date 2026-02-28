"""
Master Sweep V3 — comprehensive backtest of all new approaches.

Tests:
- 12 NEW strategies (3 ML + 4 HF + 5 alt-alpha)
- 15 pairs (top liquidity USDC zero-fee pairs)
- 6 timeframes (1m, 5m, 15m, 1h, 4h, 1d)
- 365-day backtest window (1 year of data)

Smart routing:
- HF strategies (MicroMomentum, MeanReversion_HF, OrderFlow_Imbalance, Breakout_Micro)
  → only run on 1m, 5m, 15m
- ML strategies (XGBoost, LightGBM, LSTM)
  → run on 1h, 4h, 1d (need enough bars for training, but not too many)
- Alt-alpha strategies
  → run on 1h, 4h, 1d
- For 1m data: limit to 30 days (43,200 bars) to avoid memory issues
- For 5m data: limit to 90 days
- For 15m+: full 365 days

Parallelism: sequential to respect API rate limits on data fetch.
"""
import sys
import os
import time
import logging
import traceback
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.runner import get_v3_strategies, BacktestRunner
from backtesting.engine import BacktestEngine, BacktestResult
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR, DEFAULT_INITIAL_CAPITAL

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Top-liquidity zero-fee USDC pairs
PAIRS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "DOGEUSDC",
    "BNBUSDC", "AVAXUSDC", "LINKUSDC", "ADAUSDC", "SUIUSDC",
    "PEPEUSDC", "NEARUSDC", "LTCUSDC", "INJUSDC", "APTUSDC",
]

# Strategy → allowed timeframes
HF_STRATEGIES = {"MicroMomentum", "MeanReversion_HF", "OrderFlow_Imbalance", "Breakout_Micro"}
ML_STRATEGIES = {"XGBoost", "LightGBM", "LSTM"}
ALT_STRATEGIES = {"RegimeAdaptive", "CrossTF_Momentum", "Vol_Breakout", "StatArb", "AdaptiveTrend"}

HF_TIMEFRAMES = ["5m", "15m"]        # 1m skipped (too much data to fetch)
ML_TIMEFRAMES = ["1h", "4h"]          # ML needs enough bars but not too many
ALT_TIMEFRAMES = ["1h", "4h", "1d"]

# Data windows (days of history to fetch)
DATA_WINDOWS = {
    "5m": 90,
    "15m": 180,
    "1h": 365,
    "4h": 365,
    "1d": 365,
}


def get_allowed_timeframes(strategy_name: str) -> list:
    if strategy_name in HF_STRATEGIES:
        return HF_TIMEFRAMES
    elif strategy_name in ML_STRATEGIES:
        return ML_TIMEFRAMES
    else:
        return ALT_TIMEFRAMES


def run_sweep():
    print("=" * 80)
    print("  MASTER SWEEP V3 — ML + High-Frequency + Alternative Alpha")
    print("=" * 80)

    strategies = get_v3_strategies()
    fetcher = DataFetcher()

    # Calculate total combos
    total = 0
    combos = []
    for name in strategies:
        tfs = get_allowed_timeframes(name)
        for pair in PAIRS:
            for tf in tfs:
                combos.append((name, pair, tf))
                total += 1

    print(f"\nStrategies: {len(strategies)}")
    print(f"Pairs: {len(PAIRS)}")
    print(f"Total combos: {total}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    results = []
    errors = []
    data_cache = {}

    for idx, (strat_name, pair, tf) in enumerate(combos, 1):
        elapsed_pct = idx / total * 100
        print(f"\r[{idx}/{total}] ({elapsed_pct:.0f}%) {strat_name:25s} {pair:12s} {tf:5s}", end="", flush=True)

        try:
            # Fetch data (cached per pair+tf)
            cache_key = f"{pair}_{tf}"
            if cache_key not in data_cache:
                days = DATA_WINDOWS.get(tf, 365)
                start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                # Try cache first, but verify it covers the needed window
                df = fetcher.fetch_klines_cached(pair, tf, start_date=start_date)
                if df is not None and not df.empty:
                    # Check if cached data covers enough of the window
                    needed_start = datetime.utcnow() - timedelta(days=days)
                    data_start = df.index[0].to_pydatetime().replace(tzinfo=None)
                    gap_days = (data_start - needed_start).days
                    if gap_days > 30:  # cache is missing >30 days
                        print(f" [refetching {pair} {tf}]", end="", flush=True)
                        df = fetcher.fetch_klines(pair, tf, start_date=start_date)
                    data_cache[cache_key] = df if df is not None and not df.empty else None
                else:
                    df = fetcher.fetch_klines(pair, tf, start_date=start_date)
                    data_cache[cache_key] = df if df is not None and not df.empty else None

            df = data_cache.get(cache_key)
            if df is None or df.empty:
                results.append({
                    "Strategy": strat_name, "Symbol": pair, "Interval": tf,
                    "Trades": 0, "Win Rate %": 0, "Return %": 0,
                    "Max DD %": 0, "Sharpe": 0, "Profit Factor": 0,
                    "Status": "NO DATA",
                })
                continue

            # Create strategy instance
            strat_class = strategies[strat_name]
            strategy = strat_class()

            # Run backtest
            engine = BacktestEngine(
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                commission_pct=0.0,
                slippage_pct=0.0005,
                risk_manager=RiskManager(),
            )
            result = engine.run(strategy, df, pair, tf)

            results.append({
                "Strategy": strat_name,
                "Symbol": pair,
                "Interval": tf,
                "Trades": result.total_trades,
                "Win Rate %": round(result.win_rate, 2),
                "Return %": round(result.total_return, 4),
                "Max DD %": round(result.max_drawdown, 4),
                "Sharpe": round(result.sharpe_ratio, 4),
                "Profit Factor": round(result.profit_factor, 4),
                "Avg Trade $": round(result.avg_trade_pnl, 4),
                "Status": "OK",
            })

        except Exception as e:
            err_msg = f"{strat_name} {pair} {tf}: {str(e)[:100]}"
            errors.append(err_msg)
            results.append({
                "Strategy": strat_name, "Symbol": pair, "Interval": tf,
                "Trades": 0, "Win Rate %": 0, "Return %": 0,
                "Max DD %": 0, "Sharpe": 0, "Profit Factor": 0,
                "Avg Trade $": 0, "Status": f"ERROR: {str(e)[:50]}",
            })

    # ── Save results ──────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    csv_path = LOG_DIR / "master_sweep_v3.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SWEEP V3 SUMMARY")
    print("=" * 80)

    ok = df_results[df_results["Status"] == "OK"]
    no_data = df_results[df_results["Status"] == "NO DATA"]
    err = df_results[~df_results["Status"].isin(["OK", "NO DATA"])]

    print(f"  Total combos:     {len(df_results)}")
    print(f"  Successful:       {len(ok)}")
    print(f"  No data:          {len(no_data)}")
    print(f"  Errors:           {len(err)}")

    if len(ok) > 0:
        profitable = ok[ok["Return %"] > 0]
        print(f"\n  Profitable:       {len(profitable)} / {len(ok)} ({len(profitable)/len(ok)*100:.1f}%)")
        print(f"  Avg Return:       {ok['Return %'].mean():.4f}%")
        print(f"  Median Return:    {ok['Return %'].median():.4f}%")
        print(f"  Avg Trades:       {ok['Trades'].mean():.1f}")

        # Top 30 by Sharpe
        ok_with_trades = ok[ok["Trades"] >= 5].copy()
        if len(ok_with_trades) > 0:
            top30 = ok_with_trades.nlargest(30, "Sharpe")
            print(f"\n  TOP 30 BY SHARPE (min 5 trades):")
            print("-" * 100)
            print(f"  {'Strategy':25s} {'Symbol':12s} {'TF':5s} {'Trades':>7s} {'Win%':>7s} {'Return%':>10s} {'DD%':>8s} {'Sharpe':>10s} {'PF':>8s}")
            print("-" * 100)
            for _, row in top30.iterrows():
                print(f"  {row['Strategy']:25s} {row['Symbol']:12s} {row['Interval']:5s} "
                      f"{row['Trades']:7d} {row['Win Rate %']:7.1f} {row['Return %']:10.4f} "
                      f"{row['Max DD %']:8.4f} {row['Sharpe']:10.4f} {row['Profit Factor']:8.4f}")

        # Bottom 10
        if len(ok_with_trades) > 0:
            bottom10 = ok_with_trades.nsmallest(10, "Return %")
            print(f"\n  BOTTOM 10 BY RETURN:")
            print("-" * 100)
            for _, row in bottom10.iterrows():
                print(f"  {row['Strategy']:25s} {row['Symbol']:12s} {row['Interval']:5s} "
                      f"{row['Trades']:7d} {row['Win Rate %']:7.1f} {row['Return %']:10.4f} "
                      f"{row['Max DD %']:8.4f} {row['Sharpe']:10.4f}")

        # By strategy type
        print(f"\n  BY STRATEGY CATEGORY:")
        print("-" * 60)
        for cat_name, cat_strats in [("ML", ML_STRATEGIES), ("HF", HF_STRATEGIES), ("Alt-Alpha", ALT_STRATEGIES)]:
            cat_data = ok[ok["Strategy"].isin(cat_strats)]
            if len(cat_data) > 0:
                cat_profitable = cat_data[cat_data["Return %"] > 0]
                print(f"  {cat_name:12s}: {len(cat_profitable):3d}/{len(cat_data):3d} profitable "
                      f"| Avg return: {cat_data['Return %'].mean():+.4f}% "
                      f"| Avg trades: {cat_data['Trades'].mean():.0f} "
                      f"| Best Sharpe: {cat_data['Sharpe'].max():.2f}")

        # By strategy
        print(f"\n  BY INDIVIDUAL STRATEGY (avg across all pairs/TF):")
        print("-" * 80)
        strat_summary = ok.groupby("Strategy").agg({
            "Return %": "mean",
            "Sharpe": "mean",
            "Trades": "mean",
            "Win Rate %": "mean",
        }).sort_values("Sharpe", ascending=False)
        for name, row in strat_summary.iterrows():
            print(f"  {name:25s} | Return: {row['Return %']:+.4f}% | Sharpe: {row['Sharpe']:+.4f} "
                  f"| Trades: {row['Trades']:.0f} | Win%: {row['Win Rate %']:.1f}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_sweep()
