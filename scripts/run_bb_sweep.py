"""
BB Variants Sweep — test all 8 Bollinger Bands strategy variants
across multiple pairs and timeframes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from datetime import datetime, timedelta
from itertools import product

from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics
from strategies.bb_variants import get_bb_variants
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Configuration ──────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC",
    "DOGEUSDC", "ADAUSDC", "USDCUSDT",
]

INTERVALS = ["15m", "1h", "4h"]

DAYS = 90
CAPITAL = 10_000


def main():
    variants = get_bb_variants()
    fetcher = DataFetcher()

    combos = list(product(variants.keys(), SYMBOLS, INTERVALS))
    total = len(combos)

    logger.info(f"BB Variants Sweep: {len(variants)} variants x {len(SYMBOLS)} pairs x {len(INTERVALS)} TFs = {total}")
    logger.info(f"Variants: {', '.join(variants.keys())}")

    # Pre-load data
    data_cache = {}
    end = datetime.utcnow()
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            days = 30 if interval == "15m" else DAYS
            start = end - timedelta(days=days)
            df = fetcher.fetch_klines_cached(symbol, interval,
                                              start.strftime("%Y-%m-%d"),
                                              end.strftime("%Y-%m-%d"))
            data_cache[(symbol, interval)] = df
            if not df.empty:
                logger.info(f"  Loaded {symbol} {interval}: {len(df)} candles")

    logger.info(f"\nRunning {total} backtests...\n")

    results = []
    done = 0
    errors = 0

    for strat_name, symbol, interval in combos:
        done += 1
        df = data_cache.get((symbol, interval))

        if df is None or df.empty:
            errors += 1
            continue

        try:
            strategy = variants[strat_name]()
            engine = BacktestEngine(
                initial_capital=CAPITAL,
                commission_pct=0.0,
                risk_manager=RiskManager(),
            )
            result = engine.run(strategy, df.copy(), symbol, interval)
            metrics = compute_metrics(result)

            results.append({
                "Variant": strat_name,
                "Level": strat_name.replace("BB_", ""),
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
                "Best Trade": metrics.get("best_trade_pnl", 0),
                "Worst Trade": metrics.get("worst_trade_pnl", 0),
            })

            if done % 20 == 0:
                logger.info(f"  Progress: {done}/{total}")

        except Exception as e:
            errors += 1
            logger.warning(f"  FAIL [{strat_name}|{symbol}|{interval}]: {e}")

    logger.info(f"\nCompleted: {done} backtests, {errors} errors\n")

    if not results:
        logger.error("No results!")
        return

    df_r = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(LOG_DIR, "bb_variants_sweep.csv")
    df_r.to_csv(csv_path, index=False)
    logger.info(f"Saved to {csv_path}")

    # ── Display ────────────────────────────────────────────────────────────
    W = 120
    print("\n" + "=" * W)
    print("  BOLLINGER BANDS VARIANTS SWEEP")
    print(f"  {len(variants)} variants x {len(SYMBOLS)} pairs x {len(INTERVALS)} TFs = {total} tests")
    print(f"  Capital: ${CAPITAL:,} | Commission: 0% | Period: {DAYS}d (15m: 30d)")
    print("=" * W)

    # Top 20 by Sharpe
    print("\n--- TOP 20 BY SHARPE RATIO ---")
    top = df_r.nlargest(20, "Sharpe")
    print(top[["Variant","Symbol","Interval","Trades","Win Rate %",
               "Return %","Max DD %","Sharpe","Profit Factor"]].to_string(index=False))

    # Top 20 by Return
    print("\n--- TOP 20 BY RETURN % ---")
    top_ret = df_r.nlargest(20, "Return %")
    print(top_ret[["Variant","Symbol","Interval","Trades","Win Rate %",
                    "Return %","Max DD %","Sharpe","Profit Factor"]].to_string(index=False))

    # Bottom 10
    print("\n--- BOTTOM 10 (WORST) ---")
    bottom = df_r.nsmallest(10, "Return %")
    print(bottom[["Variant","Symbol","Interval","Trades","Win Rate %",
                   "Return %","Max DD %","Sharpe"]].to_string(index=False))

    # ── By Variant (the key comparison) ────────────────────────────────────
    print("\n--- AVERAGE PERFORMANCE BY BB VARIANT (simple -> complex) ---")
    variant_order = ["BB_Naive","BB_RSI","BB_Trend","BB_Volume","BB_MACD",
                     "BB_Double","BB_Squeeze","BB_MultiConf"]
    agg = df_r.groupby("Variant").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Sortino": "mean",
        "Trades": "mean",
        "Max DD %": "mean",
        "Profit Factor": "mean",
    }).round(3)
    # Reorder
    agg = agg.reindex([v for v in variant_order if v in agg.index])
    print(agg.to_string())

    # ── By Symbol ──────────────────────────────────────────────────────────
    print("\n--- AVERAGE PERFORMANCE BY SYMBOL ---")
    sym_agg = df_r.groupby("Symbol").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
    }).round(3).sort_values("Return %", ascending=False)
    print(sym_agg.to_string())

    # ── By Timeframe ───────────────────────────────────────────────────────
    print("\n--- AVERAGE PERFORMANCE BY TIMEFRAME ---")
    tf_agg = df_r.groupby("Interval").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
    }).round(3).sort_values("Return %", ascending=False)
    print(tf_agg.to_string())

    # ── Best combo per variant ─────────────────────────────────────────────
    print("\n--- BEST COMBO PER VARIANT ---")
    for v in variant_order:
        sub = df_r[df_r["Variant"] == v]
        if sub.empty:
            continue
        best = sub.loc[sub["Sharpe"].idxmax()]
        print(f"  {v:15s} -> {best['Symbol']:10s} {best['Interval']:4s} | "
              f"Sharpe {best['Sharpe']:+7.3f} | Return {best['Return %']:+.3f}% | "
              f"WR {best['Win Rate %']:.1f}% | {int(best['Trades'])} trades")

    # ── Summary ────────────────────────────────────────────────────────────
    profitable = df_r[df_r["Return %"] > 0]
    print(f"\n--- SUMMARY ---")
    print(f"  Total backtests:  {len(df_r)}")
    print(f"  Profitable:       {len(profitable)} ({len(profitable)/len(df_r)*100:.1f}%)")
    print(f"  Avg Return:       {df_r['Return %'].mean():+.4f}%")
    print(f"  Avg Sharpe:       {df_r['Sharpe'].mean():.3f}")

    if not top.empty:
        b = top.iloc[0]
        print(f"  Best overall:     {b['Variant']} on {b['Symbol']} {b['Interval']} "
              f"(Sharpe {b['Sharpe']:.3f}, Return {b['Return %']:+.3f}%)")
    if not bottom.empty:
        w = bottom.iloc[0]
        print(f"  Worst overall:    {w['Variant']} on {w['Symbol']} {w['Interval']} "
              f"(Return {w['Return %']:+.3f}%)")

    print("=" * W)


if __name__ == "__main__":
    main()
