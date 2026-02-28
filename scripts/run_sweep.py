"""
Comprehensive backtest sweep — runs all strategies across multiple pairs and timeframes.
Outputs a full CSV and prints a ranked summary.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from datetime import datetime, timedelta
from itertools import product

from backtesting.engine import BacktestEngine
from backtesting.runner import get_all_strategies
from backtesting.metrics import compute_metrics
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from config.settings import LOG_DIR

logging.basicConfig(
    level=logging.WARNING,  # Suppress per-trade noise
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

# Skip strategies that need special data (pairs_trading needs 2 symbols)
SKIP_STRATEGIES = ["pairs_trading", "multi_timeframe"]

DAYS = 90
CAPITAL = 10_000

def main():
    all_strats = get_all_strategies()
    strat_names = [s for s in all_strats if s not in SKIP_STRATEGIES]
    fetcher = DataFetcher()
    
    combos = list(product(strat_names, SYMBOLS, INTERVALS))
    total = len(combos)
    
    logger.info(f"Starting sweep: {len(strat_names)} strategies x {len(SYMBOLS)} pairs x {len(INTERVALS)} timeframes = {total} backtests")
    
    results = []
    done = 0
    errors = 0
    
    # Pre-load all data
    data_cache = {}
    end = datetime.utcnow()
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            days = 30 if interval == "15m" else DAYS
            start = end - timedelta(days=days)
            key = (symbol, interval)
            df = fetcher.fetch_klines_cached(symbol, interval,
                                              start.strftime("%Y-%m-%d"),
                                              end.strftime("%Y-%m-%d"))
            data_cache[key] = df
            if not df.empty:
                logger.info(f"  Loaded {symbol} {interval}: {len(df)} candles")
            else:
                logger.warning(f"  No data for {symbol} {interval}")

    logger.info(f"\nRunning {total} backtests...\n")
    
    for strat_name, symbol, interval in combos:
        done += 1
        df = data_cache.get((symbol, interval))
        
        if df is None or df.empty:
            errors += 1
            continue
        
        try:
            strategy = all_strats[strat_name]()
            engine = BacktestEngine(
                initial_capital=CAPITAL,
                commission_pct=0.0,
                risk_manager=RiskManager(),
            )
            result = engine.run(strategy, df.copy(), symbol, interval)
            metrics = compute_metrics(result)
            
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
                "Best Trade": metrics.get("best_trade_pnl", 0),
                "Worst Trade": metrics.get("worst_trade_pnl", 0),
            })
            
            if done % 25 == 0:
                logger.info(f"  Progress: {done}/{total} ({errors} errors)")
                
        except Exception as e:
            errors += 1
            logger.warning(f"  FAIL [{strat_name}|{symbol}|{interval}]: {e}")
    
    logger.info(f"\nCompleted: {done} backtests, {errors} errors\n")
    
    if not results:
        logger.error("No results to display!")
        return
    
    # Build DataFrame
    df_results = pd.DataFrame(results)
    
    # Save full results
    csv_path = os.path.join(LOG_DIR, "full_sweep_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Full results saved to {csv_path}")
    
    # ── Print Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("  COMPREHENSIVE BACKTEST SWEEP RESULTS")
    print(f"  {len(strat_names)} strategies x {len(SYMBOLS)} pairs x {len(INTERVALS)} timeframes = {total} tests")
    print(f"  Capital: ${CAPITAL:,} | Commission: 0% (zero-fee pairs) | Period: {DAYS} days")
    print("=" * 120)
    
    # Top 20 by Sharpe
    print("\n--- TOP 20 BY SHARPE RATIO ---")
    top_sharpe = df_results.nlargest(20, "Sharpe")
    print(top_sharpe.to_string(index=False))
    
    # Top 20 by Return
    print("\n--- TOP 20 BY RETURN % ---")
    top_return = df_results.nlargest(20, "Return %")
    print(top_return.to_string(index=False))
    
    # Bottom 10 (worst)
    print("\n--- BOTTOM 10 (WORST PERFORMING) ---")
    bottom = df_results.nsmallest(10, "Return %")
    print(bottom.to_string(index=False))
    
    # ── Aggregate by Strategy ──────────────────────────────────────────────
    print("\n--- AVERAGE PERFORMANCE BY STRATEGY ---")
    strat_agg = df_results.groupby("Strategy").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
        "Max DD %": "mean",
        "Profit Factor": "mean",
    }).round(3).sort_values("Sharpe", ascending=False)
    print(strat_agg.to_string())
    
    # ── Aggregate by Symbol ────────────────────────────────────────────────
    print("\n--- AVERAGE PERFORMANCE BY SYMBOL ---")
    sym_agg = df_results.groupby("Symbol").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
    }).round(3).sort_values("Return %", ascending=False)
    print(sym_agg.to_string())
    
    # ── Aggregate by Interval ──────────────────────────────────────────────
    print("\n--- AVERAGE PERFORMANCE BY TIMEFRAME ---")
    tf_agg = df_results.groupby("Interval").agg({
        "Return %": "mean",
        "Win Rate %": "mean",
        "Sharpe": "mean",
        "Trades": "mean",
    }).round(3).sort_values("Return %", ascending=False)
    print(tf_agg.to_string())
    
    # ── Profitable vs Unprofitable ─────────────────────────────────────────
    profitable = df_results[df_results["Return %"] > 0]
    unprofitable = df_results[df_results["Return %"] <= 0]
    print(f"\n--- SUMMARY ---")
    print(f"  Total backtests:  {len(df_results)}")
    print(f"  Profitable:       {len(profitable)} ({len(profitable)/len(df_results)*100:.1f}%)")
    print(f"  Unprofitable:     {len(unprofitable)} ({len(unprofitable)/len(df_results)*100:.1f}%)")
    print(f"  Avg Return:       {df_results['Return %'].mean():+.3f}%")
    print(f"  Avg Sharpe:       {df_results['Sharpe'].mean():.3f}")
    print(f"  Best combo:       {top_sharpe.iloc[0]['Strategy']} on {top_sharpe.iloc[0]['Symbol']} {top_sharpe.iloc[0]['Interval']} (Sharpe {top_sharpe.iloc[0]['Sharpe']:.3f})")
    print(f"  Worst combo:      {bottom.iloc[0]['Strategy']} on {bottom.iloc[0]['Symbol']} {bottom.iloc[0]['Interval']} (Return {bottom.iloc[0]['Return %']:+.3f}%)")
    print("=" * 120)

if __name__ == "__main__":
    main()
