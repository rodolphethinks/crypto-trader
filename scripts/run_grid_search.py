"""
Grid search parameter optimizer for top-performing strategies.
Tests parameter combinations and ranks by Sharpe ratio.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product as iterproduct

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


def grid_search(strategy_class, param_grid: dict, symbol: str, interval: str,
                df: pd.DataFrame) -> pd.DataFrame:
    """
    Run a grid search over param_grid for a strategy on given data.
    Returns DataFrame of results sorted by Sharpe.
    """
    # Build all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(iterproduct(*values))

    logger.info(f"Grid search: {strategy_class.__name__} on {symbol} {interval} "
                f"— {len(combos)} param combos")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_class(params=params)
            engine = BacktestEngine(
                initial_capital=CAPITAL,
                commission_pct=0.0,
                risk_manager=RiskManager(),
            )
            result = engine.run(strategy, df.copy(), symbol, interval)
            metrics = compute_metrics(result)

            row = {**params,
                   "trades": metrics["total_trades"],
                   "win_rate": metrics["win_rate"],
                   "return_pct": metrics["total_return_pct"],
                   "sharpe": metrics["sharpe_ratio"],
                   "sortino": metrics["sortino_ratio"],
                   "max_dd": metrics["max_drawdown_pct"],
                   "profit_factor": metrics["profit_factor"]}
            results.append(row)
        except Exception as e:
            logger.warning(f"  Failed combo {i}: {e}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(combos)} done")

    df_r = pd.DataFrame(results)
    if not df_r.empty:
        df_r = df_r.sort_values("sharpe", ascending=False)
    return df_r


def main():
    from strategies.bb_variants import BB_Squeeze, BB_Double, BB_MultiConf, BB_MACD, BB_Trend
    from strategies.bollinger_bands import BollingerBandsStrategy
    from strategies.smc_liquidity import SMCLiquidityStrategy
    from strategies.vwap_strategy import VWAPStrategy

    fetcher = DataFetcher()
    end = datetime.utcnow()

    # Load best-performing data combos from sweep
    datasets = {
        ("DOGEUSDC", "4h"): 90,
        ("AVAXUSDC", "4h"): 90,
        ("BNBUSDC", "4h"): 90,
        ("XRPUSDC", "4h"): 90,
        ("SOLUSDC", "4h"): 90,
        ("XRPUSDC", "1d"): 90,
        ("LINKUSDC", "1d"): 90,
    }

    data_cache = {}
    for (sym, intv), days in datasets.items():
        start = end - timedelta(days=days)
        df = fetcher.fetch_klines_cached(sym, intv, start.strftime("%Y-%m-%d"),
                                          end.strftime("%Y-%m-%d"))
        data_cache[(sym, intv)] = df
        logger.info(f"Loaded {sym} {intv}: {len(df)} candles")

    all_results = []

    # ── 1) BB_Squeeze optimization ──
    logger.info("\n=== Optimizing BB_Squeeze ===")
    grid_bb_squeeze = {
        "bb_period": [15, 20, 25],
        "bb_std": [1.8, 2.0, 2.2],
        "kc_mult": [1.3, 1.5, 1.8],
        "sl_atr_mult": [1.5, 2.0, 2.5],
        "rr_ratio": [2.0, 2.5, 3.0],
        "min_squeeze_bars": [2, 3, 5],
    }
    for (sym, intv), df in data_cache.items():
        if intv == "4h":
            res = grid_search(BB_Squeeze, grid_bb_squeeze, sym, intv, df)
            if not res.empty:
                res["strategy"] = "BB_Squeeze"
                res["symbol"] = sym
                res["interval"] = intv
                all_results.append(res.head(5))

    # ── 2) BB_Double optimization ──
    logger.info("\n=== Optimizing BB_Double ===")
    grid_bb_double = {
        "bb_period": [15, 20, 25],
        "bb_inner_std": [1.0, 1.2, 1.5],
        "bb_outer_std": [2.0, 2.5, 3.0],
        "rsi_oversold": [20, 25, 30],
        "rsi_overbought": [70, 75, 80],
        "sl_atr_mult": [1.5, 2.0, 2.5],
    }
    for (sym, intv), df in data_cache.items():
        if intv == "4h":
            res = grid_search(BB_Double, grid_bb_double, sym, intv, df)
            if not res.empty:
                res["strategy"] = "BB_Double"
                res["symbol"] = sym
                res["interval"] = intv
                all_results.append(res.head(5))

    # ── 3) BB_Trend optimization ──
    logger.info("\n=== Optimizing BB_Trend ===")
    grid_bb_trend = {
        "bb_period": [15, 20, 25],
        "bb_std": [1.8, 2.0, 2.2],
        "rsi_oversold": [35, 40],
        "rsi_overbought": [60, 65],
        "bb_pct_buy": [0.10, 0.15, 0.20],
        "bb_pct_sell": [0.80, 0.85, 0.90],
        "ema_period": [30, 50],
    }
    for (sym, intv), df in data_cache.items():
        if intv == "4h":
            res = grid_search(BB_Trend, grid_bb_trend, sym, intv, df)
            if not res.empty:
                res["strategy"] = "BB_Trend"
                res["symbol"] = sym
                res["interval"] = intv
                all_results.append(res.head(5))

    # ── 4) BB_MACD optimization ──
    logger.info("\n=== Optimizing BB_MACD ===")
    grid_bb_macd = {
        "bb_period": [15, 20, 25],
        "bb_std": [1.8, 2.0, 2.2],
        "bb_pct_buy": [0.10, 0.15, 0.20],
        "bb_pct_sell": [0.80, 0.85, 0.90],
        "rsi_oversold": [35, 40],
        "rsi_overbought": [60, 65],
        "sl_atr_mult": [1.5, 2.0],
        "rr_ratio": [2.0, 2.5],
    }
    for (sym, intv), df in data_cache.items():
        if intv == "4h":
            res = grid_search(BB_MACD, grid_bb_macd, sym, intv, df)
            if not res.empty:
                res["strategy"] = "BB_MACD"
                res["symbol"] = sym
                res["interval"] = intv
                all_results.append(res.head(5))

    # Combine and save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        csv_path = os.path.join(LOG_DIR, "grid_search_results.csv")
        combined.to_csv(csv_path, index=False)
        logger.info(f"\nSaved {len(combined)} top results to {csv_path}")

        print("\n" + "=" * 120)
        print("  GRID SEARCH — TOP 30 OVERALL (by Sharpe)")
        print("=" * 120)
        top = combined.nlargest(30, "sharpe")
        display_cols = ["strategy", "symbol", "interval", "trades", "win_rate",
                        "return_pct", "sharpe", "sortino", "max_dd", "profit_factor"]
        # Add param columns that exist
        param_cols = [c for c in top.columns if c not in display_cols + ["level"]]
        print(top[display_cols + param_cols[:6]].to_string(index=False))
        print("=" * 120)
    else:
        logger.error("No results!")


if __name__ == "__main__":
    main()
