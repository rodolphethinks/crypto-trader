"""
Strategy parameter optimization via grid search.

Usage:
    python scripts/optimize.py --strategy trend_following --symbol BTCUSDC --interval 1h
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import LOG_DIR, DEFAULT_INITIAL_CAPITAL
from backtesting.runner import BacktestRunner, get_all_strategies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "optimize.log")),
    ],
)
logger = logging.getLogger(__name__)

# ── Predefined parameter grids for each strategy ──────────────────────────────
PARAM_GRIDS = {
    "trend_following": {
        "ema_fast": [5, 8, 13],
        "ema_slow": [18, 21, 34],
        "adx_period": [14, 20],
        "adx_threshold": [20, 25, 30],
    },
    "mean_reversion": {
        "bb_period": [15, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_period": [10, 14, 20],
    },
    "breakout": {
        "donchian_period": [15, 20, 30],
        "squeeze_length": [15, 20],
        "volume_mult": [1.2, 1.5, 2.0],
    },
    "scalping": {
        "ema_fast": [3, 5],
        "ema_slow": [8, 13],
        "stoch_rsi_period": [10, 14],
    },
    "momentum": {
        "rsi_period": [10, 14, 20],
        "rsi_overbought": [65, 70, 75],
        "rsi_oversold": [25, 30, 35],
    },
    "bollinger_bands": {
        "bb_period": [15, 20, 30],
        "bb_std": [1.5, 2.0, 2.5, 3.0],
    },
    "macd_divergence": {
        "macd_fast": [8, 12, 16],
        "macd_slow": [20, 26, 30],
        "macd_signal": [7, 9, 12],
    },
    "vwap": {
        "dev_mult": [1.0, 1.5, 2.0],
        "rsi_period": [10, 14],
    },
    "ichimoku": {
        "conv_period": [7, 9, 12],
        "base_period": [22, 26, 30],
    },
    "grid_trading": {
        "num_grids": [5, 8, 10, 15],
        "grid_range_atr_mult": [1.0, 1.5, 2.0],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--strategy", required=True, help="Strategy to optimize")
    parser.add_argument("--symbol", default="BTCUSDC", help="Trading pair")
    parser.add_argument("--interval", default="1h", help="Kline interval")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe", "return", "profit_factor", "win_rate"],
                        help="Optimization metric")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    args = parser.parse_args()

    all_strats = get_all_strategies()
    if args.strategy not in all_strats:
        logger.error(f"Unknown strategy: {args.strategy}")
        return

    param_grid = PARAM_GRIDS.get(args.strategy, {})
    if not param_grid:
        logger.error(f"No parameter grid defined for {args.strategy}")
        logger.info(f"Available grids: {list(PARAM_GRIDS.keys())}")
        return

    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(days=args.days)

    runner = BacktestRunner(initial_capital=args.capital, commission_pct=0.0)

    from itertools import product
    total = 1
    for vals in param_grid.values():
        total *= len(vals)

    logger.info(f"Optimizing {args.strategy} on {args.symbol} ({args.interval}): "
                f"{total} combinations, metric={args.metric}")

    results_df = runner.run_parameter_optimization(
        strategy_class=all_strats[args.strategy],
        param_grid=param_grid,
        symbol=args.symbol,
        interval=args.interval,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        metric=args.metric,
    )

    if results_df.empty:
        logger.error("No optimization results")
        return

    # Display top 10
    print("\n" + "=" * 80)
    print(f"  OPTIMIZATION RESULTS — {args.strategy.upper()} on {args.symbol} ({args.interval})")
    print(f"  Metric: {args.metric} | {total} combinations tested")
    print("=" * 80)
    print(results_df.head(10).to_string(index=False))
    print("=" * 80)

    # Save
    outfile = os.path.join(LOG_DIR, f"optimize_{args.strategy}_{args.symbol}_{args.interval}.csv")
    results_df.to_csv(outfile, index=False)
    logger.info(f"Full results saved to {outfile}")

    # Best params
    best = results_df.iloc[0]
    param_cols = [c for c in results_df.columns if c not in
                  ["return_pct", "win_rate", "sharpe", "sortino", "profit_factor", "max_dd", "trades"]]
    best_params = {c: best[c] for c in param_cols}
    print(f"\nBest parameters: {best_params}")


if __name__ == "__main__":
    main()
