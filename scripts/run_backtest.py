"""
Run backtests across strategies, symbols, and timeframes.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --strategy trend_following --symbol BTCUSDC --interval 1h
    python scripts/run_backtest.py --sweep --symbols BTCUSDC ETHUSDC --intervals 1h 4h
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pairs import MAJOR_PAIRS, STABLECOIN_PAIRS
from config.settings import LOG_DIR, DEFAULT_INITIAL_CAPITAL
from backtesting.runner import BacktestRunner, get_all_strategies
from backtesting.metrics import compute_metrics, format_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "backtest.log")),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument("--strategy", default=None,
                        help="Strategy name (omit for all)")
    parser.add_argument("--symbol", default="BTCUSDC",
                        help="Trading pair")
    parser.add_argument("--interval", default="1h",
                        help="Kline interval")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
                        help="Initial capital")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep across multiple combos")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols for sweep")
    parser.add_argument("--intervals", nargs="+", default=None,
                        help="Intervals for sweep")
    args = parser.parse_args()

    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(days=args.days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    runner = BacktestRunner(initial_capital=args.capital, commission_pct=0.0)

    if args.sweep:
        symbols = args.symbols or (MAJOR_PAIRS[:5] + STABLECOIN_PAIRS)
        intervals = args.intervals or ["1h", "4h"]
        strategies = [args.strategy] if args.strategy else None

        comp = runner.run_sweep(
            strategies=strategies,
            symbols=symbols,
            intervals=intervals,
            start_date=start_str,
            end_date=end_str,
        )

        if not comp.empty:
            comp.to_csv(os.path.join(LOG_DIR, "sweep_results.csv"), index=False)
            logger.info(f"Results saved to {LOG_DIR}/sweep_results.csv")

    else:
        all_strats = get_all_strategies()
        strat_names = [args.strategy] if args.strategy else list(all_strats.keys())

        for name in strat_names:
            if name not in all_strats:
                logger.error(f"Unknown strategy: {name}")
                continue

            strategy = all_strats[name]()
            result = runner.run_single(
                strategy, args.symbol, args.interval,
                start_str, end_str,
            )

            metrics = compute_metrics(result)
            print(format_report(metrics))


if __name__ == "__main__":
    main()
