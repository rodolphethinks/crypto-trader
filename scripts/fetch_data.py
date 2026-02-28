"""
Fetch historical kline data for all zero-fee pairs.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --symbols BTCUSDC ETHUSDC --interval 1h --days 90
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.pairs import ALL_NOFEE_PAIRS, MAJOR_PAIRS
from config.settings import KLINE_INTERVALS, LOG_DIR
from data.fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "fetch_data.log")),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch MEXC kline data")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to fetch (default: major pairs)")
    parser.add_argument("--all", action="store_true",
                        help="Fetch all zero-fee pairs")
    parser.add_argument("--interval", default="1h",
                        help="Kline interval (default: 1h)")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of days of history (default: 90)")
    args = parser.parse_args()

    if args.all:
        symbols = ALL_NOFEE_PAIRS
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = MAJOR_PAIRS

    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(days=args.days)

    fetcher = DataFetcher()
    logger.info(f"Fetching {len(symbols)} symbols, interval={args.interval}, "
                f"period={start.date()} → {end.date()}")

    results = fetcher.fetch_multiple_pairs(
        symbols=symbols,
        interval=args.interval,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )

    total_candles = sum(len(df) for df in results.values())
    logger.info(f"Done! Fetched {len(results)} symbols, {total_candles} total candles")


if __name__ == "__main__":
    main()
