"""
Run live/paper trading with a selected strategy.

Usage:
    python scripts/run_live.py --strategy trend_following --symbol BTCUSDC --interval 5m
    python scripts/run_live.py --strategy mean_reversion --symbol USDCUSDT --mode paper
"""
import sys
import os
import time
import argparse
import logging
import signal as signal_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import LOG_DIR, TRADING_MODE
from data.fetcher import DataFetcher
from trading.executor import TradeExecutor
from trading.portfolio import Portfolio
from trading.order_manager import OrderManager
from risk.manager import RiskManager
from backtesting.runner import get_all_strategies
from strategies.base import Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "live_trading.log")),
    ],
)
logger = logging.getLogger(__name__)

running = True


def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received — closing gracefully...")
    running = False


signal_module.signal(signal_module.SIGINT, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Live/paper trading")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--symbol", required=True, help="Trading pair")
    parser.add_argument("--interval", default="5m", help="Kline interval")
    parser.add_argument("--mode", default=None, help="Trading mode: paper or live")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    parser.add_argument("--lookback", type=int, default=200,
                        help="Number of candles to load for indicator warmup")
    args = parser.parse_args()

    mode = args.mode or TRADING_MODE
    all_strats = get_all_strategies()

    if args.strategy not in all_strats:
        logger.error(f"Unknown strategy: {args.strategy}")
        return

    strategy = all_strats[args.strategy]()
    fetcher = DataFetcher()
    executor = TradeExecutor(mode=mode)
    portfolio = Portfolio(initial_capital=args.capital)
    risk_mgr = RiskManager()
    order_mgr = OrderManager(executor=executor, portfolio=portfolio)

    # Map intervals to seconds for polling
    interval_seconds = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400,
    }
    poll_interval = interval_seconds.get(args.interval, 300)

    logger.info(f"Starting {mode.upper()} trading: {args.strategy} on {args.symbol} "
                f"({args.interval}), polling every {poll_interval}s")

    global running
    while running:
        try:
            # Fetch latest candles
            df = fetcher.fetch_klines(args.symbol, args.interval, limit=args.lookback)

            if df.empty:
                logger.warning("No data returned — retrying...")
                time.sleep(10)
                continue

            # Generate signals
            signals_df = strategy.generate_signals(df.copy())
            last_signal = int(signals_df["signal"].iloc[-1])
            last_close = signals_df["close"].iloc[-1]
            sl = signals_df["stop_loss"].iloc[-1] if "stop_loss" in signals_df else 0
            tp = signals_df["take_profit"].iloc[-1] if "take_profit" in signals_df else 0

            import pandas as pd
            if pd.isna(sl):
                sl = 0
            if pd.isna(tp):
                tp = 0

            # Update prices for SL/TP monitoring
            order_mgr.tick({args.symbol: last_close})

            # Act on signal
            if last_signal == Signal.BUY and args.symbol not in portfolio.positions:
                qty = risk_mgr.calculate_position_size(portfolio.total_equity, last_close, sl or last_close * 0.98)
                if qty > 0:
                    order_mgr.submit_entry(args.symbol, "BUY", qty, "MARKET",
                                            stop_loss=sl, take_profit=tp)

            elif last_signal == Signal.SELL and args.symbol in portfolio.positions:
                order_mgr.submit_exit(args.symbol, "signal")

            # Log status
            snap = portfolio.take_snapshot()
            logger.info(f"[{args.symbol}] Price: {last_close:.8f} | "
                        f"Signal: {last_signal} | "
                        f"Equity: ${snap.total_equity:.2f} | "
                        f"Positions: {snap.open_positions}")

        except Exception as e:
            logger.error(f"Error in trading loop: {e}", exc_info=True)

        time.sleep(poll_interval)

    # Shutdown — close all positions
    logger.info("Closing all positions...")
    for symbol in list(portfolio.positions.keys()):
        order_mgr.submit_exit(symbol, "shutdown")

    # Final report
    summary = portfolio.summary()
    logger.info(f"Final portfolio: {summary}")


if __name__ == "__main__":
    main()
