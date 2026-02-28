"""
Paper trading runner — continuously monitors the market on the best
strategy/pair/interval combos and executes paper trades.

Reads top combos from the latest sweep CSV, then loops every bar period:
  1. Fetch latest candles
  2. Generate signals for each combo
  3. If signal fires, place a paper market order (BUY or SELL)
  4. Track open positions and manage SL/TP
  5. Log all activity to logs/paper_trades.csv

Usage:
    python scripts/run_paper_trader.py [--combos 5] [--capital 10000]
"""
import sys, os, argparse, time, json, signal as os_signal, logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from api.client import MEXCClient
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from trading.executor import TradeExecutor
from backtesting.runner import get_all_strategies
from strategies.bb_variants import get_bb_variants
from strategies.base import Signal
from config.settings import LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trader")

# Interval → approximate seconds between bars
INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "8h": 28800, "1d": 86400,
}

# ── Open Position Tracker ──────────────────────────────────────────────────────

class PaperPosition:
    def __init__(self, combo_key: str, side: str, entry_price: float,
                 quantity: float, stop_loss: float, take_profit: float,
                 entry_time: str):
        self.combo_key = combo_key
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time


class PaperTrader:
    """Manages a paper-trading loop from sweep results."""

    def __init__(self, capital: float = 10_000, top_n: int = 5):
        self.capital = capital
        self.equity = capital
        self.top_n = top_n
        self.fetcher = DataFetcher()
        self.risk_manager = RiskManager(risk_per_trade_pct=1.0, max_drawdown_pct=20.0)
        self.risk_manager.update_equity(capital)
        self.risk_manager.peak_equity = capital

        self.positions: Dict[str, PaperPosition] = {}  # combo_key -> position
        self.closed_trades: List[Dict] = []
        self.strategies_cache: Dict[str, object] = {}
        self.combos: List[Dict] = []

        self._running = True
        os_signal.signal(os_signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        logger.info("Ctrl+C received — shutting down gracefully...")
        self._running = False

    def load_combos_from_csv(self, csv_path: Optional[str] = None):
        """Load top N combos by Sharpe from the latest sweep CSV."""
        if csv_path is None:
            # Find latest sweep csv
            csvs = sorted(LOG_DIR.glob("*sweep*.csv"), key=os.path.getmtime, reverse=True)
            if not csvs:
                raise FileNotFoundError("No sweep CSV found in logs/")
            csv_path = str(csvs[0])

        logger.info(f"Loading combos from {csv_path}")
        df = pd.read_csv(csv_path)

        # Take top N by Sharpe that have > 0 trades and > 0 Sharpe
        df = df[(df["Trades"] > 0) & (df["Sharpe"] > 0)]
        top = df.nlargest(self.top_n, "Sharpe")

        for _, row in top.iterrows():
            self.combos.append({
                "strategy": row["Strategy"],
                "symbol": row["Symbol"],
                "interval": row["Interval"],
            })

        logger.info(f"Loaded {len(self.combos)} combos:")
        for c in self.combos:
            logger.info(f"  {c['strategy']} on {c['symbol']} {c['interval']}")

    def _get_strategy(self, name: str):
        """Instantiate strategy by name (cached)."""
        if name not in self.strategies_cache:
            all_strats = {**get_all_strategies(), **get_bb_variants()}
            if name in all_strats:
                self.strategies_cache[name] = all_strats[name]()
            else:
                raise KeyError(f"Strategy '{name}' not found")
        return self.strategies_cache[name]

    def _fetch_latest(self, symbol: str, interval: str, lookback_bars: int = 200):
        """Fetch recent candles for signal generation."""
        days = max(1, (lookback_bars * INTERVAL_SECONDS.get(interval, 3600)) // 86400 + 1)
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        df = self.fetcher.fetch_klines(
            symbol, interval,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        return df

    def _check_sl_tp(self, pos: PaperPosition, current_price: float,
                     current_low: float, current_high: float) -> Optional[str]:
        """Check if SL or TP has been hit."""
        if pos.side == "BUY":
            if pos.stop_loss > 0 and current_low <= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit > 0 and current_high >= pos.take_profit:
                return "take_profit"
        else:  # SELL
            if pos.stop_loss > 0 and current_high >= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit > 0 and current_low <= pos.take_profit:
                return "take_profit"
        return None

    def _close_position(self, combo_key: str, exit_price: float, reason: str):
        """Close a paper position."""
        pos = self.positions.pop(combo_key)
        if pos.side == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = (pnl / self.equity) * 100
        self.equity += pnl
        self.risk_manager.update_equity(self.equity)
        self.risk_manager.open_positions = max(0, self.risk_manager.open_positions - 1)

        trade = {
            "combo": combo_key,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 4),
            "entry_time": pos.entry_time,
            "exit_time": datetime.utcnow().isoformat(),
            "reason": reason,
        }
        self.closed_trades.append(trade)
        self._save_trades()

        emoji = "🟢" if pnl >= 0 else "🔴"
        logger.info(f"{emoji} CLOSED {combo_key} [{reason}]: "
                     f"{pos.side} {pos.quantity:.6f} @ {pos.entry_price:.8f} → "
                     f"{exit_price:.8f} | PnL: {pnl:+.4f} ({pnl_pct:+.3f}%)")

    def _save_trades(self):
        """Save closed trades to CSV."""
        path = LOG_DIR / "paper_trades.csv"
        pd.DataFrame(self.closed_trades).to_csv(path, index=False)

    def _tick(self):
        """One iteration of the paper trading loop."""
        for combo in self.combos:
            combo_key = f"{combo['strategy']}|{combo['symbol']}|{combo['interval']}"

            try:
                df = self._fetch_latest(combo["symbol"], combo["interval"])
                if df.empty or len(df) < 30:
                    continue

                # Check existing position's SL/TP
                if combo_key in self.positions:
                    pos = self.positions[combo_key]
                    last = df.iloc[-1]
                    exit_reason = self._check_sl_tp(
                        pos, last["close"], last["low"], last["high"])
                    if exit_reason == "stop_loss":
                        self._close_position(combo_key, pos.stop_loss, "stop_loss")
                    elif exit_reason == "take_profit":
                        self._close_position(combo_key, pos.take_profit, "take_profit")
                    continue  # one action per tick per combo

                # Generate signals
                strategy = self._get_strategy(combo["strategy"])
                sig_df = strategy.generate_signals(df.copy())
                last_signal = sig_df["signal"].iloc[-1]

                if last_signal == Signal.BUY or last_signal == Signal.SELL:
                    side = "BUY" if last_signal == Signal.BUY else "SELL"
                    entry_price = df["close"].iloc[-1]
                    sl = sig_df["stop_loss"].iloc[-1] if "stop_loss" in sig_df else 0
                    tp = sig_df["take_profit"].iloc[-1] if "take_profit" in sig_df else 0

                    if pd.isna(sl): sl = 0
                    if pd.isna(tp): tp = 0

                    # Position size via risk manager
                    if sl > 0:
                        qty = self.risk_manager.calculate_position_size(
                            self.equity, entry_price, sl)
                    else:
                        qty = (self.equity * 0.01) / entry_price  # fallback 1%

                    if qty <= 0:
                        continue

                    if not self.risk_manager.can_open_position():
                        logger.warning(f"Risk check blocked: {combo_key}")
                        continue

                    pos = PaperPosition(
                        combo_key=combo_key,
                        side=side,
                        entry_price=entry_price,
                        quantity=qty,
                        stop_loss=sl,
                        take_profit=tp,
                        entry_time=datetime.utcnow().isoformat(),
                    )
                    self.positions[combo_key] = pos
                    self.risk_manager.open_positions += 1

                    logger.info(f"📊 OPEN {combo_key}: {side} {qty:.6f} @ {entry_price:.8f} "
                                 f"SL={sl:.8f} TP={tp:.8f}")

            except Exception as e:
                logger.error(f"Error processing {combo_key}: {e}")

    def run(self):
        """Main paper trading loop."""
        if not self.combos:
            logger.error("No combos loaded. Load from CSV first.")
            return

        # Determine smallest interval for sleep
        intervals = [c["interval"] for c in self.combos]
        min_secs = min(INTERVAL_SECONDS.get(i, 3600) for i in intervals)
        # Poll at half the smallest bar interval (or at least 30s)
        poll_secs = max(30, min_secs // 2)

        logger.info(f"Starting paper trading loop (poll every {poll_secs}s)")
        logger.info(f"Capital: ${self.equity:,.2f} | "
                     f"Risk: {self.risk_manager.risk_per_trade_pct}%/trade | "
                     f"Max DD: {self.risk_manager.max_drawdown_pct}%")
        logger.info(f"Monitoring {len(self.combos)} combos")
        print("=" * 60)
        print("  Press Ctrl+C to stop gracefully")
        print("=" * 60)

        while self._running:
            self._tick()

            # Status
            drawdown = self.risk_manager.current_drawdown_pct()
            open_count = len(self.positions)
            closed_count = len(self.closed_trades)
            total_pnl = sum(t["pnl"] for t in self.closed_trades)

            logger.info(f"💰 Equity: ${self.equity:,.2f} | DD: {drawdown:.2f}% | "
                         f"Open: {open_count} | Closed: {closed_count} | "
                         f"PnL: {total_pnl:+.4f}")

            time.sleep(poll_secs)

        # Shutdown
        logger.info("Saving final state...")
        self._save_trades()
        logger.info(f"Final equity: ${self.equity:,.2f} ({len(self.closed_trades)} trades)")


def main():
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--combos", type=int, default=5,
                        help="Number of top combos to trade (default: 5)")
    parser.add_argument("--capital", type=float, default=10_000,
                        help="Starting capital (default: 10000)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to sweep CSV (default: latest in logs/)")
    args = parser.parse_args()

    trader = PaperTrader(capital=args.capital, top_n=args.combos)
    trader.load_combos_from_csv(args.csv)
    trader.run()


if __name__ == "__main__":
    main()
