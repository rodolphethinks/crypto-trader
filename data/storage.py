"""
Data storage layer — CSV file cache and SQLite for trade logs.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from config.settings import DATA_DIR, DB_PATH

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles reading/writing market data to disk and database."""

    def __init__(self, data_dir: Optional[Path] = None, db_path: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.db_path = db_path or DB_PATH
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(f"sqlite:///{self.db_path}")
        return self._engine

    # ── CSV Cache ─────────────────────────────────────────────────────────
    def _kline_path(self, symbol: str, interval: str) -> Path:
        return self.data_dir / f"{symbol}_{interval}.csv"

    def save_klines(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save kline DataFrame to CSV."""
        path = self._kline_path(symbol, interval)
        df.to_csv(path)
        logger.debug(f"Saved {len(df)} candles to {path}")

    def load_klines(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load kline DataFrame from CSV cache."""
        path = self._kline_path(symbol, interval)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col="open_time", parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def list_cached_pairs(self) -> list:
        """List all cached symbol-interval combos."""
        results = []
        for f in self.data_dir.glob("*.csv"):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                results.append({"symbol": parts[0], "interval": parts[1]})
        return results

    # ── SQLite Trade Log ──────────────────────────────────────────────────
    def init_db(self):
        """Initialize the trade log database tables."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    total_trades INTEGER,
                    win_rate REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    profit_factor REAL,
                    avg_trade_return REAL,
                    params TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    duration_minutes REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        logger.info("Database initialized")

    def save_backtest_result(self, result: dict):
        """Save a backtest result to the database."""
        import json
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO backtest_results
                (strategy, symbol, interval, start_date, end_date,
                 total_trades, win_rate, total_return, max_drawdown,
                 sharpe_ratio, sortino_ratio, profit_factor, avg_trade_return, params)
                VALUES (:strategy, :symbol, :interval, :start_date, :end_date,
                        :total_trades, :win_rate, :total_return, :max_drawdown,
                        :sharpe_ratio, :sortino_ratio, :profit_factor, :avg_trade_return, :params)
            """), {
                **result,
                "params": json.dumps(result.get("params", {})),
            })
            conn.commit()

    def load_backtest_results(self) -> pd.DataFrame:
        """Load all backtest results."""
        return pd.read_sql("SELECT * FROM backtest_results ORDER BY created_at DESC",
                           self.engine)

    def save_trade(self, trade: dict):
        """Save a trade to the database."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO trades
                (strategy, symbol, side, entry_price, exit_price, quantity,
                 pnl, pnl_pct, entry_time, exit_time, duration_minutes,
                 stop_loss, take_profit, tags)
                VALUES (:strategy, :symbol, :side, :entry_price, :exit_price, :quantity,
                        :pnl, :pnl_pct, :entry_time, :exit_time, :duration_minutes,
                        :stop_loss, :take_profit, :tags)
            """), trade)
            conn.commit()

    def load_trades(self, strategy: Optional[str] = None,
                    symbol: Optional[str] = None) -> pd.DataFrame:
        """Load trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = {}
        if strategy:
            query += " AND strategy = :strategy"
            params["strategy"] = strategy
        if symbol:
            query += " AND symbol = :symbol"
            params["symbol"] = symbol
        query += " ORDER BY entry_time DESC"
        return pd.read_sql(text(query), self.engine, params=params)
