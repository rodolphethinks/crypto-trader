"""
Historical data fetcher for MEXC.

Downloads kline data with pagination, respects rate limits,
and stores data for backtesting.
"""
import time
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd

from api.client import MEXCClient
from config.settings import API_KLINE_INTERVALS
from data.storage import DataStorage

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and manage historical market data from MEXC."""

    def __init__(self, client: Optional[MEXCClient] = None,
                 storage: Optional[DataStorage] = None):
        self.client = client or MEXCClient()
        self.storage = storage or DataStorage()

    def fetch_klines(self, symbol: str, interval: str = "1h",
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     limit_per_request: int = 500,
                     max_retries: int = 3) -> pd.DataFrame:
        """
        Fetch kline data with automatic pagination.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDC')
            interval: Candle interval ('1m','5m','15m','30m','1h','4h','8h','1d','1w','1M')
            start_date: Start date string 'YYYY-MM-DD' (default: 90 days ago)
            end_date: End date string 'YYYY-MM-DD' (default: now)
            limit_per_request: Max candles per request (max 500)
        
        Returns:
            DataFrame with OHLCV data
        """
        api_interval = API_KLINE_INTERVALS.get(interval, interval)

        # Parse dates
        if start_date:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            start_ts = int((datetime.utcnow() - timedelta(days=90)).timestamp() * 1000)

        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.utcnow().timestamp() * 1000)

        all_klines = []
        current_start = start_ts
        retries = 0

        logger.info(f"Fetching {symbol} {interval} klines from "
                     f"{datetime.fromtimestamp(start_ts/1000)} to "
                     f"{datetime.fromtimestamp(end_ts/1000)}")

        while current_start < end_ts:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=api_interval,
                    start_time=current_start,
                    end_time=end_ts,
                    limit=limit_per_request,
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Move start to after last candle close time
                last_close_time = klines[-1][6]
                if last_close_time <= current_start:
                    break
                current_start = last_close_time + 1

                # Rate limit: ~2 requests per second
                time.sleep(0.5)

                if len(klines) < limit_per_request:
                    break

            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) reached for {symbol} {interval}")
                    break
                time.sleep(2)
                continue

        if not all_klines:
            logger.warning(f"No kline data returned for {symbol} {interval}")
            return pd.DataFrame()

        df = self._klines_to_dataframe(all_klines)
        logger.info(f"Fetched {len(df)} candles for {symbol} {interval}")

        # Store for caching
        self.storage.save_klines(symbol, interval, df)

        return df

    def fetch_klines_cached(self, symbol: str, interval: str = "1h",
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Load from cache if available, otherwise fetch from API."""
        cached = self.storage.load_klines(symbol, interval)
        if cached is not None and not cached.empty:
            logger.info(f"Loaded {len(cached)} cached candles for {symbol} {interval}")
            return cached
        return self.fetch_klines(symbol, interval, start_date, end_date)

    def fetch_multiple_pairs(self, symbols: List[str], interval: str = "1h",
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> dict:
        """Fetch kline data for multiple pairs."""
        results = {}
        for symbol in symbols:
            try:
                df = self.fetch_klines(symbol, interval, start_date, end_date)
                if not df.empty:
                    results[symbol] = df
                time.sleep(1)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results

    def fetch_order_book_snapshot(self, symbol: str, limit: int = 1000) -> dict:
        """Fetch current order book."""
        return self.client.get_order_book(symbol, limit)

    def fetch_recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch recent trades as DataFrame."""
        trades = self.client.get_recent_trades(symbol, limit)
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["price"] = pd.to_numeric(df["price"])
        df["qty"] = pd.to_numeric(df["qty"])
        df["quoteQty"] = pd.to_numeric(df["quoteQty"])
        return df

    @staticmethod
    def _klines_to_dataframe(klines: List) -> pd.DataFrame:
        """Convert raw kline data to a clean DataFrame."""
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close",
            "volume", "close_time", "quote_volume"
        ])
        
        # Type conversions
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.set_index("open_time")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        
        return df
