"""
Custom / composite indicators for advanced strategies.
"""
import numpy as np
import pandas as pd

from indicators.trend import ema, sma
from indicators.momentum import rsi
from indicators.volatility import atr, bollinger_bands


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic Pivot Points (daily).
    Returns: pivot, r1, r2, r3, s1, s2, s3
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return pd.DataFrame({
        "pivot": pivot, "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3,
    })


def fibonacci_retracements(high: float, low: float) -> dict:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "level_0": high,
        "level_236": high - 0.236 * diff,
        "level_382": high - 0.382 * diff,
        "level_500": high - 0.500 * diff,
        "level_618": high - 0.618 * diff,
        "level_786": high - 0.786 * diff,
        "level_100": low,
    }


def market_regime(df: pd.DataFrame, adx_period: int = 14,
                  adx_threshold: float = 25.0, bb_period: int = 20) -> pd.Series:
    """
    Classify market into regimes:
    - 'trending_up': Strong uptrend
    - 'trending_down': Strong downtrend
    - 'ranging': Low volatility range
    - 'volatile': High volatility no clear trend
    """
    from indicators.trend import adx as adx_fn
    
    adx_data = adx_fn(df, adx_period)
    bb = bollinger_bands(df["close"], bb_period)

    regime = pd.Series("ranging", index=df.index)

    # Trending
    trending = adx_data["adx"] > adx_threshold
    up = adx_data["plus_di"] > adx_data["minus_di"]
    
    regime[trending & up] = "trending_up"
    regime[trending & ~up] = "trending_down"
    
    # Volatile but not trending
    high_vol = bb["bb_width"] > bb["bb_width"].rolling(50).quantile(0.75)
    regime[~trending & high_vol] = "volatile"

    return regime


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Relative volume compared to average — spike detection."""
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] / avg_vol


def price_rate_of_change_momentum(series: pd.Series, fast: int = 5,
                                   slow: int = 20) -> pd.Series:
    """Dual ROC momentum — fast ROC minus slow ROC."""
    fast_roc = ((series - series.shift(fast)) / series.shift(fast)) * 100
    slow_roc = ((series - series.shift(slow)) / series.shift(slow)) * 100
    return fast_roc - slow_roc


def range_detector(df: pd.DataFrame, lookback: int = 50,
                   threshold_pct: float = 2.0) -> pd.Series:
    """
    Detect if price is in a narrow range.
    Returns True when the high-low range over lookback is < threshold_pct of mid-price.
    Useful for USDCUSDT mean reversion.
    """
    rolling_high = df["high"].rolling(lookback).max()
    rolling_low = df["low"].rolling(lookback).min()
    mid = (rolling_high + rolling_low) / 2
    range_pct = (rolling_high - rolling_low) / mid * 100
    return range_pct < threshold_pct


def session_indicator(df: pd.DataFrame) -> pd.Series:
    """
    Classify by trading session based on hour (UTC).
    - 'asian': 0-8 UTC
    - 'london': 8-16 UTC
    - 'new_york': 13-21 UTC
    - 'overlap': 13-16 UTC (London+NY overlap)
    """
    hours = df.index.hour
    session = pd.Series("off_hours", index=df.index)
    session[(hours >= 0) & (hours < 8)] = "asian"
    session[(hours >= 8) & (hours < 13)] = "london"
    session[(hours >= 13) & (hours < 16)] = "overlap"
    session[(hours >= 16) & (hours < 21)] = "new_york"
    session[(hours >= 21)] = "asian"
    return session
