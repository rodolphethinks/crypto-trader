"""
Candlestick pattern recognition.
Detects common single and multi-candle patterns.
"""
import pandas as pd
import numpy as np
from typing import List, Dict


def _body(df: pd.DataFrame) -> pd.Series:
    return df["close"] - df["open"]


def _body_abs(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def _avg_body(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _body_abs(df).rolling(period).mean()


def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """Doji: body is very small relative to range."""
    body = _body_abs(df)
    range_ = df["high"] - df["low"]
    return body < threshold * range_


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer (bullish reversal): small body at top, long lower wick."""
    body = _body_abs(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    avg = _avg_body(df)
    return (lower >= 2 * body) & (upper < body * 0.3) & (body > 0)


def detect_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    """Inverted Hammer: small body at bottom, long upper wick."""
    body = _body_abs(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    return (upper >= 2 * body) & (lower < body * 0.3) & (body > 0)


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting Star (bearish reversal): like inverted hammer at top of uptrend."""
    body = _body_abs(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    return (upper >= 2 * body) & (lower < body * 0.3) & (_body(df) < 0)


def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect bullish and bearish engulfing patterns.
    Returns DataFrame with 'bullish_engulfing' and 'bearish_engulfing' boolean columns.
    """
    body = _body(df)
    body_abs = _body_abs(df)
    prev_body = body.shift(1)
    prev_body_abs = body_abs.shift(1)

    bullish = (prev_body < 0) & (body > 0) & (body_abs > prev_body_abs)
    bearish = (prev_body > 0) & (body < 0) & (body_abs > prev_body_abs)

    return pd.DataFrame({
        "bullish_engulfing": bullish,
        "bearish_engulfing": bearish,
    })


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """Morning Star (bullish reversal): 3-candle pattern."""
    body = _body(df)
    body_abs = _body_abs(df)
    avg = _avg_body(df)

    cond1 = body.shift(2) < 0  # 1st: bearish
    cond2 = body_abs.shift(1) < avg.shift(1) * 0.3  # 2nd: small body
    cond3 = body > 0  # 3rd: bullish
    cond4 = df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2  # Closes above mid of 1st

    return cond1 & cond2 & cond3 & cond4


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """Evening Star (bearish reversal): 3-candle pattern."""
    body = _body(df)
    body_abs = _body_abs(df)
    avg = _avg_body(df)

    cond1 = body.shift(2) > 0  # 1st: bullish
    cond2 = body_abs.shift(1) < avg.shift(1) * 0.3  # 2nd: small body
    cond3 = body < 0  # 3rd: bearish
    cond4 = df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2

    return cond1 & cond2 & cond3 & cond4


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three White Soldiers (bullish continuation)."""
    body = _body(df)
    return (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) & \
           (df["close"] > df["close"].shift(1)) & \
           (df["close"].shift(1) > df["close"].shift(2))


def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three Black Crows (bearish continuation)."""
    body = _body(df)
    return (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) & \
           (df["close"] < df["close"].shift(1)) & \
           (df["close"].shift(1) < df["close"].shift(2))


def detect_all_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Run all candlestick pattern detectors."""
    result = df.copy()
    result["doji"] = detect_doji(df)
    result["hammer"] = detect_hammer(df)
    result["inverted_hammer"] = detect_inverted_hammer(df)
    result["shooting_star"] = detect_shooting_star(df)
    
    engulfing = detect_engulfing(df)
    result["bullish_engulfing"] = engulfing["bullish_engulfing"]
    result["bearish_engulfing"] = engulfing["bearish_engulfing"]
    
    result["morning_star"] = detect_morning_star(df)
    result["evening_star"] = detect_evening_star(df)
    result["three_white_soldiers"] = detect_three_white_soldiers(df)
    result["three_black_crows"] = detect_three_black_crows(df)

    return result
