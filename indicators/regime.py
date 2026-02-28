"""
Regime detection module.

Classifies the current market into regimes:
- TRENDING_UP / TRENDING_DOWN / RANGING
- HIGH_VOL / LOW_VOL
- RISK_ON / RISK_OFF

Used by ML strategies and adaptive strategies to switch behavior.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from enum import IntEnum


class MarketRegime(IntEnum):
    TRENDING_UP = 2
    RANGING_UP = 1
    FLAT = 0
    RANGING_DOWN = -1
    TRENDING_DOWN = -2


class VolRegime(IntEnum):
    HIGH = 1
    NORMAL = 0
    LOW = -1


def detect_trend_regime(close: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Detect trend regime using ADX + directional movement + slope.
    
    Returns Series of MarketRegime values.
    """
    # EMA slope
    ema = close.ewm(span=lookback, adjust=False).mean()
    slope = (ema - ema.shift(5)) / ema.shift(5)
    
    # Price vs EMA
    position = (close - ema) / ema
    
    # Rolling R-squared of linear trend
    r2 = close.rolling(lookback).apply(_rolling_r2, raw=True)
    
    regime = pd.Series(MarketRegime.FLAT, index=close.index, dtype=int)
    
    # Strong trend: high R2 + consistent slope
    strong_up = (r2 > 0.65) & (slope > 0.005) & (position > 0)
    strong_down = (r2 > 0.65) & (slope < -0.005) & (position < 0)
    weak_up = (~strong_up) & (slope > 0.001) & (position > 0)
    weak_down = (~strong_down) & (slope < -0.001) & (position < 0)
    
    regime[strong_up] = MarketRegime.TRENDING_UP
    regime[strong_down] = MarketRegime.TRENDING_DOWN
    regime[weak_up] = MarketRegime.RANGING_UP
    regime[weak_down] = MarketRegime.RANGING_DOWN
    
    return regime


def detect_volatility_regime(close: pd.Series, lookback: int = 20, 
                              long_lookback: int = 100) -> pd.Series:
    """
    Detect volatility regime by comparing short-term vol to long-term vol.
    
    Returns Series of VolRegime values.
    """
    returns = close.pct_change()
    short_vol = returns.rolling(lookback).std()
    long_vol = returns.rolling(long_lookback).std()
    
    vol_ratio = short_vol / long_vol.replace(0, np.nan)
    
    regime = pd.Series(VolRegime.NORMAL, index=close.index, dtype=int)
    regime[vol_ratio > 1.5] = VolRegime.HIGH
    regime[vol_ratio < 0.5] = VolRegime.LOW
    
    return regime


def detect_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime columns to a DataFrame.
    
    Args:
        df: DataFrame with 'close' column
    
    Returns:
        DataFrame with added regime columns
    """
    result = df.copy()
    result["regime_trend"] = detect_trend_regime(df["close"])
    result["regime_vol"] = detect_volatility_regime(df["close"])
    
    # Regime duration (how many bars since last regime change)
    trend_change = result["regime_trend"].diff().ne(0).cumsum()
    result["regime_trend_duration"] = trend_change.groupby(trend_change).cumcount() + 1
    
    vol_change = result["regime_vol"].diff().ne(0).cumsum()
    result["regime_vol_duration"] = vol_change.groupby(vol_change).cumcount() + 1
    
    # Composite regime score
    result["regime_composite"] = result["regime_trend"] + result["regime_vol"]
    
    return result


def get_regime_summary(df: pd.DataFrame) -> dict:
    """Get a summary of regime distribution."""
    if "regime_trend" not in df.columns:
        df = detect_regime_features(df)
    
    trend_counts = df["regime_trend"].value_counts(normalize=True)
    vol_counts = df["regime_vol"].value_counts(normalize=True)
    
    return {
        "trend_distribution": trend_counts.to_dict(),
        "vol_distribution": vol_counts.to_dict(),
        "current_trend": int(df["regime_trend"].iloc[-1]),
        "current_vol": int(df["regime_vol"].iloc[-1]),
    }


def _rolling_r2(x: np.ndarray) -> float:
    """R-squared of linear regression on a rolling window."""
    n = len(x)
    if n < 5:
        return 0.0
    t = np.arange(n)
    corr = np.corrcoef(t, x)[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr ** 2
