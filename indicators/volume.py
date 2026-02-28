"""
Volume indicators: OBV, VWAP, Volume SMA, Accumulation/Distribution, CMF, VPVR.
"""
import numpy as np
import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff())
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price.
    Resets daily (when date changes).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (tp * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """
    Intraday VWAP that resets each day.
    Uses date from the index.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    dates = df.index.date
    
    result = pd.Series(np.nan, index=df.index)
    
    for date in np.unique(dates):
        mask = dates == date
        day_tp = tp[mask]
        day_vol = df["volume"][mask]
        cum_tp_vol = (day_tp * day_vol).cumsum()
        cum_vol = day_vol.cumsum()
        result[mask] = cum_tp_vol / cum_vol
    
    return result


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average — useful for volume spike detection."""
    return df["volume"].rolling(period).mean()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume / average volume ratio."""
    avg = volume_sma(df, period)
    return df["volume"] / avg


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    clv = clv.fillna(0)
    ad = (clv * df["volume"]).cumsum()
    return ad


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    clv = clv.fillna(0)
    mf_volume = clv * df["volume"]
    return mf_volume.rolling(period).sum() / df["volume"].rolling(period).sum()


def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Force Index."""
    fi = df["close"].diff() * df["volume"]
    return fi.ewm(span=period, adjust=False).mean()


def elder_ray(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    """
    Elder Ray Index (Bull/Bear Power).
    Returns DataFrame with: bull_power, bear_power
    """
    from indicators.trend import ema as ema_fn
    
    ema_val = ema_fn(df["close"], period)
    return pd.DataFrame({
        "bull_power": df["high"] - ema_val,
        "bear_power": df["low"] - ema_val,
    })
