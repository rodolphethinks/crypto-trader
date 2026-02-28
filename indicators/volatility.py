"""
Volatility indicators: ATR, Bollinger Bands, Keltner Channel, Donchian Channel,
                       Standard Deviation, Historical Volatility.
"""
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with columns: bb_upper, bb_middle, bb_lower, bb_width, bb_pct
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle
    pct = (series - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": width,
        "bb_pct": pct
    })


def keltner_channel(df: pd.DataFrame, ema_period: int = 20,
                    atr_period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
    """
    Keltner Channel.
    Returns DataFrame with columns: kc_upper, kc_middle, kc_lower
    """
    from indicators.trend import ema as ema_fn

    middle = ema_fn(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return pd.DataFrame({
        "kc_upper": upper,
        "kc_middle": middle,
        "kc_lower": lower,
    })


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channel.
    Returns DataFrame with columns: dc_upper, dc_middle, dc_lower
    """
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    middle = (upper + lower) / 2

    return pd.DataFrame({
        "dc_upper": upper,
        "dc_middle": middle,
        "dc_lower": lower,
    })


def historical_volatility(series: pd.Series, period: int = 20,
                          annualize: bool = True) -> pd.Series:
    """Historical volatility (annualized by default, assuming 365 crypto days)."""
    log_returns = np.log(series / series.shift(1))
    hv = log_returns.rolling(period).std()
    if annualize:
        hv = hv * np.sqrt(365)
    return hv


def squeeze_momentum(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
                     kc_period: int = 20, kc_atr_period: int = 14,
                     kc_mult: float = 1.5) -> pd.DataFrame:
    """
    Squeeze Momentum Indicator (TTM Squeeze).
    Detects when Bollinger Bands are inside Keltner Channel (low volatility squeeze).
    Returns DataFrame with: squeeze_on (bool), momentum (value)
    """
    bb = bollinger_bands(df["close"], bb_period, bb_std)
    kc = keltner_channel(df, kc_period, kc_atr_period, kc_mult)

    squeeze_on = (bb["bb_lower"] > kc["kc_lower"]) & (bb["bb_upper"] < kc["kc_upper"])

    # Momentum = linearReg(close - avg(highest(high,kc_period), lowest(low,kc_period)), kc_period)
    highest = df["high"].rolling(kc_period).max()
    lowest = df["low"].rolling(kc_period).min()
    mean_hl = (highest + lowest) / 2
    mean_close = df["close"].rolling(kc_period).mean()
    momentum = df["close"] - (mean_hl + mean_close) / 2

    return pd.DataFrame({
        "squeeze_on": squeeze_on,
        "momentum": momentum,
    })
