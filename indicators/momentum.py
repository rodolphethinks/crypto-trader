"""
Momentum indicators: RSI, Stochastic, CCI, Williams %R, ROC, MFI, Awesome Oscillator.
"""
import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
               smooth_k: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    Returns DataFrame with columns: stoch_k, stoch_d
    """
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()

    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min)
    stoch_k = stoch_k.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(d_period).mean()

    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d})


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change."""
    return ((series - series.shift(period)) / series.shift(period)) * 100


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    pos_flow = pd.Series(0.0, index=df.index)
    neg_flow = pd.Series(0.0, index=df.index)

    tp_diff = tp.diff()
    pos_flow[tp_diff > 0] = mf[tp_diff > 0]
    neg_flow[tp_diff < 0] = mf[tp_diff < 0]

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()

    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator."""
    midpoint = (df["high"] + df["low"]) / 2
    return midpoint.rolling(fast).mean() - midpoint.rolling(slow).mean()


def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13,
        signal_period: int = 13) -> pd.DataFrame:
    """
    True Strength Index.
    Returns DataFrame with columns: tsi, signal
    """
    diff = series.diff()

    double_smoothed_diff = diff.ewm(span=long_period, adjust=False).mean().ewm(
        span=short_period, adjust=False).mean()
    double_smoothed_abs = diff.abs().ewm(span=long_period, adjust=False).mean().ewm(
        span=short_period, adjust=False).mean()

    tsi_val = 100 * double_smoothed_diff / double_smoothed_abs
    signal_val = tsi_val.ewm(span=signal_period, adjust=False).mean()

    return pd.DataFrame({"tsi": tsi_val, "signal": signal_val})


def stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14,
              k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """
    Stochastic RSI.
    Returns DataFrame with columns: stoch_rsi_k, stoch_rsi_d
    """
    rsi_val = rsi(series, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period).max()
    stoch_rsi_k = ((rsi_val - rsi_min) / (rsi_max - rsi_min)).rolling(k_smooth).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(d_smooth).mean()
    return pd.DataFrame({"stoch_rsi_k": stoch_rsi_k, "stoch_rsi_d": stoch_rsi_d})
