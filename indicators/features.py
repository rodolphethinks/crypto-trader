"""
Feature engineering module for ML-based trading strategies.

Generates 80+ features from OHLCV data covering:
- Price action (returns, gaps, body ratios)
- Trend indicators (EMA crossovers, ADX, Aroon)
- Momentum (RSI, MACD, Stochastic, ROC, Williams %R, CCI, MFI)
- Volatility (ATR, Bollinger bandwidth, Keltner width, historical vol)
- Volume (OBV, VWAP deviation, volume ratios, Chaikin MF)
- Microstructure (price efficiency, autocorrelation, fractal dimension)
- Time features (hour, day of week — for intraday)
- Regime features (rolling Sharpe, drawdown state, trend strength)
"""
import numpy as np
import pandas as pd
from typing import Optional


def compute_all_features(df: pd.DataFrame, include_time: bool = True) -> pd.DataFrame:
    """
    Compute all features from OHLCV DataFrame.
    
    Args:
        df: DataFrame with open, high, low, close, volume columns (DatetimeIndex)
        include_time: include hour/day-of-week features (for intraday TFs)
    
    Returns:
        DataFrame with all feature columns added
    """
    feat = df[["open", "high", "low", "close", "volume"]].copy()
    
    # ── Price action features ─────────────────────────────────────────
    feat["return_1"] = feat["close"].pct_change(1)
    feat["return_2"] = feat["close"].pct_change(2)
    feat["return_5"] = feat["close"].pct_change(5)
    feat["return_10"] = feat["close"].pct_change(10)
    feat["return_20"] = feat["close"].pct_change(20)
    
    feat["log_return_1"] = np.log(feat["close"] / feat["close"].shift(1))
    feat["log_return_5"] = np.log(feat["close"] / feat["close"].shift(5))
    
    feat["gap"] = (feat["open"] - feat["close"].shift(1)) / feat["close"].shift(1)
    feat["body_ratio"] = (feat["close"] - feat["open"]) / (feat["high"] - feat["low"]).replace(0, np.nan)
    feat["upper_shadow"] = (feat["high"] - feat[["open", "close"]].max(axis=1)) / (feat["high"] - feat["low"]).replace(0, np.nan)
    feat["lower_shadow"] = (feat[["open", "close"]].min(axis=1) - feat["low"]) / (feat["high"] - feat["low"]).replace(0, np.nan)
    feat["range_pct"] = (feat["high"] - feat["low"]) / feat["close"]
    
    feat["higher_high"] = (feat["high"] > feat["high"].shift(1)).astype(int)
    feat["lower_low"] = (feat["low"] < feat["low"].shift(1)).astype(int)
    feat["higher_close"] = (feat["close"] > feat["close"].shift(1)).astype(int)
    
    # ── Moving averages & trend ───────────────────────────────────────
    for period in [5, 10, 20, 50]:
        feat[f"sma_{period}"] = feat["close"].rolling(period).mean()
        feat[f"ema_{period}"] = feat["close"].ewm(span=period, adjust=False).mean()
        feat[f"close_vs_sma_{period}"] = (feat["close"] - feat[f"sma_{period}"]) / feat[f"sma_{period}"]
    
    # EMA crossover signals
    feat["ema_5_10_cross"] = (feat["ema_5"] > feat["ema_10"]).astype(int) - (feat["ema_5"] < feat["ema_10"]).astype(int)
    feat["ema_10_20_cross"] = (feat["ema_10"] > feat["ema_20"]).astype(int) - (feat["ema_10"] < feat["ema_20"]).astype(int)
    feat["ema_20_50_cross"] = (feat["ema_20"] > feat["ema_50"]).astype(int) - (feat["ema_20"] < feat["ema_50"]).astype(int)
    
    # ADX (Average Directional Index)
    feat["adx"] = _adx(feat["high"], feat["low"], feat["close"], 14)
    
    # Aroon
    feat["aroon_up"], feat["aroon_down"] = _aroon(feat["high"], feat["low"], 25)
    feat["aroon_osc"] = feat["aroon_up"] - feat["aroon_down"]
    
    # ── Momentum ──────────────────────────────────────────────────────
    # RSI at multiple periods
    for period in [7, 14, 21]:
        feat[f"rsi_{period}"] = _rsi(feat["close"], period)
    
    # MACD
    ema12 = feat["close"].ewm(span=12, adjust=False).mean()
    ema26 = feat["close"].ewm(span=26, adjust=False).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
    feat["macd_hist_diff"] = feat["macd_hist"].diff()
    
    # Stochastic
    for period in [14]:
        low_min = feat["low"].rolling(period).min()
        high_max = feat["high"].rolling(period).max()
        feat[f"stoch_k_{period}"] = 100 * (feat["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        feat[f"stoch_d_{period}"] = feat[f"stoch_k_{period}"].rolling(3).mean()
    
    # ROC (Rate of Change)
    for period in [5, 10, 20]:
        feat[f"roc_{period}"] = (feat["close"] - feat["close"].shift(period)) / feat["close"].shift(period) * 100
    
    # Williams %R
    high_14 = feat["high"].rolling(14).max()
    low_14 = feat["low"].rolling(14).min()
    feat["williams_r"] = -100 * (high_14 - feat["close"]) / (high_14 - low_14).replace(0, np.nan)
    
    # CCI (Commodity Channel Index)
    tp = (feat["high"] + feat["low"] + feat["close"]) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    feat["cci"] = (tp - tp_sma) / (0.015 * tp_mad).replace(0, np.nan)
    
    # MFI (Money Flow Index)
    feat["mfi"] = _mfi(feat["high"], feat["low"], feat["close"], feat["volume"], 14)
    
    # ── Volatility ────────────────────────────────────────────────────
    # ATR
    for period in [7, 14, 21]:
        feat[f"atr_{period}"] = _atr(feat["high"], feat["low"], feat["close"], period)
        feat[f"atr_{period}_pct"] = feat[f"atr_{period}"] / feat["close"]
    
    # Historical volatility
    for period in [10, 20, 50]:
        feat[f"hvol_{period}"] = feat["log_return_1"].rolling(period).std() * np.sqrt(252)
    
    # Bollinger Band width and %B
    for period in [20]:
        sma = feat["close"].rolling(period).mean()
        std = feat["close"].rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        feat[f"bb_width_{period}"] = (upper - lower) / sma
        feat[f"bb_pctb_{period}"] = (feat["close"] - lower) / (upper - lower).replace(0, np.nan)
    
    # Keltner Channel width
    ema_20 = feat["close"].ewm(span=20, adjust=False).mean()
    atr_10 = _atr(feat["high"], feat["low"], feat["close"], 10)
    feat["kc_width"] = (2 * 1.5 * atr_10) / ema_20
    
    # ── Volume ────────────────────────────────────────────────────────
    feat["volume_sma_10"] = feat["volume"].rolling(10).mean()
    feat["volume_sma_20"] = feat["volume"].rolling(20).mean()
    feat["volume_ratio_10"] = feat["volume"] / feat["volume_sma_10"].replace(0, np.nan)
    feat["volume_ratio_20"] = feat["volume"] / feat["volume_sma_20"].replace(0, np.nan)
    
    # OBV normalized
    obv = (np.sign(feat["close"].diff()) * feat["volume"]).cumsum()
    feat["obv_norm"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0, np.nan)
    
    # Chaikin Money Flow
    mfv = ((feat["close"] - feat["low"]) - (feat["high"] - feat["close"])) / (feat["high"] - feat["low"]).replace(0, np.nan) * feat["volume"]
    feat["cmf_20"] = mfv.rolling(20).sum() / feat["volume"].rolling(20).sum().replace(0, np.nan)
    
    # VWAP deviation (rolling)
    cum_vol = feat["volume"].rolling(20).sum()
    cum_vwap = (feat["close"] * feat["volume"]).rolling(20).sum()
    vwap = cum_vwap / cum_vol.replace(0, np.nan)
    feat["vwap_dev"] = (feat["close"] - vwap) / vwap
    
    # ── Microstructure ────────────────────────────────────────────────
    # Price efficiency ratio (directional move / total path)
    for period in [10, 20]:
        direction = abs(feat["close"] - feat["close"].shift(period))
        path = feat["close"].diff().abs().rolling(period).sum()
        feat[f"efficiency_{period}"] = direction / path.replace(0, np.nan)
    
    # Autocorrelation of returns
    feat["autocorr_5"] = feat["return_1"].rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False)
    
    # ── Regime features ───────────────────────────────────────────────
    # Rolling Sharpe
    ret_mean = feat["return_1"].rolling(20).mean()
    ret_std = feat["return_1"].rolling(20).std().replace(0, np.nan)
    feat["rolling_sharpe_20"] = ret_mean / ret_std
    
    # Current drawdown from rolling peak
    rolling_peak = feat["close"].rolling(50, min_periods=1).max()
    feat["drawdown_from_peak"] = (feat["close"] - rolling_peak) / rolling_peak
    
    # Trend strength: % of last N bars that were up
    for period in [10, 20]:
        feat[f"up_ratio_{period}"] = feat["higher_close"].rolling(period).mean()
    
    # ── Time features ─────────────────────────────────────────────────
    if include_time and hasattr(feat.index, 'hour'):
        try:
            feat["hour"] = feat.index.hour
            feat["day_of_week"] = feat.index.dayofweek
            # Cyclical encoding
            feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
            feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
            feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7)
            feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7)
        except AttributeError:
            pass
    
    # ── Target variable (for training) ────────────────────────────────
    # Forward return (shift -1 because we predict the NEXT bar)
    feat["target_return_1"] = feat["close"].pct_change(1).shift(-1)
    feat["target_return_5"] = feat["close"].pct_change(5).shift(-5)
    feat["target_class_1"] = (feat["target_return_1"] > 0).astype(int)  # 1=up, 0=down
    
    # Drop raw price columns (keep only features)
    drop_cols = ["open", "high", "low", "close", "volume",
                 "sma_5", "sma_10", "sma_20", "sma_50",
                 "ema_5", "ema_10", "ema_20", "ema_50",
                 "volume_sma_10", "volume_sma_20"]
    feat.drop(columns=[c for c in drop_cols if c in feat.columns], inplace=True, errors='ignore')
    
    return feat


def get_feature_columns(feat_df: pd.DataFrame) -> list:
    """Return feature column names (excluding targets and raw OHLCV)."""
    exclude = {"target_return_1", "target_return_5", "target_class_1",
               "open", "high", "low", "close", "volume",
               "hour", "day_of_week"}
    return [c for c in feat_df.columns if c not in exclude]


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    
    # Zero out when the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    atr = _atr(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def _aroon(high: pd.Series, low: pd.Series, period: int = 25):
    aroon_up = high.rolling(period + 1).apply(lambda x: x.argmax() / period * 100, raw=True)
    aroon_down = low.rolling(period + 1).apply(lambda x: x.argmin() / period * 100, raw=True)
    return aroon_up, aroon_down


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mr))
