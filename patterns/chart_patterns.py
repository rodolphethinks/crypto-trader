"""
Chart pattern recognition: Triangles, Wedges, Flags, Head & Shoulders,
Double Tops/Bottoms, etc.

Uses swing high/low detection and geometric fitting.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy.signal import argrelextrema


def find_swing_highs(series: pd.Series, order: int = 5) -> pd.Series:
    """Find local maxima (swing highs)."""
    indices = argrelextrema(series.values, np.greater, order=order)[0]
    result = pd.Series(np.nan, index=series.index)
    result.iloc[indices] = series.iloc[indices]
    return result


def find_swing_lows(series: pd.Series, order: int = 5) -> pd.Series:
    """Find local minima (swing lows)."""
    indices = argrelextrema(series.values, np.less, order=order)[0]
    result = pd.Series(np.nan, index=series.index)
    result.iloc[indices] = series.iloc[indices]
    return result


def find_swing_points(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """Find both swing highs and lows, adding columns to df."""
    result = df.copy()
    result["swing_high"] = find_swing_highs(df["high"], order)
    result["swing_low"] = find_swing_lows(df["low"], order)
    return result


def detect_double_top(df: pd.DataFrame, order: int = 10,
                      tolerance: float = 0.02) -> List[Dict]:
    """
    Detect Double Top pattern.
    Two peaks at roughly the same level with a valley in between.
    """
    patterns = []
    highs = find_swing_highs(df["high"], order).dropna()

    for i in range(len(highs) - 1):
        peak1 = highs.iloc[i]
        peak2 = highs.iloc[i + 1]

        if abs(peak1 - peak2) / peak1 < tolerance:
            idx1 = highs.index[i]
            idx2 = highs.index[i + 1]

            # Find neckline (lowest point between peaks)
            between = df.loc[idx1:idx2]
            neckline = between["low"].min()

            patterns.append({
                "type": "double_top",
                "peak1_idx": idx1, "peak1_price": peak1,
                "peak2_idx": idx2, "peak2_price": peak2,
                "neckline": neckline,
                "target": neckline - (peak1 - neckline),
                "signal": "bearish",
            })

    return patterns


def detect_double_bottom(df: pd.DataFrame, order: int = 10,
                         tolerance: float = 0.02) -> List[Dict]:
    """
    Detect Double Bottom pattern.
    Two troughs at roughly the same level with a peak in between.
    """
    patterns = []
    lows = find_swing_lows(df["low"], order).dropna()

    for i in range(len(lows) - 1):
        trough1 = lows.iloc[i]
        trough2 = lows.iloc[i + 1]

        if abs(trough1 - trough2) / trough1 < tolerance:
            idx1 = lows.index[i]
            idx2 = lows.index[i + 1]

            between = df.loc[idx1:idx2]
            neckline = between["high"].max()

            patterns.append({
                "type": "double_bottom",
                "trough1_idx": idx1, "trough1_price": trough1,
                "trough2_idx": idx2, "trough2_price": trough2,
                "neckline": neckline,
                "target": neckline + (neckline - trough1),
                "signal": "bullish",
            })

    return patterns


def detect_head_and_shoulders(df: pd.DataFrame, order: int = 10,
                              tolerance: float = 0.02) -> List[Dict]:
    """
    Detect Head and Shoulders pattern (bearish reversal).
    Left shoulder, head (higher), right shoulder (roughly equal to left).
    """
    patterns = []
    highs = find_swing_highs(df["high"], order).dropna()

    for i in range(len(highs) - 2):
        left = highs.iloc[i]
        head = highs.iloc[i + 1]
        right = highs.iloc[i + 2]

        # Head must be highest, shoulders roughly equal
        if (head > left and head > right and
                abs(left - right) / left < tolerance):
            idx_l = highs.index[i]
            idx_h = highs.index[i + 1]
            idx_r = highs.index[i + 2]

            # Neckline from lows between shoulders and head
            low1 = df.loc[idx_l:idx_h, "low"].min()
            low2 = df.loc[idx_h:idx_r, "low"].min()
            neckline = (low1 + low2) / 2

            patterns.append({
                "type": "head_and_shoulders",
                "left_idx": idx_l, "head_idx": idx_h, "right_idx": idx_r,
                "left_price": left, "head_price": head, "right_price": right,
                "neckline": neckline,
                "target": neckline - (head - neckline),
                "signal": "bearish",
            })

    return patterns


def detect_inverse_head_and_shoulders(df: pd.DataFrame, order: int = 10,
                                       tolerance: float = 0.02) -> List[Dict]:
    """Detect Inverse Head and Shoulders (bullish reversal)."""
    patterns = []
    lows = find_swing_lows(df["low"], order).dropna()

    for i in range(len(lows) - 2):
        left = lows.iloc[i]
        head = lows.iloc[i + 1]
        right = lows.iloc[i + 2]

        if (head < left and head < right and
                abs(left - right) / left < tolerance):
            idx_l = lows.index[i]
            idx_h = lows.index[i + 1]
            idx_r = lows.index[i + 2]

            high1 = df.loc[idx_l:idx_h, "high"].max()
            high2 = df.loc[idx_h:idx_r, "high"].max()
            neckline = (high1 + high2) / 2

            patterns.append({
                "type": "inverse_head_and_shoulders",
                "left_idx": idx_l, "head_idx": idx_h, "right_idx": idx_r,
                "left_price": left, "head_price": head, "right_price": right,
                "neckline": neckline,
                "target": neckline + (neckline - head),
                "signal": "bullish",
            })

    return patterns


def _fit_trendline(points: List[Tuple[int, float]]) -> Tuple[float, float]:
    """Fit a linear trendline to a set of (index, price) points. Returns (slope, intercept)."""
    if len(points) < 2:
        return (0.0, 0.0)
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    coeffs = np.polyfit(x, y, 1)
    return (coeffs[0], coeffs[1])


def detect_ascending_triangle(df: pd.DataFrame, order: int = 10,
                               tolerance: float = 0.015) -> List[Dict]:
    """
    Ascending Triangle: flat resistance line, rising support line.
    Bullish breakout pattern.
    """
    patterns = []
    highs = find_swing_highs(df["high"], order).dropna()
    lows = find_swing_lows(df["low"], order).dropna()

    if len(highs) < 2 or len(lows) < 2:
        return patterns

    # Check for flat highs (resistance)
    for i in range(len(highs) - 1):
        h1, h2 = highs.iloc[i], highs.iloc[i + 1]
        if abs(h1 - h2) / h1 < tolerance:
            # Look for rising lows in the same window
            idx1, idx2 = highs.index[i], highs.index[i + 1]
            window_lows = lows.loc[idx1:idx2]
            if len(window_lows) >= 2:
                if window_lows.iloc[-1] > window_lows.iloc[0]:
                    patterns.append({
                        "type": "ascending_triangle",
                        "resistance": (h1 + h2) / 2,
                        "start_idx": idx1,
                        "end_idx": idx2,
                        "signal": "bullish",
                    })

    return patterns


def detect_descending_triangle(df: pd.DataFrame, order: int = 10,
                                tolerance: float = 0.015) -> List[Dict]:
    """
    Descending Triangle: flat support line, falling resistance line.
    Bearish breakdown pattern.
    """
    patterns = []
    highs = find_swing_highs(df["high"], order).dropna()
    lows = find_swing_lows(df["low"], order).dropna()

    if len(highs) < 2 or len(lows) < 2:
        return patterns

    for i in range(len(lows) - 1):
        l1, l2 = lows.iloc[i], lows.iloc[i + 1]
        if abs(l1 - l2) / l1 < tolerance:
            idx1, idx2 = lows.index[i], lows.index[i + 1]
            window_highs = highs.loc[idx1:idx2]
            if len(window_highs) >= 2:
                if window_highs.iloc[-1] < window_highs.iloc[0]:
                    patterns.append({
                        "type": "descending_triangle",
                        "support": (l1 + l2) / 2,
                        "start_idx": idx1,
                        "end_idx": idx2,
                        "signal": "bearish",
                    })

    return patterns


def detect_flag(df: pd.DataFrame, order: int = 5, lookback: int = 50) -> List[Dict]:
    """
    Detect Bull Flag and Bear Flag patterns.
    A strong move (pole) followed by a tight consolidation (flag) 
    that slopes against the pole direction.
    """
    patterns = []
    close = df["close"]

    for i in range(lookback, len(df) - 10):
        # Check for pole (strong move)
        window = close.iloc[i - lookback:i]
        pole_return = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]

        if abs(pole_return) > 0.03:  # At least 3% move for the pole
            # Look at consolidation after pole
            flag = close.iloc[i:i + 10]
            flag_return = (flag.iloc[-1] - flag.iloc[0]) / flag.iloc[0]
            flag_range = (flag.max() - flag.min()) / flag.mean()

            if flag_range < 0.02:  # Tight consolidation
                if pole_return > 0 and flag_return < 0:
                    patterns.append({
                        "type": "bull_flag",
                        "pole_start": df.index[i - lookback],
                        "flag_end": df.index[min(i + 9, len(df) - 1)],
                        "signal": "bullish",
                    })
                elif pole_return < 0 and flag_return > 0:
                    patterns.append({
                        "type": "bear_flag",
                        "pole_start": df.index[i - lookback],
                        "flag_end": df.index[min(i + 9, len(df) - 1)],
                        "signal": "bearish",
                    })

    return patterns


def detect_wedge(df: pd.DataFrame, order: int = 10) -> List[Dict]:
    """
    Detect Rising Wedge (bearish) and Falling Wedge (bullish).
    Both trendlines converge.
    """
    patterns = []
    highs = find_swing_highs(df["high"], order).dropna()
    lows = find_swing_lows(df["low"], order).dropna()

    if len(highs) < 3 or len(lows) < 3:
        return patterns

    # Use last few swing points for trendline fitting
    high_points = [(i, v) for i, v in enumerate(highs.values[-4:])]
    low_points = [(i, v) for i, v in enumerate(lows.values[-4:])]

    if len(high_points) >= 2 and len(low_points) >= 2:
        high_slope, _ = _fit_trendline(high_points)
        low_slope, _ = _fit_trendline(low_points)

        # Rising wedge: both slopes positive but converging
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
            patterns.append({
                "type": "rising_wedge",
                "signal": "bearish",
                "high_slope": high_slope,
                "low_slope": low_slope,
            })

        # Falling wedge: both slopes negative but converging
        if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            patterns.append({
                "type": "falling_wedge",
                "signal": "bullish",
                "high_slope": high_slope,
                "low_slope": low_slope,
            })

    return patterns


def detect_all_patterns(df: pd.DataFrame, order: int = 10) -> List[Dict]:
    """Run all pattern detectors and aggregate results."""
    patterns = []
    patterns.extend(detect_double_top(df, order))
    patterns.extend(detect_double_bottom(df, order))
    patterns.extend(detect_head_and_shoulders(df, order))
    patterns.extend(detect_inverse_head_and_shoulders(df, order))
    patterns.extend(detect_ascending_triangle(df, order))
    patterns.extend(detect_descending_triangle(df, order))
    patterns.extend(detect_flag(df, order // 2))
    patterns.extend(detect_wedge(df, order))
    return patterns
