"""
Market structure analysis: Break of Structure (BOS), Change of Character (ChoCH),
Support/Resistance, Supply/Demand Zones, Liquidity Sweeps, Order Blocks.

Core of Smart Money Concepts (SMC) approach.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from patterns.chart_patterns import find_swing_highs, find_swing_lows


def detect_support_resistance(df: pd.DataFrame, order: int = 10,
                               merge_threshold: float = 0.005) -> Dict[str, List[float]]:
    """
    Detect horizontal support and resistance levels from swing highs/lows.
    Merges nearby levels within merge_threshold percentage.
    """
    highs = find_swing_highs(df["high"], order).dropna().values
    lows = find_swing_lows(df["low"], order).dropna().values

    all_levels = sorted(np.concatenate([highs, lows]))

    # Merge nearby levels
    merged = []
    if len(all_levels) > 0:
        cluster = [all_levels[0]]
        for level in all_levels[1:]:
            if abs(level - cluster[0]) / cluster[0] < merge_threshold:
                cluster.append(level)
            else:
                merged.append(np.mean(cluster))
                cluster = [level]
        merged.append(np.mean(cluster))

    current_price = df["close"].iloc[-1]
    support = [l for l in merged if l < current_price]
    resistance = [l for l in merged if l >= current_price]

    return {
        "support": sorted(support, reverse=True)[:5],  # Closest first
        "resistance": sorted(resistance)[:5],
    }


def detect_supply_demand_zones(df: pd.DataFrame, order: int = 10,
                                min_strength: float = 0.02) -> List[Dict]:
    """
    Identify supply and demand zones.
    
    Demand zone: Area where price dropped to, consolidated, then rallied strongly.
    Supply zone: Area where price rallied to, consolidated, then dropped strongly.
    """
    zones = []
    close = df["close"]
    high = df["high"]
    low = df["low"]

    swing_highs = find_swing_highs(high, order).dropna()
    swing_lows = find_swing_lows(low, order).dropna()

    # Demand zones: around swing lows followed by strong moves up
    for i, (idx, val) in enumerate(swing_lows.items()):
        pos = df.index.get_loc(idx)
        if pos + order < len(df):
            future_close = close.iloc[pos + order]
            move_pct = (future_close - val) / val
            if move_pct > min_strength:
                zone_high = high.iloc[max(0, pos-2):pos+3].max()
                zone_low = low.iloc[max(0, pos-2):pos+3].min()
                zones.append({
                    "type": "demand",
                    "zone_high": zone_high,
                    "zone_low": zone_low,
                    "idx": idx,
                    "strength": move_pct,
                })

    # Supply zones: around swing highs followed by strong moves down
    for i, (idx, val) in enumerate(swing_highs.items()):
        pos = df.index.get_loc(idx)
        if pos + order < len(df):
            future_close = close.iloc[pos + order]
            move_pct = (val - future_close) / val
            if move_pct > min_strength:
                zone_high = high.iloc[max(0, pos-2):pos+3].max()
                zone_low = low.iloc[max(0, pos-2):pos+3].min()
                zones.append({
                    "type": "supply",
                    "zone_high": zone_high,
                    "zone_low": zone_low,
                    "idx": idx,
                    "strength": move_pct,
                })

    return zones


def detect_bos(df: pd.DataFrame, order: int = 5) -> List[Dict]:
    """
    Detect Break of Structure (BOS).
    
    Bullish BOS: Price breaks above a previous swing high in an uptrend.
    Bearish BOS: Price breaks below a previous swing low in a downtrend.
    """
    events = []
    swing_highs = find_swing_highs(df["high"], order).dropna()
    swing_lows = find_swing_lows(df["low"], order).dropna()

    # Check for bullish BOS (break above swing high)
    for i in range(1, len(swing_highs)):
        prev_high = swing_highs.iloc[i - 1]
        curr_idx = swing_highs.index[i]
        pos = df.index.get_loc(curr_idx)

        # Look for a candle that closed above prev swing high
        for j in range(max(0, pos - order), min(pos + order, len(df))):
            if df["close"].iloc[j] > prev_high and df["close"].iloc[j - 1] <= prev_high:
                events.append({
                    "type": "bullish_bos",
                    "idx": df.index[j],
                    "level": prev_high,
                    "price": df["close"].iloc[j],
                })
                break

    # Check for bearish BOS (break below swing low)
    for i in range(1, len(swing_lows)):
        prev_low = swing_lows.iloc[i - 1]
        curr_idx = swing_lows.index[i]
        pos = df.index.get_loc(curr_idx)

        for j in range(max(0, pos - order), min(pos + order, len(df))):
            if df["close"].iloc[j] < prev_low and df["close"].iloc[j - 1] >= prev_low:
                events.append({
                    "type": "bearish_bos",
                    "idx": df.index[j],
                    "level": prev_low,
                    "price": df["close"].iloc[j],
                })
                break

    return events


def detect_choch(df: pd.DataFrame, order: int = 5) -> List[Dict]:
    """
    Detect Change of Character (ChoCH).
    
    Bullish ChoCH: In a downtrend, price breaks above a recent swing high 
                   (first sign of potential reversal).
    Bearish ChoCH: In an uptrend, price breaks below a recent swing low.
    """
    events = []
    swing_highs = find_swing_highs(df["high"], order).dropna()
    swing_lows = find_swing_lows(df["low"], order).dropna()

    # Determine trend at each point using higher highs/lower lows
    for i in range(2, min(len(swing_highs), len(swing_lows))):
        # Downtrend: lower highs and lower lows
        if (i < len(swing_highs) and i < len(swing_lows)):
            h_prev = swing_highs.iloc[i - 1] if i - 1 < len(swing_highs) else None
            h_curr = swing_highs.iloc[i] if i < len(swing_highs) else None
            l_prev = swing_lows.iloc[i - 1] if i - 1 < len(swing_lows) else None
            l_curr = swing_lows.iloc[i] if i < len(swing_lows) else None

            if h_prev and h_curr and l_prev and l_curr:
                # Was in downtrend (lower highs), now breaks above prev high = bullish ChoCH
                if h_curr < h_prev and l_curr < l_prev:
                    idx = swing_highs.index[i]
                    pos = df.index.get_loc(idx)
                    for j in range(pos, min(pos + order * 2, len(df))):
                        if df["close"].iloc[j] > h_prev:
                            events.append({
                                "type": "bullish_choch",
                                "idx": df.index[j],
                                "level": h_prev,
                            })
                            break

                # Was in uptrend (higher lows), now breaks below prev low = bearish ChoCH
                elif h_curr > h_prev and l_curr > l_prev:
                    idx = swing_lows.index[i]
                    pos = df.index.get_loc(idx)
                    for j in range(pos, min(pos + order * 2, len(df))):
                        if df["close"].iloc[j] < l_prev:
                            events.append({
                                "type": "bearish_choch",
                                "idx": df.index[j],
                                "level": l_prev,
                            })
                            break

    return events


def detect_liquidity_sweep(df: pd.DataFrame, order: int = 10,
                            wick_threshold: float = 0.003) -> List[Dict]:
    """
    Detect Liquidity Sweeps.
    
    Price wicks beyond a key level (previous high/low) but closes back inside,
    indicating stop-loss hunting / liquidity grab.
    """
    events = []
    swing_highs = find_swing_highs(df["high"], order).dropna()
    swing_lows = find_swing_lows(df["low"], order).dropna()

    for i in range(len(df)):
        close = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # Check for sweep above resistance (wick above, close below)
        for level in swing_highs.values:
            if not np.isnan(level):
                if high > level and close < level:
                    wick = (high - level) / level
                    if wick > wick_threshold:
                        events.append({
                            "type": "bearish_liquidity_sweep",
                            "idx": df.index[i],
                            "level": level,
                            "wick_above": high,
                            "close": close,
                        })

        # Check for sweep below support (wick below, close above)
        for level in swing_lows.values:
            if not np.isnan(level):
                if low < level and close > level:
                    wick = (level - low) / level
                    if wick > wick_threshold:
                        events.append({
                            "type": "bullish_liquidity_sweep",
                            "idx": df.index[i],
                            "level": level,
                            "wick_below": low,
                            "close": close,
                        })

    return events


def detect_order_blocks(df: pd.DataFrame, order: int = 5) -> List[Dict]:
    """
    Detect Order Blocks (the last opposing candle before a strong move).
    
    Bullish OB: Last bearish candle before a strong bullish move.
    Bearish OB: Last bullish candle before a strong bearish move.
    """
    blocks = []
    close = df["close"]
    open_ = df["open"] if "open" in df.columns else close.shift(0)

    for i in range(order, len(df) - order):
        # Check for strong bullish move after this candle
        future_return = (close.iloc[i + order] - close.iloc[i]) / close.iloc[i]

        if future_return > 0.02:  # Strong up move
            # Find last bearish candle before
            for j in range(i, max(i - order, 0), -1):
                if close.iloc[j] < open_.iloc[j]:
                    blocks.append({
                        "type": "bullish_order_block",
                        "idx": df.index[j],
                        "ob_high": df["high"].iloc[j],
                        "ob_low": df["low"].iloc[j],
                    })
                    break

        elif future_return < -0.02:  # Strong down move
            for j in range(i, max(i - order, 0), -1):
                if close.iloc[j] > open_.iloc[j]:
                    blocks.append({
                        "type": "bearish_order_block",
                        "idx": df.index[j],
                        "ob_high": df["high"].iloc[j],
                        "ob_low": df["low"].iloc[j],
                    })
                    break

    return blocks


def full_structure_analysis(df: pd.DataFrame, order: int = 10) -> Dict:
    """Run complete market structure analysis."""
    return {
        "support_resistance": detect_support_resistance(df, order),
        "supply_demand": detect_supply_demand_zones(df, order),
        "bos": detect_bos(df, order // 2),
        "choch": detect_choch(df, order // 2),
        "liquidity_sweeps": detect_liquidity_sweep(df, order),
        "order_blocks": detect_order_blocks(df, order // 2),
    }
