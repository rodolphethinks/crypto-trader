"""
Ensemble strategy — combines signals from multiple strategies using
majority voting, weighted voting, or unanimity.
"""
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseStrategy):
    """
    Combines signals from multiple child strategies.

    Modes:
        - 'majority':  signal fires when >50% of children agree
        - 'weighted':  each strategy has a weight; weighted sum > threshold fires
        - 'unanimous': all children must agree
        - 'any':       any one child signal fires (union)
    """

    name = "Ensemble"
    description = "Multi-strategy ensemble with configurable voting"
    version = "1.0"

    default_params: Dict[str, Any] = {
        "mode": "majority",       # majority | weighted | unanimous | any
        "threshold": 0.5,         # for weighted mode
        "sl_atr_mult": 1.5,
        "atr_period": 14,
    }

    def __init__(self, strategies: List[BaseStrategy],
                 weights: Optional[List[float]] = None,
                 params: Optional[Dict] = None):
        super().__init__(params)
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)

        names = "+".join(s.name for s in strategies)
        self.name = f"Ensemble({names})"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params
        mode = p["mode"]

        # Collect signals from all child strategies
        child_signals = []
        child_sl = []
        child_tp = []

        for strat in self.strategies:
            sdf = strat.generate_signals(df.copy())
            child_signals.append(sdf["signal"])
            child_sl.append(sdf.get("stop_loss", pd.Series(np.nan, index=df.index)))
            child_tp.append(sdf.get("take_profit", pd.Series(np.nan, index=df.index)))

        signals_df = pd.concat(child_signals, axis=1)
        signals_df.columns = [f"s{i}" for i in range(len(self.strategies))]

        # Combine signals
        if mode == "majority":
            df["signal"] = signals_df.apply(self._majority_vote, axis=1)
        elif mode == "weighted":
            df["signal"] = signals_df.apply(
                lambda row: self._weighted_vote(row, p["threshold"]), axis=1)
        elif mode == "unanimous":
            df["signal"] = signals_df.apply(self._unanimous_vote, axis=1)
        elif mode == "any":
            df["signal"] = signals_df.apply(self._any_vote, axis=1)
        else:
            df["signal"] = Signal.HOLD

        # Use average of child SL/TP where signal fires
        sl_df = pd.concat(child_sl, axis=1)
        tp_df = pd.concat(child_tp, axis=1)

        df["stop_loss"] = sl_df.mean(axis=1)
        df["take_profit"] = tp_df.mean(axis=1)

        # Zero out SL/TP where no signal
        no_signal = df["signal"] == Signal.HOLD
        df.loc[no_signal, "stop_loss"] = np.nan
        df.loc[no_signal, "take_profit"] = np.nan

        # Confidence = fraction of children that agree
        df["confidence"] = 0.0
        for idx in df.index[df["signal"] != Signal.HOLD]:
            sig = df.at[idx, "signal"]
            agreement = sum(1 for col in signals_df.columns
                           if signals_df.at[idx, col] == sig)
            df.at[idx, "confidence"] = round(agreement / len(self.strategies), 2)

        n_buy = (df["signal"] == Signal.BUY).sum()
        n_sell = (df["signal"] == Signal.SELL).sum()
        logger.info(f"{self.name} [{mode}]: {n_buy} BUY, {n_sell} SELL on {len(df)} bars")

        return df

    def _majority_vote(self, row):
        n = len(row)
        buys = (row == Signal.BUY).sum()
        sells = (row == Signal.SELL).sum()
        if buys > n / 2:
            return Signal.BUY
        elif sells > n / 2:
            return Signal.SELL
        return Signal.HOLD

    def _weighted_vote(self, row, threshold):
        buy_weight = sum(w for w, s in zip(self.weights, row) if s == Signal.BUY)
        sell_weight = sum(w for w, s in zip(self.weights, row) if s == Signal.SELL)
        total_weight = sum(self.weights)
        if buy_weight / total_weight >= threshold:
            return Signal.BUY
        elif sell_weight / total_weight >= threshold:
            return Signal.SELL
        return Signal.HOLD

    def _unanimous_vote(self, row):
        vals = row.values
        if all(v == Signal.BUY for v in vals):
            return Signal.BUY
        elif all(v == Signal.SELL for v in vals):
            return Signal.SELL
        return Signal.HOLD

    def _any_vote(self, row):
        if (row == Signal.BUY).any():
            return Signal.BUY
        elif (row == Signal.SELL).any():
            return Signal.SELL
        return Signal.HOLD
