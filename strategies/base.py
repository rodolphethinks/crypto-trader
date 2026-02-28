"""
Base strategy class that all strategies inherit from.

Provides a unified interface for the backtesting engine and live executor.
Each strategy must implement `generate_signals()` which returns
a DataFrame with a 'signal' column: 1 (buy), -1 (sell), 0 (hold).
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Signal:
    BUY = 1
    SELL = -1
    HOLD = 0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "BaseStrategy"
    description: str = ""
    version: str = "1.0"
    
    # Default parameters — override in subclass
    default_params: Dict[str, Any] = {}

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.default_params, **(params or {})}
        self._signals: Optional[pd.DataFrame] = None

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core method: analyze data and generate trading signals.
        
        Must return the input DataFrame with added columns:
        - 'signal': 1 (buy), -1 (sell), 0 (hold)
        - 'stop_loss': suggested stop loss price (optional)
        - 'take_profit': suggested take profit price (optional)
        - Additional indicator columns as needed
        """
        pass

    def get_signal_at(self, df: pd.DataFrame, index: int) -> Dict:
        """Get the signal and metadata at a specific index."""
        if self._signals is None:
            self._signals = self.generate_signals(df)
        
        row = self._signals.iloc[index]
        return {
            "signal": int(row.get("signal", Signal.HOLD)),
            "stop_loss": row.get("stop_loss", None),
            "take_profit": row.get("take_profit", None),
            "confidence": row.get("confidence", 0.5),
        }

    def backtest_summary(self) -> str:
        """Return a text summary of the strategy for display."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name} v{self.version} ({param_str})"

    def __repr__(self):
        return f"<{self.name}({self.params})>"


def combine_signals(*signal_series: pd.Series, method: str = "majority") -> pd.Series:
    """
    Combine multiple signal series into one.
    
    Methods:
    - 'majority': signal = majority vote (ties = hold)
    - 'unanimous': all must agree
    - 'any': any buy/sell triggers
    """
    combined = pd.concat(signal_series, axis=1)

    if method == "majority":
        return combined.apply(
            lambda row: Signal.BUY if (row == Signal.BUY).sum() > len(row) / 2
            else Signal.SELL if (row == Signal.SELL).sum() > len(row) / 2
            else Signal.HOLD,
            axis=1,
        )
    elif method == "unanimous":
        return combined.apply(
            lambda row: row.iloc[0] if (row == row.iloc[0]).all() else Signal.HOLD,
            axis=1,
        )
    elif method == "any":
        return combined.apply(
            lambda row: Signal.BUY if (row == Signal.BUY).any()
            else Signal.SELL if (row == Signal.SELL).any()
            else Signal.HOLD,
            axis=1,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
