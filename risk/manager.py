"""
Risk management: position sizing, stop-loss/take-profit, drawdown limits, exposure control.

Risk model: each trade risks at most *risk_per_trade_pct* of current equity.
Position size is calculated so that the distance to the stop-loss equals that
risk amount.  No daily-loss halt is applied — risk is controlled per-trade.
"""
import logging
from typing import Optional, Dict
from config.settings import MAX_POSITION_SIZE_PCT, MAX_OPEN_POSITIONS

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk parameters for each trade and overall portfolio."""

    def __init__(self,
                 max_position_pct: float = MAX_POSITION_SIZE_PCT,
                 max_open_positions: int = MAX_OPEN_POSITIONS,
                 max_drawdown_pct: float = 20.0,
                 risk_per_trade_pct: float = 1.0):
        self.max_position_pct = max_position_pct
        self.max_open_positions = max_open_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        # kept for backward compat — alias
        self.default_risk_per_trade_pct = risk_per_trade_pct

        # Track state
        self.open_positions = 0
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0

    def update_equity(self, equity: float):
        """Update current equity and track peak."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        return ((self.peak_equity - self.current_equity) / self.peak_equity) * 100

    def can_open_position(self) -> bool:
        """Check if a new position can be opened.
        
        Only checks position count and overall drawdown.
        Per-trade risk is enforced via position sizing, not a daily halt.
        """
        if self.open_positions >= self.max_open_positions:
            logger.warning("Max open positions reached")
            return False
        if self.current_drawdown_pct() >= self.max_drawdown_pct:
            logger.warning("Max drawdown reached — trading halted")
            return False
        return True

    def calculate_position_size(self, capital: float, entry_price: float,
                                 stop_loss: float, 
                                 risk_pct: Optional[float] = None) -> float:
        """
        Calculate position size so that the trade risks at most *risk_pct* %
        of current capital.

        Formula:  size = (capital * risk%) / |entry - stop_loss|
        Capped at max_position_pct of capital.
        """
        risk = risk_pct if risk_pct is not None else self.risk_per_trade_pct
        risk_amount = capital * (risk / 100)
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0

        size = risk_amount / price_risk

        # Cap at max position size
        max_size = (capital * self.max_position_pct / 100) / entry_price
        size = min(size, max_size)

        return round(size, 8)

    def calculate_stop_loss(self, entry_price: float, side: str,
                            atr_value: float, multiplier: float = 1.5) -> float:
        """Calculate ATR-based stop loss."""
        if side.upper() == "BUY":
            return entry_price - (atr_value * multiplier)
        else:
            return entry_price + (atr_value * multiplier)

    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                               risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(entry_price - stop_loss)
        if entry_price > stop_loss:  # Long
            return entry_price + (risk * risk_reward_ratio)
        else:  # Short
            return entry_price - (risk * risk_reward_ratio)

    def validate_trade(self, entry_price: float, stop_loss: float,
                       take_profit: float, side: str) -> Dict:
        """Validate a trade setup and return risk metrics."""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        valid = True
        reasons = []

        if rr_ratio < 1.0:
            valid = False
            reasons.append(f"Poor risk-reward ratio: {rr_ratio:.2f}")

        if side.upper() == "BUY":
            if stop_loss >= entry_price:
                valid = False
                reasons.append("Stop loss above entry for long")
            if take_profit <= entry_price:
                valid = False
                reasons.append("Take profit below entry for long")
        else:
            if stop_loss <= entry_price:
                valid = False
                reasons.append("Stop loss below entry for short")
            if take_profit >= entry_price:
                valid = False
                reasons.append("Take profit above entry for short")

        return {
            "valid": valid,
            "risk_reward": round(rr_ratio, 2),
            "risk_pct": round((risk / entry_price) * 100, 4),
            "reasons": reasons,
        }

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0

    def register_trade_result(self, pnl_pct: float):
        """Register a completed trade."""
        self.daily_pnl += pnl_pct
        if pnl_pct >= 0:
            logger.info(f"Trade closed with +{pnl_pct:.2f}% profit")
        else:
            logger.info(f"Trade closed with {pnl_pct:.2f}% loss")
