"""
Production 2-Layer Trading System for MEXC Spot (Long-Only).

Architecture:
  Layer 1 — Strategy Execution (PRIMARY): 7 walk-forward validated configs
  Layer 2 — Catastrophic Risk Filter (OVERRIDE ONLY): slow BTC kill switch

Capital is allocated across strategies with tier-based percentages.
No regime filters, no RS ranking — strategies proved to work better raw.

Long-only only. No shorting on MEXC spot.
"""
import sys
import os
import json
import time
import logging
import signal as os_signal
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.client import MEXCClient
from data.fetcher import DataFetcher
from strategies.v6_aggressive import RegimeMomentumV2Strategy
from strategies.v7_diverse import AdaptiveChannelStrategy, VolatilityCaptureStrategy
from strategies.v8_research import VolBreakoutMomentumStrategy, TSMOMStrategy
from strategies.base import Signal
from config.settings import LOG_DIR

logger = logging.getLogger("production_trader")

# ──────────────────────────────────────────────────────────────────────────────
#  Strategy Map
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_MAP = {
    "AdaptiveChannelStrategy": AdaptiveChannelStrategy,
    "VolatilityCaptureStrategy": VolatilityCaptureStrategy,
    "RegimeMomentumV2Strategy": RegimeMomentumV2Strategy,
    "VolBreakoutMomentumStrategy": VolBreakoutMomentumStrategy,
    "TSMOMStrategy": TSMOMStrategy,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy Configs — Long-Only Validated
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """A validated strategy configuration with allocation."""
    name: str
    strategy_class: str
    symbol: str
    interval: str
    tier: int                   # 1=core, 2=active, 3=reduced
    allocation_pct: float       # % of total capital allocated
    risk_pct: float             # % of allocated capital risked per trade
    oos_weekly: float           # Walk-forward OOS weekly return (long-only)
    folds_positive: str         # e.g. "5/5"
    max_consecutive_losses: int = 5   # disable after N consecutive losses
    enabled: bool = True
    params: Optional[Dict] = None     # strategy param overrides


# Long-only walk-forward validated configs, ranked by reliability
STRATEGY_CONFIGS = [
    # ── Tier 1: Core — Always active (5/5 validation) ──
    StrategyConfig(
        name="AdaptChan_NEAR",
        strategy_class="AdaptiveChannelStrategy",
        symbol="NEARUSDC",
        interval="4h",
        tier=1,
        allocation_pct=25.0,   # capped at 25% (not 35-40%, concentration risk)
        risk_pct=5.0,
        oos_weekly=1.15,
        folds_positive="5/5",
    ),
    StrategyConfig(
        name="AdaptChan_SOL",
        strategy_class="AdaptiveChannelStrategy",
        symbol="SOLUSDC",
        interval="4h",
        tier=1,
        allocation_pct=15.0,
        risk_pct=5.0,
        oos_weekly=0.70,
        folds_positive="5/5",
    ),
    # ── Tier 2: Active (4/5 validation) ──
    StrategyConfig(
        name="VBM_XRP",
        strategy_class="VolBreakoutMomentumStrategy",
        symbol="XRPUSDC",
        interval="4h",
        tier=2,
        allocation_pct=12.0,
        risk_pct=5.0,
        oos_weekly=1.17,
        folds_positive="4/5",
    ),
    StrategyConfig(
        name="VolCapture_AVAX",
        strategy_class="VolatilityCaptureStrategy",
        symbol="AVAXUSDC",
        interval="4h",
        tier=2,
        allocation_pct=10.0,
        risk_pct=5.0,
        oos_weekly=0.84,
        folds_positive="4/5",
    ),
    StrategyConfig(
        name="TSMOM_ADA",
        strategy_class="TSMOMStrategy",
        symbol="ADAUSDC",
        interval="4h",
        tier=2,
        allocation_pct=8.0,
        risk_pct=5.0,
        oos_weekly=0.94,
        folds_positive="4/5",
    ),
    # ── Tier 3: Reduced (3/5 validation) ──
    StrategyConfig(
        name="VBM_DOGE",
        strategy_class="VolBreakoutMomentumStrategy",
        symbol="DOGEUSDC",
        interval="4h",
        tier=3,
        allocation_pct=5.0,
        risk_pct=5.0,
        oos_weekly=1.05,
        folds_positive="3/5",
    ),
    StrategyConfig(
        name="RegimeMomV2_XRP",
        strategy_class="RegimeMomentumV2Strategy",
        symbol="XRPUSDC",
        interval="4h",
        tier=3,
        allocation_pct=5.0,
        risk_pct=5.0,
        oos_weekly=0.99,
        folds_positive="3/5",
    ),
]

# Total allocation: 80%. Remaining 20% stays in USDC reserve.
# Max per symbol: NEAR 25%, XRP 17% (VBM+RegMom), AVAX 10%, SOL 15%, ADA 8%, DOGE 5%


# ──────────────────────────────────────────────────────────────────────────────
#  Catastrophic Risk Filter (Layer 2 — Override Only)
# ──────────────────────────────────────────────────────────────────────────────

class CatastrophicFilter:
    """
    Slow kill switch that activates ONLY during severe BTC downturns.
    ALL 4 conditions must be true for >= persistence_candles in a row.
    
    Conditions (all evaluated on t-1 data to prevent lookahead):
      1. BTC below 200-period SMA
      2. BTC making lower lows (3 consecutive lower lows)
      3. BTC volatility contracting (ATR declining)
      4. Breakout failure rate > 70% over last 20 bars
    
    This is NOT for optimization. It's for capital preservation in crashes.
    """

    def __init__(self, sma_period: int = 200, ll_count: int = 3,
                 atr_period: int = 14, breakout_lookback: int = 20,
                 persistence_candles: int = 3):
        self.sma_period = sma_period
        self.ll_count = ll_count
        self.atr_period = atr_period
        self.breakout_lookback = breakout_lookback
        self.persistence_candles = persistence_candles
        self._consecutive_active = 0
        self._active = False

    def evaluate(self, btc_df: pd.DataFrame) -> bool:
        """
        Evaluate catastrophic conditions on BTC data.
        Returns True if kill switch should be ACTIVE (reduce exposure).
        """
        if btc_df is None or len(btc_df) < self.sma_period + 20:
            return False

        c = btc_df["close"]
        h = btc_df["high"]
        l = btc_df["low"]

        # All on t-1 to prevent lookahead
        sma200 = c.rolling(self.sma_period).mean().shift(1)
        atr = self._atr(btc_df, self.atr_period).shift(1)

        # Latest available values (t-1)
        idx = len(btc_df) - 1

        # Condition 1: BTC below 200 SMA
        below_sma = c.iloc[idx - 1] < sma200.iloc[idx]

        # Condition 2: 3 consecutive lower lows
        lows = l.iloc[idx - self.ll_count - 1:idx]
        lower_lows = all(
            lows.iloc[i] < lows.iloc[i - 1]
            for i in range(1, len(lows))
        ) if len(lows) >= self.ll_count else False

        # Condition 3: ATR declining (current ATR < ATR 10 bars ago)
        vol_contracting = False
        if idx >= 11:
            vol_contracting = atr.iloc[idx] < atr.iloc[idx - 10]

        # Condition 4: Breakout failure rate > 70%
        breakout_fail = self._breakout_failure_rate(btc_df) > 0.70

        all_conditions = below_sma and lower_lows and vol_contracting and breakout_fail

        if all_conditions:
            self._consecutive_active += 1
        else:
            self._consecutive_active = 0

        was_active = self._active
        self._active = self._consecutive_active >= self.persistence_candles

        if self._active and not was_active:
            logger.warning("CATASTROPHIC FILTER ACTIVATED — reducing exposure")
        elif not self._active and was_active:
            logger.info("Catastrophic filter deactivated — resuming normal operations")

        return self._active

    @property
    def is_active(self) -> bool:
        return self._active

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _breakout_failure_rate(self, df: pd.DataFrame) -> float:
        """
        Over the last breakout_lookback bars, how many 5-bar high breakouts
        failed to hold after 3 bars?
        """
        n = self.breakout_lookback
        if len(df) < n + 10:
            return 0.0

        c = df["close"].values
        h = df["high"].values
        end = len(df) - 1  # use t-1

        breakouts = 0
        failures = 0

        for i in range(end - n, end):
            if i < 6:
                continue
            # Breakout: high[i] > max(high[i-5:i])
            prev_high = max(h[i - 5:i])
            if h[i] > prev_high:
                breakouts += 1
                # Failure: close 3 bars later < breakout close
                if i + 3 < len(c) and c[i + 3] < c[i]:
                    failures += 1

        if breakouts == 0:
            return 0.0
        return failures / breakouts


# ──────────────────────────────────────────────────────────────────────────────
#  Capital Allocator
# ──────────────────────────────────────────────────────────────────────────────

class CapitalAllocator:
    """
    Manages capital allocation across strategies with dynamic adjustments.

    Base allocations come from config tiers. Adjustments:
      - Increase +5% if strategy performing well (last 20 trades WR > 50%)
      - Decrease -10% after 2-3 consecutive losses
      - Disable after max_consecutive_losses
      - Catastrophic mode: reduce total deployment to 30%
    """

    def __init__(self, total_capital: float, configs: List[StrategyConfig]):
        self.total_capital = total_capital
        self.peak_capital = total_capital
        self.configs = {c.name: c for c in configs}
        # Track base + adjusted allocations
        self.base_allocations = {c.name: c.allocation_pct for c in configs}
        self.adjustments = {c.name: 0.0 for c in configs}
        self.disabled = {c.name: False for c in configs}

    def update_capital(self, capital: float):
        self.total_capital = capital
        if capital > self.peak_capital:
            self.peak_capital = capital

    def get_drawdown_pct(self) -> float:
        if self.peak_capital == 0:
            return 0.0
        return (self.peak_capital - self.total_capital) / self.peak_capital * 100

    def get_allocation(self, config_name: str, catastrophic: bool = False) -> float:
        """
        Get current capital allocation in dollars for a strategy.
        Returns 0 if strategy is disabled.
        """
        if self.disabled.get(config_name, False):
            return 0.0

        base_pct = self.base_allocations.get(config_name, 0.0)
        adj = self.adjustments.get(config_name, 0.0)
        effective_pct = max(0.0, min(base_pct + adj, base_pct * 2))  # cap at 2× base

        if catastrophic:
            # Only Tier 1 strategies at reduced allocation
            cfg = self.configs.get(config_name)
            if cfg and cfg.tier == 1:
                effective_pct = effective_pct * 0.5  # 50% of normal
            else:
                return 0.0

        # Global drawdown check: if equity DD >= 15%, cap total at 30%
        if self.get_drawdown_pct() >= 15.0 and not catastrophic:
            effective_pct = effective_pct * 0.4  # roughly reduces ~80% total to ~30%

        return self.total_capital * effective_pct / 100

    def adjust_for_performance(self, config_name: str, recent_trades: List[Dict]):
        """Adjust allocation based on recent trade performance."""
        if not recent_trades:
            return

        wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        total = len(recent_trades)
        wr = wins / total if total > 0 else 0.5

        # Count consecutive losses from the end
        consec_losses = 0
        for t in reversed(recent_trades):
            if t.get("pnl", 0) <= 0:
                consec_losses += 1
            else:
                break

        cfg = self.configs.get(config_name)
        if not cfg:
            return

        # Disable if too many consecutive losses
        if consec_losses >= cfg.max_consecutive_losses:
            self.disabled[config_name] = True
            logger.warning(
                f"DISABLED {config_name}: {consec_losses} consecutive losses "
                f"(threshold: {cfg.max_consecutive_losses})"
            )
            return

        base = self.base_allocations.get(config_name, 0.0)

        if consec_losses >= 3:
            # Decrease by 10% of base
            self.adjustments[config_name] = -base * 0.10
        elif consec_losses >= 2:
            # Decrease by 5% of base
            self.adjustments[config_name] = -base * 0.05
        elif total >= 10 and wr > 0.55:
            # Increase by 5% of base
            self.adjustments[config_name] = base * 0.05
        else:
            self.adjustments[config_name] = 0.0

    def reenable_strategy(self, config_name: str):
        """Manually re-enable a disabled strategy."""
        self.disabled[config_name] = False
        self.adjustments[config_name] = 0.0
        logger.info(f"Re-enabled {config_name}")

    def get_summary(self) -> Dict[str, Any]:
        """Summary of all allocations."""
        result = {}
        for name, cfg in self.configs.items():
            base = self.base_allocations[name]
            adj = self.adjustments[name]
            disabled = self.disabled[name]
            result[name] = {
                "tier": cfg.tier,
                "base_pct": base,
                "adjustment": adj,
                "effective_pct": max(0, base + adj) if not disabled else 0,
                "disabled": disabled,
                "allocation_usd": self.get_allocation(name),
            }
        return result


# ──────────────────────────────────────────────────────────────────────────────
#  State Management
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    config_name: str
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    allocated_capital: float    # capital allocated at entry


@dataclass
class ClosedTrade:
    config_name: str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    reason: str


@dataclass
class ConfigState:
    """Per-config state tracking."""
    name: str
    position: Optional[Dict] = None
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    recent_trades: List[Dict] = field(default_factory=list)  # last 20 trades
    halted: bool = False
    cooldown_until: Optional[str] = None  # ISO timestamp


class TradingState:
    """Persistent state for the entire trading system."""

    def __init__(self, state_dir: Path):
        self.state_file = state_dir / "production_state.json"
        self.trades_file = state_dir / "production_trades.csv"
        state_dir.mkdir(parents=True, exist_ok=True)

        self.total_capital: float = 10_000
        self.peak_capital: float = 10_000
        self.configs: Dict[str, ConfigState] = {}
        self.catastrophic_active: bool = False
        self.global_halted: bool = False
        self.start_time: str = datetime.now(timezone.utc).isoformat()
        self.last_tick: str = ""

    def init_config(self, cfg: StrategyConfig):
        if cfg.name not in self.configs:
            self.configs[cfg.name] = ConfigState(name=cfg.name)

    def save(self):
        state = {
            "total_capital": self.total_capital,
            "peak_capital": self.peak_capital,
            "catastrophic_active": self.catastrophic_active,
            "global_halted": self.global_halted,
            "start_time": self.start_time,
            "last_tick": self.last_tick,
            "configs": {},
        }
        for name, cs in self.configs.items():
            state["configs"][name] = {
                "name": cs.name,
                "position": cs.position,
                "trade_count": cs.trade_count,
                "win_count": cs.win_count,
                "total_pnl": cs.total_pnl,
                "consecutive_losses": cs.consecutive_losses,
                "recent_trades": cs.recent_trades[-20:],  # keep last 20
                "halted": cs.halted,
                "cooldown_until": cs.cooldown_until,
            }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> bool:
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            self.total_capital = state.get("total_capital", 10_000)
            self.peak_capital = state.get("peak_capital", 10_000)
            self.catastrophic_active = state.get("catastrophic_active", False)
            self.global_halted = state.get("global_halted", False)
            self.start_time = state.get("start_time", "")
            self.last_tick = state.get("last_tick", "")
            for name, cdata in state.get("configs", {}).items():
                cs = ConfigState(
                    name=cdata["name"],
                    position=cdata.get("position"),
                    trade_count=cdata.get("trade_count", 0),
                    win_count=cdata.get("win_count", 0),
                    total_pnl=cdata.get("total_pnl", 0),
                    consecutive_losses=cdata.get("consecutive_losses", 0),
                    recent_trades=cdata.get("recent_trades", []),
                    halted=cdata.get("halted", False),
                    cooldown_until=cdata.get("cooldown_until"),
                )
                self.configs[name] = cs
            logger.info(
                f"Loaded state: {len(self.configs)} configs, "
                f"capital=${self.total_capital:,.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def record_trade(self, trade: ClosedTrade, cs: ConfigState):
        """Record a closed trade and update state."""
        trade_dict = asdict(trade)

        # Update config state
        cs.trade_count += 1
        cs.total_pnl += trade.pnl
        if trade.pnl > 0:
            cs.win_count += 1
            cs.consecutive_losses = 0
        else:
            cs.consecutive_losses += 1

        cs.recent_trades.append(trade_dict)
        cs.recent_trades = cs.recent_trades[-20:]  # keep last 20

        # Update total capital
        self.total_capital += trade.pnl
        if self.total_capital > self.peak_capital:
            self.peak_capital = self.total_capital

        # Append to CSV
        df = pd.DataFrame([trade_dict])
        write_header = not self.trades_file.exists()
        df.to_csv(self.trades_file, mode="a", header=write_header, index=False)

        self.save()


# ──────────────────────────────────────────────────────────────────────────────
#  Interval Helpers
# ──────────────────────────────────────────────────────────────────────────────

INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "8h": 28800, "1d": 86400,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Production Trader
# ──────────────────────────────────────────────────────────────────────────────

class ProductionTrader:
    """
    2-Layer Production Trading Engine.

    Layer 1: Strategy Execution — runs 7 validated configs with capital allocation.
    Layer 2: Catastrophic Filter — BTC-based kill switch (override only).

    Long-only. No shorting.
    """

    def __init__(
        self,
        configs: Optional[List[StrategyConfig]] = None,
        capital: float = 10_000,
        mode: str = "paper",
    ):
        self.configs = configs or STRATEGY_CONFIGS
        self.initial_capital = capital
        self.mode = mode
        self.client = MEXCClient()
        self.fetcher = DataFetcher(client=self.client)

        # Strategy instances (cached)
        self._strategies: Dict[str, Any] = {}

        # State
        self.state = TradingState(LOG_DIR)

        # Capital allocation
        self.allocator = CapitalAllocator(capital, self.configs)

        # Catastrophic filter
        self.cat_filter = CatastrophicFilter()

        # BTC data cache
        self._btc_df: Optional[pd.DataFrame] = None
        self._btc_cache_time: float = 0

        # Data cache per symbol (avoid re-fetching within same tick)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._data_cache_time: float = 0

        self._running = True
        os_signal.signal(os_signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        logger.info("Shutdown signal received...")
        self._running = False

    # ── Strategy Factory ───────────────────────────────────────────────────

    def _get_strategy(self, cfg: StrategyConfig) -> Any:
        key = f"{cfg.strategy_class}_{cfg.name}"
        if key not in self._strategies:
            cls = STRATEGY_MAP.get(cfg.strategy_class)
            if cls is None:
                raise ValueError(f"Unknown strategy: {cfg.strategy_class}")
            params = cfg.params or {}
            self._strategies[key] = cls(params=params if params else None)
        return self._strategies[key]

    # ── Data Fetching ──────────────────────────────────────────────────────

    def _fetch_data(self, symbol: str, interval: str,
                    lookback_bars: int = 300) -> pd.DataFrame:
        now = time.time()
        cache_key = f"{symbol}_{interval}"

        # Use cache within same tick (60s window)
        if (cache_key in self._data_cache and
                now - self._data_cache_time < 60):
            return self._data_cache[cache_key]

        secs = INTERVAL_SECONDS.get(interval, 3600)
        days = max(2, (lookback_bars * secs) // 86400 + 2)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        try:
            df = self.fetcher.fetch_klines(
                symbol, interval,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            if not df.empty:
                self._data_cache[cache_key] = df
                self._data_cache_time = now
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {interval}: {e}")
            return pd.DataFrame()

    def _fetch_btc_data(self) -> pd.DataFrame:
        """Fetch BTC 4h data with 5-minute cache."""
        now = time.time()
        if self._btc_df is not None and now - self._btc_cache_time < 300:
            return self._btc_df

        df = self._fetch_data("BTCUSDC", "4h", lookback_bars=300)
        if not df.empty:
            self._btc_df = df
            self._btc_cache_time = now
        return df if not df.empty else (self._btc_df or pd.DataFrame())

    # ── Extreme Candle Check ───────────────────────────────────────────────

    @staticmethod
    def _is_extreme_candle(df: pd.DataFrame, atr_mult: float = 2.0) -> bool:
        """
        Check if the latest candle is extreme (body > atr_mult × ATR).
        Avoid entering after extreme moves.
        """
        if len(df) < 15:
            return False
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        latest_range = abs(c.iloc[-1] - df["open"].iloc[-1])
        return latest_range > atr_mult * atr.iloc[-1]

    # ── Cooldown Check ─────────────────────────────────────────────────────

    @staticmethod
    def _in_cooldown(cs: ConfigState) -> bool:
        if cs.cooldown_until is None:
            return False
        try:
            until = datetime.fromisoformat(cs.cooldown_until)
            return datetime.now(timezone.utc) < until
        except (ValueError, TypeError):
            return False

    def _set_cooldown(self, cs: ConfigState, bars: int, interval: str):
        """Set cooldown for N bars after a stop-loss."""
        secs = INTERVAL_SECONDS.get(interval, 14400) * bars
        until = datetime.now(timezone.utc) + timedelta(seconds=secs)
        cs.cooldown_until = until.isoformat()

    # ── Correlation Check ──────────────────────────────────────────────────

    def _check_correlation_limit(self, cfg: StrategyConfig) -> bool:
        """
        Ensure we don't exceed 60% allocation to highly correlated positions.
        Correlated groups: [XRP pair configs], [NEAR+SOL = L1 alts]
        """
        CORR_GROUPS = {
            "L1_ALTS": ["NEARUSDC", "SOLUSDC", "AVAXUSDC"],
            "XRP": ["XRPUSDC"],
            "MEME": ["DOGEUSDC"],
            "ADA": ["ADAUSDC"],
        }

        # Find which group this symbol belongs to
        my_group = None
        for group, symbols in CORR_GROUPS.items():
            if cfg.symbol in symbols:
                my_group = group
                break

        if my_group is None:
            return True  # no group, no limit

        # Sum allocated capital for all open positions in same group
        group_symbols = CORR_GROUPS[my_group]
        group_deployed = 0.0
        for name, cs in self.state.configs.items():
            if cs.position:
                pos_cfg = self.configs_by_name.get(name)
                if pos_cfg and pos_cfg.symbol in group_symbols:
                    group_deployed += cs.position.get("allocated_capital", 0)

        # Check if adding this config would exceed 60% of total capital
        new_alloc = self.allocator.get_allocation(
            cfg.name, catastrophic=self.cat_filter.is_active
        )
        total = group_deployed + new_alloc
        limit = self.state.total_capital * 0.60

        if total > limit:
            logger.info(
                f"Correlation limit: {cfg.name} blocked "
                f"(group {my_group} would be {total/self.state.total_capital*100:.0f}%)"
            )
            return False
        return True

    # ── Position Sizing ────────────────────────────────────────────────────

    def _calculate_position_size(
        self, cfg: StrategyConfig, allocated_capital: float,
        entry_price: float, stop_loss: float
    ) -> float:
        """
        Position size based on risk % of allocated capital.
        Risk amount / (entry - stop) = quantity.
        """
        risk_amount = allocated_capital * (cfg.risk_pct / 100)

        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0

        size = risk_amount / price_risk

        # Cap: don't use more than allocated capital
        max_size = allocated_capital / entry_price
        size = min(size, max_size)

        return round(size, 8)

    # ── Trade Execution ────────────────────────────────────────────────────

    def _open_position(
        self, cfg: StrategyConfig, cs: ConfigState,
        entry_price: float, quantity: float,
        stop_loss: float, take_profit: float,
        allocated_capital: float
    ):
        """Open a LONG position."""
        if self.mode == "paper":
            logger.info(
                f"[PAPER] OPEN {cfg.name}: BUY {quantity:.6f} {cfg.symbol} "
                f"@ ${entry_price:.4f} (SL=${stop_loss:.4f} TP=${take_profit:.4f} "
                f"alloc=${allocated_capital:.2f})"
            )
        else:
            try:
                result = self.client.place_order(
                    symbol=cfg.symbol,
                    side="BUY",
                    order_type="MARKET",
                    quantity=quantity,
                )
                if result:
                    logger.info(
                        f"[LIVE] OPEN {cfg.name}: BUY {quantity:.6f} {cfg.symbol} "
                        f"— Order: {result.get('orderId')}"
                    )
                else:
                    logger.error(f"[LIVE] Failed to open {cfg.name}")
                    return
            except Exception as e:
                logger.error(f"[LIVE] Order error for {cfg.name}: {e}")
                return

        cs.position = asdict(OpenPosition(
            config_name=cfg.name,
            symbol=cfg.symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc).isoformat(),
            allocated_capital=allocated_capital,
        ))

    def _close_position(self, cfg: StrategyConfig, cs: ConfigState,
                        exit_price: float, reason: str):
        """Close an existing LONG position."""
        if not cs.position:
            return

        pos = cs.position
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]
        pnl = (exit_price - entry_price) * quantity
        alloc = pos.get("allocated_capital", 1)
        pnl_pct = (pnl / alloc) * 100 if alloc > 0 else 0

        if self.mode == "live":
            try:
                self.client.place_order(
                    symbol=cfg.symbol,
                    side="SELL",
                    order_type="MARKET",
                    quantity=quantity,
                )
            except Exception as e:
                logger.error(f"[LIVE] Close order error: {e}")
                return

        trade = ClosedTrade(
            config_name=cfg.name,
            symbol=cfg.symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            entry_time=pos["entry_time"],
            exit_time=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self.state.record_trade(trade, cs)
        cs.position = None

        # Set cooldown after stop-loss
        if reason == "stop_loss":
            self._set_cooldown(cs, 2, cfg.interval)

        icon = "+" if pnl >= 0 else "-"
        logger.info(
            f"[{icon}] CLOSED {cfg.name}: BUY @ ${entry_price:.4f} -> "
            f"${exit_price:.4f} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) [{reason}]"
        )

    # ── Signal Processing ──────────────────────────────────────────────────

    def _process_config(self, cfg: StrategyConfig, cs: ConfigState):
        """Process one strategy config for the current tick."""

        # === Halted check ===
        if cs.halted:
            return

        # === Fetch data ===
        df = self._fetch_data(cfg.symbol, cfg.interval)
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for {cfg.name}")
            return

        current_price = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]

        # === EXISTING POSITION: Check exits ===
        if cs.position:
            pos = cs.position
            sl = pos["stop_loss"]
            tp = pos["take_profit"]

            exit_reason = None
            exit_price = current_price

            # Stop-loss
            if sl > 0 and current_low <= sl:
                exit_reason = "stop_loss"
                exit_price = sl
            # Take-profit
            elif tp > 0 and current_high >= tp:
                exit_reason = "take_profit"
                exit_price = tp

            # Signal-based exit (strategy says SELL or HOLD with specific logic)
            if not exit_reason:
                strategy = self._get_strategy(cfg)
                sig_df = strategy.generate_signals(df.copy())
                last_sig = sig_df["signal"].iloc[-1]
                if last_sig == Signal.SELL:
                    exit_reason = "signal_exit"
                    exit_price = current_price

            if exit_reason:
                self._close_position(cfg, cs, exit_price, exit_reason)
            return

        # === NO POSITION: Check for new entry ===

        # Cooldown check
        if self._in_cooldown(cs):
            return

        # Generate signal
        strategy = self._get_strategy(cfg)
        sig_df = strategy.generate_signals(df.copy())
        last_signal = sig_df["signal"].iloc[-1]

        # Long-only: only act on BUY signals
        if last_signal != Signal.BUY:
            return

        # Extreme candle filter
        if self._is_extreme_candle(df):
            logger.info(f"Extreme candle — skipping entry for {cfg.name}")
            return

        # Correlation limit
        if not self._check_correlation_limit(cfg):
            return

        # Get allocated capital
        catastrophic = self.cat_filter.is_active
        allocated = self.allocator.get_allocation(cfg.name, catastrophic=catastrophic)
        if allocated <= 0:
            return

        # Get SL/TP from strategy
        sl = sig_df["stop_loss"].iloc[-1] if "stop_loss" in sig_df else 0
        tp = sig_df["take_profit"].iloc[-1] if "take_profit" in sig_df else 0
        if pd.isna(sl):
            sl = 0
        if pd.isna(tp):
            tp = 0

        # Long-only: SL must be below entry
        entry_price = current_price
        if sl >= entry_price:
            sl = 0

        # Position sizing
        if sl > 0:
            quantity = self._calculate_position_size(
                cfg, allocated, entry_price, sl
            )
        else:
            quantity = allocated / entry_price

        if quantity <= 0:
            return

        # Final cost check
        cost = quantity * entry_price
        if cost > allocated * 1.01:
            quantity = allocated / entry_price

        self._open_position(
            cfg, cs, entry_price, quantity, sl, tp, allocated
        )

    # ── Main Tick ──────────────────────────────────────────────────────────

    def tick(self):
        """Process one full tick across all strategies."""
        self.state.last_tick = datetime.now(timezone.utc).isoformat()

        # Clear data cache for new tick
        self._data_cache.clear()

        # Update allocator with current capital
        self.allocator.update_capital(self.state.total_capital)

        # === Layer 2: Catastrophic filter ===
        btc_df = self._fetch_btc_data()
        catastrophic = self.cat_filter.evaluate(btc_df) if not btc_df.empty else False
        self.state.catastrophic_active = catastrophic

        # === Global drawdown check ===
        dd = self.allocator.get_drawdown_pct()
        if dd >= 15.0:
            if not self.state.global_halted:
                logger.warning(
                    f"GLOBAL DRAWDOWN {dd:.1f}% >= 15% — "
                    "reducing exposure, no new trades"
                )
                self.state.global_halted = True
        elif dd < 10.0 and self.state.global_halted:
            logger.info("Global drawdown recovered below 10% — resuming")
            self.state.global_halted = False

        # === Layer 1: Process each strategy ===
        for cfg in self.configs:
            if not cfg.enabled:
                continue
            cs = self.state.configs.get(cfg.name)
            if not cs:
                continue

            # Update allocations based on recent performance
            self.allocator.adjust_for_performance(cfg.name, cs.recent_trades)

            # Global halt: only allow closing existing positions
            if self.state.global_halted and not cs.position:
                continue

            try:
                self._process_config(cfg, cs)
            except Exception as e:
                logger.error(f"Error processing {cfg.name}: {e}", exc_info=True)

        self.state.save()

    # ── Initialization ─────────────────────────────────────────────────────

    def initialize(self):
        """Initialize state and strategy configs."""
        loaded = self.state.load()
        if loaded:
            self.allocator.update_capital(self.state.total_capital)
            logger.info("Resumed from saved state")
        else:
            self.state.total_capital = self.initial_capital
            self.state.peak_capital = self.initial_capital
            logger.info("Starting fresh")

        # Build name lookup
        self.configs_by_name = {c.name: c for c in self.configs}

        for cfg in self.configs:
            self.state.init_config(cfg)

        self.state.save()

    # ── Status Display ─────────────────────────────────────────────────────

    def print_status(self):
        print()
        print("=" * 90)
        print(
            f"  PRODUCTION TRADER — "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        print(
            f"  Mode: {self.mode.upper()} | Capital: ${self.state.total_capital:,.2f} | "
            f"Peak: ${self.state.peak_capital:,.2f} | "
            f"DD: {self.allocator.get_drawdown_pct():.1f}%"
        )
        cat_str = "ACTIVE" if self.cat_filter.is_active else "inactive"
        halt_str = " | GLOBAL HALT" if self.state.global_halted else ""
        print(f"  Catastrophic Filter: {cat_str}{halt_str}")
        print("=" * 90)

        total_pnl = 0
        total_deployed = 0

        for cfg in self.configs:
            cs = self.state.configs.get(cfg.name)
            if not cs:
                continue

            alloc = self.allocator.get_allocation(
                cfg.name, catastrophic=self.cat_filter.is_active
            )
            disabled = self.allocator.disabled.get(cfg.name, False)

            pos_str = "FLAT"
            deployed = 0
            if cs.position:
                p = cs.position
                unrealized = (self._get_current_price(cfg.symbol) or p["entry_price"])
                unrealized_pnl = (unrealized - p["entry_price"]) * p["quantity"]
                deployed = p.get("allocated_capital", 0)
                pos_str = (
                    f"LONG {p['quantity']:.4f} @ ${p['entry_price']:.4f} "
                    f"(uPnL: ${unrealized_pnl:+.2f})"
                )

            status = "DISABLED" if disabled else ("HALTED" if cs.halted else "ACTIVE")
            wr = (cs.win_count / cs.trade_count * 100) if cs.trade_count > 0 else 0

            print(
                f"  T{cfg.tier} {cfg.name:<25} [{status}] "
                f"Alloc: ${alloc:,.0f} ({cfg.allocation_pct}%)"
            )
            print(
                f"     PnL: ${cs.total_pnl:+,.2f} | Trades: {cs.trade_count} "
                f"(WR: {wr:.0f}%) | ConsecL: {cs.consecutive_losses} | "
                f"Pos: {pos_str}"
            )

            total_pnl += cs.total_pnl
            total_deployed += deployed

        reserve = self.state.total_capital - total_deployed
        print(f"\n  Total PnL: ${total_pnl:+,.2f} | "
              f"Deployed: ${total_deployed:,.2f} | "
              f"USDC Reserve: ${reserve:,.2f}")
        print("=" * 90)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.client.get_ticker_price(symbol)
            if ticker and "price" in ticker:
                return float(ticker["price"])
        except Exception:
            pass
        return None

    # ── Main Loop ──────────────────────────────────────────────────────────

    def run(self, poll_interval: Optional[int] = None):
        """Main trading loop."""
        self.initialize()

        if poll_interval is None:
            intervals = [
                INTERVAL_SECONDS.get(c.interval, 3600)
                for c in self.configs if c.enabled
            ]
            min_interval = min(intervals) if intervals else 3600
            poll_interval = max(60, min_interval // 3)

        logger.info(f"Starting {self.mode.upper()} trading loop")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info(
            f"Active configs: {len([c for c in self.configs if c.enabled])}"
        )
        logger.info(
            f"Total allocation: "
            f"{sum(c.allocation_pct for c in self.configs if c.enabled):.0f}% "
            f"+ {100 - sum(c.allocation_pct for c in self.configs if c.enabled):.0f}% USDC reserve"
        )

        self.print_status()
        print(f"\n  Polling every {poll_interval}s. Press Ctrl+C to stop.\n")

        tick_count = 0
        while self._running:
            try:
                self.tick()
                tick_count += 1
                if tick_count % 10 == 0:
                    self.print_status()
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)

            for _ in range(poll_interval):
                if not self._running:
                    break
                time.sleep(1)

        self.print_status()
        self.state.save()
        logger.info("Trader stopped. State saved.")

    # ── One-Shot Signal Check ──────────────────────────────────────────────

    def check_signals(self) -> Dict[str, Any]:
        """
        One-shot signal check for all configs.
        Useful for testing without running the loop.
        """
        self.initialize()

        # Check catastrophic filter
        btc_df = self._fetch_btc_data()
        catastrophic = self.cat_filter.evaluate(btc_df) if not btc_df.empty else False

        results = {
            "_system": {
                "capital": self.state.total_capital,
                "catastrophic_filter": catastrophic,
                "global_halted": self.state.global_halted,
                "drawdown_pct": self.allocator.get_drawdown_pct(),
            }
        }

        for cfg in self.configs:
            if not cfg.enabled:
                continue

            try:
                df = self._fetch_data(cfg.symbol, cfg.interval)
                if df.empty or len(df) < 50:
                    results[cfg.name] = {"signal": "NO_DATA"}
                    continue

                strategy = self._get_strategy(cfg)
                sig_df = strategy.generate_signals(df.copy())
                last_sig = sig_df["signal"].iloc[-1]

                sig_name = "HOLD"
                if last_sig == Signal.BUY:
                    sig_name = "BUY"
                elif last_sig == Signal.SELL:
                    sig_name = "SELL"

                alloc = self.allocator.get_allocation(
                    cfg.name, catastrophic=catastrophic
                )
                cs = self.state.configs.get(cfg.name)

                results[cfg.name] = {
                    "signal": sig_name,
                    "price": float(df["close"].iloc[-1]),
                    "time": str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else str(len(df)),
                    "bars": len(df),
                    "allocation_usd": round(alloc, 2),
                    "extreme_candle": self._is_extreme_candle(df),
                    "in_cooldown": self._in_cooldown(cs) if cs else False,
                    "consecutive_losses": cs.consecutive_losses if cs else 0,
                    "position": "LONG" if (cs and cs.position) else "FLAT",
                }

            except Exception as e:
                results[cfg.name] = {"signal": "ERROR", "error": str(e)}

        return results


# ──────────────────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Production 2-Layer Trader")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000,
        help="Starting capital in USDC"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="One-shot signal check (no trading loop)"
    )
    parser.add_argument(
        "--poll", type=int, default=None,
        help="Poll interval in seconds (default: auto)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current status and exit"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "production_trader.log"),
        ],
    )

    trader = ProductionTrader(
        capital=args.capital,
        mode=args.mode,
    )

    if args.status:
        trader.initialize()
        trader.print_status()
        return

    if args.check:
        results = trader.check_signals()
        print("\n" + "=" * 70)
        print("  SIGNAL CHECK")
        print("=" * 70)
        sys_info = results.pop("_system", {})
        print(
            f"  Capital: ${sys_info.get('capital', 0):,.2f} | "
            f"DD: {sys_info.get('drawdown_pct', 0):.1f}% | "
            f"Catastrophic: {sys_info.get('catastrophic_filter', False)}"
        )
        print("-" * 70)
        for name, info in results.items():
            sig = info.get("signal", "?")
            price = info.get("price", 0)
            alloc = info.get("allocation_usd", 0)
            pos = info.get("position", "FLAT")
            extreme = " [EXTREME]" if info.get("extreme_candle") else ""
            cool = " [COOLDOWN]" if info.get("in_cooldown") else ""
            print(
                f"  {name:<25} {sig:>5} @ ${price:<10.4f} "
                f"Alloc: ${alloc:<8,.0f} Pos: {pos}{extreme}{cool}"
            )
        print("=" * 70)
        return

    if args.mode == "live":
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real orders will be placed on MEXC.")
        print("!" * 60)
        confirm = input("  Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("  Aborted.")
            return

    trader.run(poll_interval=args.poll)


if __name__ == "__main__":
    main()
