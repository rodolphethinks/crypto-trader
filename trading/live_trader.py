"""
Production paper/live trader for walk-forward validated V6 strategies.

Deploys the top OOS-validated configurations with aggressive position sizing.
Each configuration runs independently with its own risk parameters.

Features:
  - State persistence (JSON) — survives restarts
  - BTC data injection for CrossPairLeader
  - Aggressive position sizing (10-20% risk, 75-100% position)
  - Per-config equity tracking  
  - Trade logging to CSV
  - Health monitoring
"""
import sys, os, json, time, logging, signal as os_signal
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.client import MEXCClient
from data.fetcher import DataFetcher
from strategies.v6_aggressive import (
    CrossPairLeaderStrategy,
    MomentumAcceleratorStrategy,
    MultiEdgeCompositeStrategy,
    RegimeMomentumV2Strategy,
)
from strategies.v7_diverse import (
    AdaptiveChannelStrategy,
    VolatilityCaptureStrategy,
)
from strategies.base import Signal
from config.settings import LOG_DIR

logger = logging.getLogger("live_trader")

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration — Walk-Forward Validated Configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradingConfig:
    """A single validated strategy configuration."""
    name: str
    strategy_class: str       # Class name string
    symbol: str
    interval: str
    risk_pct: float           # % capital to risk per trade
    max_position_pct: float   # % capital max per position
    max_drawdown_pct: float   # Max drawdown before halting
    oos_weekly: float         # Walk-forward OOS weekly return
    folds_positive: str       # e.g. "4/4" or "3/4"
    enabled: bool = True


# Walk-forward validated: ALL 5/5 folds positive, all exceed 1%/week OOS
# Verified on full 2-year data (2024-2026) with 5-fold time-series CV
VALIDATED_CONFIGS = [
    # #1: MomAccel AVAX 4h — +1.81%/wk OOS, +295% full data, Sharpe 1.53
    TradingConfig(
        name="MomAccel_AVAX_4h",
        strategy_class="MomentumAcceleratorStrategy",
        symbol="AVAXUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.814,
        folds_positive="5/5",
    ),
    # #2: MultiEdge XRP 4h — +1.33%/wk OOS, +317% full data, Sharpe 1.57
    TradingConfig(
        name="MultiEdge_XRP_4h",
        strategy_class="MultiEdgeCompositeStrategy",
        symbol="XRPUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.327,
        folds_positive="5/5",
    ),
    # #3: RegimeMomV2 XRP 4h — +1.32%/wk OOS, +282% full data, BEST risk-adj (18.9% MaxDD, Sharpe 1.58)
    TradingConfig(
        name="RegimeMomV2_XRP_4h",
        strategy_class="RegimeMomentumV2Strategy",
        symbol="XRPUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.320,
        folds_positive="5/5",
    ),
    # #4: MultiEdge DOGE 4h — +1.13%/wk OOS, +278% full data
    TradingConfig(
        name="MultiEdge_DOGE_4h",
        strategy_class="MultiEdgeCompositeStrategy",
        symbol="DOGEUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.133,
        folds_positive="5/5",
    ),
    # #5: CrossPair AVAX 4h — +1.04%/wk OOS, +126% full data
    TradingConfig(
        name="CrossPair_AVAX_4h",
        strategy_class="CrossPairLeaderStrategy",
        symbol="AVAXUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.041,
        folds_positive="5/5",
    ),
    # #6: CrossPair BTC 4h — +1.01%/wk OOS, +143% full data
    TradingConfig(
        name="CrossPair_BTC_4h",
        strategy_class="CrossPairLeaderStrategy",
        symbol="BTCUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.005,
        folds_positive="5/5",
    ),
    # ── V7 Walk-Forward Validated (5/5 folds positive) ──
    # #7: AdaptChan NEAR 4h — +1.74%/wk OOS, +396% full data, Sharpe 1.63
    TradingConfig(
        name="AdaptChan_NEAR_4h",
        strategy_class="AdaptiveChannelStrategy",
        symbol="NEARUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.740,
        folds_positive="5/5",
    ),
    # #8: AdaptChan SOL 4h — +1.54%/wk OOS, +236% full data, Sharpe 1.61
    TradingConfig(
        name="AdaptChan_SOL_4h",
        strategy_class="AdaptiveChannelStrategy",
        symbol="SOLUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.540,
        folds_positive="5/5",
    ),
    # #9: VolCapture AVAX 4h — +1.53%/wk OOS, +281% full data, BEST risk-adj (6.9% DD, Sharpe 1.99)
    TradingConfig(
        name="VolCapture_AVAX_4h",
        strategy_class="VolatilityCaptureStrategy",
        symbol="AVAXUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.530,
        folds_positive="5/5",
    ),
    # #10: VolCapture NEAR 4h — +1.02%/wk OOS, +128% full data
    TradingConfig(
        name="VolCapture_NEAR_4h",
        strategy_class="VolatilityCaptureStrategy",
        symbol="NEARUSDC",
        interval="4h",
        risk_pct=5.0,
        max_position_pct=100.0,
        max_drawdown_pct=35.0,
        oos_weekly=1.020,
        folds_positive="5/5",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
#  Position & State Management
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    config_name: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    capital_at_entry: float


@dataclass
class ClosedTrade:
    config_name: str
    symbol: str
    side: str
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
    """Per-config equity and position tracking."""
    name: str
    capital: float
    peak_capital: float
    position: Optional[Dict] = None  # Serialized OpenPosition or None
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    halted: bool = False


class TradingState:
    """Persistent state for the entire trading system."""
    
    STATE_FILE = LOG_DIR / "live_trader_state.json"
    TRADES_FILE = LOG_DIR / "live_trades.csv"
    
    def __init__(self, initial_capital: float = 10_000):
        self.initial_capital = initial_capital
        self.configs: Dict[str, ConfigState] = {}
        self.closed_trades: List[Dict] = []
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.last_tick_time = ""
        
    def init_config(self, config: TradingConfig):
        """Initialize state for a config (capital shared across ALL configs)."""
        if config.name not in self.configs:
            self.configs[config.name] = ConfigState(
                name=config.name,
                capital=self.initial_capital,
                peak_capital=self.initial_capital,
            )
    
    def save(self):
        """Persist state to JSON."""
        state = {
            "initial_capital": self.initial_capital,
            "start_time": self.start_time,
            "last_tick_time": self.last_tick_time,
            "configs": {},
            "closed_trades": self.closed_trades,
        }
        for name, cs in self.configs.items():
            state["configs"][name] = {
                "name": cs.name,
                "capital": cs.capital,
                "peak_capital": cs.peak_capital,
                "position": cs.position,
                "trade_count": cs.trade_count,
                "win_count": cs.win_count,
                "total_pnl": cs.total_pnl,
                "halted": cs.halted,
            }
        
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self) -> bool:
        """Load state from JSON. Returns True if loaded."""
        if not self.STATE_FILE.exists():
            return False
        try:
            with open(self.STATE_FILE) as f:
                state = json.load(f)
            self.initial_capital = state.get("initial_capital", 10000)
            self.start_time = state.get("start_time", "")
            self.last_tick_time = state.get("last_tick_time", "")
            self.closed_trades = state.get("closed_trades", [])
            for name, cs_data in state.get("configs", {}).items():
                self.configs[name] = ConfigState(**cs_data)
            logger.info(f"Loaded state: {len(self.configs)} configs, "
                       f"{len(self.closed_trades)} historical trades")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def save_trade(self, trade: ClosedTrade):
        """Append trade to CSV log."""
        trade_dict = asdict(trade)
        self.closed_trades.append(trade_dict)
        
        df = pd.DataFrame([trade_dict])
        write_header = not self.TRADES_FILE.exists()
        df.to_csv(self.TRADES_FILE, mode="a", header=write_header, index=False)
        self.save()  # also persist JSON


# ──────────────────────────────────────────────────────────────────────────────
#  Strategy Factory
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_MAP = {
    "CrossPairLeaderStrategy": CrossPairLeaderStrategy,
    "MomentumAcceleratorStrategy": MomentumAcceleratorStrategy,
    "MultiEdgeCompositeStrategy": MultiEdgeCompositeStrategy,
    "RegimeMomentumV2Strategy": RegimeMomentumV2Strategy,
    "AdaptiveChannelStrategy": AdaptiveChannelStrategy,
    "VolatilityCaptureStrategy": VolatilityCaptureStrategy,
}


# ──────────────────────────────────────────────────────────────────────────────
#  Core Trader
# ──────────────────────────────────────────────────────────────────────────────

INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "8h": 28800, "1d": 86400,
}


class LiveTrader:
    """
    Production trading engine.
    
    Runs a polling loop that:
    1. Fetches latest OHLCV data for each config
    2. Injects BTC data for cross-pair strategies
    3. Generates signals
    4. Manages positions (entry, SL/TP, trailing)
    5. Logs everything
    """

    def __init__(self, 
                 configs: List[TradingConfig] = None,
                 capital: float = 10_000,
                 mode: str = "paper"):
        self.configs = configs or VALIDATED_CONFIGS
        self.capital = capital
        self.mode = mode
        self.client = MEXCClient()
        self.fetcher = DataFetcher(client=self.client)
        
        # Strategy instances (cached)
        self._strategies: Dict[str, Any] = {}
        
        # State
        self.state = TradingState(initial_capital=capital)
        self._running = True
        
        # BTC data cache
        self._btc_cache: Dict[str, pd.DataFrame] = {}  # interval -> df
        self._btc_cache_time: Dict[str, float] = {}
        
        os_signal.signal(os_signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, *_):
        logger.info("Shutdown signal received...")
        self._running = False

    def _get_strategy(self, class_name: str):
        """Get or create strategy instance."""
        if class_name not in self._strategies:
            cls = STRATEGY_MAP.get(class_name)
            if cls is None:
                raise ValueError(f"Unknown strategy: {class_name}")
            self._strategies[class_name] = cls()
        return self._strategies[class_name]

    def _fetch_data(self, symbol: str, interval: str, lookback_bars: int = 300) -> pd.DataFrame:
        """Fetch recent OHLCV data."""
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
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {interval}: {e}")
            return pd.DataFrame()

    def _fetch_btc_data(self, interval: str) -> pd.DataFrame:
        """Fetch BTC data (cached for 5 minutes)."""
        now = time.time()
        cache_ttl = 300  # 5 min cache
        
        if (interval in self._btc_cache and 
            now - self._btc_cache_time.get(interval, 0) < cache_ttl):
            return self._btc_cache[interval]
        
        df = self._fetch_data("BTCUSDC", interval, lookback_bars=300)
        if not df.empty:
            self._btc_cache[interval] = df
            self._btc_cache_time[interval] = now
        return df

    def _inject_btc(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Inject btc_close column into DataFrame for CrossPairLeader."""
        btc_df = self._fetch_btc_data(interval)
        if btc_df.empty:
            return df
        
        # Align by index (timestamp)
        btc_close = btc_df["close"].rename("btc_close")
        df = df.join(btc_close, how="left")
        # Forward-fill any gaps
        df["btc_close"] = df["btc_close"].ffill()
        return df

    def _calculate_position_size(self, config: TradingConfig, cs: ConfigState,
                                  entry_price: float, stop_loss: float) -> float:
        """Calculate aggressive position size."""
        capital = cs.capital
        risk_amount = capital * (config.risk_pct / 100)
        
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0
        
        size = risk_amount / price_risk
        
        # Cap at max position % of capital
        max_size = (capital * config.max_position_pct / 100) / entry_price
        size = min(size, max_size)
        
        return round(size, 8)

    def _check_drawdown(self, config: TradingConfig, cs: ConfigState) -> bool:
        """Check if config has exceeded max drawdown."""
        if cs.peak_capital == 0:
            return False
        dd = (cs.peak_capital - cs.capital) / cs.peak_capital * 100
        return dd >= config.max_drawdown_pct

    def _get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        try:
            ticker = self.client.get_ticker_price(symbol)
            if ticker and "price" in ticker:
                return float(ticker["price"])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
        return 0.0

    # ── Trade Execution ────────────────────────────────────────────────────

    def _open_position(self, config: TradingConfig, cs: ConfigState,
                        side: str, entry_price: float, quantity: float,
                        stop_loss: float, take_profit: float):
        """Open a new position."""
        cost = quantity * entry_price
        
        if self.mode == "paper":
            logger.info(f"[PAPER] OPEN {config.name}: {side} {quantity:.6f} "
                       f"{config.symbol} @ ${entry_price:.4f} "
                       f"(SL=${stop_loss:.4f} TP=${take_profit:.4f})")
        else:
            # Live mode — place actual order
            try:
                result = self.client.place_order(
                    symbol=config.symbol,
                    side=side,
                    order_type="MARKET",
                    quantity=quantity,
                )
                if result:
                    logger.info(f"[LIVE] OPEN {config.name}: {side} {quantity:.6f} "
                               f"{config.symbol} — Order: {result.get('orderId')}")
                else:
                    logger.error(f"[LIVE] Failed to open {config.name}")
                    return
            except Exception as e:
                logger.error(f"[LIVE] Order error for {config.name}: {e}")
                return
        
        # Record position in state
        pos = OpenPosition(
            config_name=config.name,
            symbol=config.symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc).isoformat(),
            capital_at_entry=cs.capital,
        )
        cs.position = asdict(pos)

    def _close_position(self, config: TradingConfig, cs: ConfigState,
                         exit_price: float, reason: str):
        """Close an existing position."""
        if not cs.position:
            return
        
        pos = cs.position
        side = pos["side"]
        quantity = pos["quantity"]
        entry_price = pos["entry_price"]
        
        if side == "BUY":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        pnl_pct = (pnl / pos["capital_at_entry"]) * 100
        
        if self.mode == "live":
            close_side = "SELL" if side == "BUY" else "BUY"
            try:
                self.client.place_order(
                    symbol=config.symbol,
                    side=close_side,
                    order_type="MARKET",
                    quantity=quantity,
                )
            except Exception as e:
                logger.error(f"[LIVE] Close order error for {config.name}: {e}")
                return
        
        # Update capital
        cs.capital += pnl
        if cs.capital > cs.peak_capital:
            cs.peak_capital = cs.capital
        cs.trade_count += 1
        if pnl > 0:
            cs.win_count += 1
        cs.total_pnl += pnl
        cs.position = None
        
        # Log trade
        trade = ClosedTrade(
            config_name=config.name,
            symbol=config.symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            entry_time=pos["entry_time"],
            exit_time=datetime.now(timezone.utc).isoformat(),
            reason=reason,
        )
        self.state.save_trade(trade)
        
        icon = "+" if pnl >= 0 else "-"
        logger.info(f"[{icon}] CLOSED {config.name}: {side} @ ${entry_price:.4f} → "
                    f"${exit_price:.4f} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) [{reason}]")

    # ── Signal Processing ──────────────────────────────────────────────────

    def _process_config(self, config: TradingConfig, cs: ConfigState):
        """Process one configuration: check existing position or generate new signal."""
        
        # Check if halted
        if cs.halted:
            return
        if self._check_drawdown(config, cs):
            logger.warning(f"MAX DRAWDOWN reached for {config.name} — halting")
            cs.halted = True
            return
        
        # Fetch data
        df = self._fetch_data(config.symbol, config.interval)
        if df.empty or len(df) < 50:
            logger.warning(f"Insufficient data for {config.name}")
            return
        
        # Inject BTC data for cross-pair strategies
        if config.strategy_class == "CrossPairLeaderStrategy":
            df = self._inject_btc(df, config.interval)
        
        # Get current market price from latest bar
        current_price = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]
        
        # === EXISTING POSITION: Check SL/TP ===
        if cs.position:
            pos = cs.position
            side = pos["side"]
            sl = pos["stop_loss"]
            tp = pos["take_profit"]
            
            exit_reason = None
            exit_price = current_price
            
            if side == "BUY":
                if sl > 0 and current_low <= sl:
                    exit_reason = "stop_loss"
                    exit_price = sl
                elif tp > 0 and current_high >= tp:
                    exit_reason = "take_profit"
                    exit_price = tp
            else:
                if sl > 0 and current_high >= sl:
                    exit_reason = "stop_loss"
                    exit_price = sl
                elif tp > 0 and current_low <= tp:
                    exit_reason = "take_profit"
                    exit_price = tp
            
            # Also check for signal reversal
            if not exit_reason:
                strategy = self._get_strategy(config.strategy_class)
                sig_df = strategy.generate_signals(df.copy())
                last_sig = sig_df["signal"].iloc[-1]
                
                if side == "BUY" and last_sig == Signal.SELL:
                    exit_reason = "signal_reversal"
                    exit_price = current_price
                elif side == "SELL" and last_sig == Signal.BUY:
                    exit_reason = "signal_reversal"
                    exit_price = current_price
            
            if exit_reason:
                self._close_position(config, cs, exit_price, exit_reason)
            return
        
        # === NO POSITION: Generate signals ===
        strategy = self._get_strategy(config.strategy_class)
        sig_df = strategy.generate_signals(df.copy())
        last_signal = sig_df["signal"].iloc[-1]
        
        if last_signal not in (Signal.BUY, Signal.SELL):
            return
        
        side = "BUY" if last_signal == Signal.BUY else "SELL"
        entry_price = current_price
        
        # Get SL/TP from strategy
        sl = sig_df["stop_loss"].iloc[-1] if "stop_loss" in sig_df else 0
        tp = sig_df["take_profit"].iloc[-1] if "take_profit" in sig_df else 0
        if pd.isna(sl): sl = 0
        if pd.isna(tp): tp = 0
        
        # Calculate position size with aggressive parameters
        if sl > 0:
            quantity = self._calculate_position_size(config, cs, entry_price, sl)
        else:
            # Fallback: use max position pct
            quantity = (cs.capital * config.max_position_pct / 100) / entry_price
        
        if quantity <= 0:
            return
        
        # Validate trade makes sense
        cost = quantity * entry_price
        if cost > cs.capital * (config.max_position_pct / 100) * 1.01:
            quantity = (cs.capital * config.max_position_pct / 100) / entry_price
        
        self._open_position(config, cs, side, entry_price, quantity, sl, tp)

    # ── Main Loop ──────────────────────────────────────────────────────────

    def initialize(self):
        """Initialize state and configs."""
        loaded = self.state.load()
        
        for config in self.configs:
            if not config.enabled:
                continue
            self.state.init_config(config)
        
        if loaded:
            logger.info("Resumed from saved state")
        else:
            logger.info("Starting fresh")
        
        self.state.save()

    def tick(self):
        """Process one tick for all configs."""
        self.state.last_tick_time = datetime.now(timezone.utc).isoformat()
        
        for config in self.configs:
            if not config.enabled:
                continue
            cs = self.state.configs.get(config.name)
            if not cs:
                continue
            
            try:
                self._process_config(config, cs)
            except Exception as e:
                logger.error(f"Error processing {config.name}: {e}", exc_info=True)
        
        self.state.save()

    def print_status(self):
        """Print current status summary."""
        print()
        print("=" * 80)
        print(f"  LIVE TRADER STATUS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Mode: {self.mode.upper()} | Initial Capital: ${self.capital:,.2f}")
        print("=" * 80)
        
        total_pnl = 0
        for config in self.configs:
            if not config.enabled:
                continue
            cs = self.state.configs.get(config.name)
            if not cs:
                continue
            
            dd = 0
            if cs.peak_capital > 0:
                dd = (cs.peak_capital - cs.capital) / cs.peak_capital * 100
            
            pos_str = "FLAT"
            if cs.position:
                p = cs.position
                pos_str = f"{p['side']} {p['quantity']:.4f} @ ${p['entry_price']:.4f}"
            
            status = "HALTED" if cs.halted else "ACTIVE"
            wr = (cs.win_count / cs.trade_count * 100) if cs.trade_count > 0 else 0
            
            print(f"  {config.name:<35} [{status}]")
            print(f"    Capital: ${cs.capital:,.2f} | PnL: ${cs.total_pnl:+,.2f} | "
                  f"DD: {dd:.1f}% | Trades: {cs.trade_count} (WR: {wr:.0f}%)")
            print(f"    Position: {pos_str}")
            print(f"    OOS Weekly: +{config.oos_weekly:.3f}% | Folds: {config.folds_positive}")
            total_pnl += cs.total_pnl
        
        print(f"\n  TOTAL PnL: ${total_pnl:+,.2f}")
        print("=" * 80)

    def run(self, poll_interval: Optional[int] = None):
        """Main trading loop."""
        self.initialize()
        
        # Determine poll interval from fastest config
        if poll_interval is None:
            intervals = [INTERVAL_SECONDS.get(c.interval, 3600) for c in self.configs if c.enabled]
            min_interval = min(intervals) if intervals else 3600
            poll_interval = max(60, min_interval // 3)  # Poll at 1/3 of fastest bar
        
        logger.info(f"Starting {'PAPER' if self.mode == 'paper' else 'LIVE'} trading loop")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info(f"Configs: {len([c for c in self.configs if c.enabled])} active")
        
        self.print_status()
        
        print(f"\n  Polling every {poll_interval}s. Press Ctrl+C to stop.\n")
        
        tick_count = 0
        while self._running:
            try:
                self.tick()
                tick_count += 1
                
                # Print status every 10 ticks or on position change
                if tick_count % 10 == 0:
                    self.print_status()
                
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
            
            # Sleep in small increments to allow Ctrl+C
            for _ in range(poll_interval):
                if not self._running:
                    break
                time.sleep(1)
        
        # Shutdown
        self.print_status()
        self.state.save()
        logger.info("Trader stopped. State saved.")

    def check_signals_now(self) -> Dict[str, Any]:
        """
        One-shot signal check across all configs.
        Returns dict of config_name -> signal info.
        Useful for testing without running the full loop.
        """
        self.initialize()
        results = {}
        
        for config in self.configs:
            if not config.enabled:
                continue
            
            try:
                df = self._fetch_data(config.symbol, config.interval)
                if df.empty or len(df) < 50:
                    results[config.name] = {"signal": "NO_DATA", "bars": len(df)}
                    continue
                
                if config.strategy_class == "CrossPairLeaderStrategy":
                    df = self._inject_btc(df, config.interval)
                
                strategy = self._get_strategy(config.strategy_class)
                sig_df = strategy.generate_signals(df.copy())
                
                last_sig = sig_df["signal"].iloc[-1]
                last_sl = sig_df["stop_loss"].iloc[-1] if "stop_loss" in sig_df else 0
                last_tp = sig_df["take_profit"].iloc[-1] if "take_profit" in sig_df else 0
                last_conf = sig_df["confidence"].iloc[-1] if "confidence" in sig_df else 0.5
                
                # Find last non-hold signal
                non_hold = sig_df[sig_df["signal"] != Signal.HOLD]
                last_active = None
                if len(non_hold) > 0:
                    last_active = {
                        "signal": "BUY" if non_hold["signal"].iloc[-1] == Signal.BUY else "SELL",
                        "time": str(non_hold.index[-1]),
                        "bars_ago": len(df) - df.index.get_loc(non_hold.index[-1]) - 1,
                    }
                
                sig_name = "HOLD"
                if last_sig == Signal.BUY: sig_name = "BUY"
                elif last_sig == Signal.SELL: sig_name = "SELL"
                
                results[config.name] = {
                    "signal": sig_name,
                    "price": float(df["close"].iloc[-1]),
                    "time": str(df.index[-1]),
                    "bars": len(df),
                    "stop_loss": float(last_sl) if not pd.isna(last_sl) else 0,
                    "take_profit": float(last_tp) if not pd.isna(last_tp) else 0,
                    "confidence": float(last_conf) if not pd.isna(last_conf) else 0.5,
                    "last_active_signal": last_active,
                }
                
            except Exception as e:
                results[config.name] = {"signal": "ERROR", "error": str(e)}
        
        return results
