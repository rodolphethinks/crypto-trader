"""
Launch the production paper/live trader with validated V6 strategies.

Usage:
    # Check current signals (one-shot, no trading loop):
    python scripts/run_production.py --check-signals
    
    # Start paper trading (default):
    python scripts/run_production.py
    
    # Start live trading (caution!):
    python scripts/run_production.py --mode live
    
    # Custom capital:
    python scripts/run_production.py --capital 5000
    
    # With specific configs:
    python scripts/run_production.py --configs "CrossPair_AVAX_1d_R10_P100,MomAccel_AVAX_4h_R5_P100"
"""
import sys, os, argparse, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.live_trader import LiveTrader, VALIDATED_CONFIGS
from config.settings import LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "live_trader.log"),
    ],
)
logger = logging.getLogger("production")


def print_banner():
    print()
    print("=" * 80)
    print("  MEXC PRODUCTION TRADER — Walk-Forward Validated V6 Strategies")
    print("=" * 80)
    print()
    print("  Validated configurations (OOS walk-forward):")
    for c in VALIDATED_CONFIGS:
        marker = "***" if c.folds_positive == "4/4" else "   "
        print(f"  {marker} {c.name:<38} OOS: +{c.oos_weekly:.3f}%/wk  "
              f"Folds: {c.folds_positive}  Risk: {c.risk_pct}%  Pos: {c.max_position_pct}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Production trading launcher")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"],
                        help="Trading mode (default: paper)")
    parser.add_argument("--capital", type=float, default=10_000,
                        help="Starting capital (default: 10000)")
    parser.add_argument("--check-signals", action="store_true",
                        help="Check current signals and exit (no trading loop)")
    parser.add_argument("--poll", type=int, default=None,
                        help="Poll interval in seconds (default: auto)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to enable (default: all)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset state (start fresh)")
    args = parser.parse_args()

    print_banner()

    # Filter configs if specified
    configs = VALIDATED_CONFIGS.copy()
    if args.configs:
        names = [n.strip() for n in args.configs.split(",")]
        configs = [c for c in configs if c.name in names]
        if not configs:
            print(f"  ERROR: No matching configs found for: {args.configs}")
            print(f"  Available: {[c.name for c in VALIDATED_CONFIGS]}")
            return

    # Reset state if requested
    if args.reset:
        state_file = LOG_DIR / "live_trader_state.json"
        if state_file.exists():
            state_file.unlink()
            print("  State reset.")
        trades_file = LOG_DIR / "live_trades.csv"
        if trades_file.exists():
            trades_file.unlink()
            print("  Trade log reset.")

    # Create trader
    trader = LiveTrader(
        configs=configs,
        capital=args.capital,
        mode=args.mode,
    )

    if args.check_signals:
        # One-shot signal check
        print("  Checking current signals...\n")
        results = trader.check_signals_now()
        
        for name, info in results.items():
            sig = info.get("signal", "UNKNOWN")
            price = info.get("price", 0)
            bars = info.get("bars", 0)
            
            if sig == "BUY":
                marker = ">> BUY  <<"
            elif sig == "SELL":
                marker = ">> SELL <<"
            elif sig == "HOLD":
                marker = "   HOLD  "
            else:
                marker = f"   {sig}   "
            
            print(f"  {name:<38} {marker}  Price: ${price:,.4f}  ({bars} bars)")
            
            # Show last active signal
            last_active = info.get("last_active_signal")
            if last_active:
                print(f"    Last signal: {last_active['signal']} at {last_active['time']} "
                      f"({last_active['bars_ago']} bars ago)")
            
            if sig in ("BUY", "SELL"):
                sl = info.get("stop_loss", 0)
                tp = info.get("take_profit", 0)
                conf = info.get("confidence", 0.5)
                print(f"    SL: ${sl:,.4f}  TP: ${tp:,.4f}  Confidence: {conf:.2f}")
        
        print()
        return

    # Safety check for live mode
    if args.mode == "live":
        print("  WARNING: LIVE TRADING MODE")
        print("  This will place REAL orders with REAL money!")
        confirm = input("  Type 'YES' to confirm: ")
        if confirm.strip() != "YES":
            print("  Aborted.")
            return

    # Start trading loop
    trader.run(poll_interval=args.poll)


if __name__ == "__main__":
    main()
