"""
Launch the WebSocket real-time signal alert engine.
Monitors the top 5 strategy/pair/interval combos from the latest
sweep CSV and prints alerts when signals fire.

Usage:
    python scripts/run_alerts.py [--combos 5]
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from trading.alerts import RealtimeAlertEngine
from backtesting.runner import get_all_strategies
from strategies.bb_variants import get_bb_variants
from config.settings import LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("alerts_runner")


def load_top_combos(csv_path=None, n=5):
    """Load top N combos by Sharpe from sweep CSV."""
    if csv_path is None:
        csvs = sorted(LOG_DIR.glob("*sweep*.csv"), key=os.path.getmtime, reverse=True)
        if not csvs:
            raise FileNotFoundError("No sweep CSV in logs/")
        csv_path = str(csvs[0])

    df = pd.read_csv(csv_path)
    df = df[(df["Trades"] > 0) & (df["Sharpe"] > 0)]
    return df.nlargest(n, "Sharpe")[["Strategy", "Symbol", "Interval"]].to_dict("records")


def main():
    parser = argparse.ArgumentParser(description="Real-time signal alerts")
    parser.add_argument("--combos", type=int, default=5)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    combos = load_top_combos(args.csv, args.combos)
    all_strats = {**get_all_strategies(), **get_bb_variants()}

    engine = RealtimeAlertEngine()

    for c in combos:
        name = c["Strategy"]
        if name in all_strats:
            strategy = all_strats[name]()
            engine.add_combo(strategy, c["Symbol"], c["Interval"])
        else:
            logger.warning(f"Strategy '{name}' not found, skipping")

    print("=" * 60)
    print(f"  Real-Time Signal Alerts — {len(combos)} combos")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)
    for c in combos:
        print(f"  {c['Strategy']:25s} {c['Symbol']:10s} {c['Interval']}")
    print("=" * 60)

    try:
        engine.start()
    except KeyboardInterrupt:
        engine.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
