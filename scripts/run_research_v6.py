"""
V6 Research Pipeline — Aggressive strategies + Portfolio simulation.

Phases:
1. Sweep all V6 strategies across pairs/timeframes
2. Walk-forward validate top performers
3. Build optimal portfolio from ALL validated strategies (V3-V6)
4. Simulate portfolio with compounding
5. Report weekly return expectations

Target: 1% per week (~52% annualized)
"""
import sys
import os
import logging
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.runner import BacktestRunner, get_v6_strategies, get_v3_strategies, get_v4_strategies
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import compute_metrics
from backtesting.portfolio import PortfolioBacktester, PortfolioAllocation
from risk.manager import RiskManager
from data.fetcher import DataFetcher
from config.settings import LOG_DIR, DEFAULT_INITIAL_CAPITAL

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "v6_research.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

PAIRS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "BNBUSDC",
    "ADAUSDC", "DOGEUSDC", "AVAXUSDC", "LINKUSDC", "NEARUSDC",
    "APTUSDC", "SUIUSDC", "INJUSDC", "LTCUSDC",
]

TIMEFRAMES = ["4h", "1d"]

START_DATE = "2023-06-01"
END_DATE = None  # Up to now


def phase1_sweep():
    """Phase 1: Run all V6 strategies across all pairs and timeframes."""
    logger.info("=" * 60)
    logger.info("PHASE 1: V6 Strategy Sweep")
    logger.info("=" * 60)

    strategies = get_v6_strategies()
    fetcher = DataFetcher()
    rows = []
    total = len(strategies) * len(PAIRS) * len(TIMEFRAMES)
    done = 0

    for strat_name, strat_class in strategies.items():
        for symbol in PAIRS:
            for interval in TIMEFRAMES:
                done += 1
                try:
                    strategy = strat_class()
                    runner = BacktestRunner(fetcher=fetcher)
                    df = fetcher.fetch_klines_cached(symbol, interval, START_DATE, END_DATE)
                    
                    if df.empty or len(df) < 100:
                        logger.warning(f"[{done}/{total}] Skipping {strat_name} {symbol} {interval}: insufficient data")
                        continue

                    engine = BacktestEngine(
                        initial_capital=DEFAULT_INITIAL_CAPITAL,
                        commission_pct=0.0,
                        risk_manager=RiskManager(risk_per_trade_pct=2.0),
                    )
                    result = engine.run(strategy, df, symbol, interval)
                    m = compute_metrics(result)

                    row = {
                        "strategy": strat_name,
                        "symbol": symbol,
                        "interval": interval,
                        "return_pct": m["total_return_pct"],
                        "win_rate": m["win_rate"],
                        "sharpe": m["sharpe_ratio"],
                        "sortino": m["sortino_ratio"],
                        "profit_factor": m["profit_factor"],
                        "max_dd": m["max_drawdown_pct"],
                        "trades": m["total_trades"],
                        "avg_trade": m.get("avg_trade_return_pct", 0),
                    }
                    rows.append(row)

                    logger.info(
                        f"[{done}/{total}] {strat_name:20s} {symbol:10s} {interval:3s} → "
                        f"{m['total_return_pct']:+7.2f}%  Sharpe {m['sharpe_ratio']:6.2f}  "
                        f"Trades {m['total_trades']:4d}  WR {m['win_rate']:5.1f}%"
                    )

                except Exception as e:
                    logger.error(f"[{done}/{total}] FAILED: {strat_name} {symbol} {interval}: {e}")

    df_results = pd.DataFrame(rows)
    outpath = LOG_DIR / "master_sweep_v6.csv"
    df_results.to_csv(outpath, index=False)
    logger.info(f"Saved {len(rows)} results to {outpath}")

    # Print top 30 by Sharpe
    if not df_results.empty:
        top = df_results[df_results["trades"] >= 5].nlargest(30, "sharpe")
        print("\n" + "=" * 80)
        print("  TOP 30 V6 STRATEGIES (by Sharpe, min 5 trades)")
        print("=" * 80)
        for _, r in top.iterrows():
            print(f"  {r['strategy']:20s} {r['symbol']:10s} {r['interval']:3s}  "
                  f"Ret {r['return_pct']:+7.2f}%  Sharpe {r['sharpe']:6.2f}  "
                  f"Trades {r['trades']:4.0f}  WR {r['win_rate']:5.1f}%  "
                  f"MaxDD {r['max_dd']:5.2f}%  PF {r['profit_factor']:5.2f}")
        print("=" * 80)

    return df_results


def phase2_walk_forward(sweep_results: pd.DataFrame, top_n: int = 40):
    """Phase 2: Walk-forward validate top performers."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Walk-Forward Validation")
    logger.info("=" * 60)

    if sweep_results.empty:
        logger.error("No sweep results to validate")
        return pd.DataFrame()

    # Select top candidates: min 5 trades, top by Sharpe
    candidates = sweep_results[sweep_results["trades"] >= 5].nlargest(top_n, "sharpe")
    
    strategies = get_v6_strategies()
    fetcher = DataFetcher()
    wf_rows = []
    n_folds = 3

    for idx, row in candidates.iterrows():
        strat_name = row["strategy"]
        symbol = row["symbol"]
        interval = row["interval"]
        
        if strat_name not in strategies:
            continue

        try:
            df = fetcher.fetch_klines_cached(symbol, interval, START_DATE, END_DATE)
            if df.empty or len(df) < 100:
                continue

            n = len(df)
            fold_size = n // (n_folds + 1)
            
            if fold_size < 30:
                continue

            oos_returns = []
            for fold in range(n_folds):
                # Train on first (fold+1) chunks, test on next chunk
                train_end = fold_size * (fold + 1)
                test_start = train_end
                test_end = min(test_start + fold_size, n)

                if test_end - test_start < 20:
                    continue

                train_df = df.iloc[:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()

                # Generate signals on train (for strategies that need it)
                strat_class = strategies[strat_name]
                strategy = strat_class()
                
                engine = BacktestEngine(
                    initial_capital=DEFAULT_INITIAL_CAPITAL,
                    commission_pct=0.0,
                    risk_manager=RiskManager(risk_per_trade_pct=2.0),
                )
                result = engine.run(strategy, test_df, symbol, interval)
                m = compute_metrics(result)
                oos_returns.append(m["total_return_pct"])

            if not oos_returns:
                continue

            oos_mean = np.mean(oos_returns)
            folds_positive = sum(1 for r in oos_returns if r > 0)
            all_positive = folds_positive == len(oos_returns)

            wf_rows.append({
                "strategy": strat_name,
                "symbol": symbol,
                "interval": interval,
                "original_return": row["return_pct"],
                "original_sharpe": row["sharpe"],
                "oos_mean_return": oos_mean,
                "folds_positive": f"{folds_positive}/{len(oos_returns)}",
                "all_positive": all_positive,
                "oos_returns": str(oos_returns),
            })

            status = "✓ PASSED" if all_positive else f"  {folds_positive}/{len(oos_returns)}"
            logger.info(f"  {strat_name:20s} {symbol:10s} {interval:3s}  "
                        f"OOS mean: {oos_mean:+.4f}%  {status}")

        except Exception as e:
            logger.error(f"  WF failed: {strat_name} {symbol} {interval}: {e}")

    wf_df = pd.DataFrame(wf_rows)
    outpath = LOG_DIR / "walk_forward_v6.csv"
    wf_df.to_csv(outpath, index=False)
    logger.info(f"Saved {len(wf_rows)} walk-forward results to {outpath}")

    # Print summary
    if not wf_df.empty:
        all_pos = wf_df[wf_df["all_positive"] == True]
        print(f"\n  Walk-Forward: {len(all_pos)}/{len(wf_df)} passed ALL folds positive")
        if not all_pos.empty:
            print("\n  ALL-POSITIVE walk-forward strategies:")
            for _, r in all_pos.iterrows():
                print(f"    {r['strategy']:20s} {r['symbol']:10s} {r['interval']:3s}  "
                      f"OOS mean: {r['oos_mean_return']:+.4f}%")

    return wf_df


def phase3_portfolio_simulation(wf_v6: pd.DataFrame):
    """
    Phase 3: Build optimal portfolio from ALL validated strategies across V3-V6.
    
    Loads previous walk-forward results and combines with V6.
    Selects strategies with positive OOS mean return.
    Runs portfolio backtest with compounding.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Portfolio Simulation")
    logger.info("=" * 60)

    # Collect validated strategies from all versions
    all_validated = []
    
    # Load V3 walk-forward results  
    for wf_file, version, get_strats_fn in [
        ("walk_forward_v3.csv", "V3", get_v3_strategies),
        ("walk_forward_v4.csv", "V4", get_v4_strategies),
    ]:
        wf_path = LOG_DIR / wf_file
        if wf_path.exists():
            try:
                wf_prev = pd.read_csv(wf_path)
                strats = get_strats_fn()
                for _, row in wf_prev.iterrows():
                    if row.get("oos_mean_return", 0) > 0 and row.get("strategy") in strats:
                        all_validated.append({
                            "version": version,
                            "strategy_name": row["strategy"],
                            "strategy_class": strats[row["strategy"]],
                            "symbol": row["symbol"],
                            "interval": row["interval"],
                            "oos_mean": row["oos_mean_return"],
                            "original_sharpe": row.get("original_sharpe", 0),
                        })
            except Exception as e:
                logger.warning(f"Could not load {wf_file}: {e}")

    # Add V6 validated
    v6_strats = get_v6_strategies()
    if not wf_v6.empty:
        for _, row in wf_v6.iterrows():
            if row.get("oos_mean_return", 0) > 0 and row.get("strategy") in v6_strats:
                all_validated.append({
                    "version": "V6",
                    "strategy_name": row["strategy"],
                    "strategy_class": v6_strats[row["strategy"]],
                    "symbol": row["symbol"],
                    "interval": row["interval"],
                    "oos_mean": row["oos_mean_return"],
                    "original_sharpe": row.get("original_sharpe", 0),
                })

    logger.info(f"Total validated strategy-pair combos: {len(all_validated)}")

    if not all_validated:
        logger.error("No validated strategies found!")
        return None

    # Sort by OOS mean return, take top 20 for portfolio
    all_validated.sort(key=lambda x: x["oos_mean"], reverse=True)
    portfolio_picks = all_validated[:20]

    print(f"\n  Portfolio components ({len(portfolio_picks)} strategies):")
    for p in portfolio_picks:
        print(f"    [{p['version']}] {p['strategy_name']:20s} {p['symbol']:10s} {p['interval']:3s}  "
              f"OOS: {p['oos_mean']:+.4f}%  Sharpe: {p['original_sharpe']:.2f}")

    # Build portfolio allocations
    allocations = []
    for p in portfolio_picks:
        allocations.append(PortfolioAllocation(
            strategy_class=p["strategy_class"],
            strategy_params={},
            symbol=p["symbol"],
            interval=p["interval"],
            weight=max(p["oos_mean"], 0.01),  # Weight by OOS return
            name=f"{p['strategy_name']}_{p['symbol']}_{p['interval']}",
        ))

    # Run portfolio backtest
    pb = PortfolioBacktester(
        initial_capital=10000,
        commission_pct=0.0,
        risk_per_trade_pct=2.0,
    )
    
    logger.info("Running portfolio backtest...")
    result = pb.run(allocations, start_date=START_DATE, end_date=END_DATE)
    pb.print_report(result)

    # Save portfolio equity curve
    if result.equity_curve is not None:
        result.equity_curve.to_csv(LOG_DIR / "portfolio_equity_v6.csv")
        logger.info("Saved portfolio equity curve")

    # Save component details
    comp_df = pd.DataFrame(result.component_results)
    comp_df.to_csv(LOG_DIR / "portfolio_components_v6.csv", index=False)

    # Weekly return analysis
    if result.weekly_returns is not None and len(result.weekly_returns) > 0:
        wr = result.weekly_returns
        print(f"\n  Weekly Return Distribution:")
        print(f"    Mean:     {wr.mean()*100:+.3f}%")
        print(f"    Median:   {wr.median()*100:+.3f}%")
        print(f"    Std Dev:  {wr.std()*100:.3f}%")
        print(f"    Min:      {wr.min()*100:+.3f}%")
        print(f"    Max:      {wr.max()*100:+.3f}%")
        print(f"    % Positive: {(wr > 0).mean()*100:.0f}%")
        
        target_weeks = (wr >= 0.01).mean() * 100
        print(f"    % ≥ 1%/week: {target_weeks:.0f}%")

    return result


def phase4_aggressive_portfolio():
    """
    Phase 4: Run even more aggressive portfolio configurations.
    
    - Higher risk per trade (3-5%)
    - Kelly-based position sizing  
    - Concentrated portfolio (top 10 only)
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Aggressive Portfolio Configurations")
    logger.info("=" * 60)

    # Load V6 walk-forward results
    wf_path = LOG_DIR / "walk_forward_v6.csv"
    if not wf_path.exists():
        logger.error("No V6 walk-forward results found")
        return

    wf_v6 = pd.read_csv(wf_path)
    
    # Collect ALL validated strategies
    all_validated = []
    
    for wf_file, get_fn in [
        ("walk_forward_v3.csv", get_v3_strategies),
        ("walk_forward_v4.csv", get_v4_strategies),
    ]:
        path = LOG_DIR / wf_file
        if path.exists():
            try:
                wf = pd.read_csv(path)
                strats = get_fn()
                for _, row in wf.iterrows():
                    if row.get("oos_mean_return", 0) > 0.05 and row["strategy"] in strats:
                        all_validated.append({
                            "strategy_class": strats[row["strategy"]],
                            "name": f"{row['strategy']}_{row['symbol']}_{row['interval']}",
                            "symbol": row["symbol"],
                            "interval": row["interval"],
                            "oos_mean": row["oos_mean_return"],
                        })
            except:
                pass

    v6_strats = get_v6_strategies()
    for _, row in wf_v6.iterrows():
        if row.get("oos_mean_return", 0) > 0.05 and row["strategy"] in v6_strats:
            all_validated.append({
                "strategy_class": v6_strats[row["strategy"]],
                "name": f"{row['strategy']}_{row['symbol']}_{row['interval']}",
                "symbol": row["symbol"],
                "interval": row["interval"],
                "oos_mean": row["oos_mean_return"],
            })

    all_validated.sort(key=lambda x: x["oos_mean"], reverse=True)
    
    # Test different risk levels
    for risk_pct in [2.0, 3.0, 5.0]:
        for n_strats in [5, 10, 15]:
            picks = all_validated[:n_strats]
            if not picks:
                continue

            allocations = [
                PortfolioAllocation(
                    strategy_class=p["strategy_class"],
                    strategy_params={},
                    symbol=p["symbol"],
                    interval=p["interval"],
                    weight=p["oos_mean"],
                    name=p["name"],
                )
                for p in picks
            ]

            pb = PortfolioBacktester(
                initial_capital=10000,
                commission_pct=0.0,
                risk_per_trade_pct=risk_pct,
            )

            result = pb.run(allocations, start_date=START_DATE)
            
            weekly_avg = 0
            if result.weekly_returns is not None and len(result.weekly_returns) > 0:
                weekly_avg = result.weekly_returns.mean() * 100
            
            print(f"  Risk {risk_pct}% | Top {n_strats:2d} strats | "
                  f"Return {result.total_return_pct:+7.2f}% | "
                  f"Sharpe {result.sharpe_ratio:5.2f} | "
                  f"MaxDD {result.max_drawdown_pct:5.2f}% | "
                  f"Trades {result.total_trades:4d} | "
                  f"Avg Week {weekly_avg:+.3f}%")


if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("  V6 RESEARCH PIPELINE — Aggressive Strategies + Portfolio")
    print("  Target: 1% per week (~52% annualized)")
    print("=" * 80 + "\n")

    # Phase 1: Sweep
    sweep_results = phase1_sweep()
    
    # Phase 2: Walk-Forward Validation
    wf_results = phase2_walk_forward(sweep_results)
    
    # Phase 3: Portfolio Simulation
    portfolio_result = phase3_portfolio_simulation(wf_results)
    
    # Phase 4: Aggressive portfolio configs
    phase4_aggressive_portfolio()

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("  Done!")
