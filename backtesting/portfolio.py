"""
Portfolio backtester — simulates running multiple strategy-pair combos
simultaneously with proper capital allocation, compounding, and
correlation-aware diversification.

This is the KEY piece for achieving higher returns: combining many
small-edge strategies that are uncorrelated produces a much smoother
and larger aggregate return.
"""
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult, Trade
from backtesting.metrics import compute_metrics
from risk.manager import RiskManager
from data.fetcher import DataFetcher
from config.settings import DEFAULT_INITIAL_CAPITAL, DEFAULT_COMMISSION

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Defines a strategy-pair allocation within the portfolio."""
    strategy_class: type
    strategy_params: Dict
    symbol: str
    interval: str
    weight: float = 1.0  # Relative weight in portfolio
    name: str = ""


@dataclass
class PortfolioResult:
    """Result of a portfolio backtest."""
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float
    equity_curve: pd.Series = None
    component_results: List[Dict] = field(default_factory=list)
    weekly_returns: pd.Series = None
    monthly_returns: pd.Series = None


class PortfolioBacktester:
    """
    Simulates a portfolio of multiple strategies running simultaneously.
    
    Capital allocation modes:
    - 'equal': Equal weight to all components 
    - 'sharpe': Weight by historical Sharpe ratio
    - 'kelly': Weight by Kelly criterion
    - 'custom': Use provided weights
    
    Key features:
    - Compound returns across all strategies
    - Track aggregate equity curve
    - Measure portfolio-level metrics
    - Correlation analysis between components
    """

    def __init__(self, 
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 commission_pct: float = DEFAULT_COMMISSION,
                 allocation_mode: str = "equal",
                 risk_per_trade_pct: float = 2.0,  # More aggressive
                 max_portfolio_risk_pct: float = 15.0,  # Max concurrent risk
                 rebalance_period: int = 0):  # 0 = no rebalancing
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.allocation_mode = allocation_mode
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.rebalance_period = rebalance_period
        self.fetcher = DataFetcher()

    def run(self, allocations: List[PortfolioAllocation],
            start_date: str = "2023-06-01",
            end_date: str = None) -> PortfolioResult:
        """
        Run portfolio backtest.
        
        1. Fetch data for all pairs
        2. Run each strategy independently 
        3. Merge trade timelines
        4. Build aggregate equity curve with compounding
        """
        n_components = len(allocations)
        logger.info(f"Running portfolio backtest: {n_components} components")

        # Normalize weights
        total_weight = sum(a.weight for a in allocations)
        weights = [a.weight / total_weight for a in allocations]

        # Run each component
        component_results = []
        component_trades = []  # (allocation_idx, trades_list, capital_fraction)

        for idx, alloc in enumerate(allocations):
            try:
                strategy = alloc.strategy_class(params=alloc.strategy_params)
                alloc_capital = self.initial_capital * weights[idx]
                
                engine = BacktestEngine(
                    initial_capital=alloc_capital,
                    commission_pct=self.commission_pct,
                    risk_manager=RiskManager(
                        risk_per_trade_pct=self.risk_per_trade_pct,
                        max_drawdown_pct=25.0,
                    ),
                )
                
                df = self.fetcher.fetch_klines_cached(
                    alloc.symbol, alloc.interval, start_date, end_date
                )
                
                if df.empty:
                    logger.warning(f"No data for {alloc.symbol} {alloc.interval}")
                    continue

                result = engine.run(strategy, df, alloc.symbol, alloc.interval)
                metrics = compute_metrics(result)
                
                component_results.append({
                    "name": alloc.name or f"{strategy.name}_{alloc.symbol}_{alloc.interval}",
                    "strategy": strategy.name,
                    "symbol": alloc.symbol,
                    "interval": alloc.interval,
                    "weight": weights[idx],
                    "allocated_capital": alloc_capital,
                    "final_equity": result.final_equity,
                    "return_pct": metrics["total_return_pct"],
                    "sharpe": metrics["sharpe_ratio"],
                    "trades": metrics["total_trades"],
                    "win_rate": metrics["win_rate"],
                    "max_dd": metrics["max_drawdown_pct"],
                    "profit_factor": metrics["profit_factor"],
                    "equity_curve": result.equity_curve,
                })
                
                component_trades.append((idx, result.trades, weights[idx]))
                
                logger.info(
                    f"  [{idx+1}/{n_components}] {strategy.name} {alloc.symbol} {alloc.interval}: "
                    f"{metrics['total_return_pct']:+.2f}%, {metrics['total_trades']} trades"
                )

            except Exception as e:
                logger.error(f"Failed component {idx}: {e}")
                import traceback
                traceback.print_exc()

        if not component_results:
            return PortfolioResult(
                total_return_pct=0, annualized_return_pct=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown_pct=0, total_trades=0,
                win_rate=0, profit_factor=0,
            )

        # Build aggregate equity curve
        # Normalize all equity curves to same date range and resample to daily
        equity_curves = []
        for comp in component_results:
            if comp["equity_curve"] is not None and len(comp["equity_curve"]) > 0:
                ec = comp["equity_curve"].copy()
                # Normalize to returns
                ec_returns = ec.pct_change().fillna(0)
                ec_returns.name = comp["name"]
                equity_curves.append(ec_returns)

        if not equity_curves:
            return PortfolioResult(
                total_return_pct=0, annualized_return_pct=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown_pct=0, total_trades=0,
                win_rate=0, profit_factor=0,
            )

        # Combine weighted returns
        # Align all curves to common index via outer join
        returns_df = pd.DataFrame({ec.name: ec for ec in equity_curves})
        returns_df = returns_df.fillna(0)
        
        # Portfolio return = weighted sum of component returns
        weights_arr = []
        for comp in component_results:
            weights_arr.append(comp["weight"])
        
        # Build portfolio returns
        port_returns = pd.Series(0.0, index=returns_df.index)
        for i, col in enumerate(returns_df.columns):
            if i < len(weights_arr):
                port_returns += returns_df[col] * weights_arr[i]

        # Build equity curve from returns
        port_equity = (1 + port_returns).cumprod() * self.initial_capital

        # Calculate portfolio metrics
        total_return = ((port_equity.iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Annualize
        n_bars = len(port_equity)
        if n_bars > 1:
            # Estimate timespan
            time_span = (port_equity.index[-1] - port_equity.index[0]).total_seconds() / (365.25 * 86400)
            if time_span > 0:
                annualized = ((port_equity.iloc[-1] / self.initial_capital) ** (1 / time_span) - 1) * 100
            else:
                annualized = total_return
        else:
            annualized = 0
            time_span = 0

        # Sharpe
        if port_returns.std() > 0:
            sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(365 * 24)
        else:
            sharpe = 0

        # Sortino
        downside = port_returns[port_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (port_returns.mean() / downside.std()) * np.sqrt(365 * 24)
        else:
            sortino = float('inf') if port_returns.mean() > 0 else 0

        # Max drawdown
        peak = port_equity.expanding().max()
        dd = (port_equity - peak) / peak * 100
        max_dd = abs(dd.min())

        # Aggregate trade stats
        total_trades = sum(c["trades"] for c in component_results)
        total_wins = sum(c["trades"] * c["win_rate"] / 100 for c in component_results)
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        total_gross_profit = sum(
            c["allocated_capital"] * max(0, c["return_pct"] / 100) 
            for c in component_results
        )
        total_gross_loss = sum(
            c["allocated_capital"] * abs(min(0, c["return_pct"] / 100))
            for c in component_results
        )
        profit_factor = total_gross_profit / max(total_gross_loss, 0.01)

        # Weekly returns
        weekly = port_returns.resample("W").sum() if hasattr(port_returns.index, 'freq') or len(port_returns) > 7 else None
        monthly = port_returns.resample("ME").sum() if hasattr(port_returns.index, 'freq') or len(port_returns) > 30 else None

        result = PortfolioResult(
            total_return_pct=total_return,
            annualized_return_pct=annualized,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            equity_curve=port_equity,
            component_results=component_results,
            weekly_returns=weekly,
            monthly_returns=monthly,
        )

        return result

    def correlation_matrix(self, component_results: List[Dict]) -> pd.DataFrame:
        """Calculate return correlation between portfolio components."""
        equity_curves = {}
        for comp in component_results:
            if comp["equity_curve"] is not None and len(comp["equity_curve"]) > 0:
                equity_curves[comp["name"]] = comp["equity_curve"].pct_change().fillna(0)
        
        if len(equity_curves) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(equity_curves)
        return returns_df.corr()

    def print_report(self, result: PortfolioResult):
        """Print detailed portfolio report."""
        print("\n" + "=" * 80)
        print("  PORTFOLIO BACKTEST REPORT")
        print("=" * 80)
        
        print(f"\n  Total Return:       {result.total_return_pct:+.2f}%")
        print(f"  Annualized Return:  {result.annualized_return_pct:+.2f}%")
        print(f"  Sharpe Ratio:       {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:      {result.sortino_ratio:.2f}")
        print(f"  Max Drawdown:       {result.max_drawdown_pct:.2f}%")
        print(f"  Total Trades:       {result.total_trades}")
        print(f"  Win Rate:           {result.win_rate:.1f}%")
        print(f"  Profit Factor:      {result.profit_factor:.2f}")
        
        if result.weekly_returns is not None and len(result.weekly_returns) > 0:
            avg_weekly = result.weekly_returns.mean() * 100
            median_weekly = result.weekly_returns.median() * 100
            print(f"\n  Avg Weekly Return:  {avg_weekly:+.3f}%")
            print(f"  Median Weekly Ret:  {median_weekly:+.3f}%")
            pct_positive = (result.weekly_returns > 0).mean() * 100
            print(f"  Positive Weeks:     {pct_positive:.0f}%")

        print(f"\n  {'Component':<40} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'MaxDD':>8}")
        print("  " + "-" * 78)
        for comp in result.component_results:
            print(f"  {comp['name']:<40} {comp['return_pct']:>+7.2f}% "
                  f"{comp['sharpe']:>7.2f} {comp['trades']:>7} "
                  f"{comp['win_rate']:>7.1f}% {comp['max_dd']:>7.2f}%")
        
        # Correlation matrix
        if len(result.component_results) >= 2:
            corr = self.correlation_matrix(result.component_results)
            if not corr.empty:
                print(f"\n  Return Correlation Matrix:")
                # Truncate names for display
                corr.index = [n[:20] for n in corr.index]
                corr.columns = [n[:20] for n in corr.columns]
                print(corr.round(2).to_string())
        
        print("\n" + "=" * 80)
