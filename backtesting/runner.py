"""
Backtest runner — orchestrates running multiple strategies across
multiple pairs and timeframes. Stores results for comparison.
"""
import logging
from typing import List, Optional, Dict, Any
from itertools import product

import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import compute_metrics, compare_results, format_report
from strategies.base import BaseStrategy
from data.fetcher import DataFetcher
from data.storage import DataStorage
from risk.manager import RiskManager
from config.settings import DEFAULT_INITIAL_CAPITAL, DEFAULT_COMMISSION

logger = logging.getLogger(__name__)


# ── Strategy Registry ─────────────────────────────────────────────────────────
def get_all_strategies() -> Dict[str, type]:
    """Import and return all available strategy classes."""
    from strategies.smc_liquidity import SMCLiquidityStrategy
    from strategies.mean_reversion import MeanReversionStrategy
    from strategies.trend_following import TrendFollowingStrategy
    from strategies.breakout import BreakoutRetestStrategy
    from strategies.scalping import ScalpingStrategy
    from strategies.momentum_strategy import MomentumStrategy
    from strategies.grid_trading import GridTradingStrategy
    from strategies.bollinger_bands import BollingerBandsStrategy
    from strategies.macd_divergence import MACDDivergenceStrategy
    from strategies.vwap_strategy import VWAPStrategy
    from strategies.ichimoku_strategy import IchimokuCloudStrategy
    from strategies.multi_timeframe import MultiTimeframeStrategy
    from strategies.dca_strategy import DCAStrategy
    from strategies.pairs_trading import PairsTradingStrategy

    return {
        "smc_liquidity": SMCLiquidityStrategy,
        "mean_reversion": MeanReversionStrategy,
        "trend_following": TrendFollowingStrategy,
        "breakout": BreakoutRetestStrategy,
        "scalping": ScalpingStrategy,
        "momentum": MomentumStrategy,
        "grid_trading": GridTradingStrategy,
        "bollinger_bands": BollingerBandsStrategy,
        "macd_divergence": MACDDivergenceStrategy,
        "vwap": VWAPStrategy,
        "ichimoku": IchimokuCloudStrategy,
        "multi_timeframe": MultiTimeframeStrategy,
        "dca": DCAStrategy,
        "pairs_trading": PairsTradingStrategy,
    }


def get_v3_strategies() -> Dict[str, type]:
    """Import and return all V3 strategies (ML + HF + alt-alpha)."""
    from strategies.ml_gbm import XGBoostStrategy, LightGBMStrategy
    from strategies.ml_lstm import LSTMStrategy
    from strategies.hf_strategies import (
        MicroMomentumStrategy, MeanReversionHFStrategy,
        OrderFlowImbalanceStrategy, BreakoutMicroStrategy,
    )
    from strategies.alt_alpha import (
        RegimeAdaptiveStrategy, CrossTFMomentumStrategy,
        VolatilityBreakoutStrategy, StatArbStrategy, AdaptiveTrendStrategy,
    )

    return {
        # ML strategies
        "XGBoost": XGBoostStrategy,
        "LightGBM": LightGBMStrategy,
        "LSTM": LSTMStrategy,
        # High-frequency strategies
        "MicroMomentum": MicroMomentumStrategy,
        "MeanReversion_HF": MeanReversionHFStrategy,
        "OrderFlow_Imbalance": OrderFlowImbalanceStrategy,
        "Breakout_Micro": BreakoutMicroStrategy,
        # Alternative alpha strategies
        "RegimeAdaptive": RegimeAdaptiveStrategy,
        "CrossTF_Momentum": CrossTFMomentumStrategy,
        "Vol_Breakout": VolatilityBreakoutStrategy,
        "StatArb": StatArbStrategy,
        "AdaptiveTrend": AdaptiveTrendStrategy,
    }


def get_v4_strategies() -> Dict[str, type]:
    """Import and return V4 research strategies (ensembles + tuned + fixes)."""
    from strategies.v4_research import (
        EnsembleTop3Strategy, EnsembleWeightedStrategy,
        AdaptiveTrendTunedStrategy, CrossTFTunedStrategy,
        StatArbRelaxedStrategy,
    )
    return {
        "Ensemble_Top3": EnsembleTop3Strategy,
        "Ensemble_Weighted": EnsembleWeightedStrategy,
        "AdaptiveTrend_Tuned": AdaptiveTrendTunedStrategy,
        "CrossTF_Tuned": CrossTFTunedStrategy,
        "StatArb_Relaxed": StatArbRelaxedStrategy,
    }


def get_v6_strategies() -> Dict[str, type]:
    """Import and return V6 aggressive strategies."""
    from strategies.v6_aggressive import (
        TrendRiderStrategy,
        MomentumAcceleratorStrategy,
        RegimeMomentumV2Strategy,
        BreakoutAccumulatorStrategy,
        DynamicKellyStrategy,
        MultiEdgeCompositeStrategy,
        CrossPairLeaderStrategy,
    )
    return {
        "TrendRider": TrendRiderStrategy,
        "MomAccelerator": MomentumAcceleratorStrategy,
        "RegimeMomV2": RegimeMomentumV2Strategy,
        "BreakoutAccum": BreakoutAccumulatorStrategy,
        "DynamicKelly": DynamicKellyStrategy,
        "MultiEdge": MultiEdgeCompositeStrategy,
        "CrossPairLead": CrossPairLeaderStrategy,
    }


class BacktestRunner:
    """
    High-level runner that executes backtests across multiple
    strategy / pair / timeframe combinations.
    """

    def __init__(self,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 commission_pct: float = DEFAULT_COMMISSION,
                 fetcher: Optional[DataFetcher] = None,
                 storage: Optional[DataStorage] = None):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.fetcher = fetcher or DataFetcher()
        self.storage = storage or DataStorage()
        self.results: List[BacktestResult] = []

    def run_single(self, strategy: BaseStrategy, symbol: str,
                   interval: str = "1h", start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """Run a single backtest."""
        if df is None:
            df = self.fetcher.fetch_klines_cached(symbol, interval, start_date, end_date)

        if df.empty:
            logger.error(f"No data for {symbol} {interval}")
            return BacktestResult(
                strategy_name=strategy.name, symbol=symbol, interval=interval,
                start_date=start_date or "", end_date=end_date or "",
                initial_capital=self.initial_capital, final_equity=self.initial_capital,
            )

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            risk_manager=RiskManager(),
        )

        result = engine.run(strategy, df, symbol, interval)
        self.results.append(result)

        # Save to DB
        try:
            self.storage.init_db()
            self.storage.save_backtest_result(result.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save result to DB: {e}")

        return result

    def run_sweep(self,
                  strategies: Optional[List[str]] = None,
                  symbols: Optional[List[str]] = None,
                  intervals: Optional[List[str]] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  print_results: bool = True) -> pd.DataFrame:
        """
        Run backtests across multiple strategies, symbols, and intervals.
        
        Args:
            strategies: List of strategy names (keys from get_all_strategies)
            symbols: List of trading pairs
            intervals: List of timeframe intervals
            start_date: Start date for data
            end_date: End date for data
            print_results: Whether to print comparison table
        
        Returns:
            Comparison DataFrame sorted by Sharpe ratio
        """
        all_strategies = get_all_strategies()
        
        if strategies is None:
            strategies = list(all_strategies.keys())
        if symbols is None:
            symbols = ["BTCUSDC", "ETHUSDC"]
        if intervals is None:
            intervals = ["1h"]

        total = len(strategies) * len(symbols) * len(intervals)
        logger.info(f"Starting sweep: {total} combinations")

        completed = 0
        for strat_name, symbol, interval in product(strategies, symbols, intervals):
            if strat_name not in all_strategies:
                logger.warning(f"Unknown strategy: {strat_name}")
                continue

            try:
                strategy = all_strategies[strat_name]()
                result = self.run_single(strategy, symbol, interval,
                                          start_date, end_date)
                completed += 1
                
                metrics = compute_metrics(result)
                logger.info(f"[{completed}/{total}] {strat_name} on {symbol} {interval}: "
                           f"{metrics['total_return_pct']:+.2f}% return, "
                           f"{metrics['win_rate']:.1f}% win rate")

            except Exception as e:
                logger.error(f"Failed: {strat_name} on {symbol} {interval}: {e}")
                completed += 1

        # Compare all results
        comparison = compare_results(self.results)

        if print_results and not comparison.empty:
            print("\n" + "=" * 80)
            print("  BACKTEST SWEEP RESULTS")
            print("=" * 80)
            print(comparison.to_string(index=False))
            print("=" * 80 + "\n")

        return comparison

    def run_parameter_optimization(self, strategy_class: type,
                                     param_grid: Dict[str, List[Any]],
                                     symbol: str, interval: str = "1h",
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     metric: str = "sharpe") -> pd.DataFrame:
        """
        Grid search over strategy parameters to find optimal settings.
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dict of param_name -> list of values to try
            symbol: Trading pair
            interval: Timeframe
            metric: Optimization target ('sharpe', 'return', 'profit_factor', 'win_rate')
        
        Returns:
            DataFrame of results sorted by target metric
        """
        # Fetch data once
        df = self.fetcher.fetch_klines_cached(symbol, interval, start_date, end_date)
        if df.empty:
            logger.error(f"No data for optimization")
            return pd.DataFrame()

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Optimizing {strategy_class.__name__}: {len(combinations)} combinations")

        opt_results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            try:
                strategy = strategy_class(params=params)
                engine = BacktestEngine(
                    initial_capital=self.initial_capital,
                    commission_pct=self.commission_pct,
                )
                result = engine.run(strategy, df.copy(), symbol, interval)
                metrics = compute_metrics(result)
                
                row = {**params, **{
                    "return_pct": metrics["total_return_pct"],
                    "win_rate": metrics["win_rate"],
                    "sharpe": metrics["sharpe_ratio"],
                    "sortino": metrics["sortino_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "max_dd": metrics["max_drawdown_pct"],
                    "trades": metrics["total_trades"],
                }}
                opt_results.append(row)

            except Exception as e:
                logger.warning(f"Optimization failed for {params}: {e}")

        results_df = pd.DataFrame(opt_results)
        
        metric_map = {
            "sharpe": "sharpe",
            "return": "return_pct",
            "profit_factor": "profit_factor",
            "win_rate": "win_rate",
        }
        sort_col = metric_map.get(metric, "sharpe")
        
        if not results_df.empty:
            results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        return results_df
