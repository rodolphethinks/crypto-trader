"""
Tests for the backtesting engine and strategies.
Run:  python -m pytest tests/ -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# ── Helper: generate synthetic OHLCV data ──────────────────────────────────────

def make_ohlcv(n: int = 500, start_price: float = 100.0,
                volatility: float = 0.02, trend: float = 0.0001) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
    
    prices = [start_price]
    for i in range(1, n):
        ret = np.random.normal(trend, volatility)
        prices.append(prices[-1] * (1 + ret))

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, volatility / 2, n)))
    low = close * (1 - np.abs(np.random.normal(0, volatility / 2, n)))
    open_ = close * (1 + np.random.normal(0, volatility / 3, n))
    volume = np.random.randint(100, 10000, n).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)

    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def make_ranging_ohlcv(n: int = 500, center: float = 1.0,
                        spread: float = 0.001) -> pd.DataFrame:
    """Generate tight ranging data (for USDCUSDT-like pairs)."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")

    close = center + np.random.normal(0, spread, n)
    high = close + np.abs(np.random.normal(0, spread / 2, n))
    low = close - np.abs(np.random.normal(0, spread / 2, n))
    open_ = close + np.random.normal(0, spread / 3, n)
    volume = np.random.randint(1000, 100000, n).astype(float)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


# ── Test Indicators ────────────────────────────────────────────────────────────

class TestIndicators:
    def test_sma(self):
        from indicators.trend import sma
        df = make_ohlcv()
        result = sma(df["close"], 20)
        assert len(result) == len(df)
        assert result.iloc[19] == pytest.approx(df["close"].iloc[:20].mean(), rel=1e-6)

    def test_ema(self):
        from indicators.trend import ema
        df = make_ohlcv()
        result = ema(df["close"], 20)
        assert len(result) == len(df)
        assert not result.iloc[20:].isna().any()

    def test_macd(self):
        from indicators.trend import macd
        df = make_ohlcv()
        result = macd(df["close"])
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_rsi(self):
        from indicators.momentum import rsi
        df = make_ohlcv()
        result = rsi(df["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_bollinger_bands(self):
        from indicators.volatility import bollinger_bands
        df = make_ohlcv()
        result = bollinger_bands(df["close"], 20, 2.0)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_lower"] <= valid["bb_middle"]).all()

    def test_atr(self):
        from indicators.volatility import atr
        df = make_ohlcv()
        result = atr(df, 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_obv(self):
        from indicators.volume import obv
        df = make_ohlcv()
        result = obv(df)
        assert len(result) == len(df)

    def test_pivot_points(self):
        from indicators.custom import pivot_points
        df = make_ohlcv()
        result = pivot_points(df)
        assert "pivot" in result.columns
        assert "r1" in result.columns
        assert "s1" in result.columns


# ── Test Patterns ──────────────────────────────────────────────────────────────

class TestPatterns:
    def test_candlestick_detection(self):
        from patterns.candlestick import detect_all_candlestick_patterns
        df = make_ohlcv()
        result = detect_all_candlestick_patterns(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_support_resistance(self):
        from patterns.structure import detect_support_resistance
        df = make_ohlcv()
        levels = detect_support_resistance(df)
        assert isinstance(levels, list)

    def test_swing_points(self):
        from patterns.chart_patterns import find_swing_points
        df = make_ohlcv()
        highs, lows = find_swing_points(df)
        assert isinstance(highs, list)
        assert isinstance(lows, list)


# ── Test Strategy Base ─────────────────────────────────────────────────────────

class TestStrategyBase:
    def test_all_strategies_generate_signals(self):
        """Every strategy must produce a DataFrame with signal column."""
        from backtesting.runner import get_all_strategies
        df = make_ohlcv(300)

        for name, cls in get_all_strategies().items():
            strategy = cls()
            try:
                result = strategy.generate_signals(df.copy())
                assert isinstance(result, pd.DataFrame), f"{name} did not return DataFrame"
                assert "signal" in result.columns, f"{name} missing signal column"
                # Signal must be -1, 0, or 1
                valid_signals = result["signal"].dropna().unique()
                for s in valid_signals:
                    assert int(s) in [-1, 0, 1], f"{name} invalid signal value: {s}"
            except Exception as e:
                pytest.fail(f"Strategy {name} failed: {e}")

    def test_strategy_has_name(self):
        from backtesting.runner import get_all_strategies
        for name, cls in get_all_strategies().items():
            strategy = cls()
            assert strategy.name, f"Strategy {name} has no name"


# ── Test Backtest Engine ───────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_engine_run(self):
        from backtesting.engine import BacktestEngine, BacktestResult
        from strategies.trend_following import TrendFollowingStrategy

        df = make_ohlcv(300)
        engine = BacktestEngine(initial_capital=10000, commission_pct=0.0)
        strategy = TrendFollowingStrategy()
        result = engine.run(strategy, df, "TESTUSDC", "1h")

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 10000
        assert result.final_equity > 0
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(df)

    def test_zero_commission(self):
        from backtesting.engine import BacktestEngine
        from strategies.bollinger_bands import BollingerBandsStrategy

        df = make_ohlcv(200)
        engine = BacktestEngine(initial_capital=10000, commission_pct=0.0)
        strategy = BollingerBandsStrategy()
        result = engine.run(strategy, df, "TESTUSDC", "1h")

        # All commissions should be 0
        for t in result.trades:
            assert t.commission == 0.0

    def test_metrics_computation(self):
        from backtesting.engine import BacktestEngine
        from backtesting.metrics import compute_metrics
        from strategies.trend_following import TrendFollowingStrategy

        df = make_ohlcv(300)
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(TrendFollowingStrategy(), df, "TESTUSDC", "1h")
        metrics = compute_metrics(result)

        assert "total_return_pct" in metrics
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert metrics["total_trades"] == len(result.trades)


# ── Test Risk Manager ─────────────────────────────────────────────────────────

class TestRiskManager:
    def test_position_sizing(self):
        from risk.manager import RiskManager
        rm = RiskManager()
        size = rm.calculate_position_size(
            equity=10000, entry_price=100, stop_loss=95
        )
        assert size > 0
        assert size * 100 <= 10000  # Should not exceed capital

    def test_max_position_cap(self):
        from risk.manager import RiskManager
        rm = RiskManager()
        size = rm.calculate_position_size(
            equity=10000, entry_price=100, stop_loss=99.99
        )
        # Even with tight SL, position should be capped
        assert size * 100 <= 10000 * rm.max_position_pct / 100

    def test_can_open_position(self):
        from risk.manager import RiskManager
        rm = RiskManager()
        assert rm.can_open_position() is True
        rm.open_positions = rm.max_open_positions
        assert rm.can_open_position() is False


# ── Test Comparison ────────────────────────────────────────────────────────────

class TestComparison:
    def test_compare_results(self):
        from backtesting.engine import BacktestEngine
        from backtesting.metrics import compare_results
        from strategies.trend_following import TrendFollowingStrategy
        from strategies.bollinger_bands import BollingerBandsStrategy

        df = make_ohlcv(200)
        engine = BacktestEngine(initial_capital=10000)

        results = [
            engine.run(TrendFollowingStrategy(), df, "TESTUSDC", "1h"),
            engine.run(BollingerBandsStrategy(), df, "TESTUSDC", "1h"),
        ]

        comp = compare_results(results)
        assert len(comp) == 2
        assert "Strategy" in comp.columns
        assert "Sharpe" in comp.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
