"""
Backtest performance metrics — detailed calculations for analytics.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from backtesting.engine import BacktestResult, Trade


def compute_metrics(result: BacktestResult) -> Dict:
    """Compute comprehensive performance metrics from a backtest result."""
    trades = result.trades
    equity_curve = result.equity_curve

    if not trades:
        return _empty_metrics(result)

    # Basic metrics
    pnl_list = [t.pnl for t in trades]
    pnl_pct_list = [t.pnl_pct for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    # Duration metrics
    durations = []
    for t in trades:
        if hasattr(t.entry_time, 'timestamp') and hasattr(t.exit_time, 'timestamp'):
            dur = (t.exit_time - t.entry_time).total_seconds() / 60
            durations.append(dur)

    # Equity curve metrics
    returns = equity_curve.pct_change().dropna() if equity_curve is not None else pd.Series()
    
    # Drawdown
    if equity_curve is not None and len(equity_curve) > 0:
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak
        max_dd = abs(dd.min()) * 100
        
        # Drawdown duration
        underwater = dd < 0
        dd_periods = []
        current_dd_start = None
        for i, is_uw in enumerate(underwater):
            if is_uw and current_dd_start is None:
                current_dd_start = i
            elif not is_uw and current_dd_start is not None:
                dd_periods.append(i - current_dd_start)
                current_dd_start = None
        if current_dd_start is not None:
            dd_periods.append(len(underwater) - current_dd_start)
        max_dd_duration = max(dd_periods) if dd_periods else 0
    else:
        max_dd = 0
        max_dd_duration = 0

    # Calmar ratio
    annual_return = result.total_return  # Simplified
    calmar = annual_return / max_dd if max_dd > 0 else 0

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    metrics = {
        # Return metrics
        "total_return_pct": round(result.total_return, 4),
        "total_return_abs": round(result.final_equity - result.initial_capital, 2),
        "initial_capital": result.initial_capital,
        "final_equity": round(result.final_equity, 2),

        # Trade counts
        "total_trades": result.total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(result.win_rate, 2),

        # PnL metrics
        "avg_trade_pnl": round(np.mean(pnl_list), 4),
        "avg_trade_pnl_pct": round(np.mean(pnl_pct_list), 4),
        "median_trade_pnl_pct": round(np.median(pnl_pct_list), 4),
        "best_trade_pnl": round(max(pnl_list), 4),
        "worst_trade_pnl": round(min(pnl_list), 4),
        "avg_win": round(np.mean([t.pnl for t in wins]), 4) if wins else 0,
        "avg_loss": round(np.mean([t.pnl for t in losses]), 4) if losses else 0,
        "avg_win_pct": round(np.mean([t.pnl_pct for t in wins]), 4) if wins else 0,
        "avg_loss_pct": round(np.mean([t.pnl_pct for t in losses]), 4) if losses else 0,

        # Risk metrics
        "profit_factor": round(result.profit_factor, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "max_drawdown_duration": max_dd_duration,
        "sharpe_ratio": round(result.sharpe_ratio, 4),
        "sortino_ratio": round(result.sortino_ratio, 4),
        "calmar_ratio": round(calmar, 4),

        # Streak metrics
        "max_consecutive_wins": result.max_consecutive_wins,
        "max_consecutive_losses": result.max_consecutive_losses,

        # Duration metrics
        "avg_trade_duration_min": round(np.mean(durations), 1) if durations else 0,
        "max_trade_duration_min": round(max(durations), 1) if durations else 0,
        "min_trade_duration_min": round(min(durations), 1) if durations else 0,

        # Exit reasons
        "exit_reasons": exit_reasons,

        # Strategy info
        "strategy": result.strategy_name,
        "symbol": result.symbol,
        "interval": result.interval,
        "period": f"{result.start_date} → {result.end_date}",
        "params": result.params,
    }

    return metrics


def _empty_metrics(result: BacktestResult) -> Dict:
    """Return empty metrics when no trades were generated."""
    return {
        "total_return_pct": 0, "total_return_abs": 0,
        "initial_capital": result.initial_capital,
        "final_equity": result.initial_capital,
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": 0, "avg_trade_pnl": 0, "avg_trade_pnl_pct": 0,
        "profit_factor": 0, "max_drawdown_pct": 0, "sharpe_ratio": 0,
        "sortino_ratio": 0, "strategy": result.strategy_name,
        "symbol": result.symbol, "interval": result.interval,
        "params": result.params,
    }


def compare_results(results: List[BacktestResult]) -> pd.DataFrame:
    """Compare multiple backtest results in a summary table."""
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Symbol": r.symbol,
            "Interval": r.interval,
            "Trades": r.total_trades,
            "Win Rate %": round(r.win_rate, 1),
            "Return %": round(r.total_return, 2),
            "Max DD %": round(r.max_drawdown, 2),
            "Sharpe": round(r.sharpe_ratio, 3),
            "Sortino": round(r.sortino_ratio, 3),
            "Profit Factor": round(r.profit_factor, 2),
            "Avg Trade %": round(np.mean([t.pnl_pct for t in r.trades]), 3) if r.trades else 0,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Sharpe", ascending=False).reset_index(drop=True)


def format_report(metrics: Dict) -> str:
    """Format metrics into a readable text report."""
    lines = [
        "=" * 60,
        f"  BACKTEST REPORT: {metrics.get('strategy', 'Unknown')}",
        f"  {metrics.get('symbol', '')} | {metrics.get('interval', '')} | {metrics.get('period', '')}",
        "=" * 60,
        "",
        f"  Capital:      ${metrics['initial_capital']:,.2f} → ${metrics['final_equity']:,.2f}",
        f"  Return:       {metrics['total_return_pct']:+.2f}%  (${metrics['total_return_abs']:+,.2f})",
        "",
        f"  Trades:       {metrics['total_trades']} ({metrics['winning_trades']}W / {metrics['losing_trades']}L)",
        f"  Win Rate:     {metrics['win_rate']:.1f}%",
        f"  Avg Trade:    {metrics['avg_trade_pnl_pct']:+.3f}%",
        "",
        f"  Profit Factor: {metrics['profit_factor']:.2f}",
        f"  Max Drawdown:  {metrics['max_drawdown_pct']:.2f}%",
        f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.3f}",
        f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}",
        "",
        f"  Best Trade:    {metrics.get('best_trade_pnl', 0):+.4f}",
        f"  Worst Trade:   {metrics.get('worst_trade_pnl', 0):+.4f}",
        f"  Max Wins:      {metrics.get('max_consecutive_wins', 0)} in a row",
        f"  Max Losses:    {metrics.get('max_consecutive_losses', 0)} in a row",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)
