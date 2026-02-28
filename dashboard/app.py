"""
Streamlit Trading Dashboard

Launch:  streamlit run dashboard/app.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from config.pairs import ALL_NOFEE_PAIRS, MAJOR_PAIRS, STABLECOIN_PAIRS
from config.settings import KLINE_INTERVALS
from data.fetcher import DataFetcher
from data.storage import DataStorage
from backtesting.engine import BacktestEngine
from backtesting.runner import get_all_strategies, BacktestRunner
from backtesting.metrics import compute_metrics, compare_results, format_report
from risk.manager import RiskManager


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEXC Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Init ─────────────────────────────────────────────────────────
if "fetcher" not in st.session_state:
    st.session_state.fetcher = DataFetcher()
if "storage" not in st.session_state:
    st.session_state.storage = DataStorage()
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 MEXC Trading System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Backtest", "Strategy Comparison", "Sweep Results",
     "Live Monitor", "Data Explorer"]
)


# ── Helper Functions ───────────────────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, title: str = "", signals_df: pd.DataFrame = None):
    """Create an interactive candlestick chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Price", "Volume", ""),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    ), row=1, col=1)

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=colors,
        name="Volume", opacity=0.5,
    ), row=2, col=1)

    # Buy/Sell signals
    if signals_df is not None and "signal" in signals_df.columns:
        buys = signals_df[signals_df["signal"] == 1]
        sells = signals_df[signals_df["signal"] == -1]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["close"],
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
            ), row=1, col=1)

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["close"],
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
            ), row=1, col=1)

    fig.update_layout(
        title=title,
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve"):
    """Plot equity curve with drawdown."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity", "Drawdown %"),
    )

    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode="lines", name="Equity",
        line=dict(color="#26a69a", width=2),
        fill="tozeroy", fillcolor="rgba(38, 166, 154, 0.1)",
    ), row=1, col=1)

    # Drawdown
    peak = equity_curve.expanding().max()
    dd = ((equity_curve - peak) / peak) * 100

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", name="Drawdown",
        line=dict(color="#ef5350", width=1),
        fill="tozeroy", fillcolor="rgba(239, 83, 80, 0.2)",
    ), row=2, col=1)

    fig.update_layout(
        title=title, height=500,
        template="plotly_dark",
    )
    return fig


def plot_trade_distribution(trades):
    """Plot PnL distribution histogram."""
    pnl_list = [t.pnl_pct for t in trades]
    colors = ["#26a69a" if p > 0 else "#ef5350" for p in pnl_list]

    fig = go.Figure(go.Histogram(
        x=pnl_list, nbinsx=50,
        marker_color="#42a5f5",
        name="Trade Returns",
    ))

    fig.update_layout(
        title="Trade Return Distribution (%)",
        xaxis_title="Return %",
        yaxis_title="Count",
        height=350,
        template="plotly_dark",
    )
    return fig


# ── Page: Dashboard ────────────────────────────────────────────────────────────

def page_dashboard():
    st.title("📊 Trading Dashboard")
    st.markdown("Overview of latest backtest results and system status.")

    results = st.session_state.backtest_results
    if not results:
        st.info("No backtest results yet. Go to the **Backtest** page to run one.")
        return

    # Summary metrics
    latest = results[-1]
    metrics = compute_metrics(latest)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return", f"{metrics['total_return_pct']:+.2f}%")
    col2.metric("Trades", metrics["total_trades"])
    col3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    col4.metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
    col5.metric("Max DD", f"-{metrics['max_drawdown_pct']:.2f}%")

    # Equity curve
    if latest.equity_curve is not None:
        st.plotly_chart(
            plot_equity_curve(latest.equity_curve, f"Equity — {latest.strategy_name}"),
            use_container_width=True,
        )

    # Trade log
    if latest.trades:
        st.subheader("Recent Trades")
        trade_data = [{
            "Entry": t.entry_time,
            "Exit": t.exit_time,
            "Side": t.side,
            "Entry $": round(t.entry_price, 6),
            "Exit $": round(t.exit_price, 6),
            "PnL": round(t.pnl, 4),
            "PnL %": round(t.pnl_pct, 3),
            "Reason": t.exit_reason,
        } for t in latest.trades[-50:]]
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True)


# ── Page: Backtest ─────────────────────────────────────────────────────────────

def page_backtest():
    st.title("🧪 Backtest Engine")

    all_strats = get_all_strategies()

    col1, col2, col3 = st.columns(3)

    with col1:
        strategy_name = st.selectbox("Strategy", list(all_strats.keys()))

    with col2:
        # Build symbol list
        symbols = MAJOR_PAIRS + STABLECOIN_PAIRS
        symbol = st.selectbox("Symbol", symbols)

    with col3:
        interval = st.selectbox("Interval", list(KLINE_INTERVALS.keys()),
                                 index=list(KLINE_INTERVALS.keys()).index("1h"))

    col_d1, col_d2, col_capital = st.columns(3)
    with col_d1:
        start_date = st.date_input("Start Date",
                                    datetime.now() - timedelta(days=90))
    with col_d2:
        end_date = st.date_input("End Date", datetime.now())
    with col_capital:
        capital = st.number_input("Capital ($)", value=10000, step=1000)

    if st.button("▶ Run Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Running {strategy_name} on {symbol} ({interval})..."):
            try:
                strategy = all_strats[strategy_name]()
                fetcher = st.session_state.fetcher

                df = fetcher.fetch_klines_cached(
                    symbol, interval,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )

                if df.empty:
                    st.error("No data returned. Check the symbol and date range.")
                    return

                engine = BacktestEngine(
                    initial_capital=capital,
                    commission_pct=0.0,  # Zero-fee pairs
                    risk_manager=RiskManager(),
                )
                result = engine.run(strategy, df, symbol, interval)
                st.session_state.backtest_results.append(result)

                metrics = compute_metrics(result)

                # KPI Row
                st.markdown("---")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Return", f"{metrics['total_return_pct']:+.2f}%")
                c2.metric("Trades", metrics["total_trades"])
                c3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                c4.metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
                c5.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                c6.metric("Max DD", f"-{metrics['max_drawdown_pct']:.2f}%")

                # Charts
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📈 Chart + Signals", "💰 Equity Curve", "📊 Distribution", "📋 Report"])

                with tab1:
                    signals_df = strategy.generate_signals(df.copy())
                    fig = plot_candlestick(df, f"{symbol} — {strategy_name}", signals_df)
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    if result.equity_curve is not None:
                        st.plotly_chart(
                            plot_equity_curve(result.equity_curve),
                            use_container_width=True,
                        )

                with tab3:
                    if result.trades:
                        st.plotly_chart(
                            plot_trade_distribution(result.trades),
                            use_container_width=True,
                        )

                with tab4:
                    st.code(format_report(metrics), language="text")

                # Trade Log
                if result.trades:
                    st.subheader("Trade Log")
                    trade_data = [{
                        "Entry Time": t.entry_time,
                        "Exit Time": t.exit_time,
                        "Side": t.side,
                        "Entry": round(t.entry_price, 8),
                        "Exit": round(t.exit_price, 8),
                        "Qty": round(t.quantity, 6),
                        "PnL $": round(t.pnl, 4),
                        "PnL %": round(t.pnl_pct, 3),
                        "Reason": t.exit_reason,
                    } for t in result.trades]
                    st.dataframe(pd.DataFrame(trade_data), use_container_width=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                raise e


# ── Page: Strategy Comparison ──────────────────────────────────────────────────

def page_comparison():
    st.title("📊 Strategy Comparison")

    results = st.session_state.backtest_results
    if len(results) < 2:
        st.info("Run at least 2 backtests from the Backtest page to compare strategies.")
        return

    # Build comparison table
    comp_df = compare_results(results)
    st.dataframe(comp_df, use_container_width=True)

    # Overlay equity curves
    st.subheader("Equity Curves Overlay")
    fig = go.Figure()
    for r in results:
        if r.equity_curve is not None:
            label = f"{r.strategy_name} ({r.symbol} {r.interval})"
            fig.add_trace(go.Scatter(
                x=r.equity_curve.index,
                y=r.equity_curve.values,
                mode="lines",
                name=label,
            ))

    fig.update_layout(
        title="Strategy Equity Comparison",
        height=500,
        template="plotly_dark",
        yaxis_title="Equity ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        fig_ret = go.Figure(go.Bar(
            x=comp_df["Strategy"] + " " + comp_df["Symbol"],
            y=comp_df["Return %"],
            marker_color=["#26a69a" if v > 0 else "#ef5350" for v in comp_df["Return %"]],
        ))
        fig_ret.update_layout(title="Return %", template="plotly_dark", height=350)
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        fig_wr = go.Figure(go.Bar(
            x=comp_df["Strategy"] + " " + comp_df["Symbol"],
            y=comp_df["Win Rate %"],
            marker_color="#42a5f5",
        ))
        fig_wr.update_layout(title="Win Rate %", template="plotly_dark", height=350)
        st.plotly_chart(fig_wr, use_container_width=True)


# ── Page: Live Monitor ─────────────────────────────────────────────────────────

def page_live_monitor():
    st.title("📡 Live Monitor")
    st.markdown("Real-time price monitoring and signal alerts.")

    col1, col2 = st.columns(2)
    with col1:
        symbols = MAJOR_PAIRS + STABLECOIN_PAIRS
        symbol = st.selectbox("Symbol", symbols, key="live_symbol")
    with col2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h"],
                                 key="live_interval")

    if st.button("📡 Fetch Latest Data", use_container_width=True):
        with st.spinner("Fetching..."):
            fetcher = st.session_state.fetcher
            df = fetcher.fetch_klines(symbol, interval, limit=200)

            if df.empty:
                st.error("No data.")
                return

            # Chart
            fig = plot_candlestick(df, f"{symbol} — Live ({interval})")
            st.plotly_chart(fig, use_container_width=True)

            # Price stats
            latest = df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${latest['close']:.8f}")
            c2.metric("24h Volume", f"{latest['volume']:,.0f}")

            change_1h = ((df["close"].iloc[-1] / df["close"].iloc[-2]) - 1) * 100
            c3.metric("Change (1 bar)", f"{change_1h:+.4f}%")

            high_low_range = ((df["high"].max() - df["low"].min()) / df["low"].min()) * 100
            c4.metric("Range", f"{high_low_range:.2f}%")

            # Quick signals from all strategies
            st.subheader("Strategy Signals (Current Bar)")
            all_strats = get_all_strategies()
            signal_rows = []

            for name, strat_cls in all_strats.items():
                try:
                    strat = strat_cls()
                    sig_df = strat.generate_signals(df.copy())
                    last_sig = int(sig_df["signal"].iloc[-1])
                    conf = sig_df["confidence"].iloc[-1] if "confidence" in sig_df else 0

                    signal_rows.append({
                        "Strategy": name,
                        "Signal": {1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪ HOLD"}.get(last_sig, "⚪ HOLD"),
                        "Confidence": f"{conf:.2f}" if conf else "-",
                    })
                except Exception:
                    signal_rows.append({
                        "Strategy": name,
                        "Signal": "⚠ Error",
                        "Confidence": "-",
                    })

            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True)


# ── Page: Data Explorer ────────────────────────────────────────────────────────

def page_sweep_results():
    """Display sweep CSV results with interactive charts and filters."""
    st.title("📋 Sweep Results Explorer")

    from config.settings import LOG_DIR
    import glob

    csv_files = sorted(glob.glob(str(LOG_DIR / "*.csv")), reverse=True)
    if not csv_files:
        st.info("No sweep result CSVs found in logs/. Run a sweep script first.")
        return

    csv_names = [os.path.basename(f) for f in csv_files]
    selected = st.selectbox("Choose a sweep result", csv_names)
    csv_path = csv_files[csv_names.index(selected)]
    df = pd.read_csv(csv_path)

    if df.empty:
        st.warning("CSV is empty.")
        return

    st.caption(f"Loaded **{len(df)}** rows from `{selected}`")

    # ── Filters ────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        if "Strategy" in df.columns:
            strats = st.multiselect("Strategies", sorted(df["Strategy"].unique()),
                                     default=sorted(df["Strategy"].unique()))
            df = df[df["Strategy"].isin(strats)]
    with col_f2:
        if "Symbol" in df.columns:
            syms = st.multiselect("Symbols", sorted(df["Symbol"].unique()),
                                   default=sorted(df["Symbol"].unique()))
            df = df[df["Symbol"].isin(syms)]
    with col_f3:
        if "Interval" in df.columns:
            ivs = st.multiselect("Intervals", sorted(df["Interval"].unique()),
                                  default=sorted(df["Interval"].unique()))
            df = df[df["Interval"].isin(ivs)]

    if df.empty:
        st.warning("No data matches the filters.")
        return

    # ── KPI summary ────────────────────────────────────────────────────────
    st.markdown("---")
    ret_col = "Return %" if "Return %" in df.columns else None
    sharpe_col = "Sharpe" if "Sharpe" in df.columns else None
    trades_col = "Trades" if "Trades" in df.columns else None

    if ret_col:
        profitable = df[df[ret_col] > 0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total combos", len(df))
        c2.metric("Profitable", f"{len(profitable)} ({len(profitable)/len(df)*100:.0f}%)")
        c3.metric("Avg Return", f"{df[ret_col].mean():+.4f}%")
        if sharpe_col:
            c4.metric("Avg Sharpe", f"{df[sharpe_col].mean():.3f}")
        if trades_col:
            c5.metric("Avg Trades", f"{df[trades_col].mean():.1f}")

    # ── Top/bottom tables ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🏆 Top 30 Sharpe", "💰 Top 30 Return", "📉 Bottom 10",
         "📊 Aggregated", "📄 Full Table"])

    show_cols = [c for c in ["Strategy","Symbol","Interval","Trades","Win Rate %",
                              "Return %","Max DD %","Sharpe","Sortino",
                              "Profit Factor","Avg Trade %"] if c in df.columns]

    with tab1:
        if sharpe_col:
            st.dataframe(df.nlargest(30, sharpe_col)[show_cols],
                          use_container_width=True, hide_index=True)

    with tab2:
        if ret_col:
            st.dataframe(df.nlargest(30, ret_col)[show_cols],
                          use_container_width=True, hide_index=True)

    with tab3:
        if ret_col:
            st.dataframe(df.nsmallest(10, ret_col)[show_cols],
                          use_container_width=True, hide_index=True)

    with tab4:
        agg_tabs = st.tabs(["By Strategy", "By Symbol", "By Interval"])
        agg_cols = {c: "mean" for c in [ret_col, sharpe_col, "Win Rate %",
                                          "Max DD %", "Trades", "Profit Factor"]
                     if c and c in df.columns}
        with agg_tabs[0]:
            if "Strategy" in df.columns and agg_cols:
                agg = df.groupby("Strategy").agg(agg_cols).round(3)
                agg = agg.sort_values(sharpe_col or ret_col, ascending=False)
                st.dataframe(agg, use_container_width=True)

                # Bar chart — avg return by strategy
                if ret_col:
                    fig = go.Figure(go.Bar(
                        x=agg.index, y=agg[ret_col],
                        marker_color=["#26a69a" if v > 0 else "#ef5350"
                                       for v in agg[ret_col]],
                    ))
                    fig.update_layout(title="Avg Return % by Strategy",
                                       template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

        with agg_tabs[1]:
            if "Symbol" in df.columns and agg_cols:
                agg = df.groupby("Symbol").agg(agg_cols).round(3)
                agg = agg.sort_values(ret_col or sharpe_col, ascending=False)
                st.dataframe(agg, use_container_width=True)

        with agg_tabs[2]:
            if "Interval" in df.columns and agg_cols:
                agg = df.groupby("Interval").agg(agg_cols).round(3)
                agg = agg.sort_values(ret_col or sharpe_col, ascending=False)
                st.dataframe(agg, use_container_width=True)

    with tab5:
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_out = df.to_csv(index=False)
        st.download_button("📥 Download filtered CSV", csv_out,
                            f"filtered_{selected}", "text/csv",
                            use_container_width=True)

    # ── Heatmap: Strategy × Symbol ────────────────────────────────────────
    if ret_col and "Strategy" in df.columns and "Symbol" in df.columns:
        st.markdown("---")
        st.subheader("Return % Heatmap  (Strategy × Symbol)")
        pivot = df.pivot_table(values=ret_col, index="Strategy",
                                columns="Symbol", aggfunc="mean")
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(pivot.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(title="Avg Return % (green=profit, red=loss)",
                           template="plotly_dark", height=max(400, len(pivot)*25))
        st.plotly_chart(fig, use_container_width=True)


def page_data_explorer():
    st.title("🔍 Data Explorer")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbols = MAJOR_PAIRS + STABLECOIN_PAIRS + ["Custom..."]
        sel = st.selectbox("Symbol", symbols, key="de_symbol")
        if sel == "Custom...":
            sel = st.text_input("Enter symbol", "BTCUSDC")
    with col2:
        interval = st.selectbox("Interval", list(KLINE_INTERVALS.keys()),
                                 key="de_interval")
    with col3:
        limit = st.number_input("Candles", value=500, min_value=50, max_value=5000)

    if st.button("📥 Fetch Data", use_container_width=True):
        with st.spinner("Fetching..."):
            fetcher = st.session_state.fetcher
            df = fetcher.fetch_klines(sel, interval, limit=limit)

            if df.empty:
                st.error("No data returned.")
                return

            st.plotly_chart(
                plot_candlestick(df, f"{sel} — {interval}"),
                use_container_width=True,
            )

            # Stats
            st.subheader("Statistics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(df.describe())
            with col_b:
                st.write(f"**Period:** {df.index[0]} → {df.index[-1]}")
                st.write(f"**Candles:** {len(df)}")
                st.write(f"**Price range:** ${df['low'].min():.8f} — ${df['high'].max():.8f}")
                daily_return = df["close"].pct_change().dropna()
                st.write(f"**Avg bar return:** {daily_return.mean() * 100:.4f}%")
                st.write(f"**Volatility (std):** {daily_return.std() * 100:.4f}%")

            # Download
            csv = df.to_csv()
            st.download_button("📥 Download CSV", csv, f"{sel}_{interval}.csv",
                                "text/csv", use_container_width=True)


# ── Route ──────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    page_dashboard()
elif page == "Backtest":
    page_backtest()
elif page == "Strategy Comparison":
    page_comparison()
elif page == "Sweep Results":
    page_sweep_results()
elif page == "Live Monitor":
    page_live_monitor()
elif page == "Data Explorer":
    page_data_explorer()
